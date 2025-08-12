# argo/gen_models/__init__.py

import os, subprocess, shutil, time
import requests
import json
import pandas as pd
import logging
import random
from abc import ABC, abstractmethod
from typing import List, Callable, Optional, Dict, Any, Literal, Union
from dataclasses import dataclass, field
import torch
from rdkit import Chem

# Suppress warnings from libraries for a cleaner user experience
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.generation.configuration_utils")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pandas.core.dtypes.cast")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="numpy")

# Backend model imports
from argo.gen_models.f_rag.model import f_RAG
from argo.gen_models.gem.model import GEM

def validate_smiles(smiles: str) -> bool:
    if not isinstance(smiles, str) or not smiles.strip():
        return False
    
    smiles = smiles.strip()

    try:
        # Parse SMILES to RDKit molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        
        # Convert back to SMILES to ensure consistency
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        if not canonical_smiles:
            return False

        return True
    except Exception:
        return False

# --- A structured way to define a generation task ---
@dataclass
class GenerationTask:
    """
    A structured configuration for a molecular generation task.
    This allows for a unified `generate()` entry point across all models.

    Attributes:
        mode: The generation mode to use. This is a required field.
            - 'de_novo': Generate molecules from scratch.
            - 'biased_generation': Generate molecules biased towards a certain chemical space, defined by `seed_smiles`.
            - 'scaffold_decoration': Generate molecules by decorating a given `scaffold`.
            - 'linker_generation': Generate a linker to connect two `fragments`.
            - 'property_optimization': Optimize a `seed_smiles` towards a certain `objective`.
        scaffold: The scaffold to decorate. Required for 'scaffold_decoration' mode.
                  Example: '[*]c1ccccc1[*]'
        fragments: A list of two fragments to be linked. Required for 'linker_generation' mode.
                   Example: ['[*]C', '[*]N']
        seed_smiles: A single SMILES string or a list of SMILES strings.
                     - For 'biased_generation' with GEM, this is a list of SMILES to fine-tune the model on.
                     - For 'property_optimization' with MolMIM, this is the starting molecule for optimization.
        labels: A list of labels for the `seed_smiles`. Used in some fine-tuning tasks.
        objective: The objective function to optimize for. Required for 'property_optimization' mode.
                   Can be a string (e.g., 'QED' or 'plogP' for MolMIM) or a callable function
                   that takes a list of SMILES and returns a list of scores.
        config: A dictionary of additional configuration options for the generation task.
                This can include parameters like `n_samples`, `batch_size`, `epochs`, etc.
    """
    mode: Literal[
        'de_novo',
        'biased_generation',
        'scaffold_decoration',
        'linker_generation',
        'property_optimization'
    ]
    scaffold: Optional[Union[str, List[str]]] = None
    fragments: Optional[List[str]] = None
    seed_smiles: Optional[Union[str, List[str]]] = None
    objective: Optional[Union[str, Callable[[List[str]], List[float]]]] = None
    config: Dict[str, Any] = field(default_factory=dict)


class BaseGenerator(ABC):
    """Abstract base class for all generation models."""
    def __init__(self, use_cuda: bool = True):
        self.use_cuda = use_cuda
        self.model = None

    @abstractmethod
    def generate(self, task: GenerationTask) -> Union[List[str], pd.DataFrame]:
        """
        A unified entry point for all generation tasks.
        """
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}"


class SAFEGenerator(BaseGenerator):
    """
    Interface for the SAFE-GPT model for de novo, scaffold, and linker generation.
    """
    def __init__(self, model_path: Optional[str] = None, use_cuda: bool = True):
        super().__init__(use_cuda=use_cuda)
        try:
            import safe as sf
            from safe.trainer.model import SAFEDoubleHeadsModel
            from safe.tokenizer import SAFETokenizer
        except ImportError:
            raise ImportError("The 'safe-mol' package must be installed for SAFE-GPT usage. Please run 'pip install safe-mol'.")

        device = 'cuda' if self.use_cuda else 'cpu'
        if model_path:
            tokenizer = SAFETokenizer.from_pretrained(model_path)
            model = SAFEDoubleHeadsModel.from_pretrained(model_path)
            designer = sf.SAFEDesign(model=model, tokenizer=tokenizer)
            designer.model = designer.model.to(device)
        else:
            designer = sf.SAFEDesign.load_default(device=device, verbose=False)
        self.designer = designer

    def de_novo(self, batch_size: int = 100, **kwargs) -> List[str]:
        """Generates a batch of molecules from scratch."""
        return self.designer.de_novo_generation(n_samples_per_trial=batch_size, n_trials=1, **kwargs)

    def scaffold_decoration(self, scaffold: str, batch_size: int = 100, **kwargs) -> List[str]:
        """Generates a batch of molecules by decorating a scaffold."""
        return self.designer.scaffold_decoration(scaffold=scaffold, n_samples_per_trial=batch_size, n_trials=1, **kwargs)

    def linker_generation(self, fragment1: str, fragment2: str, batch_size: int = 100, **kwargs) -> List[str]:
        """Generates a batch of molecules by linking two fragments."""
        return self.designer.linker_generation(fragment1, fragment2, n_samples_per_trial=batch_size, n_trials=1, **kwargs)

    def generate(self, task: GenerationTask) -> List[str]:
        config = task.config or {}
        n_samples = config.get('n_samples', 1000)
        batch_size = config.get('batch_size', 100)
        kwargs = {k: v for k, v in config.items() if k not in ['n_samples', 'batch_size', 'processing_mode']}

        valid_smiles = []
        generated_count = 0

        if task.mode == 'de_novo':
            while len(valid_smiles) < n_samples:
                batch = self.de_novo(batch_size=batch_size, **kwargs)
                if not batch:
                    logging.warning("SAFEGenerator.de_novo returned no SMILES. Stopping generation.")
                    break
                generated_count += len(batch)

                for smi in batch:
                    if validate_smiles(smi):
                        valid_smiles.append(smi)
                        if len(valid_smiles) >= n_samples:
                            break
            
        elif task.mode == 'scaffold_decoration':
            if not task.scaffold:
                raise ValueError("A 'scaffold' must be provided for this task.")

            scaffolds = [task.scaffold] if isinstance(task.scaffold, str) else task.scaffold
            processing_mode = config.get('processing_mode', 'iterate')
            scaffold_idx = 0

            while len(valid_smiles) < n_samples:
                if processing_mode == 'iterate':
                    scaffold = scaffolds[scaffold_idx % len(scaffolds)]
                    scaffold_idx += 1
                elif processing_mode == 'sample':
                    scaffold = random.choice(scaffolds)

                batch = self.scaffold_decoration(scaffold, batch_size=batch_size, **kwargs)

                if not batch:
                    logging.warning(f"SAFEGenerator.scaffold_decoration returned no SMILES for scaffold {scaffold}.")
                    continue

                generated_count += len(batch)

                for smi in batch:
                    if validate_smiles(smi):
                        valid_smiles.append(smi)
                        if len(valid_smiles) >= n_samples:
                            break

        elif task.mode == 'linker_generation':
            if not task.fragments or len(task.fragments) < 2:
                raise ValueError("A list of at least two 'fragments' must be provided for this task.")
            
            processing_mode = config.get('processing_mode', 'iterate')

            fragment_pairs = []
            if len(task.fragments) >= 2:
                for i, frag1 in enumerate(task.fragments):
                    for j, frag2 in enumerate(task.fragments[i+1:], i+1):
                        fragment_pairs.append([frag1, frag2])

            if not fragment_pairs:
                raise ValueError("Could not form any fragment pairs from the provided fragments.")

            pair_idx = 0

            while len(valid_smiles) < n_samples:
                if processing_mode == 'iterate':
                    frag1, frag2 = fragment_pairs[pair_idx % len(fragment_pairs)]
                    pair_idx += 1
                elif processing_mode == 'sample':
                    frag1, frag2 = random.choice(fragment_pairs)

                batch = self.linker_generation(frag1, frag2, batch_size=batch_size, **kwargs)

                if not batch:
                    logging.warning(f"SAFEGenerator.linker_generation returned no SMILES for fragments {frag1}, {frag2}.")
                    continue

                generated_count += len(batch)
                
                for smi in batch:
                    if validate_smiles(smi):
                        valid_smiles.append(smi)
                        if len(valid_smiles) >= n_samples:
                            break
        else:
            raise NotImplementedError(f"SAFE-GPT does not support the '{task.mode}' generation mode.")

        if generated_count > 0:
            validity = len(valid_smiles) / generated_count * 100
            logging.info(f"SAFEGenerator.{task.mode}: validity: {validity:.2f}% ({len(valid_smiles)} valid SMILES from {generated_count} generated)")

        return valid_smiles[:n_samples]

class MolMIMClient(BaseGenerator):
    """
    A client for an already running MolMIM container service.

    This class connects to a specified server address, verifies the service is
    healthy, and provides methods to interact with the MolMIM API.

    The server process must be started and managed separately.
    """

    def __init__(self, server_address: str):
        """
        Initializes the client and connects to the specified server address.

        Args:
            server_address (str): The address of the running MolMIM service,
                                  formatted as "hostname:port".
        """
        super().__init__(use_cuda=True)

        if not server_address or ":" not in server_address:
            raise ValueError("Invalid server_address format. Expected 'hostname:port'.")

        self.base_url = f"http://{server_address}"
        logging.info(f"Attempting to connect to MolMIM service at: {self.base_url}")

        # Immediately perform a health check to ensure the server is ready.
        self._health_check()

    def _health_check(self):
        """
        Verifies that the server is running and ready to accept requests.
        Raises an error if the health check fails.
        """
        health_endpoint = f"{self.base_url}/v1/health/ready"
        try:
            response = requests.get(health_endpoint, timeout=5)

            # Check for a successful HTTP status code
            response.raise_for_status() # Raises HTTPError for 4xx/5xx responses

            # Check the content of the response
            if response.json().get("status") == "ready":
                 logging.info("Connection successful. MolMIM service is ready.")
            else:
                raise ConnectionError(
                    f"Service at {self.base_url} is running but not ready. "
                    f"Response: {response.text}"
                )

        except requests.exceptions.RequestException as e:
            # Catches connection errors, timeouts, and HTTP errors
            raise ConnectionError(
                f"Failed to connect or verify server at {health_endpoint}. "
                f"Please ensure the server is running and accessible. Error: {e}"
            ) from e

    def _call_api(self, payload: Dict[str, Any], endpoint: str) -> List[str]:
        """Internal method to handle the API request to the server."""
        url = f"{self.base_url}{endpoint}"
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        session = requests.Session()
        response = session.post(url, headers=headers, json=payload)
        response.raise_for_status()

        if response.status_code != 200:
            raise requests.HTTPError(
                f"MolMIM API Error: {response.status_code} {response.text}"
            )

        response_body = response.json()

        # Extract molecules from the 'generated' key
        if 'generated' in response_body:
            molecules = response_body['generated']
            return [mol['smiles'] for mol in molecules]
        else:
            raise ValueError(f"Unexpected API response format. Expected 'generated' key, got: {list(response_body.keys())}")

    def optimize(self,
                 seed_smiles: str,
                 algorithm: str = 'CMA-ES',
                 iterations: int = 10,
                 min_similarity: float = 0.7,
                 minimize: bool = False,
                 n_samples: int = 10,
                 particles: int = 30,
                 property_name: str = 'QED',
                 scaled_radius: float = 1.0
    ) -> List[str]:
        """Optimizes a starting molecule towards a desired property."""

        if algorithm not in ['CMA-ES', 'none']:
            raise ValueError("algorithm must be either 'CMA-ES' or 'none'")
        if property_name not in ['QED', 'plogP']:
            raise ValueError("property_name must be either 'QED' or 'plogP'")

        # Validate parameters are within allowed ranges
        if iterations < 1 or iterations > 1000:
            raise ValueError("iterations must be between 1 and 1000")
        if min_similarity < 0.0 or min_similarity > 0.7:
            raise ValueError("min_similarity must be between 0.0 and 0.7")
        if n_samples < 1 or n_samples > 1000:
            raise ValueError("n_samples must be between 1 and 1000")
        if particles < 2 or particles > 3000:
            raise ValueError("particles must be between 2 and 3000")
        if scaled_radius < 0.0 or scaled_radius > 2.0:
            raise ValueError("scaled_radius must be between 0.0 and 2.0")

        payload = {
                   "smi": seed_smiles,
                   "algorithm": algorithm,
                   "iterations": iterations,
                   "min_similarity": min_similarity,
                   "minimize": minimize,
                   "num_molecules": n_samples,
                   "particles": particles,
                   "property_name": property_name,
                   "scaled_radius": scaled_radius
        }
        return self._call_api(payload, endpoint="/generate")

    def generate(self, task: GenerationTask) -> List[str]:
        """Generate molecules using MolMiM with support for property optimization and biased generation."""
        if not task.seed_smiles:
            raise ValueError("A 'seed_smiles' string or list must be provided for MolMiM.")

        config = task.config or {}
        n_samples = config.get('n_samples', 10)
        batch_size = config.get('batch_size', 10)
        processing_mode = config.get('processing_mode', 'iterate')
        objective = task.objective or 'QED'

        seed_smiles_list = [task.seed_smiles] if isinstance(task.seed_smiles, str) else task.seed_smiles
        if not seed_smiles_list:
            raise ValueError("'seed_smiles' list cannot be empty.")

        if task.mode == 'property_optimization':
            algorithm = 'CMA-ES'
        elif task.mode == 'biased_generation':
            algorithm = 'none'
        else:
            raise NotImplementedError(f"MolMiM does not support the '{task.mode}' generation mode.")

        optimize_params = {
            "iterations": config.get('iterations', 10),
            "min_similarity": config.get('min_similarity', 0.7),
            "minimize": config.get('minimize', False),
            "particles": config.get('particles', 30),
            "property_name": objective,
            "scaled_radius": config.get('scaled_radius', 1.0)
        }

        valid_smiles = []
        generated_count = 0
        seed_idx = 0
        while len(valid_smiles) < n_samples:
            if processing_mode == 'iterate':
                seed = seed_smiles_list[seed_idx % len(seed_smiles_list)]
                seed_idx += 1
            elif processing_mode == 'sample':
                seed = random.choice(seed_smiles_list)

            try:
                smiles_batch = self.optimize(
                    seed_smiles=seed,
                    algorithm=algorithm,
                    n_samples=batch_size,
                    **optimize_params
                )
                
                if not smiles_batch:
                    logging.warning(f"MolMiM returned no SMILES for seed: {seed}")
                    continue
                
                generated_count += len(smiles_batch)
                
                for smi in smiles_batch:
                    if validate_smiles(smi):
                        valid_smiles.append(smi)
                        if len(valid_smiles) >= n_samples:
                            break

            except requests.exceptions.RequestException as e:
                logging.error(f"MolMiM API call failed for seed {seed}: {e}")
                continue

        if generated_count > 0:
            validity = len(valid_smiles) / generated_count * 100
            logging.info(f"MolMIMClient.{task.mode}: validity: {validity:.2f}% ({len(valid_smiles)} valid SMILES from {generated_count} generated)")

        return valid_smiles[:n_samples]

class GEMGenerator(BaseGenerator):
    """
    Interface for the GEM model workflow for fine-tuning and generation.
    """
    def __init__(self, model_path: str, use_cuda: bool = True, finetuned: bool = False):
        super().__init__(use_cuda=use_cuda)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"GEM model path not found: {model_path}")
        self.gem = GEM(model_path, device=torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu"))
        self.finetuned = finetuned

    def save(self, save_path: str):
        self.gem.save_checkpoint(save_path)
        return

    def finetune(self, smiles: list, lr: float = 1e-5, n_epochs: int = 10, save_path: str = None):
        self.gem.fine_tune(smiles, lr=lr, n_epochs=n_epochs, save_path=save_path)
        self.finetuned = True
        return

    def generate(self, task: GenerationTask) -> list:
        config = task.config or {}
        n_samples = config.get('n_samples', 1000)
        batch_size = config.get('batch_size', 100)

        if task.mode == 'de_novo':
            if self.finetuned:
                logging.info("GEM is finetuned. De novo generation is biased generation.")
        elif task.mode == 'biased_generation':
            if not task.seed_smiles:
                raise ValueError("'seed_smiles', list of SMILES for biasing, must be provided for this task.")
            if not isinstance(task.seed_smiles, list):
                task.seed_smiles = [task.seed_smiles]
            self.gem.fine_tune(task.seed_smiles, lr=1e-5, n_epochs=10, save_path=None)
            self.finetuned = True
        else:
            raise NotImplementedError(f"GEM does not support the '{task.mode}' generation mode.")

        valid_smiles = []
        generated_count = 0
        while len(valid_smiles) < n_samples:
            smiles_batch = self.gem.generate(n_samples=batch_size, batch_size=batch_size)
            if not smiles_batch:
                logging.warning("GEMGenerator.generate returned no SMILES. Stopping generation.")
                break
            generated_count += len(smiles_batch)

            for smi in smiles_batch:
                if validate_smiles(smi):
                    valid_smiles.append(smi)

        if generated_count > 0:
            validity = len(valid_smiles) / generated_count * 100
            logging.info(f"GEMGenerator.{task.mode}: validity: {validity:.2f}% ({len(valid_smiles)} valid SMILES from {generated_count} generated)")

        return valid_smiles[:n_samples]

class F_RAGGenerator(BaseGenerator):
    """
    Interface for the f-RAG model, an evolutionary algorithm for de novo design.
    """
    def __init__(self, injection_model_path: str, vocab: "str | pd.DataFrame", frag_population_size: int = 50, mol_population_size: int = 100, min_frag_size: int = 1, max_frag_size: int = 15, min_mol_size: int = 10, max_mol_size: int = 100, use_cuda: bool = True):
        super().__init__(use_cuda=use_cuda)
        self.f_rag = f_RAG(
            injection_model_path=injection_model_path,
            vocab=vocab,
            frag_population_size=frag_population_size,
            mol_population_size=mol_population_size,
            min_frag_size=min_frag_size,
            max_frag_size=max_frag_size,
            min_mol_size=min_mol_size,
            max_mol_size=max_mol_size,
            use_cuda=use_cuda
        )

    def generate(self, task: GenerationTask) -> list:
        config = task.config or {}
        generation_fn = None

        if task.mode == "linker_generation":
            n_samples = config.get('n_samples', 10)
            random_seed = config.get('random_seed', 42)
            generation_fn = lambda n: self.f_rag.linker_generation(n_samples=n, random_seed=random_seed)
        elif task.mode == "scaffold_decoration":
            n_samples = config.get('n_samples', 10)
            random_seed = config.get('random_seed', 42)
            generation_fn = lambda n: self.f_rag.scaffold_decoration(n_samples=n, scaffold=task.scaffold, random_seed=random_seed)
        elif task.mode == "property_optimization":
            n_samples = config.get('n_samples', 10)
            if not task.objective:
                raise ValueError("'objective' must be provided for this task.")
            threshold = config.get('threshold', 0.8)
            max_iter = config.get('max_iter', 10)
            higher_is_better = config.get('higher_is_better', True)
            opt_batch_size = config.get('batch_size', 50)
            mutation_rate = config.get('mutation_rate', 0.01)
            init_lg_wt = config.get('init_lg_wt', 0.5)
            init_sd_wt = config.get('init_sd_wt', 0.5)
            init_ga_wt = config.get('init_ga_wt', 0.0)
            generation_fn = lambda n: self.f_rag.optimize(
                n_samples=n, oracle_name=task.objective, threshold=threshold, max_iter=max_iter,
                higher_is_better=higher_is_better, batch_size=opt_batch_size,
                mutation_rate=mutation_rate, init_lg_wt=init_lg_wt, init_sd_wt=init_sd_wt,
                init_ga_wt=init_ga_wt
            )
        else:
            raise NotImplementedError(f"f-RAG does not support the '{task.mode}' generation mode.")

        batch_size = config.get('batch_size', 10)
        valid_smiles = []
        generated_count = 0
        while len(valid_smiles) < n_samples:
            smiles_batch = generation_fn(batch_size)
            if not smiles_batch:
                logging.warning("F_RAGGenerator generation function returned no SMILES. Stopping generation.")
                break
            generated_count += len(smiles_batch)

            for smi in smiles_batch:
                if validate_smiles(smi):
                    valid_smiles.append(smi)

        if generated_count > 0:
            validity = len(valid_smiles) / generated_count * 100
            logging.info(f"F_RAGGenerator.{task.mode}: validity: {validity:.2f}% ({len(valid_smiles)} valid SMILES from {generated_count} generated)")

        return valid_smiles[:n_samples]


def GenerationModel(
    model_type: str,
    **kwargs: Any
) -> BaseGenerator:
    """
    Factory function to instantiate a generative model interface.

    Args:
        model_type: The type of model to load.
                    One of ['safegpt', 'molmim', 'gem', 'f-rag'].
        **kwargs: Arguments specific to each model.
                  - for 'safegpt': model_path (optional), use_cuda (optional)
                  - for 'molmim': api_token (required)
                  - for 'gem': model_path (required), use_cuda (optional), finetuned (optional)
                  - for 'f-rag': injection_model_path (required), vocab (str or pd.DataFrame, required), use_cuda (optional), and other population/size params.

    Returns:
        An instance of the appropriate generator class.
    """
    model_type = model_type.lower()
    if model_type == 'safegpt':
        return SAFEGenerator(**kwargs)
    if model_type == 'molmim':
        #return MolMIMGenerator(**kwargs)
        return MolMIMClient(**kwargs)
    if model_type == 'gem':
        return GEMGenerator(**kwargs)
    if model_type == 'f-rag':
        return F_RAGGenerator(**kwargs)

    raise ValueError(f"Unknown model type: {model_type}. Must be one of ['safegpt', 'molmim', 'gem', 'f-rag']")