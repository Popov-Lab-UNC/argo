# argo/gen_models/api.py

import os
import requests
import json
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Callable, Optional, Dict, Any, Literal, Union
from dataclasses import dataclass, field
import torch

# Suppress warnings from libraries for a cleaner user experience
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.generation.configuration_utils")

# Backend model imports
from argo.gen_models.f_rag.model import f_RAG
from argo.gen_models.gem.model import GEM

# --- A structured way to define a generation task ---
@dataclass
class GenerationTask:
    """
    A structured configuration for a molecular generation task.
    This allows for a unified `generate()` entry point across all models.
    """
    mode: Literal[
        'de_novo',
        'biased_generation',
        'scaffold_decoration',
        'linker_generation',
        'property_optimization'
    ]
    # Inputs for different modes
    scaffold: Optional[str] = None
    fragments: Optional[List[str]] = None
    # For MolMiM
    seed_smiles: Optional[Union[str, List[str]]] = None
    labels: Optional[List[float]] = None
    # For optimization tasks
    objective: Optional[Union[str, Callable[[List[str]], List[float]]]] = None
    # General configuration for samples, epochs, etc.
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

    def de_novo(self, 
                n_samples: int = 10, 
                n_trials: int = 1,
                sanitize: bool = True
    ) -> List[str]:
        """Generates molecules from scratch."""
        return self.designer.de_novo_generation(n_samples_per_trial=n_samples, 
                                                n_trials=n_trials, 
                                                sanitize=sanitize)

    def scaffold_decoration(self, 
                          scaffold: str, 
                          n_samples: int = 10, 
                          n_trials: int = 1,
                          sanitize: bool = True,
                          random_seed: int = 42
    ) -> List[str]:
        """Generates molecules by decorating a given scaffold."""
        return self.designer.scaffold_decoration(scaffold=scaffold, 
                                                 n_samples_per_trial=n_samples, 
                                                 n_trials=n_trials, 
                                                 sanitize=sanitize, 
                                                 random_seed=random_seed)

    def linker_generation(self, 
                       fragment1: str, 
                       fragment2: str, 
                       n_samples: int = 10, 
                       n_trials: int = 1, 
                       sanitize: bool = True,
                       random_seed: int = 42
    ) -> List[str]:
        """Generates linkers to connect two molecular fragments."""
        return self.designer.linker_generation(fragment1, 
                                               fragment2, 
                                               n_samples_per_trial=n_samples, 
                                               n_trials=n_trials, 
                                               sanitize=sanitize,
                                               random_seed=random_seed)

    def generate(self, task: GenerationTask) -> List[str]:
        if task.mode == 'de_novo':
            return self.de_novo(**task.config)
        elif task.mode == 'scaffold_decoration':
            if not task.scaffold:
                raise ValueError("A 'scaffold' must be provided for this task.")
            return self.scaffold_decoration(task.scaffold, **task.config)
        elif task.mode == 'linker_generation':
            if not task.fragments or len(task.fragments) != 2:
                raise ValueError("A list of two 'fragments' must be provided for this task.")
            return self.linker_generation(task.fragments[0], task.fragments[1], **task.config)
        else:
            raise NotImplementedError(f"SAFE-GPT does not support the '{task.mode}' generation mode.")


class MolMIMGenerator(BaseGenerator):
    """
    Interface for the MolMiM API for property-guided molecule optimization.
    """
    def __init__(self, api_token: str):
        if not api_token:
            raise ValueError("An 'api_token' is required for the MolMIM model.")
        super().__init__(use_cuda=False)
        self.api_token = api_token
        self.generate_url = "https://health.api.nvidia.com/v1/biology/nvidia/molmim/generate"
        self.sample_url = "https://health.api.nvidia.com/v1/biology/nvidia/molmim/sample"

    def _call_api(self, payload: Dict[str, Any], url: str) -> List[str]:
        """Internal method to handle the API request."""
        headers = {"Authorization": f"Bearer {self.api_token}", "Accept": "application/json"}
        session = requests.Session()
        response = session.post(url, headers=headers, json=payload)
        response.raise_for_status()
        response_body = response.json()
        molecules = json.loads(response_body['molecules'])
        return [mol['sample'] for mol in molecules]

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
        return self._call_api(payload, url=self.generate_url)
    
    '''
    def sample(self, 
               seed_smiles: str, 
               n_samples: int = 10, 
               beam_size: int = 1, 
               scaled_radius: float = 0.7
    ) -> List[str]:
        """Samples molecules from the MolMiM API using the 'sample' endpoint."""
        
        # Validate parameters are within allowed ranges
        if n_samples < 1 or n_samples > 10:
            raise ValueError("n_samples must be between 1 and 10")
        if beam_size < 1 or beam_size > 10:
            raise ValueError("beam_size must be between 1 and 10") 
        if scaled_radius < 0.0 or scaled_radius > 2.0:
            raise ValueError("scaled_radius must be between 0.0 and 2.0")

        payload = {
            "sequences": [seed_smiles],
            "num_molecules": n_samples,
            "beam_size": beam_size,
            "scaled_radius": scaled_radius
        }
        return self._call_api(payload, url=self.sample_url)
    '''

    def generate(self, task: GenerationTask) -> List[str]:
        if not task.seed_smiles or not isinstance(task.seed_smiles, str):
            raise ValueError("A single 'seed_smiles' string must be provided for MolMiM.")
        
        if task.mode == 'property_optimization':
            return self.optimize(seed_smiles=task.seed_smiles, algorithm='CMA-ES', **task.config)
        elif task.mode == 'biased_generation':
            return self.optimize(seed_smiles=task.seed_smiles, algorithm='none', **task.config)
            #return self.sample(seed_smiles=task.seed_smiles, **task.config)
        else:
            raise NotImplementedError(f"MolMiM does not support the '{task.mode}' generation mode.")


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
        if task.mode == 'de_novo':
            if self.finetuned:
                print("GEM is finetuned. De novo generation is biased generation.")
            return self.gem.generate(**task.config)
        elif task.mode == 'biased_generation':
            if not task.seed_smiles:
                raise ValueError("'seed_smiles', list of SMILES for biasing, must be provided for this task.")
            if not isinstance(task.seed_smiles, list):
                task.seed_smiles = [task.seed_smiles]
            self.gem.fine_tune(task.seed_smiles, lr=1e-5, n_epochs=10, save_path=None)
            self.finetuned = True
            return self.gem.generate(**task.config)
        else:
            raise NotImplementedError(f"GEM does not support the '{task.mode}' generation mode.")

class FRAGGenerator(BaseGenerator):
    """
    Interface for the f-RAG model, an evolutionary algorithm for de novo design.
    """
    def __init__(self, injection_model_path: str = None, vocab_path: str = None, frag_population_size: int = 50, mol_population_size: int = 100, min_frag_size: int = 1, max_frag_size: int = 15, min_mol_size: int = 10, max_mol_size: int = 100, mutation_rate: float = 0.01):
        super().__init__()
        self.frag = f_RAG(
            injection_model_path=injection_model_path,
            vocab_path=vocab_path,
            frag_population_size=frag_population_size,
            mol_population_size=mol_population_size,
            min_frag_size=min_frag_size,
            max_frag_size=max_frag_size,
            min_mol_size=min_mol_size,
            max_mol_size=max_mol_size,
            mutation_rate=mutation_rate
        )

    def generate(self, task: GenerationTask) -> list:
        if task.mode == 'property_optimization':
            config = task.config or {}
            num_to_generate = config.get('num_to_generate', 10)
            random_seed = config.get('random_seed', 42)
            return self.frag.generate(num_to_generate=num_to_generate, random_seed=random_seed)
        else:
            raise NotImplementedError(f"f-RAG does not support the '{task.mode}' generation mode.")


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
                  - for 'f-rag': vocab_path (required), and other population/size params.
    
    Returns:
        An instance of the appropriate generator class.
    """
    model_type = model_type.lower()
    if model_type == 'safegpt':
        return SAFEGenerator(**kwargs)
    if model_type == 'molmim':
        return MolMIMGenerator(**kwargs)
    if model_type == 'gem':
        return GEMGenerator(**kwargs)
    if model_type == 'f-rag':
        return FRAGGenerator(**kwargs)
    
    raise ValueError(f"Unknown model type: {model_type}. Must be one of ['safegpt', 'molmim', 'gem', 'f-rag']")