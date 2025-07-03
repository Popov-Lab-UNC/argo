# argo/gen_models/api.py

import os
import requests
import json
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Callable, Optional, Dict, Any, Literal, Union
from dataclasses import dataclass, field

# Suppress warnings from libraries for a cleaner user experience
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.generation.configuration_utils")

# Backend model imports
from argo.gen_models.f_rag.optimizer import f_RAG
from argo.gen_models.gem.workflow import run_generation_workflow

# --- A structured way to define a generation task ---
@dataclass
class GenerationTask:
    """
    A structured configuration for a molecular generation task.
    This allows for a unified `generate()` entry point across all models.
    """
    mode: Literal[
        'de_novo',
        'scaffold_decoration',
        'linker_generation',
        'property_optimization',
        'finetune_and_generate'
    ]
    # Inputs for different modes
    scaffold: Optional[str] = None
    fragments: Optional[List[str]] = None
    starting_smiles: Optional[Union[str, List[str]]] = None
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
            raise ImportError("The 'safe-generative-models' package must be installed for SAFE-GPT usage. Please run 'pip install safe-generative-models'.")

        device = 'cuda' if self.use_cuda else 'cpu'
        if model_path:
            tokenizer = SAFETokenizer.from_pretrained(model_path)
            model = SAFEDoubleHeadsModel.from_pretrained(model_path)
            designer = sf.SAFEDesign(model=model, tokenizer=tokenizer)
            designer.model = designer.model.to(device)
        else:
            designer = sf.SAFEDesign.load_default(device=device, verbose=False)
        self.designer = designer

    def de_novo(self, n_samples: int = 12, n_trials: int = 1, sanitize: bool = True) -> List[str]:
        """Generates molecules from scratch."""
        return self.designer.de_novo_generation(n_samples_per_trial=n_samples, n_trials=n_trials, sanitize=sanitize)

    def decorate_scaffold(self, scaffold: str, n_samples: int = 12, n_trials: int = 1, sanitize: bool = True) -> List[str]:
        """Generates molecules by decorating a given scaffold."""
        return self.designer.scaffold_decoration(scaffold=scaffold, n_samples_per_trial=n_samples, n_trials=n_trials, sanitize=sanitize)

    def link_fragments(self, fragment1: str, fragment2: str, n_samples: int = 12, n_trials: int = 1, sanitize: bool = True) -> List[str]:
        """Generates linkers to connect two molecular fragments."""
        return self.designer.linker_generation(fragment1, fragment2, n_samples_per_trial=n_samples, n_trials=n_trials, sanitize=sanitize)

    def generate(self, task: GenerationTask) -> List[str]:
        if task.mode == 'de_novo':
            return self.de_novo(**task.config)
        elif task.mode == 'scaffold_decoration':
            if not task.scaffold:
                raise ValueError("A 'scaffold' must be provided for this task.")
            return self.decorate_scaffold(task.scaffold, **task.config)
        elif task.mode == 'linker_generation':
            if not task.fragments or len(task.fragments) != 2:
                raise ValueError("A list of two 'fragments' must be provided for this task.")
            return self.link_fragments(task.fragments[0], task.fragments[1], **task.config)
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
        self.invoke_url = "https://health.api.nvidia.com/v1/biology/nvidia/molmim/generate"

    def _call_api(self, payload: Dict[str, Any]) -> List[str]:
        """Internal method to handle the API request."""
        headers = {"Authorization": f"Bearer {self.api_token}", "Accept": "application/json"}
        session = requests.Session()
        response = session.post(self.invoke_url, headers=headers, json=payload)
        response.raise_for_status()
        response_body = response.json()
        molecules = json.loads(response_body['molecules'])
        return [mol['sample'] for mol in molecules]

    def optimize(self, starting_smiles: str, property_name: str, num_molecules: int = 10, **api_params) -> List[str]:
        """Optimizes a starting molecule towards a desired property."""
        payload = {"algorithm": api_params.pop('algorithm', 'CMA-ES'), "smi": starting_smiles, "property_name": property_name, "num_molecules": num_molecules, **api_params}
        return self._call_api(payload)

    def generate(self, task: GenerationTask) -> List[str]:
        if task.mode == 'property_optimization':
            if not task.starting_smiles or not isinstance(task.starting_smiles, str):
                raise ValueError("A single 'starting_smiles' string must be provided for this task.")
            if not task.objective or not isinstance(task.objective, str):
                raise ValueError("The 'objective' (property name string) must be provided for this task.")
            return self.optimize(starting_smiles=task.starting_smiles, property_name=task.objective, **task.config)
        else:
            raise NotImplementedError(f"MolMiM does not support the '{task.mode}' generation mode.")


class GEMGenerator(BaseGenerator):
    """
    Interface for the GEM model workflow for fine-tuning and generation.
    """
    def __init__(self, model_path: str, use_cuda: bool = True):
        super().__init__(use_cuda=use_cuda)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"GEM model path not found: {model_path}")
        self.model_path = model_path

    def finetune_and_generate(self, smiles: List[str], labels: List[float], num_to_generate: int = 100, **workflow_params) -> List[str]:
        """Fine-tunes the model on the best examples from the input and generates new molecules."""
        return run_generation_workflow(mode='filter', 
                                       smiles=smiles, 
                                       labels=labels, 
                                       gen_model_path=self.model_path, 
                                       use_cuda=self.use_cuda,
                                       **workflow_params
        )

    def generate(self, task: GenerationTask) -> List[str]:
        if task.mode == 'finetune_and_generate':
            if not task.starting_smiles or not task.labels:
                raise ValueError("'starting_smiles' (a list) and 'labels' must be provided for this task.")
            return self.finetune_and_generate(smiles=task.starting_smiles, labels=task.labels, **task.config)
        else:
            raise NotImplementedError(f"GEM does not support the '{task.mode}' generation mode.")


class FRAGGenerator(BaseGenerator):
    """
    Interface for the f-RAG model, an evolutionary algorithm for de novo design.
    """
    def __init__(self, vocab_path: str, **kwargs):
        super().__init__()
        # The backend f_RAG class is initialized with all keyword arguments
        self._backend = f_RAG(vocab_path=vocab_path, **kwargs)
        # Store necessary parameters for the run_optimization method
        self.num_safe_per_gen = self._backend.num_safe_per_gen
        self.num_ga_per_gen = self._backend.num_ga_per_gen
        self.mutation_rate = self._backend.mutation_rate

    def run_optimization(self, objective_fn: Callable[[List[str]], List[float]], num_generations: int = 10) -> pd.DataFrame:
        """Runs the full f-RAG evolutionary optimization loop."""
        print("--- Starting f-RAG Generative Process ---")
        all_results = []
        for generation in range(num_generations):
            print(f"\n>> Generation {generation + 1}/{num_generations}")

            safe_smiles = self._backend.generate(num_to_generate=self.num_safe_per_gen)
            ga_smiles = []
            if len(self._backend.molecule_population) > 0:
                 ga_smiles = [self._backend.reproduce(self._backend.molecule_population, self.mutation_rate) for _ in range(self.num_ga_per_gen)]

            new_smiles = list(filter(None, safe_smiles + ga_smiles))
            if not new_smiles:
                print("No valid molecules were generated. Skipping generation.")
                continue

            scores = objective_fn(new_smiles)
            all_results.extend(zip(new_smiles, scores))
            self._backend.update_population(scores, new_smiles)

            if self._backend.molecule_population:
                best_score, best_smiles = self._backend.molecule_population[0]
                print(f"Current best molecule: {best_smiles} (Score: {best_score:.4f})")

        print("\n--- f-RAG Process Finished ---")
        return pd.DataFrame(all_results, columns=['smiles', 'score']).sort_values(by='score', ascending=False)

    def generate(self, task: GenerationTask) -> pd.DataFrame:
        if task.mode == 'property_optimization':
            if not callable(task.objective):
                raise ValueError("A callable 'objective' function must be provided for this task.")
            return self.run_optimization(objective_fn=task.objective, **task.config)
        else:
            raise NotImplementedError(f"f-RAG does not support the '{task.mode}' generation mode.")


# --- The Main Factory Function ---

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
                  - for 'safegpt': model_path (optional), use_cuda
                  - for 'molmim': api_token (required)
                  - for 'gem': model_path (required), use_cuda
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