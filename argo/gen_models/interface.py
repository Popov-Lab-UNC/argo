import requests
import json
from typing import Any, Dict, Optional, List

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.generation.configuration_utils")

from .gem import GEMModel
from argo.gen_models.f_rag.optimizer import f_RAG

class GenModelInterface:
    def __init__(self, model_type: str, model_path: Optional[str] = None, use_cuda: bool = True, api_token: Optional[str] = None):
        """
        model_type: 'molmim', 'safegpt', 'gem', or 'custom'
        model_path: Optional path to the model or config. Only required for 'custom' or for SAFE-GPT custom weights.
        api_token: API token for MolMiM (required if using MolMiM)
        use_cuda: Whether to use CUDA for SAFE-GPT and GEM models
        """
        self.model_type = model_type.lower()
        if self.model_type == 'custom' and model_path is None:
            raise ValueError("model_path is required for custom model type")
        self.model_path = model_path
        self.use_cuda = use_cuda
        self.api_token = api_token
        self.model = self._load_model()

    def _load_model(self):
        if self.model_type == 'molmim':
            # No local model to load for MolMiM
            return None
        elif self.model_type == 'safegpt':
            try:
                import safe as sf
                from safe.trainer.model import SAFEDoubleHeadsModel
                from safe.tokenizer import SAFETokenizer
            except ImportError:
                raise ImportError("safe and its dependencies must be installed for SAFE-GPT usage.")
            if self.model_path:
                tokenizer = SAFETokenizer.from_pretrained(self.model_path)
                model = SAFEDoubleHeadsModel.from_pretrained(self.model_path)
                designer = sf.SAFEDesign(model=model, tokenizer=tokenizer)
                designer.model = designer.model.to('cuda' if self.use_cuda else 'cpu')
            else:
                designer = sf.SAFEDesign.load_default(device='cuda' if self.use_cuda else 'cpu', verbose=False)
            return designer
        elif self.model_type == 'gem':
            return GEMModel(
                gen_model_path=self.model_path,
                use_cuda=self.use_cuda,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def generate(self, mode: str = 'de_novo', **kwargs) -> Any:
        """
        mode: 'de_novo', 'linker', 'morphing', etc.
        kwargs: additional flags/options for the model
        """
        if self.model_type == 'molmim':
            return self._generate_molmim(mode, **kwargs)
        elif self.model_type == 'safegpt':
            return self._generate_safegpt(mode, **kwargs)
        elif self.model_type == 'gem':
            return self._generate_gem(mode, **kwargs)
        elif self.model_type == 'f-rag':
            args = kwargs.get('args')
            if args is None:
                raise ValueError("'args' must be provided for f-rag model_type.")
            return f_RAG(args)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _generate_molmim(self, mode: str, **kwargs) -> List[str]:
        """
        Calls the NVIDIA MolMiM API.

        Args:
            mode (str): The generation algorithm to use (e.g., 'CMA-ES'). In the payload, this is 'algorithm'.
            **kwargs: Must include 'smi' and other API parameters like 'num_molecules'.
        """
        invoke_url = "https://health.api.nvidia.com/v1/biology/nvidia/molmim/generate"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Accept": "application/json",
        }
        
        # 'smi' is required for the API endpoint
        if 'smi' not in kwargs:
            raise ValueError("'smi' (a starting SMILES string) must be provided in kwargs for MolMiM generation.")
        
        # The 'mode' for this interface maps to the 'algorithm' parameter in the MolMiM API
        payload = {"algorithm": mode, **kwargs}
        
        session = requests.Session()
        response = session.post(invoke_url, headers=headers, json=payload)
        response.raise_for_status()
        response_body = response.json()

        # The API returns a JSON string within the 'molecules' key, which needs to be parsed again.
        molecules = json.loads(response_body['molecules'])
        
        # Extract SMILES strings from each molecule dict
        smiles_list = [mol['sample'] for mol in molecules]

        return smiles_list

    def _generate_safegpt(self, mode: str, **kwargs):
        """
        mode: 'de_novo' (scaffold decoration), 'linker', 'morphing'
        kwargs: options for each mode
        """
        designer = self.model
        if mode == 'de_novo':
            n_samples_per_trial = kwargs.get('n_samples_per_trial', 12)
            n_trials = kwargs.get('n_trials', 1)
            sanitize = kwargs.get('sanitize', True)
            return designer.de_novo_generation(
                n_samples_per_trial=n_samples_per_trial,
                n_trials=n_trials,
                sanitize=sanitize
            )
        elif mode == 'scaffold_decoration':
            scaffold = kwargs.get('scaffold')
            if not scaffold:
                raise ValueError("'scaffold' must be provided for scaffold decoration.")
            n_samples_per_trial = kwargs.get('n_samples_per_trial', 12)
            n_trials = kwargs.get('n_trials', 1)
            sanitize = kwargs.get('sanitize', True)
            return designer.scaffold_decoration(
                scaffold=scaffold,
                n_samples_per_trial=n_samples_per_trial,
                n_trials=n_trials,
                sanitize=sanitize
            )
        elif mode == 'linker_generation':
            fragment1 = kwargs.get('fragment1')
            fragment2 = kwargs.get('fragment2')
            if not fragment1 or not fragment2:
                raise ValueError("'fragment1' and 'fragment2' must be provided for linker generation.")
            n_samples_per_trial = kwargs.get('n_samples_per_trial', 12)
            n_trials = kwargs.get('n_trials', 1)
            sanitize = kwargs.get('sanitize', True)
            do_not_fragment_further = kwargs.get('do_not_fragment_further', False)
            random_seed = kwargs.get('random_seed', 42)
            return designer.linker_generation(
                fragment1,
                fragment2,
                n_samples_per_trial=n_samples_per_trial,
                n_trials=n_trials,
                sanitize=sanitize,
                do_not_fragment_further=do_not_fragment_further,
                random_seed=random_seed
            )
        else:
            raise ValueError(f"Unknown SAFE-GPT generation mode: {mode}")

    def _generate_gem(self, mode: str, **kwargs):
        """
        mode: 'filter' (default, uses filter_and_generate) or 'bias_set' (uses bias_set_filter_and_generate)
        kwargs: options for each mode, see GEMModelInterface.generate
        """
        gem_model = self.model
        return gem_model.generate(mode=mode, **kwargs) 