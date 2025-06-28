from typing import Optional, List
from .workflow import run_generation_workflow

class GEMModel:
    """
    Public interface for the GEM workflow.
    """
    def __init__(self,
                 gen_model_path: str,
                 clf_model_path: Optional[str] = None,
                 use_cuda: bool = True):
        """
        Initializes the GEM model wrapper.

        Args:
            gen_model_path (str): Path to the pre-trained generative Transformer model.
            clf_model_path (Optional[str]): Path to a pre-trained classifier model (optional).
            use_cuda (bool): Whether to use CUDA if available.
        """
        self.gen_model_path = gen_model_path
        self.clf_model_path = clf_model_path
        self.use_cuda = use_cuda

    def generate(self,
                 mode: str,
                 smiles: List[str],
                 labels: Optional[List[float]] = None,
                 **kwargs) -> List[str]:
        """
        Runs the full generation workflow.

        Args:
            mode (str): The generation mode. Either 'filter' or 'bias_set'.
            smiles (List[str]): Input SMILES for fine-tuning.
            labels (Optional[List[float]]): Corresponding scores/labels for 'filter' mode.
            **kwargs: Additional parameters for the workflow (e.g., tot_hits, fine_tune_epochs).

        Returns:
            List[str]: A list of newly generated, valid SMILES strings.
        """
        return run_generation_workflow(
            mode=mode,
            smiles=smiles,
            labels=labels,
            gen_model_path=self.gen_model_path,
            clf_model_path=self.clf_model_path,
            use_cuda=self.use_cuda,
            **kwargs
        )