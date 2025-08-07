import pytest
import torch
from pathlib import Path
from argo.gen_models import GenerationModel, GenerationTask

# Use CUDA if available
use_cuda = torch.cuda.is_available()
model_path = Path(__file__).parent.parent.parent / "argo" / "gen_models" / "pretrained" / "gem_chembl.pt"

@pytest.fixture(scope="module")
def gem_model():
    """Fixture to initialize the GEM model."""
    if not model_path.exists():
        pytest.skip("GEM model file not found.")
    return GenerationModel(model_type='gem', model_path=str(model_path), use_cuda=use_cuda)

def test_gem_de_novo(gem_model):
    """Test GEM de novo generation."""
    task = GenerationTask(
        mode='de_novo',
        config={
            "n_samples": 5,
            "batch_size": 5
        }
    )
    result = gem_model.generate(task)
    assert isinstance(result, list)
    assert len(result) == 5
    for smi in result:
        assert isinstance(smi, str)

def test_gem_biased_generation(gem_model):
    """Test GEM biased generation."""
    task = GenerationTask(
        mode='biased_generation',
        seed_smiles="[H][C@@]12Cc3c[nH]c4cccc(C1=C[C@H](NC(=O)N(CC)CC)CN2C)c34",
        config={
            "n_samples": 5,
            "batch_size": 5
        }
    )
    result = gem_model.generate(task)
    assert isinstance(result, list)
    assert len(result) == 5
    for smi in result:
        assert isinstance(smi, str)
