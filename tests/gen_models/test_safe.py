import pytest
import torch
from argo.gen_models import GenerationModel, GenerationTask

# Use CUDA if available
use_cuda = torch.cuda.is_available()

@pytest.fixture(scope="module")
def safegpt_model():
    """Fixture to initialize the SAFE-GPT model once for all tests in this module."""
    try:
        return GenerationModel(model_type='safegpt', use_cuda=use_cuda)
    except ImportError as e:
        pytest.skip(str(e))

def test_safegpt_de_novo(safegpt_model):
    """Test SAFE-GPT de novo generation."""
    task = GenerationTask(
        mode='de_novo',
        config={
            "n_samples": 5,
            "batch_size": 5,
            "sanitize": True
        }
    )
    result = safegpt_model.generate(task)
    assert isinstance(result, list)
    assert len(result) == 5
    for smi in result:
        assert isinstance(smi, str)

def test_safegpt_scaffold_decoration(safegpt_model):
    """Test SAFE-GPT scaffold decoration."""
    scaffold = "[*]N-c1ccc2ncnc(-N[*])c2c1"
    task = GenerationTask(
        mode='scaffold_decoration',
        scaffold=scaffold,
        config={
            "n_samples": 5,
            "batch_size": 5,
            "sanitize": True
        }
    )
    result = safegpt_model.generate(task)
    assert isinstance(result, list)
    assert len(result) == 5, f"Expected exactly 5 valid molecules, got {len(result)}"
    for smi in result:
        assert isinstance(smi, str)

def test_safegpt_linker_generation(safegpt_model):
    """Test SAFE-GPT linker generation."""
    fragment1 = "[*]N1CCCCC1"
    fragment2 = "Brc1cccc(Nc2ncnc3ccc(-[*])cc23)c1"
    task = GenerationTask(
        mode='linker_generation',
        fragments=[fragment1, fragment2],
        config={
            "n_samples": 5,
            "batch_size": 5,
            "sanitize": True
        }
    )
    result = safegpt_model.generate(task)
    assert isinstance(result, list)
    assert len(result) == 5, f"Expected exactly 5 valid molecules, got {len(result)}"
    for smi in result:
        assert isinstance(smi, str)
