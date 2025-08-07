import pytest
import os
from argo.gen_models import GenerationModel, GenerationTask

MOLMIM_SERVER_ADDRESS = "g0310:8000"

@pytest.fixture(scope="module")
def molmim_model():
    """Fixture to initialize the MolMIM client."""
    if not MOLMIM_SERVER_ADDRESS:
        pytest.skip("MOLMIM_SERVER_ADDRESS environment variable not set.")

    try:
        model = GenerationModel(model_type='molmim', server_address=MOLMIM_SERVER_ADDRESS)
        return model
    except ConnectionError as e:
        pytest.skip(f"Could not connect to MolMIM server at {MOLMIM_SERVER_ADDRESS}: {e}")

@pytest.mark.skipif(not MOLMIM_SERVER_ADDRESS, reason="MOLMIM_SERVER_ADDRESS not set")
def test_molmim_property_optimization(molmim_model):
    """Test MolMIM property optimization."""
    task = GenerationTask(
        mode='property_optimization',
        seed_smiles="[H][C@@]12Cc3c[nH]c4cccc(C1=C[C@H](NC(=O)N(CC)CC)CN2C)c34",
        objective="QED",
        config={
            "n_samples": 5,
        }
    )
    result = molmim_model.generate(task)
    assert isinstance(result, list)
    assert len(result) > 0
    for smi in result:
        assert isinstance(smi, str)

@pytest.mark.skipif(not MOLMIM_SERVER_ADDRESS, reason="MOLMIM_SERVER_ADDRESS not set")
def test_molmim_biased_generation(molmim_model):
    """Test MolMIM biased generation."""
    task = GenerationTask(
        mode='biased_generation',
        seed_smiles="[H][C@@]12Cc3c[nH]c4cccc(C1=C[C@H](NC(=O)N(CC)CC)CN2C)c34",
        config={
            "n_samples": 5,
        }
    )
    result = molmim_model.generate(task)
    assert isinstance(result, list)
    assert len(result) > 0
    for smi in result:
        assert isinstance(smi, str)
