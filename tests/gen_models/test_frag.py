import pytest
from pathlib import Path
from argo.gen_models import GenerationModel, GenerationTask

# Paths to model and vocab
vocab_path = Path(__file__).parent.parent.parent / "argo" / "gen_models" / "f_rag" / "example_vocab.csv"
injection_model_path = Path(__file__).parent.parent.parent / "argo" / "gen_models" / "pretrained" / "model.safetensors"

@pytest.fixture(scope="module")
def frag_model():
    """Fixture to initialize the f-RAG model."""
    if not vocab_path.exists():
        pytest.skip("f-RAG vocab file not found.")
    if not injection_model_path.exists():
        pytest.skip("f-RAG injection model not found.")

    return GenerationModel(
        model_type='f-rag',
        vocab=str(vocab_path),
        injection_model_path=str(injection_model_path),
        frag_population_size=15,
        min_frag_size=1,
        max_frag_size=15,
        min_mol_size=10,
        max_mol_size=100,
    )

def test_f_rag_scaffold_decoration(frag_model):
    """Test f-RAG scaffold decoration mode."""
    task = GenerationTask(
        mode='scaffold_decoration',
        scaffold="[*]c1ccccc1[*]",
        config={
            "n_samples": 5,
            "random_seed": 42
        }
    )
    result = frag_model.generate(task)
    assert isinstance(result, list)
    assert len(result) > 0  # f-RAG might not always return the exact number
    for smi in result:
        assert isinstance(smi, str)

def test_f_rag_linker_generation(frag_model):
    """Test f-RAG linker generation mode."""
    task = GenerationTask(
        mode='linker_generation',
        config={
            "n_samples": 5,
            "random_seed": 42
        }
    )
    result = frag_model.generate(task)
    assert isinstance(result, list)
    assert len(result) > 0
    for smi in result:
        assert isinstance(smi, str)

def test_f_rag_property_optimization(frag_model):
    """Test f-RAG property optimization mode."""
    pytest.importorskip("tdc")
    pytest.importorskip("numpy")

    task = GenerationTask(
        mode='property_optimization',
        objective='qed',
        config={
            "n_samples": 5,
            "random_seed": 42,
            "batch_size": 5,
            "max_iter": 2, # Keep it short for testing
        }
    )
    result = frag_model.generate(task)
    assert isinstance(result, list)
    assert len(result) > 0
    for smi in result:
        assert isinstance(smi, str)
