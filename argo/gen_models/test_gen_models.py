from argo.gen_models.api import GenerationModel, GenerationTask
import torch
from pathlib import Path
from dotenv import load_dotenv
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED

load_dotenv()

# Use CUDA if available
use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")

# Test MolMiM (requires a valid API key)
def test_molmim_property_optimization():
    api_key = os.environ.get("MOLMIM_API_KEY")
    if not api_key:
        raise RuntimeError("MOLMIM_API_KEY environment variable not set.")
    molmim = GenerationModel(model_type='molmim', api_token=api_key)
    task = GenerationTask(
        mode='property_optimization',
        seed_smiles="[H][C@@]12Cc3c[nH]c4cccc(C1=C[C@H](NC(=O)N(CC)CC)CN2C)c34",
        objective="QED",
        config={
            "n_samples": 2,
        }
    )
    try:
        result = molmim.generate(task)
        print("MolMiM Property Optimization result:")
        print(result)
    except Exception as e:
        print(f"MolMiM Property Optimization test failed: {e}")

# Test MolMiM 'biased_generation' mode
def test_molmim_biased_generation():
    api_key = os.environ.get("MOLMIM_API_KEY")
    if not api_key:
        raise RuntimeError("MOLMIM_API_KEY environment variable not set.")
    molmim = GenerationModel(model_type='molmim', api_token=api_key)
    task = GenerationTask(
        mode='biased_generation',
        seed_smiles="[H][C@@]12Cc3c[nH]c4cccc(C1=C[C@H](NC(=O)N(CC)CC)CN2C)c34",
        config={
            "n_samples": 2,
        }
    )
    try:
        result = molmim.generate(task)
        print("MolMiM Biased Generation result:")
        print(result)
    except Exception as e:
        print(f"MolMiM Biased Generation test failed: {e}")

# Test SAFE-GPT (de novo generation, scaffold decoration, linker generation)
def test_safegpt():
    safegpt = GenerationModel(model_type='safegpt', use_cuda=use_cuda)

    # De Novo Generation
    try:
        task = GenerationTask(
            mode='de_novo',
            config={
                "n_samples": 2,
                "n_trials": 1,
                "sanitize": True
            }
        )
        result = safegpt.generate(task)
        print("SAFE-GPT De Novo Generation result:")
        print(result)
    except Exception as e:
        print(f"SAFE-GPT De Novo Generation test failed: {e}")

    # Scaffold Decoration
    try:
        scaffold = "[*]N-c1ccc2ncnc(-N[*])c2c1"
        task = GenerationTask(
            mode='scaffold_decoration',
            scaffold=scaffold,
            config={
                "n_samples": 2,
                "n_trials": 1,
                "sanitize": True
            }
        )
        result = safegpt.generate(task)
        print("SAFE-GPT Scaffold Decoration result:")
        print(result)
    except Exception as e:
        print(f"SAFE-GPT Scaffold Decoration test failed: {e}")

    # Linker Generation
    try:
        fragment1 = "[*]N1CCCCC1"
        fragment2 = "Brc1cccc(Nc2ncnc3ccc(-[*])cc23)c1"
        task = GenerationTask(
            mode='linker_generation',
            fragments=[fragment1, fragment2],
            config={
                "n_samples": 2,
                "n_trials": 1,
                "sanitize": True
            }
        )
        result = safegpt.generate(task)
        print("SAFE-GPT Linker Generation result:")
        print(result)
    except Exception as e:
        print(f"SAFE-GPT Linker Generation test failed: {e}")

def test_gem_de_novo():
    model_path = str(Path(__file__).parent / "pretrained" / "gem_chembl.pt")
    gem = GenerationModel(model_type='gem', model_path=model_path, use_cuda=use_cuda)

    task = GenerationTask(
        mode='de_novo',
        config={
            "n_samples": 2,
            "n_trials": 1
        }
    )
    try:
        result = gem.generate(task)
        print("GEM result:")
        print(result)
    except Exception as e:
        print(f"GEM test failed: {e}")

def test_gem_biased_generation():
    model_path = str(Path(__file__).parent / "pretrained" / "gem_chembl.pt")
    gem = GenerationModel(model_type='gem', model_path=model_path, use_cuda=use_cuda)

    task = GenerationTask(
        mode='biased_generation',
        seed_smiles="[H][C@@]12Cc3c[nH]c4cccc(C1=C[C@H](NC(=O)N(CC)CC)CN2C)c34",
        config={
            "n_samples": 2,
            "n_trials": 1
        }
    )
    try:
        result = gem.generate(task)
        print("GEM result:")
        print(result)
    except Exception as e:
        print(f"GEM test failed: {e}")

def test_f_rag():
    # Use the provided vocab file
    vocab_path = str(Path(__file__).parent / "f_rag" / "example_vocab.csv")
    f_rag = GenerationModel(
        model_type='f-rag',
        vocab_path=vocab_path,
        injection_model_path="pretrained/model.safetensors",
        frag_population_size=15,
        mol_population_size=20,
        min_frag_size=1,
        max_frag_size=15,
        min_mol_size=10,
        max_mol_size=100,
        mutation_rate=0.01
    )
    task = GenerationTask(
        mode='scaffold_decoration',
        config={
            "n_samples": 5,
            "random_seed": 42
        }
    )
    try:
        result = f_rag.generate(task)
        print("f-RAG generated SMILES:")
        print(result)
    except Exception as e:
        print(f"f-RAG test failed: {e}")

if __name__ == "__main__":
    print("\nTesting MolMiM (property optimization)...")
    #test_molmim_property_optimization()
    print("\nTesting MolMiM (biased generation)...")
    #test_molmim_biased_generation()
    print("\nTesting SAFE-GPT...")
    #test_safegpt()
    print("\nTesting GEM (de novo generation)...")
    #test_gem_de_novo()
    print("\nTesting GEM (biased generation)...")
    #test_gem_biased_generation()
    print("\nTesting f-RAG...")
    test_f_rag() 