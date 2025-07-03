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
def test_molmim():
    api_key = os.environ.get("MOLMIM_API_KEY")
    if not api_key:
        raise RuntimeError("MOLMIM_API_KEY environment variable not set.")
    molmim = GenerationModel(model_type='molmim', api_token=api_key)
    task = GenerationTask(
        mode='property_optimization',
        starting_smiles="[H][C@@]12Cc3c[nH]c4cccc(C1=C[C@H](NC(=O)N(CC)CC)CN2C)c34",
        objective="QED",
        config={
            "algorithm": "CMA-ES",
            "num_molecules": 5,
            "minimize": False,
            "min_similarity": 0.3,
            "particles": 5,
            "iterations": 2
        }
    )
    try:
        result = molmim.generate(task)
        print("MolMiM result:")
        print(result)
    except Exception as e:
        print(f"MolMiM test failed: {e}")

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

def test_gem():
    # Dummy data for testing
    smiles = ["c1ccccc1", "CC(=O)Nc1ccc(O)cc1", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"]
    labels = [-4.0, -4.0, -8.0]
    # You must provide a valid model path for real tests
    model_path = str(Path(__file__).parent / "pretrained" / "gem_chembl.pt")
    gem = GenerationModel(model_type='gem', model_path=model_path, use_cuda=use_cuda)
    task = GenerationTask(
        mode='finetune_and_generate',
        starting_smiles=smiles,
        labels=labels,
        config={
            "tot_hits": 6,
            "batch_size": 2,
            "save_files": False,
            "save_models": False,
            "do_filter": False
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
    vocab_path = str(Path(__file__).parent / "f_rag" / "my_fragment_vocab.csv")
    # Simple QED objective function
    def evaluate_qed(smiles_list):
        scores = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                scores.append(float(QED.qed(mol)) if mol else -1.0)
            except Exception:
                scores.append(-1.0)
        return scores
    
    f_rag = GenerationModel(
        model_type='f-rag',
        vocab_path=vocab_path,
        results_path=str(Path(__file__).parent / "f_rag" / "output" / "frag_test_results.csv"),
        mol_population_size=20,
        num_safe_per_gen=3,
        num_ga_per_gen=2,
        seed=42
    )
    task = GenerationTask(
        mode='property_optimization',
        objective=evaluate_qed,
        config={
            "num_generations": 2
        }
    )
    try:
        result = f_rag.generate(task)
        print("f-RAG result:")
        print(result)
    except Exception as e:
        print(f"f-RAG test failed: {e}")

if __name__ == "__main__":
    print("\nTesting MolMiM...")
    test_molmim()
    print("\nTesting SAFE-GPT...")
    test_safegpt()
    print("\nTesting GEM...")
    test_gem()
    print("\nTesting f-RAG...")
    test_f_rag() 