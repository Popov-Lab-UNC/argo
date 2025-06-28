from argo.gen_models.interface import GenModelInterface
import torch
from pathlib import Path

# Test MolMiM (requires a valid API key)
def test_molmim():
    api_key = "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"
    molmim = GenModelInterface(model_type='molmim', api_token=api_key)
    try:
        result = molmim.generate(
            mode='de_novo',
            smi="[H][C@@]12Cc3c[nH]c4cccc(C1=C[C@H](NC(=O)N(CC)CC)CN2C)c34",
            algorithm="CMA-ES",
            num_molecules=5,
            property_name="QED",
            minimize=False,
            min_similarity=0.3,
            particles=5,
            iterations=2
        )
        print("MolMiM result:")
        print(result)
    except Exception as e:
        print(f"MolMiM test failed: {e}")


# Test SAFE-GPT (de novo generation, scaffold decoration, linker generation)
def test_safegpt():
    safegpt = GenModelInterface(model_type='safegpt')

    # De Novo Generation
    try:
        result = safegpt.generate(
            mode='de_novo',
            n_samples_per_trial=2,
            n_trials=1,
            sanitize=True
        )
        print("SAFE-GPT De Novo Generation result:")
        print(result)
    except Exception as e:
        print(f"SAFE-GPT De Novo Generation test failed: {e}")

    # Scaffold Decoration
    try:
        scaffold = "[*]N-c1ccc2ncnc(-N[*])c2c1"
        result = safegpt.generate(
            mode='scaffold_decoration',
            scaffold=scaffold,
            n_samples_per_trial=2,
            n_trials=1,
            sanitize=True
        )
        print("SAFE-GPT Scaffold Decoration result:")
        print(result)
    except Exception as e:
        print(f"SAFE-GPT Scaffold Decoration test failed: {e}")


    # Linker Generation
    try:
        fragment1 = "[*]N1CCCCC1"
        fragment2 = "Brc1cccc(Nc2ncnc3ccc(-[*])cc23)c1"
        result = safegpt.generate(
            mode='linker_generation',
            fragment1=fragment1,
            fragment2=fragment2,
            n_samples_per_trial=2,
            n_trials=1,
            sanitize=True
        )
        print("SAFE-GPT Linker Generation result:")
        print(result)
    except Exception as e:
        print(f"SAFE-GPT Linker Generation test failed: {e}")

def test_gem():
    # Use CUDA if available
    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    # Dummy data for testing
    smiles = ["CCO", "CCN", "CCC"]
    labels = [0.1, 0.2, 0.3]
    # You must provide a valid model path for real tests
    model_path = str(Path(__file__).parent / "pretrained" / "gem_chembl.pt")
    gem = GenModelInterface(model_type='gem', model_path=model_path, use_cuda=use_cuda)
    try:
        result = gem.generate(
            mode='filter',
            smiles=smiles,
            labels=labels,
            do_filter=False,
            tot_hits=5,
            batch_size=2,
            save_auxiliary_files=False
        )
        print("GEM result:")
        print(result)
    except Exception as e:
        print(f"GEM test failed: {e}")

if __name__ == "__main__":
    print("Testing MolMiM...")
    test_molmim()
    print("\nTesting SAFE-GPT...")
    test_safegpt()
    print("\nTesting GEM...")
    test_gem() 