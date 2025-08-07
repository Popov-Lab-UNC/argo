# Argo: A Framework for In Silico Molecular Design

Argo is a Python framework designed to facilitate in silico molecular design by providing a unified interface to a variety of generative models. It allows researchers and developers to easily perform tasks such as de novo design, scaffold decoration, linker generation, and property-guided optimization.

## Project Philosophy

The goal of Argo is to create an accessible and extensible platform for the design-test-learn cycle in drug discovery. By providing a common interface for different generative models and a standardized way to define generation tasks, Argo aims to accelerate research and development in this area.

## Example Workflow

Here is a comprehensive example that demonstrates a typical workflow with Argo. This script covers:
1.  Loading a dataset of molecules.
2.  Cleaning the SMILES strings.
3.  Training a simple filter model (e.g., QED).
4.  Generating new molecules using different models and tasks.
5.  Filtering the generated molecules using the trained model.

```python
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED
from argo.gen_models import GenerationModel, GenerationTask
from argo.utils import clean_smiles # Assuming a utility for cleaning SMILES

# --- 1. Load and Prepare Data ---
# Let's assume we have a CSV file with a 'smiles' column.
# For this example, we'll create a dummy DataFrame.
data = {'smiles': ['CCO', 'c1ccccc1', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'CC(=O)Oc1ccccc1C(=O)O']}
df = pd.DataFrame(data)

# --- 2. Clean SMILES ---
# It's good practice to standardize and clean SMILES before using them.
cleaned_smiles = [clean_smiles(smi) for smi in df['smiles']]
df['cleaned_smiles'] = [s for s in cleaned_smiles if s is not None]
print(f"Original SMILES: {len(df['smiles'])}, Cleaned SMILES: {len(df['cleaned_smiles'])}")

# --- 3. "Train" a Filter Model ---
# For this example, our "filter model" will be a simple QED (Quantitative Estimate of Drug-likeness)
# threshold. In a real-world scenario, this could be a more complex model (e.g., a machine learning
# model for predicting activity or toxicity).
def qed_filter(smiles_list, threshold=0.5):
    """Filters a list of SMILES based on a QED threshold."""
    filtered = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            qed_score = QED.qed(mol)
            if qed_score > threshold:
                filtered.append((smi, qed_score))
    return filtered

# --- 4. Generate Molecules ---
# We'll use different generative models for various tasks.

# Initialize models (assuming required model files are in place)
safegpt = GenerationModel(model_type='safegpt')
# gem = GenerationModel(model_type='gem', model_path='path/to/gem_model.pt') # Uncomment if you have GEM

# --- Task 1: De Novo Generation with SAFE-GPT ---
print("\nGenerating molecules with SAFE-GPT (de novo)...")
denovo_task = GenerationTask(
    mode='de_novo',
    config={"n_samples": 20}
)
denovo_molecules = safegpt.generate(denovo_task)
print(f"Generated {len(denovo_molecules)} molecules.")

# --- Task 2: Scaffold Decoration with SAFE-GPT ---
print("\nDecorating scaffold with SAFE-GPT...")
scaffold_task = GenerationTask(
    mode='scaffold_decoration',
    scaffold='[*]c1ccccc1[*]',
    config={"n_samples": 20}
)
decorated_molecules = safegpt.generate(scaffold_task)
print(f"Generated {len(decorated_molecules)} decorated molecules.")

# --- 5. Filter Generated Molecules ---
print("\nFiltering generated molecules...")
all_generated_molecules = denovo_molecules + decorated_molecules

# Apply the QED filter
filtered_results = qed_filter(all_generated_molecules, threshold=0.6)

print(f"\n--- Filter Results (QED > 0.6) ---")
if filtered_results:
    for smi, score in filtered_results:
        print(f"SMILES: {smi}, QED: {score:.3f}")
else:
    print("No molecules passed the filter.")

```

## Additional Suggestions

Here are some suggestions for future development and extension of the Argo framework:

*   **Workflow Manager**: Implement a workflow manager that can orchestrate iterative cycles of generation and scoring. This would allow for more complex, multi-step design campaigns where the output of one step (e.g., generated molecules) is fed into the next (e.g., a scoring or docking model), and the results are used to guide further generation.

*   **Reinforcement Learning Agent**: Develop a reinforcement learning (RL) agent that sits on top of the workflow. This agent could learn to intelligently sample from the different generative models and tasks based on the feedback from scoring functions. The agent could be optimized to balance the exploration of chemical space with the exploitation of promising regions, while also considering the computational cost of each generation or scoring step. This would enable more autonomous and efficient molecular design.
