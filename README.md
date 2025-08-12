# Argo: A Framework for In Silico Molecular Design

Argo is a Python framework designed to facilitate in silico molecular design by providing a unified interface to a variety of generative models. It allows researchers and developers to easily perform tasks such as de novo design, scaffold decoration, linker generation, and property-guided optimization.

## Project Philosophy

The goal of Argo is to create an accessible and extensible platform for the design-test-learn cycle in drug discovery. By providing a common interface for different generative models and a standardized way to define generation tasks, Argo aims to accelerate research and development in this area.

## Model Capabilities

Argo provides a unified interface for the following generative models and tasks:

| Model       | De Novo | Biased Generation | Scaffold Decoration | Linker Generation | Property Optimization |
|-------------|:-------:|:-----------------:|:-------------------:|:-----------------:|:---------------------:|
| **SAFE-GPT**|    ✅    |         -         |          ✅         |         ✅        |           -           |
| **GEM**     |    ✅    |         ✅        |          -          |         -         |           -           |
| **f-RAG**   |    -    |         -         |          ✅         |         ✅        |           ✅          |
| **MolMIM**  |    -    |         ✅        |          -          |         -         |           ✅          |

## Example Workflow

Here is a comprehensive example demonstrating a typical workflow with Argo.

```python
import pandas as pd
from rdkit import Chem
from argo.gen_models import GenerationModel, GenerationTask
from argo.utils import clean_smiles

# --- 1. Load and Prepare Data ---
# Create a dummy dataframe for demonstration
data = {'smiles': ['O=C(c1ccccc1)c1ccc(O)cc1', 'c1ccccc1C(=O)c1c(O)cccc1O']}
df = pd.DataFrame(data)

# --- 2. Clean SMILES ---
cleaned_smiles = [clean_smiles(smi) for smi in df['smiles']]

# --- 3. Generate Molecules ---
# Instantiate models
safegpt = GenerationModel(model_type='safegpt')
gem = GenerationModel(model_type='gem', model_path='argo/gen_models/pretrained/gem_chembl.pt')

# --- Task 1: De Novo Generation with SAFE-GPT ---
print("--- Running De Novo Generation with SAFE-GPT ---")
denovo_task = GenerationTask(
    mode='de_novo',
    config={"n_samples": 10, "batch_size": 5}
)
denovo_molecules = safegpt.generate(denovo_task)
print(f"Generated {len(denovo_molecules)} de novo molecules.")

# --- Task 2: Scaffold Decoration with a list of scaffolds ---
print("\n--- Running Scaffold Decoration with SAFE-GPT (List Input) ---")
scaffold_task = GenerationTask(
    mode='scaffold_decoration',
    scaffold=['[*]c1ccccc1[*]', '[*]c1n[nH]c2c1c(=O)n(C)c(=O)n2C'],
    config={"n_samples": 10, "strategy": "iterate"} # 'iterate' or 'sample'
)
decorated_molecules = safegpt.generate(scaffold_task)
print(f"Generated {len(decorated_molecules)} decorated molecules.")

# --- Task 3: Biased Generation with GEM ---
print("\n--- Running Biased Generation with GEM ---")
biased_task = GenerationTask(
    mode='biased_generation',
    seed_smiles=cleaned_smiles,
    config={"n_samples": 10, "batch_size": 5, "n_epochs": 5}
)
biased_molecules = gem.generate(biased_task)
print(f"Generated {len(biased_molecules)} biased molecules.")
```

## Future Tasks

Here are some suggestions for future development and extension of the Argo framework:

*   **Workflow Manager**: Implement a workflow manager that can orchestrate iterative cycles of generation and scoring. This would allow for more complex, multi-step design campaigns where the output of one step (e.g., generated molecules) is fed into the next (e.g., a scoring or docking model), and the results are used to guide further generation.

*   **Reinforcement Learning Agent**: Develop a reinforcement learning (RL) agent that sits on top of the workflow. This agent could learn to intelligently sample from the different generative models and tasks based on the feedback from scoring functions. The agent could be optimized to balance the exploration of chemical space with the exploitation of promising regions, while also considering the computational cost of each generation or scoring step. This would enable more autonomous and efficient molecular design.
