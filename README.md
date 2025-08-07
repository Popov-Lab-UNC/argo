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
from argo.gen_models import GenerationModel, GenerationTask
from argo.utils import clean_smiles

# --- 1. Load and Prepare Data ---
df = pd.DataFrame('example_data.csv')

# --- 2. Clean SMILES ---
cleaned_smiles = [clean_smiles(smi) for smi in df['smiles']]

# --- 3. Generate Molecules ---
safegpt = GenerationModel(model_type='safegpt')
gem = GenerationModel(model_type='gem', model_path='argo/gen_models/pretrained/gem_chembl.pt')

# --- Task 1: De Novo Generation with SAFE-GPT ---
denovo_task = GenerationTask(
    mode='de_novo',
    config={"n_samples": 20}
)
denovo_molecules = safegpt.generate(denovo_task)

# --- Task 2: Scaffold Decoration with SAFE-GPT ---
scaffold_task = GenerationTask(
    mode='scaffold_decoration',
    scaffold='[*]c1ccccc1[*]',
    config={"n_samples": 20}
)
decorated_molecules = safegpt.generate(scaffold_task)
```

## Additional Suggestions

Here are some suggestions for future development and extension of the Argo framework:

*   **Workflow Manager**: Implement a workflow manager that can orchestrate iterative cycles of generation and scoring. This would allow for more complex, multi-step design campaigns where the output of one step (e.g., generated molecules) is fed into the next (e.g., a scoring or docking model), and the results are used to guide further generation.

*   **Reinforcement Learning Agent**: Develop a reinforcement learning (RL) agent that sits on top of the workflow. This agent could learn to intelligently sample from the different generative models and tasks based on the feedback from scoring functions. The agent could be optimized to balance the exploration of chemical space with the exploitation of promising regions, while also considering the computational cost of each generation or scoring step. This would enable more autonomous and efficient molecular design.
