# Generative Models (`gen_models`)

This module provides a unified interface for various molecular generation models. It is designed to be a flexible and extensible framework for de novo design, optimization, and other generative tasks.

## Key Components

- **`GenerationModel`**: A factory function that returns a specific generator instance based on the `model_type` argument. This is the main entry point for accessing the models.
- **`GenerationTask`**: A dataclass that standardizes the input for all generation tasks. It defines the `mode` of generation and other necessary parameters like `scaffold`, `fragments`, `seed_smiles`, etc.
- **`BaseGenerator`**: An abstract base class that defines the common interface for all generators. Each generator implements the `generate` method, which takes a `GenerationTask` as input.

## Supported Models

- **`safegpt`**: The SAFE-GPT model for *de novo* generation, scaffold decoration, and linker generation.
- **`gem`**: The GEM model for *de novo* and biased generation (fine-tuning).
- **`f-rag`**: The f-RAG model for scaffold decoration, linker generation, and property-guided optimization.
- **`molmim`**: A client for the MolMIM model for property-guided optimization and biased generation.

## Usage

Here are some examples of how to use the `gen_models` module.

### Basic Setup

First, import the necessary components and instantiate a model using the `GenerationModel` factory.

```python
from argo.gen_models import GenerationModel, GenerationTask
import torch

# Check if CUDA is available
use_cuda = torch.cuda.is_available()

# Instantiate a SAFE-GPT model
safegpt = GenerationModel(model_type='safegpt', use_cuda=use_cuda)
```

### De Novo Generation with SAFE-GPT

To generate molecules from scratch:

```python
task = GenerationTask(
    mode='de_novo',
    config={"n_samples": 10, "batch_size": 5}
)
molecules = safegpt.generate(task)
print(molecules)
```

### Scaffold Decoration with f-RAG

To decorate a given scaffold using f-RAG, you need to provide a vocabulary and a pretrained injection model.

```python
# Assuming you have the necessary model and vocab files
frag_model = GenerationModel(
    model_type='f-rag',
    vocab='path/to/your/vocab.csv',
    injection_model_path='path/to/your/injection_model.safetensors'
)

task = GenerationTask(
    mode='scaffold_decoration',
    scaffold='[*]c1n[nH]c2c1c(=O)n(C)c(=O)n2C',
    config={"n_samples": 5}
)
decorated_molecules = frag_model.generate(task)
print(decorated_molecules)
```

### Property Optimization with MolMIM

To optimize a molecule for a specific property (e.g., QED) using MolMIM, you need a running MolMIM server.

```python
# The server address should be configured via an environment variable or directly.
# For this example, we assume it's running at 'localhost:8000'.
try:
    molmim = GenerationModel(model_type='molmim', server_address='localhost:8000')

    task = GenerationTask(
        mode='property_optimization',
        seed_smiles='CCO', # Starting molecule
        objective='QED',   # Target property
        config={"n_samples": 5}
    )
    optimized_molecules = molmim.generate(task)
    print(optimized_molecules)
except ConnectionError as e:
    print(f"Could not connect to MolMIM server: {e}")
```

### Biased Generation with GEM

To generate molecules biased towards a specific chemical space using GEM, you can fine-tune the model on a set of seed SMILES.

```python
# Path to the pretrained GEM model
gem_model_path = 'argo/gen_models/pretrained/gem_chembl.pt'

gem = GenerationModel(model_type='gem', model_path=gem_model_path, use_cuda=use_cuda)

task = GenerationTask(
    mode='biased_generation',
    seed_smiles=['O=C(c1ccccc1)c1ccc(O)cc1', 'c1ccccc1C(=O)c1c(O)cccc1O'],
    config={"n_samples": 10, "batch_size": 5}
)
biased_molecules = gem.generate(task)
print(biased_molecules)
``` 