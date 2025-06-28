# gen_models

This module provides a general interface to call different molecular generation models, such as MolMiM and SAFE-GPT.

## Usage

- Supports de novo generation, linker generation, and other modes.
- Allows passing optional flags used by both models.

Example:
```python
from gen_models.interface import GenerationModelInterface

# For MolMiM
gen = GenerationModelInterface('molmim', model_path='path/to/model')
gen.generate(mode='de_novo', option1='value1')

# For SAFE-GPT
gen = GenerationModelInterface('safegpt', model_path='path/to/model')
gen.generate(mode='linker', option2='value2')
``` 