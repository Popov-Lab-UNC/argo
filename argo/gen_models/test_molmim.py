from argo.gen_models.api import GenerationModel, GenerationTask
import os

try:
    molmim = GenerationModel('molmim', server_address='g0310:8000')
    
    print("Generator is ready.")
        
    # Define a task for property optimization
    opt_task = GenerationTask(
        mode='property_optimization',
        objective='plogP',
        seed_smiles='COc1ccccc1C(=O)NCc1ccc(F)cc1Br',
        config={
            'n_samples': 5,
        }
    )

    opt_task.seed_smiles = 'CCO'

    # The .generate() call now transparently talks to the local container
    optimized_molecules = molmim.generate(opt_task)
    
    print("\nOptimized Molecules:")
    for smi in optimized_molecules:
        print(smi)

except (EnvironmentError, RuntimeError) as e:
    print(f"An error occurred: {e}")