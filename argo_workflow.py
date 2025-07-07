import os
import pandas as pd
from rdkit import Chem
from argo.gen_models.f_rag.fusion.slicer import MolSlicerForSAFEEncoder
from argo.frag_utils import SAFECodec
from argo.gen_models.api import GenerationModel, GenerationTask
from tqdm import tqdm

# 1. Load docking score CSV
df = pd.read_csv('CHD1_score0.csv')

# 2. Sort by score (ascending: best first)
df = df.sort_values('score', ascending=True)

# 3. Setup fragmenter
slicer = MolSlicerForSAFEEncoder(shortest_linker=True)
sfcodec = SAFECodec(slicer=slicer, ignore_stereo=True)

# 4. Fragment molecules and accumulate stats
from collections import defaultdict
frag_counts = defaultdict(int)
frag_score_sum = defaultdict(float)

for idx, row in tqdm(df.iterrows(), total=len(df), desc='Fragmenting molecules'):
    smiles = row['smiles']
    score = row['score']
    try:
        molecule_sf = sfcodec.encode(smiles)
        if molecule_sf is None:
            continue
        for fragment_sf in molecule_sf.split('.'):
            fragment_smiles = sfcodec.decode(fragment_sf)
            if fragment_smiles is None:
                continue
            frag_counts[fragment_smiles] += 1
            frag_score_sum[fragment_smiles] += score
    except Exception as e:
        print(f"Error fragmenting {smiles}: {e}")
        continue

# 5. Compute average score and output CSV
out_rows = []
for frag, count in frag_counts.items():
    avg_score = frag_score_sum[frag] / count
    # Compute size (number of heavy atoms)
    try:
        mol = Chem.MolFromSmiles(frag)
        size = mol.GetNumAtoms() if mol is not None else None
    except Exception:
        size = None
    out_rows.append({'frag': frag, 'count': count, 'score': avg_score, 'size': size})

out_df = pd.DataFrame(out_rows)
out_df = out_df.sort_values('avg_score', ascending=True)
out_df.to_csv('fragment_scores.csv', index=False)
print('Fragment statistics written to fragment_scores.csv')

# 6. Instantiate all four generative models via the API
safegpt_model = GenerationModel('safegpt', use_cuda=True)
molmim_model = GenerationModel('molmim', api_token=os.environ.get("MOLMIM_API_KEY"))
gem_model = GenerationModel('gem', model_path='argo/gen_models/pretrained/gem_chembl.pt', use_cuda=True)

vocab_df = out_df.head(100)
f_rag_model = GenerationModel('f-rag', 
                              vocab_path=vocab_df, 
                              injection_model_path="argo/gen_models/pretrained/model.safetensors",
                              frag_population_size=50,
                              mol_population_size=20,
                              min_frag_size=5,
                              max_frag_size=30,
                              min_mol_size=10,
                              max_mol_size=100,
                              mutation_rate=0.01
)

print('All four generative models instantiated.')
