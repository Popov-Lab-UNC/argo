import numpy as np
from rdkit import DataStructs, Chem
import safe as sf
from rdkit.Chem import AllChem
import re
from argo.gen_models.f_rag.fusion.slicer import MolSlicerForSAFEEncoder

def canonicalize(smiles):
    smiles = re.sub(r'\[\*:\d+\]', '*', smiles)
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

def prepare_attach(smiles):
    smiles = re.sub(r'\[\*:\d+\]', '*', smiles)
    return re.sub(r'\*', '[1*]', smiles)

def attach(frag1, frag2, idx=0):
    rxn = AllChem.ReactionFromSmarts('[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]')
    mols = rxn.RunReactants((Chem.MolFromSmiles(frag1), Chem.MolFromSmiles(frag2)))
    return Chem.MolToSmiles(mols[idx][0])

def compute_similarity(args):
    """
    For each fragment, find the most similar fragment (excluding identicals),
    reconstruct a new molecule with the most similar fragment swapped in,
    re-encode to SAFE, and permute so the most similar fragment is last.
    """
    df_chunk, df_all, num_retrieve, frag_type, slicer_params = args
    idx_list, input_list, retrieved_list = [], [], []
    slicer = MolSlicerForSAFEEncoder(**slicer_params)
    for i, row in df_chunk.iterrows():
        sim_vec = DataStructs.BulkTanimotoSimilarity(row['fps'], df_all['fps'])
        sim_vec = np.array(sim_vec)
        # Remove identical fragments (sim == 1)
        sim_vec[sim_vec == 1] = -1
        top_idx = sim_vec.argsort()[::-1][:num_retrieve]
        retrieved = df_all['frag'].iloc[top_idx].tolist()

        if not retrieved:
            continue
        try:
            most_similar_frag = prepare_attach(retrieved[0])
            retrieved_frags = '.'.join([row['frag']] + retrieved[1:])
            frags = row['safe'].split('.')
            rest, frag = frags[:-1], frags[-1]
            if frag_type == 'arm':
                rest_decoded = sf.decode('.'.join(rest), remove_dummies=False)
                if rest_decoded is None:
                    continue
                rest_att = prepare_attach(rest_decoded)
                attached_smiles = attach(rest_att, most_similar_frag)
            elif frag_type == 'linker':
                if len(rest) != 2:
                    continue
                rest1, rest2 = sf.decode(rest[0], remove_dummies=False), sf.decode(rest[1], remove_dummies=False)
                if rest1 is None or rest2 is None:
                    continue
                rest1, rest2 = prepare_attach(rest1), prepare_attach(rest2)
                attached_smiles = attach(attach(rest1, most_similar_frag), rest2)
            else:
                continue
            new_safe = sf.encode(Chem.MolFromSmiles(attached_smiles), slicer=slicer)
            frags_new = new_safe.split('.')
            for j, frag in enumerate(frags_new):
                if canonicalize(sf.decode(frag, remove_dummies=False)) == retrieved[0]:
                    break
            else:
                continue
            # Permute to place the most_similar_frag at last
            new_safe = '.'.join(frags_new[:j] + frags_new[j + 1:] + [frags_new[j]])
        except Exception:
            continue
        idx_list.append(row['idx'])
        input_list.append(new_safe)
        retrieved_list.append(retrieved_frags)
    return idx_list, input_list, retrieved_list