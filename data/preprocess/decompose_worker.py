def decompose_row(args):
    i, smiles, slicer_params = args
    # Import inside worker
    import re
    import safe as sf
    from rdkit import Chem
    from argo.gen_models.f_rag.fusion.slicer import MolSlicerForSAFEEncoder

    def canonicalize(smiles):
        smiles = re.sub(r'\[\*:\d+\]', '*', smiles)
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

    try:
        slicer = MolSlicerForSAFEEncoder(**slicer_params)
        safestr = sf.encode(Chem.MolFromSmiles(smiles), slicer=slicer)
        frags = safestr.split('.')
        results = []
        for j, frag in enumerate(frags):
            perm_safe = '.'.join(frags[:j] + frags[j + 1:] + [frags[j]])
            frag_canon = canonicalize(sf.decode(frag, remove_dummies=False))
            if frag_canon.count('*') == 1:
                results.append(('arm', i, perm_safe, frag_canon))
            else:
                results.append(('linker', i, perm_safe, frag_canon))
        return results
    except Exception:
        return []