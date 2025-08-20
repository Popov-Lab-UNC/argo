from meeko import PDBQTMolecule, RDKitMolCreate
import gzip
import tempfile
import os
from rdkit import Chem
from typing import Union, List

def get_poses_from_dlg(filepath: str, pose_indices: Union[int, List[int]]):
    """
    Read a .dlg or .dlg.gz file and return combined_mol for specified pose indices.
    
    Args:
        filepath (str): Path to .dlg or .dlg.gz file
        pose_indices: Single index (int) or list of indices to extract
        only_cluster_leads (bool): Whether to use only cluster leads (default False)
        keep_flexres (bool): Whether to keep flexible residues (default False)
    
    Returns:
        combined_mol: RDKit molecule containing the specified poses
    """
    # Convert single index to list for consistent handling
    if isinstance(pose_indices, int):
        pose_indices = [pose_indices]
    
    temp_dlg_path = None
    
    try:
        path_to_process = filepath
        
        # Handle gzipped files
        if filepath.endswith('.gz'):
            with tempfile.NamedTemporaryFile(mode='wt', delete=False, suffix=".dlg") as tmp:
                temp_dlg_path = tmp.name
                with gzip.open(filepath, 'rt') as f_gz:
                    tmp.write(f_gz.read())
            path_to_process = temp_dlg_path
        
        # Read the PDBQT molecule
        pdbqt_mol = PDBQTMolecule.from_file(path_to_process, 
                                           name=os.path.basename(filepath).split('.')[0],
                                           is_dlg=True, 
                                           skip_typing=True)
        
        # Create RDKit molecules from PDBQT
        mol_list = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol, only_cluster_leads=False, keep_flexres=False)
        failures = [i for i, mol in enumerate(mol_list) if mol is None]
        combined_mol = RDKitMolCreate.combine_rdkit_mols(mol_list)

        # Select only the specified conformers
        selected_mol = Chem.Mol(combined_mol)
        selected_mol.RemoveAllConformers()
        for idx in pose_indices:
            conf = combined_mol.GetConformer(idx)
            selected_mol.AddConformer(conf, assignId=True)

        return selected_mol
        
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
        return None
        
    finally:
        # Clean up temporary file if created
        if temp_dlg_path and os.path.exists(temp_dlg_path):
            os.remove(temp_dlg_path)