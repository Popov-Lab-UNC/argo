import re
import random
from typing import Optional, Union, List, Callable
from rdkit import Chem
import datamol as dm

from safe import SAFEConverter

class SAFECodec:
    def __init__(self, 
                 slicer: Optional[Union[List[str], str, Callable]] = None, 
                 require_hs: Optional[bool] = False, 
                 ignore_stereo: bool = False,
                 verbose: bool = False
    ):
        """
        SAFE processor with encoder/decoder instances
        
        Args:
            slicer: Slicing algorithm for encoding, defaults to "brics"
                Supported slicers: ["hr", "rotatable", "recap", "mmpa", "attach", "brics"]
                Also custom slicers that return pairs of atom numbers
            require_hs (bool): whether the slicing algorithm require the molecule to have hydrogen explictly added
            ignore_stereo (bool): whether to remove stereochemistry before fragmenting
        """
        self.verbose = verbose
        # Encoder converter with custom parameters
        self.encoder_conv = SAFEConverter(
            slicer=slicer,
            require_hs=require_hs,
            ignore_stereo=ignore_stereo
        )
        
        # Decoder converter using default params
        self.decoder_conv = SAFEConverter()

    def encode(self, inp: Union[str, dm.Mol], canonical=True):
        """Convert SMILES/molecule to SAFE string"""
        try:
            return self.encoder_conv.encoder(inp, canonical=canonical)
        except Exception as e:
            if self.verbose:
                print(f"Unable to encode: {e}.")
            return None
        
    def encode_fragment(self, inp: Union[str, dm.Mol]):
        """Encode a fragment molecule to SAFE-formatted SMILES"""
        if isinstance(inp, str):
            inp = dm.to_mol(inp)

        if inp is None:
            return None
        
        if not isinstance(inp, dm.Mol):
            raise ValueError("Input must be a SMILES string or a RDKit molecule")
        
        non_map_atom_idxs = [
            atom.GetIdx() for atom in inp.GetAtoms() if atom.GetAtomicNum() != 0
        ]

        inp_smiles = Chem.MolToSmiles(
            inp,
            isomericSmiles=True,
            canonical=True,  # needs to always be true
            rootedAtAtom=non_map_atom_idxs[0],
        )

        # Remove any exit vector numbering
        #inp_smiles = self._canonicalize_frag(inp_smiles)

        attach_pos = set(re.findall(r"(\[\d+\*\]|!\[[^:]*:\d+\])", inp_smiles))
        for attach in attach_pos:
            inp_smiles = inp_smiles.replace(attach, '*')

        return inp_smiles # return SAFE-formatted SMILES for fragment
    
    def _canonicalize_frag(self, inp: Union[str, dm.Mol]):
        """Canonicalize a fragment molecule SMILES. Output SMILES"""
        if isinstance(inp, dm.Mol):
            inp = dm.to_smiles(inp)
        if inp is None:
            return None
        return re.sub(r'\[\*:\d+\]', '*', inp)
    
    def decode(self, 
               inp: str, 
               as_mol=False,
               canonical=False,
               fix=True,
               remove_dummies=False,
               remove_added_hs=True
    ):
        """Convert SAFE string to SMILES"""
        try:
            return self.decoder_conv.decoder(inp, 
                                             as_mol=as_mol, 
                                             canonical=canonical, 
                                             fix=fix, 
                                             remove_dummies=remove_dummies, 
                                             remove_added_hs=remove_added_hs
            )
        except Exception as e:
            print(f'Decoding failed: {e}')
            return None
        
    def link_fragments(self, frag1: str, frag2: str):
        """Link two fragments SMILES together, randomly selecting a position to link"""
        frag1_sf = self.encode_fragment(frag1)
        frag2_sf = self.encode_fragment(frag2)
        
        frag1_sf_canon = self._canonicalize_frag(frag1_sf)
        frag2_sf_canon = self._canonicalize_frag(frag2_sf)

        # Randomly select a * in frag1_sf_canon and frag2_sf_canon
        # Find all * positions in each fragment
        single_bond_pattern = r'(?<!\=)\(\*\)|(?<!\=)\(\[\*\]\)|(?<!\=)\*'  # Matches single-bonded exit vectors, ignoring (=*) and =*
        #double_bond_pattern = r'=\*|\(=\*\)'  # Matches both =* and (=*)
        
        # Count stars to check if linking is possible
        if not re.search(single_bond_pattern, frag1_sf_canon) or not re.search(single_bond_pattern, frag2_sf_canon):
            return None
            
        # Replace one random * with %99 in each fragment
        # Find all positions of single bond exit vectors
        frag1_stars = [m.start() for m in re.finditer(single_bond_pattern, frag1_sf_canon)]
        frag2_stars = [m.start() for m in re.finditer(single_bond_pattern, frag2_sf_canon)]
        
        # Randomly select one position from each fragment
        frag1_pos = random.choice(frag1_stars)
        frag2_pos = random.choice(frag2_stars)
        
        # Replace the randomly selected positions with %99
        frag1_sf_canon = frag1_sf_canon[:frag1_pos] + '%99' + frag1_sf_canon[frag1_pos + len(re.match(single_bond_pattern, frag1_sf_canon[frag1_pos:]).group()):]
        frag2_sf_canon = frag2_sf_canon[:frag2_pos] + '%99' + frag2_sf_canon[frag2_pos + len(re.match(single_bond_pattern, frag2_sf_canon[frag2_pos:]).group()):]

        return self.decode(frag1_sf_canon + '.' + frag2_sf_canon, remove_dummies=False)