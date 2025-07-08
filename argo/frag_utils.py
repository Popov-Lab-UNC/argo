import re
import random
import itertools
from typing import Optional, Union, List, Callable

import numpy as np
from rdkit import Chem, RDLogger
import datamol as dm
from safe import SAFEConverter

class SAFECodec:
    def __init__(self, 
                 slicer: Optional[Union[str, Callable]] = 'brics', 
                 require_hs: Optional[bool] = False, 
                 ignore_stereo: bool = False,
                 verbose: bool = False
    ):
        """
        SAFE processor with encoder/decoder instances
        
        Args:
            slicer: Slicing algorithm for encoding, defaults to "brics"
                Supported slicers: ["hr", "rotatable", "recap", "mmpa", "attach", "brics", "f-rag"]
                Also custom slicers that return pairs of atom numbers
            require_hs (bool): whether the slicing algorithm require the molecule to have hydrogen explictly added
            ignore_stereo (bool): whether to remove stereochemistry before fragmenting
            verbose (bool): whether to print warnings and show RDKit logs
        """
        self.verbose = verbose
        if not self.verbose:
            RDLogger.DisableLog('rdApp.*')

        # Encoder converter with custom parameters
        if slicer == 'f-rag':
            slicer = MolSlicerForSAFEEncoder(shortest_linker=True)

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
        #inp_smiles = self.canonicalize_frag(inp_smiles)

        attach_pos = set(re.findall(r"(\[\d+\*\]|!\[[^:]*:\d+\])", inp_smiles))
        for attach in attach_pos:
            inp_smiles = inp_smiles.replace(attach, '*')

        return inp_smiles # return SAFE-formatted SMILES for fragment
    
    def canonicalize_frag(self, inp: Union[str, dm.Mol]):
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
        
        frag1_sf_canon = self.canonicalize_frag(frag1_sf)
        frag2_sf_canon = self.canonicalize_frag(frag2_sf)

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

# Following classes copied from f-rag/fusion/slicer.py
class MolSlicer:
    BOND_SPLITTERS = [
        # two atoms connected by a non ring single bond, one of each is not in a ring and at least two heavy neighbor
        "[R:1]-&!@[!R;!D1:2]",
        # two atoms in different rings linked by a non-ring single bond
        "[R:1]-&!@[R:2]",
    ]
    _BOND_BUFFER = 1  # buffer around substructure match size.
    MAX_CUTS = 2  # maximum number of cuts. Here we need two cuts for head-linker-tail.

    def __init__(self, shortest_linker=False, min_linker_size=0, require_ring_system=True):
        self.bond_splitters = [Chem.MolFromSmarts(x) for x in self.BOND_SPLITTERS]
        self.shortest_linker = shortest_linker
        self.min_linker_size = min_linker_size
        self.require_ring_system = require_ring_system

    def get_ring_system(self, mol):
        """Get the list of ring system from a molecule"""
        mol.UpdatePropertyCache()
        ri = mol.GetRingInfo()
        systems = []
        for ring in ri.AtomRings():
            ring_atoms = set(ring)
            cur_system = []  # keep a track of ring system
            for system in systems:
                if len(ring_atoms.intersection(system)) > 0:
                    ring_atoms = ring_atoms.union(system)  # merge ring system that overlap
                else:
                    cur_system.append(system)
            cur_system.append(ring_atoms)
            systems = cur_system
        return systems

    def _bond_selection_from_max_cuts(self, bond_list, dist_mat):
        """Select bonds based on maximum number of cuts allowed"""
        # for now we are just implementing to 2 max cuts algorithms
        if self.MAX_CUTS != 2:
            raise ValueError(f"Only MAX_CUTS=2 is supported, got {self.MAX_CUTS}")

        bond_pdist = np.full((len(bond_list), len(bond_list)), -1)
        for i in range(len(bond_list)):
            for j in range(i, len(bond_list)):
                # we get the minimum topological distance between bond to cut
                bond_pdist[i, j] = bond_pdist[j, i] = min(
                    [dist_mat[a1, a2] for a1, a2 in itertools.product(bond_list[i], bond_list[j])]
                )

        masked_bond_pdist = np.ma.masked_less_equal(bond_pdist, self.min_linker_size)

        if self.shortest_linker:
            return np.unravel_index(np.ma.argmin(masked_bond_pdist), bond_pdist.shape)
        return np.unravel_index(np.ma.argmax(masked_bond_pdist), bond_pdist.shape)

    def _get_bonds_to_cut(self, mol):
        """Get possible bond to cuts"""
        # use this if you want to enumerate yourself the possible cuts
        ring_systems = self.get_ring_system(mol)
        candidate_bonds = []
        ring_query = Chem.rdqueries.IsInRingQueryAtom()

        for query in self.bond_splitters:
            bonds = mol.GetSubstructMatches(query, uniquify=True)
            cur_unique_bonds = [set(cbond) for cbond in candidate_bonds]
            # do not accept bonds part of the same ring system or already known
            for b in bonds:
                bond_id = mol.GetBondBetweenAtoms(*b).GetIdx()
                bond_cut = Chem.GetMolFrags(
                    Chem.FragmentOnBonds(mol, [bond_id], addDummies=False), asMols=True
                )
                can_add = not self.require_ring_system or all(
                    len(frag.GetAtomsMatchingQuery(ring_query)) > 0 for frag in bond_cut
                )
                if can_add and not (
                    set(b) in cur_unique_bonds or any(x.issuperset(set(b)) for x in ring_systems)
                ):
                    candidate_bonds.append(b)
        return candidate_bonds

    def _fragment_mol(self, mol, bonds):
        """Fragment molecules on bonds and return head, linker, tail combination"""
        tmp = Chem.rdmolops.FragmentOnBonds(mol, [b.GetIdx() for b in bonds])
        _frags = list(Chem.GetMolFrags(tmp, asMols=True))
        # linker is the one with 2 dummy atoms
        linker_pos = 0
        for pos, _frag in enumerate(_frags):
            if sum([at.GetSymbol() == "*" for at in _frag.GetAtoms()]) == 2:
                linker_pos = pos
                break
        linker = _frags.pop(linker_pos)
        head, tail = _frags
        return (head, linker, tail)

    def __call__(self, mol):
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        # remove salt and solution
        mol = dm.keep_largest_fragment(mol)
        Chem.rdDepictor.Compute2DCoords(mol)
        dist_mat = Chem.rdmolops.GetDistanceMatrix(mol)

        candidate_bonds = self._get_bonds_to_cut(mol)

        # we have all the candidate bonds we can cut
        # now we need to pick the most plausible bonds
        selected_bonds = [mol.GetBondBetweenAtoms(a1, a2) for (a1, a2) in candidate_bonds]

        # CASE 1: no bond to cut ==> only head
        if len(selected_bonds) == 0:
            return (mol, None, None)

        # CASE 2: only one bond ==> linker is empty
        if len(selected_bonds) == 1:
            # there is no linker
            tmp = Chem.rdmolops.FragmentOnBonds(mol, [b.GetIdx() for b in selected_bonds])
            head, tail = Chem.GetMolFrags(tmp, asMols=True)
            return (head, None, tail)

        # CASE 3: we select the most plausible bond to cut on ourselves
        choice = self._bond_selection_from_max_cuts(candidate_bonds, dist_mat)
        selected_bonds = [selected_bonds[c] for c in choice]
        return self._fragment_mol(mol, selected_bonds)
    

class MolSlicerForSAFEEncoder(MolSlicer):
    def __call__(self, mol):
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        # remove salt and solution
        mol = dm.keep_largest_fragment(mol)
        Chem.rdDepictor.Compute2DCoords(mol)
        dist_mat = Chem.rdmolops.GetDistanceMatrix(mol)

        candidate_bonds = self._get_bonds_to_cut(mol)
        selected_bonds = [mol.GetBondBetweenAtoms(a1, a2) for (a1, a2) in candidate_bonds]
        assert len(selected_bonds) != 0     # only head cases

        # CASE 3: we select the most plausible bond to cut on ourselves
        if len(selected_bonds) >= 2:
            choice = self._bond_selection_from_max_cuts(candidate_bonds, dist_mat)
            selected_bonds = [selected_bonds[c] for c in choice]
        
        for bond in selected_bonds:
            yield (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())