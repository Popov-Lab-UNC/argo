import pandas as pd
import numpy as np
import joblib
from typing import Optional, Union, List, Callable, Dict, Any
from collections import defaultdict
from rdkit import Chem
from tqdm import tqdm

from argo.frag_utils import SAFECodec

class Fragment:
    """
    A class to represent a molecular fragment with its properties and statistics.
    """
    def __init__(self, smiles: str):
        self.smiles = smiles
        self.count = 0
        self.score_sum = 0.0
        self.molecules = []
        
        # Determine fragment type based on number of attachment points (*)
        attachment_count = smiles.count('*')
        if attachment_count == 1:
            self.type = 'arm'
        elif attachment_count == 2:
            self.type = 'linker'
        elif attachment_count >= 3:
            self.type = 'scaffold'
        else:
            self.type = 'unknown'  # No attachment points
        
        self.attachment_count = attachment_count
        
        # Calculate size
        try:
            mol = Chem.MolFromSmiles(smiles)
            self.size = mol.GetNumAtoms() if mol is not None else None
        except Exception:
            self.size = None
    
    def add_occurrence(self, score: float, molecule: str):
        """Add an occurrence of this fragment with its score and source molecule."""
        self.count += 1
        self.score_sum += score
        self.molecules.append(molecule)
    
    def add_batch_occurrences(self, total_score: float, molecules: List[str]):
        """Add multiple occurrences of this fragment with a total score."""
        count = len(molecules)
        if count > 0:
            self.count += count
            self.score_sum += total_score
            self.molecules.extend(molecules)
    
    def get_average_score(self) -> float:
        """Get the average score of all molecules containing this fragment."""
        return self.score_sum / self.count if self.count > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert fragment to dictionary for DataFrame creation."""
        return {
            'frag': self.smiles,
            'count': self.count,
            'score': self.get_average_score(),
            'size': self.size,
            'type': self.type,
            'attachment_count': self.attachment_count
        }

class FragmentVocabulary:
    """
    A class for crafting fragment vocabularies from molecular data with scores.
    This class is designed to be more robust, understandable, and modular.
    """
    def __init__(self, 
                 slicer: Union[str, Callable],
                 data: Optional[Union[str, pd.DataFrame]] = None,
                 ignore_stereo: bool = True,
                 smiles_col: str = 'smiles',
                 score_col: str = 'score',
                 verbose: bool = False,
                 **kwargs):
        """
        Initialize FragmentVocabulary.
        The constructor initializes empty internal state variables.
        If initial data is provided, it calls the add() method with rescore=True.
        """
        self.slicer = slicer
        self.smiles_col = smiles_col
        self.score_col = score_col
        self.verbose = verbose
        self.sfcodec = SAFECodec(slicer=slicer, ignore_stereo=ignore_stereo, verbose=self.verbose)
        
        # Internal state variables
        self.all_molecules = pd.DataFrame(columns=[self.smiles_col, self.score_col])
        self.vocabulary = pd.DataFrame(columns=['frag', 'count', 'score', 'size', 'type', 'attachment_count'])
        self.fragment_counts = defaultdict(int)
        self.molecule_fragments = defaultdict(list)
        self._fragments = {}  # Dictionary of Fragment objects
        
        self._params: Dict[str, Any] = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        if data is not None:
            self.add(data, rescore=True, **kwargs)

    def add(self, new_data: Union[str, pd.DataFrame], rescore: bool = False, use_tqdm: bool = True, **kwargs):
        """
        Add new molecular data and update fragment counts.
        Does not automatically re-score fragments unless rescore is True.
        """
        df_new = self._load_and_validate_data(new_data)
        
        # Append to all_molecules
        self.all_molecules = pd.concat([self.all_molecules, df_new], ignore_index=True)
        
        # Update fragment statistics with new data
        self._update_fragment_stats(df_new, use_tqdm=use_tqdm)
        
        if rescore:
            self.rescore(**self._params)

    def rescore(self, **kwargs):
        """
        Recalculate vocabulary using the complete self.all_molecules DataFrame.
        This is the single source of truth for updating self.vocabulary.
        """
        # Update parameters if provided
        if kwargs:
            self._params.update(kwargs)
            for key, value in kwargs.items():
                if hasattr(self, key) or key not in ['slicer', 'ignore_stereo', 'smiles_col', 'score_col']:
                    setattr(self, key, value)
        
        # Score fragments with current parameters
        self.vocabulary = self._score_fragments()
        return self.vocabulary

    def get_vocab(self) -> pd.DataFrame:
        """Return the fragment vocabulary data as a DataFrame."""
        return self.vocabulary

    def get_data(self) -> pd.DataFrame:
        """Return the accumulated data as a DataFrame."""
        return self.all_molecules.copy()

    def get_params(self) -> Dict[str, Any]:
        """Return the parameters used for the last vocabulary calculation."""
        return self._params

    def __len__(self) -> int:
        return len(self.vocabulary)

    def __getitem__(self, key):
        return self.vocabulary.__getitem__(key)

    def head(self, n: int = 5) -> pd.DataFrame:
        return self.vocabulary.head(n)

    def save(self, path: str):
        """Serializes the entire FragmentVocabulary object using joblib.dump()."""
        joblib.dump(self, path)

    def export_dataframe(self, path: str):
        """Saves only the self.vocabulary DataFrame to a CSV."""
        self.vocabulary.to_csv(path, index=False)

    @classmethod
    def load(cls, path: str):
        """Load the full object using joblib.load()."""
        return joblib.load(path)

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the vocabulary state.
        """
        info = {
            'vocab_size': len(self.vocabulary),
            'data_size': len(self.all_molecules),
            'fragment_stats': {
                'unique_fragments': len(self._fragments),
                'total_fragment_occurrences': sum(frag.count for frag in self._fragments.values())
            },
            'parameters': self._params.copy(),
        }
        return info

    # Internal methods below
    def _score_fragments(self) -> pd.DataFrame:
        """
        Internal method to score fragments using current parameters and fragment statistics.
        """
        scoring_method = self._params.get('scoring_method', 'average')
        if scoring_method == 'average':
            return self._score_by_average()
        elif scoring_method == 'enrichment':
            return self._score_by_fold_enrichment(self.all_molecules)
        else:
            raise ValueError(f"Unknown scoring method: {scoring_method}. Use 'average' or 'enrichment'")

    def _load_and_validate_data(self, data: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Load and validate data from a source.
        """
        if isinstance(data, str):
            df = pd.read_csv(data)
        else:
            df = data.copy()
        
        if df.empty:
            return df
            
        required_cols = {self.smiles_col, self.score_col}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Data must contain columns: {required_cols}. Found: {set(df.columns)}")

        lower_is_better = self._params.get('lower_is_better', True)
        df = df.sort_values(self.score_col, ascending=lower_is_better)
        return df

    def _update_fragment_stats(self, df: pd.DataFrame, use_tqdm: bool = True):
        """
        Fragment molecules and update the internal fragment statistics.
        """
        if df.empty:
            return
            
        frag_stats = self._fragment_molecules(df, use_tqdm=use_tqdm)
        
        for frag_smiles, count in frag_stats['counts'].items():
            if frag_smiles not in self._fragments:
                self._fragments[frag_smiles] = Fragment(frag_smiles)
            
            molecules = frag_stats['molecules'][frag_smiles]
            total_score = frag_stats['score_sums'][frag_smiles]
            
            self._fragments[frag_smiles].add_batch_occurrences(total_score, molecules)

    def _fragment_molecules(self, df: pd.DataFrame, use_tqdm: bool = True) -> dict:
        """
        Fragment molecules and return fragment statistics.
        """
        frag_counts = defaultdict(int)
        frag_score_sum = defaultdict(float)
        frag_molecules = defaultdict(list)
        success_count = 0
        
        if df.empty:
            print("Warning: Empty DataFrame provided for fragmentation")
            return {'counts': frag_counts, 'score_sums': frag_score_sum, 'molecules': frag_molecules}
        
        iterator = tqdm(df.iterrows(), total=len(df), desc='Fragmenting molecules') if use_tqdm else df.iterrows()
        for idx, row in iterator:
            smiles = row[self.smiles_col]
            score = row[self.score_col]
            try:
                molecule_sf = self.sfcodec.encode(smiles)
                if molecule_sf is None: continue
                success_count += 1
                for fragment_sf in molecule_sf.split('.'):
                    fragment_smiles = self.sfcodec.canonicalize_frag(self.sfcodec.decode(fragment_sf))
                    if fragment_smiles is None: continue
                    frag_counts[fragment_smiles] += 1
                    frag_score_sum[fragment_smiles] += score
                    frag_molecules[fragment_smiles].append(smiles)
            except Exception as e:
                if self.verbose:
                    print(f"Error fragmenting {smiles}: {e}")
                continue

        success_rate = (success_count / len(df) * 100) if len(df) > 0 else 0
        print(f"Successfully fragmented {success_count} molecules")
        print(f"Number of fragments from {success_count} molecules: {len(frag_counts)}")
        print(f"Success rate: {success_rate:.2f}%")
        
        if success_rate < 80.0:
            print(f"WARNING: Low fragmentation success rate ({success_rate:.2f}%).")
        return {'counts': frag_counts, 'score_sums': frag_score_sum, 'molecules': frag_molecules}

    def _score_by_average(self) -> pd.DataFrame:
        """
        Score fragments by average score of compounds containing each fragment.
        """
        out_rows = []
        min_count = self._params.get('min_count', 1)
        min_frag_size = self._params.get('min_frag_size', 1)
        max_frag_size = self._params.get('max_frag_size', 50)

        for fragment in self._fragments.values():
            if fragment.count < min_count: continue
            if fragment.size is None or not (min_frag_size <= fragment.size <= max_frag_size): continue
            out_rows.append(fragment.to_dict())
        
        vocab_df = pd.DataFrame(out_rows)
        if vocab_df.empty: return vocab_df
            
        lower_is_better = self._params.get('lower_is_better', True)
        vocab_df = vocab_df.sort_values('score', ascending=lower_is_better)

        max_fragments = self._params.get('max_fragments')
        if max_fragments is not None:
            vocab_df = vocab_df.head(max_fragments)
        return vocab_df

    def _score_by_fold_enrichment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score fragments by fold enrichment in top vs bottom compounds.
        """
        top_percent = self._params.get('top_percent', 10.0)
        n_top = int(len(df) * top_percent / 100)
        n_bottom = len(df) - n_top
        
        top_molecules = set(df.head(n_top)[self.smiles_col].tolist())
        bottom_molecules = set(df.tail(n_bottom)[self.smiles_col].tolist())
        
        out_rows = []
        min_count = self._params.get('min_count', 1)
        min_frag_size = self._params.get('min_frag_size', 1)
        max_frag_size = self._params.get('max_frag_size', 50)

        for fragment in self._fragments.values():
            if fragment.count < min_count: continue
            if fragment.size is None or not (min_frag_size <= fragment.size <= max_frag_size): continue
                
            top_count = sum(1 for mol in fragment.molecules if mol in top_molecules)
            bottom_count = sum(1 for mol in fragment.molecules if mol in bottom_molecules)
            
            if bottom_count > 0 and top_count > 0:
                top_freq = top_count / len(top_molecules)
                bottom_freq = bottom_count / len(bottom_molecules)
                enrichment = top_freq / bottom_freq
            else:
                enrichment = (top_count / len(top_molecules)) * 10.0 if top_molecules else 0.0

            frag_dict = fragment.to_dict()
            frag_dict.update({'top_count': top_count, 'bottom_count': bottom_count, 'score': enrichment})
            out_rows.append(frag_dict)
        
        vocab_df = pd.DataFrame(out_rows)
        if vocab_df.empty: return vocab_df
            
        vocab_df = vocab_df.sort_values('score', ascending=False)

        max_fragments = self._params.get('max_fragments')
        if max_fragments is not None:
            vocab_df = vocab_df.head(max_fragments)
        return vocab_df