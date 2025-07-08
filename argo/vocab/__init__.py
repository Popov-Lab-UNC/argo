import pandas as pd
import numpy as np
from typing import Optional, Union, List, Callable
from collections import defaultdict
from rdkit import Chem
from tqdm import tqdm

from argo.frag_utils import SAFECodec

class FragmentVocabulary:
    """
    A class for crafting fragment vocabularies from molecular data with scores.
    Can be used directly as a vocab argument for f-RAG models.
    """
    
    def __init__(self, data: Union[str, pd.DataFrame],
                 slicer: Optional[Union[str, Callable]] = 'f-rag',
                 ignore_stereo: bool = True,
                 smiles_col: str = 'smiles',
                 score_col: str = 'score',
                 scoring_method: str = 'average',
                 top_percent: float = 10.0,
                 min_frag_size: int = 1,
                 max_frag_size: int = 50,
                 min_count: int = 1,
                 max_fragments: Optional[int] = None,
                 verbose: bool = False):
        """
        Initialize FragmentVocabulary with data and parameters.        
        Args:
            data: Path to CSV file or DataFrame with SMILES and scores
            slicer: Slicing algorithm for encoding, defaults to "f-rag"
                Supported slicers: ["f-rag", "brics", "hr", "rotatable", "recap", "mmpa", "attach"]
            smiles_col: Name of SMILES column
            score_col: Name of score column
            scoring_method: 'average' or 'enrichment'
            top_percent: For enrichment scoring, percentage of top molecules to consider
            min_frag_size: Minimum fragment size (number of atoms)
            max_frag_size: Maximum fragment size (number of atoms)
            min_count: Minimum number of appearances for a fragment
            max_fragments: Maximum number of fragments to return (None for all)
            verbose: Whether to show progress and warnings
        """
        self.data = data
        self.smiles_col = smiles_col
        self.score_col = score_col
        self.scoring_method = scoring_method
        self.top_percent = top_percent
        self.min_frag_size = min_frag_size
        self.max_frag_size = max_frag_size
        self.min_count = min_count
        self.max_fragments = max_fragments
        self.verbose = verbose

        self.sfcodec = SAFECodec(slicer=slicer, ignore_stereo=ignore_stereo, verbose=self.verbose)
        
        # Initialize as None, will be crafted when needed
        self._vocab_df = None
        
    def craft_vocabulary(self) -> pd.DataFrame:
        """Craft the fragment vocabulary and return as DataFrame."""
        if self._vocab_df is not None:
            return self._vocab_df
            
        # Load and validate data
        df = self._load_and_validate_data()
        
        # Fragment molecules
        frag_stats = self._fragment_molecules(df)
        
        # Score fragments
        if self.scoring_method == 'average':
            self._vocab_df = self._score_by_average(frag_stats)
        elif self.scoring_method == 'enrichment':
            self._vocab_df = self._score_by_enrichment(frag_stats, df)
        else:
            raise ValueError(f"Unknown scoring method: {self.scoring_method}. Use 'average' or 'enrichment'")
        
        return self._vocab_df
    
    def _load_and_validate_data(self) -> pd.DataFrame:
        """Load and validate the input data."""
        if isinstance(self.data, str):
            df = pd.read_csv(self.data)
        else:
            df = self.data.copy()
        
        # Check required columns
        required_cols = {self.smiles_col, self.score_col}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Data must contain columns: {required_cols}. Found: {set(df.columns)}")
        
        # Sort by score (ascending: best first)
        df = df.sort_values(self.score_col, ascending=True)
        
        return df
    
    def _fragment_molecules(self, df: pd.DataFrame) -> dict:
        """Fragment molecules and collect statistics."""
        frag_counts = defaultdict(int)
        frag_score_sum = defaultdict(float)
        frag_molecules = defaultdict(list)  # Track which molecules contain each fragment
        
        success_count = 0
        iterator = tqdm(df.iterrows(), total=len(df), desc='Fragmenting molecules') if self.verbose else df.iterrows()
        
        for idx, row in iterator:
            smiles = row[self.smiles_col]
            score = row[self.score_col]
            try:
                molecule_sf = self.sfcodec.encode(smiles)
                if molecule_sf is None:
                    continue
                success_count += 1
                
                for fragment_sf in molecule_sf.split('.'):
                    fragment_smiles = self.sfcodec.canonicalize_frag(self.sfcodec.decode(fragment_sf))
                    if fragment_smiles is None:
                        continue
                    frag_counts[fragment_smiles] += 1
                    frag_score_sum[fragment_smiles] += score
                    frag_molecules[fragment_smiles].append(smiles)
                    
            except Exception as e:
                if self.verbose:
                    print(f"Error fragmenting {smiles}: {e}")
                continue
        
        success_rate = success_count / len(df) * 100
        if self.verbose:
            print(f"Successfully fragmented {success_count} molecules")
            print(f"Total fragments: {len(frag_counts)}")
            print(f"Success rate: {success_rate:.2f}%")
        
        # Warning for low success rate
        if success_rate < 80.0:
            print(f"WARNING: Low fragmentation success rate ({success_rate:.2f}%). "
                  f"Consider adjusting fragmentation strategy or checking molecule validity.")
        
        return {
            'counts': frag_counts,
            'score_sums': frag_score_sum,
            'molecules': frag_molecules
        }
    
    def _score_by_average(self, frag_stats: dict) -> pd.DataFrame:
        """Score fragments by averaging the scores of molecules containing them."""
        out_rows = []
        
        for frag, count in frag_stats['counts'].items():
            if count < self.min_count:
                continue
                
            avg_score = frag_stats['score_sums'][frag] / count
            
            # Compute size
            try:
                mol = Chem.MolFromSmiles(frag)
                size = mol.GetNumAtoms() if mol is not None else None
            except Exception:
                size = None
                
            if size is None or size < self.min_frag_size or size > self.max_frag_size:
                continue
                
            out_rows.append({
                'frag': frag,
                'count': count,
                'score': avg_score,
                'size': size
            })
        
        vocab_df = pd.DataFrame(out_rows)
        vocab_df = vocab_df.sort_values('score', ascending=True)
        
        if self.max_fragments is not None:
            vocab_df = vocab_df.head(self.max_fragments)
            
        return vocab_df
    
    def _score_by_enrichment(self, frag_stats: dict, df: pd.DataFrame) -> pd.DataFrame:
        """Score fragments by enrichment in top X% of molecules."""
        # Determine top molecules
        n_top = int(len(df) * self.top_percent / 100)
        top_molecules = set(df.head(n_top)[self.smiles_col].tolist())
        
        out_rows = []
        
        for frag, count in frag_stats['counts'].items():
            if count < self.min_count:
                continue
                
            # Count appearances in top molecules
            top_count = sum(1 for mol in frag_stats['molecules'][frag] if mol in top_molecules)
            
            # Calculate enrichment score (higher is better)
            if top_count > 0:
                enrichment = (top_count / len(top_molecules)) / (count / len(df))
            else:
                enrichment = 0.0
            
            # Compute size
            try:
                mol = Chem.MolFromSmiles(frag)
                size = mol.GetNumAtoms() if mol is not None else None
            except Exception:
                size = None
                
            if size is None or size < self.min_frag_size or size > self.max_frag_size:
                continue
                
            out_rows.append({
                'frag': frag,
                'count': count,
                'top_count': top_count,
                'score': enrichment,
                'size': size
            })
        
        vocab_df = pd.DataFrame(out_rows)
        vocab_df = vocab_df.sort_values('score', ascending=False)  # Higher enrichment is better
        
        if self.max_fragments is not None:
            vocab_df = vocab_df.head(self.max_fragments)
            
        return vocab_df
    
    def to_dataframe(self) -> pd.DataFrame:
        """Return the vocabulary as a DataFrame."""
        return self.craft_vocabulary()
    
    def __len__(self) -> int:
        """Return the number of fragments in the vocabulary."""
        return len(self.craft_vocabulary())
    
    def __getitem__(self, key):
        """Allow indexing into the vocabulary DataFrame."""
        return self.craft_vocabulary().__getitem__(key)
    
    def head(self, n: int = 5) -> pd.DataFrame:
        """Return the top n fragments."""
        return self.craft_vocabulary().head(n)
    
    def save(self, path: str):
        """Save the vocabulary to a CSV file."""
        self.craft_vocabulary().to_csv(path, index=False)


# Convenience function for backward compatibility
def craft_fragment_vocabulary(data: Union[str, pd.DataFrame],
                            smiles_col: str = 'smiles',
                            score_col: str = 'score',
                            scoring_method: str = 'average',
                            top_percent: float = 10.0,
                            min_frag_size: int = 1,
                            max_frag_size: int = 50,
                            min_count: int = 1,
                            max_fragments: Optional[int] = None,
                            verbose: bool = False) -> pd.DataFrame:
    """
    Convenience function to craft a fragment vocabulary.
    
    Args:
        data: Path to CSV file or DataFrame with SMILES and scores
        smiles_col: Name of SMILES column
        score_col: Name of score column
        scoring_method: 'average' or 'enrichment'
        top_percent: For enrichment scoring, percentage of top molecules to consider
        min_frag_size: Minimum fragment size (number of atoms)
        max_frag_size: Maximum fragment size (number of atoms)
        min_count: Minimum number of appearances for a fragment
        max_fragments: Maximum number of fragments to return (None for all)
        verbose: Whether to show progress and warnings
        
    Returns:
        DataFrame with columns: frag, count, score, size
    """
    vocab = FragmentVocabulary(
        data=data,
        smiles_col=smiles_col,
        score_col=score_col,
        scoring_method=scoring_method,
        top_percent=top_percent,
        min_frag_size=min_frag_size,
        max_frag_size=max_frag_size,
        min_count=min_count,
        max_fragments=max_fragments,
        verbose=verbose
    )
    return vocab.craft_vocabulary() 