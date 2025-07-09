import pandas as pd
import numpy as np
from typing import Optional, Union, List, Callable, Dict, Any
from collections import defaultdict
from rdkit import Chem
from tqdm import tqdm

from argo.frag_utils import SAFECodec

class FragmentVocabulary:
    """
    A class for crafting fragment vocabularies from molecular data with scores.
    Can be used directly as a vocab argument for f-RAG models.
    
    Key Features:
    - One-line initialization: vocab = FragmentVocabulary('scores.csv')
    - Add data incrementally: vocab.add('new_data.csv')
    - Rescore with new parameters: vocab.rescore(scoring_method='enrichment')
    - Simple API: Add data → Rescore → Get results
    
    Usage:
        # Basic usage
        vocab = FragmentVocabulary('scores.csv')
        df = vocab.to_dataframe()
        
        # Add more data and rescore
        vocab.add('new_scores.csv')
        vocab.rescore(scoring_method='enrichment', top_percent=5.0)
        df2 = vocab.to_dataframe()
        
        # Multiple additions
        vocab.add(df_part1) # rescores with current parameters
        vocab.add(df_part2) # rescores with current parameters
    """
    def __init__(self, 
                 data: Union[str, pd.DataFrame],
                 slicer: Union[str, Callable] = 'f-rag',
                 ignore_stereo: bool = True,
                 smiles_col: str = 'smiles',
                 score_col: str = 'score',
                 verbose: bool = False,
                 **kwargs):
        """
        Initialize FragmentVocabulary with a data source (CSV path or DataFrame).
        All other parameters are set via rescore (called with defaults here).
        """
        self.data = data
        self.smiles_col = smiles_col
        self.score_col = score_col
        self.verbose = verbose
        self.sfcodec = SAFECodec(slicer=slicer, ignore_stereo=ignore_stereo, verbose=self.verbose)
        self._vocab_df = None
        self._params: Dict[str, Any] = {}
        
        # Initialize fragment statistics
        self._frag_counts = defaultdict(int)
        self._frag_score_sum = defaultdict(float)
        self._frag_molecules = defaultdict(list)
        
        self._initialize_vocab(**kwargs)

    def add(self, new_data: Union[str, pd.DataFrame], use_tqdm: bool = True):
        """
        Add new molecules to fragment list and rescores based on current parameters.
        
        Args:
            new_data: Path to CSV file or DataFrame with new SMILES and scores
            use_tqdm: Whether to show a progress bar for fragmentation
        """
        # Load and validate new data using the existing method
        df_new = self._load_and_validate_data(new_data)
        
        # Update self.data to include the new data
        if isinstance(self.data, pd.DataFrame):
            # If self.data is already a DataFrame, concatenate
            # Check that columns match
            if set(self.data.columns) != set(df_new.columns):
                raise ValueError(f"New data columns {set(df_new.columns)} must match existing data columns {set(self.data.columns)}")
            self.data = pd.concat([self.data, df_new], ignore_index=True)
            # Re-sort the combined data
            self.data = self.data.sort_values(self.score_col, ascending=self.lower_is_better)
        else:
            # If self.data is a string (CSV path), convert to DataFrame and concatenate
            if isinstance(self.data, str):
                original_df = pd.read_csv(self.data)
            else:
                original_df = self.data.copy()
            # Check that columns match
            if set(original_df.columns) != set(df_new.columns):
                raise ValueError(f"New data columns {set(df_new.columns)} must match existing data columns {set(original_df.columns)}")
            self.data = pd.concat([original_df, df_new], ignore_index=True)
            self.data = self.data.sort_values(self.score_col, ascending=self.lower_is_better)
        
        # Update fragment statistics with new data
        self._update_fragment_stats(df_new, use_tqdm=use_tqdm)
        
        # Automatically rescore with current parameters
        self.rescore()

    def rescore(self, **kwargs):
        """
        Recalculate vocabulary using existing fragment statistics.
        If no parameters provided, uses current settings.
        
        Args:
            **kwargs: Parameters to update (scoring_method, top_percent, etc.)
                    Note: slicer, ignore_stereo, smiles_col, score_col cannot be changed
        """
        # Check for forbidden parameters
        forbidden_params = ['slicer', 'ignore_stereo', 'smiles_col', 'score_col']
        forbidden_changes = [param for param in forbidden_params if param in kwargs]
        if forbidden_changes:
            raise ValueError(f"Cannot change fragmentation parameters after initialization: {forbidden_changes}. "
                          f"These affect how molecules are fragmented and must be set during __init__.")
        
        # Update parameters if provided
        if kwargs:
            self._params.update(kwargs)
            # Update instance variables for the new parameters
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        
        # Score fragments with current parameters
        self._vocab_df = self._score_fragments()
        return self._vocab_df

    def to_dataframe(self) -> pd.DataFrame:
        """Return the vocabulary as a DataFrame."""
        return self._vocab_df

    def get_data(self) -> pd.DataFrame:
        """Return the accumulated data as a DataFrame."""
        if isinstance(self.data, str):
            return pd.read_csv(self.data)
        else:
            return self.data.copy()

    def get_params(self) -> Dict[str, Any]:
        """Return the parameters used for the last vocabulary calculation."""
        return self._params

    def reset_stats(self):
        """Reset fragment statistics to start fresh."""
        self._frag_counts.clear()
        self._frag_score_sum.clear()
        self._frag_molecules.clear()

    def __len__(self) -> int:
        return len(self._vocab_df)

    def __getitem__(self, key):
        return self._vocab_df.__getitem__(key)

    def head(self, n: int = 5) -> pd.DataFrame:
        return self._vocab_df.head(n)

    def save(self, path: str):
        self._vocab_df.to_csv(path, index=False)

    # Internal methods below
    def _initialize_vocab(self,
                        scoring_method: str = 'average',
                        top_percent: float = 10.0,
                        min_frag_size: int = 1,
                        max_frag_size: int = 50,
                        min_count: int = 1,
                        max_fragments: Optional[int] = None,
                        use_tqdm: bool = True,
                        lower_is_better: bool = True) -> pd.DataFrame:
        """
        Internal method to initialize the vocabulary with initial data.
        """
        self._params = dict(
            scoring_method=scoring_method, top_percent=top_percent, min_frag_size=min_frag_size,
            max_frag_size=max_frag_size, min_count=min_count, max_fragments=max_fragments,
            lower_is_better=lower_is_better
        )
        self.scoring_method = scoring_method
        self.top_percent = top_percent
        self.min_frag_size = min_frag_size
        self.max_frag_size = max_frag_size
        self.min_count = min_count
        self.max_fragments = max_fragments
        self.lower_is_better = lower_is_better

        # Load and validate data
        df = self._load_and_validate_data(self.data)
        # Fragment molecules and update stats
        self._update_fragment_stats(df, use_tqdm=use_tqdm)
        # Score fragments
        self._vocab_df = self._score_fragments()
        return self._vocab_df

    def _score_fragments(self) -> pd.DataFrame:
        """
        Internal method to score fragments using current parameters and fragment statistics.
        """
        if self.scoring_method == 'average':
            return self._score_by_average({
                'counts': self._frag_counts, 
                'score_sums': self._frag_score_sum, 
                'molecules': self._frag_molecules
            })
        elif self.scoring_method == 'enrichment':
            # Use self.data for enrichment scoring (contains all accumulated data)
            all_data = self._load_and_validate_data(self.data)
            return self._score_by_fold_enrichment({
                'counts': self._frag_counts, 
                'score_sums': self._frag_score_sum, 
                'molecules': self._frag_molecules
            }, all_data)
        else:
            raise ValueError(f"Unknown scoring method: {self.scoring_method}. Use 'average' or 'enrichment'")

    def _load_and_validate_data(self, data: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Load and validate data from a source.
        
        Args:
            data: Data source (CSV path or DataFrame).
        """
        if isinstance(data, str):
            df = pd.read_csv(data)
        else:
            df = data.copy()
        required_cols = {self.smiles_col, self.score_col}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Data must contain columns: {required_cols}. Found: {set(df.columns)}")
        # Sort by score (ascending if lower_is_better, descending otherwise)
        df = df.sort_values(self.score_col, ascending=self.lower_is_better)
        return df

    def _update_fragment_stats(self, df: pd.DataFrame, use_tqdm: bool = True):
        """
        Fragment molecules and update the internal fragment statistics.
        """
        frag_stats = self._fragment_molecules(df, use_tqdm=use_tqdm)
        
        # Update stats with new fragments
        for frag, count in frag_stats['counts'].items():
            self._frag_counts[frag] += count
        for frag, score_sum in frag_stats['score_sums'].items():
            self._frag_score_sum[frag] += score_sum
        for frag, mols in frag_stats['molecules'].items():
            self._frag_molecules[frag].extend(mols)

    def _fragment_molecules(self, df: pd.DataFrame, use_tqdm: bool = True) -> dict:
        """
        Fragment molecules and return fragment statistics.
        """
        frag_counts = defaultdict(int)
        frag_score_sum = defaultdict(float)
        frag_molecules = defaultdict(list)
        success_count = 0
        
        # Handle empty DataFrame
        if df.empty:
            print("Warning: Empty DataFrame provided for fragmentation")
            return {
                'counts': frag_counts,
                'score_sums': frag_score_sum,
                'molecules': frag_molecules
            }
        
        iterator = tqdm(df.iterrows(), total=len(df), desc='Fragmenting molecules') if use_tqdm else df.iterrows()
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
        print(f"Successfully fragmented {success_count} molecules")
        print(f"Total fragments: {len(frag_counts)}")
        print(f"Success rate: {success_rate:.2f}%")
        
        if success_rate < 80.0:
            print(f"WARNING: Low fragmentation success rate ({success_rate:.2f}%). "
                  f"Consider adjusting fragmentation strategy or checking molecule validity.")
        return {
            'counts': frag_counts,
            'score_sums': frag_score_sum,
            'molecules': frag_molecules
        }

    def _score_by_average(self, frag_stats: dict) -> pd.DataFrame:
        """
        Score fragments by average score of compounds containing each fragment.
        """
        out_rows = []
        for frag, count in frag_stats['counts'].items():
            if count < self.min_count:
                continue
            avg_score = frag_stats['score_sums'][frag] / count
            try:
                mol = Chem.MolFromSmiles(frag)
                size = mol.GetNumAtoms() if mol is not None else None
            except Exception:
                size = None
            if size is None or size < self.min_frag_size or size > self.max_frag_size:
                continue
            out_rows.append({'frag': frag, 'count': count, 'score': avg_score, 'size': size})
        vocab_df = pd.DataFrame(out_rows)
        # Sort by score (ascending if lower_is_better, descending otherwise)
        vocab_df = vocab_df.sort_values('score', ascending=self.lower_is_better)
        if self.max_fragments is not None:
            vocab_df = vocab_df.head(self.max_fragments)
        return vocab_df

    def _score_by_fold_enrichment(self, frag_stats: dict, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score fragments by fold enrichment in top vs bottom compounds.
        """
        n_top = int(len(df) * self.top_percent / 100)
        n_bottom = int(len(df) * (100 - self.top_percent) / 100)  # Bottom (100-x)%
        
        top_molecules = set(df.head(n_top)[self.smiles_col].tolist())
        bottom_molecules = set(df.tail(n_bottom)[self.smiles_col].tolist())
        
        out_rows = []
        for frag, count in frag_stats['counts'].items():
            if count < self.min_count:
                continue
                
            # Count appearances in top and bottom
            top_count = sum(1 for mol in frag_stats['molecules'][frag] if mol in top_molecules)
            bottom_count = sum(1 for mol in frag_stats['molecules'][frag] if mol in bottom_molecules)
            
            # Calculate fold enrichment
            if bottom_count > 0:
                # Fold enrichment = (top_freq / bottom_freq)
                top_freq = top_count / len(top_molecules)
                bottom_freq = bottom_count / len(bottom_molecules)
                enrichment = top_freq / bottom_freq
            else:
                # If no appearances in bottom, enrichment is based on top frequency only
                # Use a high enrichment value to indicate strong positive enrichment
                enrichment = (top_count / len(top_molecules)) * 10.0  # Arbitrary multiplier for zero bottom case
            
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
                'bottom_count': bottom_count,
                'score': enrichment, 
                'size': size
            })
        vocab_df = pd.DataFrame(out_rows)
        # Sort by enrichment score (higher is better, so descending)
        vocab_df = vocab_df.sort_values('score', ascending=False)
        if self.max_fragments is not None:
            vocab_df = vocab_df.head(self.max_fragments)
        return vocab_df