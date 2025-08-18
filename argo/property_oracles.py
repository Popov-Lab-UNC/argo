"""
Property calculation oracles to replace PyTDC functionality.
This module provides RDKit-based implementations of common molecular properties.
"""

import numpy as np
from typing import List, Union
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED
from rdkit.Contrib.SA_Score import sascorer


class PropertyOracle:
    """Base class for property calculation oracles."""
    
    def __init__(self, name: str):
        self.name = name.lower()
        
    def __call__(self, smiles_list: Union[str, List[str]]) -> Union[float, List[float]]:
        """Calculate property values for given SMILES strings."""
        if isinstance(smiles_list, str):
            return self._calculate_single(smiles_list)
        else:
            return [self._calculate_single(smiles) for smiles in smiles_list]
    
    def _calculate_single(self, smiles: str) -> float:
        """Calculate property for a single SMILES string."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan
        return self._calculate_property(mol)
    
    def _calculate_property(self, mol: Chem.Mol) -> float:
        """Calculate the specific property. Override in subclasses."""
        raise NotImplementedError


class QEDOracle(PropertyOracle):
    """QED (Quantitative Estimate of Drug-likeness) oracle."""
    
    def __init__(self):
        super().__init__("qed")
    
    def _calculate_property(self, mol: Chem.Mol) -> float:
        """Calculate QED score."""
        try:
            return QED.qed(mol)
        except:
            return np.nan


class SAOracle(PropertyOracle):
    """Synthetic Accessibility (SA) oracle."""
    
    def __init__(self):
        super().__init__("sa")
    
    def _calculate_property(self, mol: Chem.Mol) -> float:
        """Calculate SA score (lower is more accessible)."""
        try:
            return sascorer.calculateScore(mol)
        except:
            return np.nan


class LogPOracle(PropertyOracle):
    """LogP (octanol-water partition coefficient) oracle."""
    
    def __init__(self):
        super().__init__("logp")
    
    def _calculate_property(self, mol: Chem.Mol) -> float:
        """Calculate LogP value."""
        try:
            return Crippen.MolLogP(mol)
        except:
            return np.nan


class Oracle:
    """Drop-in replacement for PyTDC Oracle class."""
    
    def __init__(self, name: str):
        self.name = name.lower()
        if self.name == "qed":
            self._oracle = QEDOracle()
        elif self.name == "sa":
            self._oracle = SAOracle()
        elif self.name == "logp":
            self._oracle = LogPOracle()
        else:
            raise ValueError(f"Unsupported oracle name: {name}. Supported: ['qed', 'sa', 'logp']")
    
    def __call__(self, smiles_list: Union[str, List[str]]) -> Union[float, List[float]]:
        """Calculate property values using the appropriate oracle."""
        return self._oracle(smiles_list)
