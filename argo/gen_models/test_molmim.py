#!/usr/bin/env python3
"""
Comprehensive tests for the MolMIM generator (self‑hosted API mode).

Two modes are exercised:

  • property_optimization   – CMA‑ES guided optimisation toward high QED
  • biased_generation       – simple, bias‑only sampling (no CMA‑ES)

For each call we:
  – check the API responds,
  – verify returned SMILES strings are valid,
  – optionally compute QED scores (TDCommons Oracle).

Edit BASE_URL or export MOLMIM_BASE_URL if your tunnel uses a
different local port.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd                # noqa: F401  (imported for parity; unused)
from rdkit import Chem
from tdc import Oracle

# ---------------------------------------------------------------------
#  add project root so   `from argo...`  works when the script is run
#  directly without installing the package
# ---------------------------------------------------------------------
sys.path.append(str(Path(__file__).parent.parent))

from argo.gen_models.api import GenerationModel, GenerationTask   # noqa: E402

# ---------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------
BASE_URL = os.getenv("MOLMIM_BASE_URL", "http://localhost:18080")
SEED = "CCO"                        # tiny ethanol – fast convergence demo


# ---------------------------------------------------------------------
#  Helper
# ---------------------------------------------------------------------
def _validate_smiles(smiles_list: List[str]) -> List[str]:
    """Return only the SMILES strings that RDKit can parse."""
    return [s for s in smiles_list if Chem.MolFromSmiles(s) is not None]


# ---------------------------------------------------------------------
#  Tests
# ---------------------------------------------------------------------
def test_property_optimisation() -> List[str]:
    """Run a quick CMA‑ES optimisation towards high QED."""
    print("\n" + "=" * 60)
    print("TESTING MolMIM – PROPERTY OPTIMISATION (QED)")
    print("=" * 60)

    # --- QED oracle ---------------------------------------------------
    qed_oracle = Oracle(name="qed")

    # --- model --------------------------------------------------------
    molmim = GenerationModel(
        model_type="molmim",
        base_url=BASE_URL,           # new arg we added in MolMIMGenerator
    )

    task = GenerationTask(
        mode="property_optimization",
        objective="qed",
        config=dict(
            iterations=3,
            n_samples=5,
            random_seed=42,
        ),
        seed_smiles=SEED,
    )

    try:
        smiles = molmim.generate(task)
        valid = _validate_smiles(smiles)

        print(f"✓ optimisation call succeeded ({len(smiles)} SMILES returned)")
        print(f"  valid molecules: {len(valid)}/{len(smiles)}")

        # evaluate QED
        if valid:
            scores = [qed_oracle(s) for s in valid]
            print(
                "  QED ⟨avg / max / min⟩ : "
                f"{np.mean(scores):.3f} / {np.max(scores):.3f} / {np.min(scores):.3f}"
            )
            print("  sample:", valid[:3])
        return valid

    except Exception as exc:                                 # pragma: no cover
        print(f"✗ optimisation failed : {exc}")
        return []


def test_biased_generation() -> List[str]:
    """Plain biased generation (no CMA‑ES, `algorithm='none'`)."""
    print("\n" + "=" * 60)
    print("TESTING MolMIM – BIASED GENERATION")
    print("=" * 60)

    molmim = GenerationModel(model_type="molmim", base_url=BASE_URL)

    task = GenerationTask(
        mode="biased_generation",
        config=dict(
            n_samples=5,
            random_seed=123,
        ),
        seed_smiles=SEED,
    )

    try:
        smiles = molmim.generate(task)
        valid = _validate_smiles(smiles)

        print(f"✓ biased generation succeeded ({len(smiles)} SMILES returned)")
        print(f"  valid molecules: {len(valid)}/{len(smiles)}")
        print("  sample:", valid[:3])
        return valid

    except Exception as exc:                                 # pragma: no cover
        print(f"✗ biased generation failed : {exc}")
        return []


# ---------------------------------------------------------------------
#  Entrypoint
# ---------------------------------------------------------------------
def run_all() -> None:
    print("\nMolMIM SELF‑HOSTED API TEST SUITE")
    print("=" * 60)

    res_opt = test_property_optimisation()
    res_bias = test_biased_generation()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f" optimisation : {'✓' if res_opt else '✗'}  ({len(res_opt)} mols)")
    print(f" biased gen   : {'✓' if res_bias else '✗'}  ({len(res_bias)} mols)")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
