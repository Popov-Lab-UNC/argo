#!/usr/bin/env python3
"""
run_molmim_local.py
-------------------
Quick CLI wrapper that sends a single GenerationTask to a *self‑hosted* MolMIM
server (the one you started on Longleaf and tunneled to your desktop).

Usage
-----

# basic use (defaults listed below)
python scripts/run_molmim_local.py \
    --seed "CCO" \
    --iterations 5 \
    --samples 5 \
    --base-url http://localhost:18080 \
    --out molecules.json

All arguments have sensible defaults, so you can omit most of them.
"""

import argparse
import json
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from argo.gen_models.api import MolMIMGenerator, GenerationTask


def main():
    parser = argparse.ArgumentParser(description="Self‑hosted MolMIM test driver")
    parser.add_argument("--seed",        required=True, help="Seed SMILES string")
    parser.add_argument("--iterations",  type=int, default=10)
    parser.add_argument("--samples",     type=int, default=10)
    parser.add_argument("--base-url",    default=os.getenv("MOLMIM_BASE_URL", "http://localhost:18080"))
    parser.add_argument("--out",         default="molmim_out.json", help="Output file")
    args = parser.parse_args()

    gen = MolMIMGenerator(base_url=args.base_url)

    task = GenerationTask(
        seed_smiles=args.seed,
        mode="property_optimization",
        config=dict(
            iterations=args.iterations,
            n_samples=args.samples,
        )
    )

    molecules = gen.generate(task)

    # Save to JSON for inspection
    with open(args.out, "w") as f:
        json.dump(molecules, f, indent=2)

    print(f"Generated {len(molecules)} molecules -> {args.out}")


if __name__ == "__main__":
    main()
