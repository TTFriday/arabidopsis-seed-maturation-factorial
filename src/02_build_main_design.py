"""Freeze the 24-sample 2x2x2 factorial design at dry seed (round 3).

Factors: A = abi3-12, D = dog1-4, B = aba2-1.
Genotypes are parsed by mutation tokens so that names containing
'Col-0' (e.g. the DOG1-Cvi line) cannot leak into the design.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "processed" / "samples.csv"
COUNTS = ROOT / "data" / "raw" / "GSE270988_raw_counts.csv"
OUT_DIR = ROOT / "data" / "processed"

FACTOR_TOKENS = {"A": "abi3-12", "D": "dog1-4", "B": "aba2-1"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    samples = pd.read_csv(SAMPLES)
    ds3 = samples[(samples["stage"] == "DS") & (samples["round"] == 3)].copy()

    def genotype_levels(name: str) -> dict:
        tokens = set(name.split("_"))
        if tokens == {"Col-0"}:
            return {"A": 0, "D": 0, "B": 0}
        if tokens - set(FACTOR_TOKENS.values()) or not tokens & set(FACTOR_TOKENS.values()):
            return {"A": None}
        return {k: int(v in tokens) for k, v in FACTOR_TOKENS.items()}

    levels = ds3["genotype"].map(genotype_levels)
    design = ds3.assign(**pd.DataFrame(list(levels), index=ds3.index))
    design = design[design["A"].notna()].copy()
    for k in FACTOR_TOKENS:
        design[k] = design[k].astype(int)

    design = design.sort_values(["A", "D", "B", "rep"])
    design = design[["sample_id", "genotype", "A", "D", "B", "rep"]]

    assert len(design) == 24, f"expected 24 samples, got {len(design)}"
    counts = design.groupby(["A", "D", "B"]).size()
    assert (counts == 3).all(), f"unbalanced design:\n{counts}"

    design.to_csv(OUT_DIR / "main_design_24.csv", index=False)

    with open(OUT_DIR / "input_hashes.json", "w", encoding="utf-8") as fh:
        json.dump({"GSE270988_raw_counts.csv": sha256(COUNTS)}, fh, indent=2)

    print(design.to_string(index=False))
    print("\n8 genotype combinations x 3 replicates: OK")
    print("counts sha256:", sha256(COUNTS))


if __name__ == "__main__":
    main()
