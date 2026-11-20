"""Build the sample table from the GSE270988 count matrix header.

Column names follow the pattern <genotype>_<stage>_round_<k>_rep<j>,
e.g. Col-0_20d_round_3_rep1 or abi3-12_dog1-4_DS_round_3_rep2.
The script only parses labels; it does not read expression values.
"""

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COUNTS = ROOT / "data" / "raw" / "GSE270988_raw_counts.csv"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATTERN = re.compile(
    r"^(?P<genotype>.+)_(?P<stage>\d+d|DS)_round_(?P<round>\d+)_rep(?P<rep>\d+)$"
)


def main() -> None:
    header = pd.read_csv(COUNTS, nrows=0)
    cols = [c for c in header.columns if c != "gene_id"]

    rows = []
    for col in cols:
        m = PATTERN.match(col)
        if m is None:
            sys.exit(f"Unparsed column name: {col}")
        d = m.groupdict()
        d["sample_id"] = col
        d["genotype"] = d["genotype"]
        d["stage"] = d["stage"]
        d["round"] = int(d["round"])
        d["rep"] = int(d["rep"])
        rows.append(d)

    samples = pd.DataFrame(rows)[
        ["sample_id", "genotype", "stage", "round", "rep"]
    ]
    samples["stage_day"] = samples["stage"].str.rstrip("d")
    samples["stage_day"] = samples["stage_day"].replace({"DS": "dry"})

    samples.to_csv(OUT_DIR / "samples.csv", index=False)

    print(f"samples: {len(samples)}")
    print("\ngenotype x stage (all rounds):")
    print(pd.crosstab(samples["genotype"], samples["stage"]).to_string())

    print("\nCol-0 stage x round:")
    col0 = samples[samples["genotype"] == "Col-0"]
    print(pd.crosstab(col0["stage"], col0["round"]).to_string())

    print("\ndry seed samples by genotype and round:")
    ds = samples[samples["stage"] == "DS"]
    print(pd.crosstab(ds["genotype"], ds["round"]).to_string())

    print("\nround-3 dry seed genotype x replicate:")
    ds3 = ds[ds["round"] == 3]
    print(pd.crosstab(ds3["genotype"], ds3["rep"]).to_string())


if __name__ == "__main__":
    main()
