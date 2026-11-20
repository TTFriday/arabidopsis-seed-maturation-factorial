"""Audit candidate stability after the pre-registered broad filter.

The pre-registered candidate rule (LRT q<0.05 and |max contrast|>=1) is kept
unchanged.  This script adds a post-hoc numerical-stability diagnostic:
a contrast is "supported" when q<0.05, |log2 effect|>=1 and its Wald SE<=1;
genes whose only large effects have SE>2 are flagged as unstable separation
artifacts.  Both counts are reported so the broad result is never hidden.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
GENE_INFO = ROOT / "data" / "raw" / "external" / "GPL198" / "Arabidopsis_thaliana.gene_info.gz"
NAMES = ["AD_given_B0", "AD_given_B1", "AB_given_D0", "AB_given_D1",
         "DB_given_A0", "DB_given_A1", "ADB_three_way"]
AGI = re.compile(r"AT[1-5]G\d{5}")


def gene_symbols():
    cols = ["tax_id", "GeneID", "Symbol", "LocusTag", "Synonyms", "dbXrefs",
            "chromosome", "map_location", "description", "type_of_gene",
            "symbol_auth", "full_auth", "status", "other", "mod", "feature"]
    g = pd.read_csv(GENE_INFO, sep="\t", header=None, names=cols, dtype=str)

    def to_agi(row):
        for field in ("LocusTag", "Synonyms", "dbXrefs"):
            val = row[field]
            if isinstance(val, str):
                hit = AGI.search(val.upper())
                if hit:
                    return hit.group(0)
        return None

    g["gene_id"] = g.apply(to_agi, axis=1)
    g = g.dropna(subset=["gene_id"]).drop_duplicates("gene_id")
    return g.set_index("gene_id")["Symbol"]


def main() -> None:
    c = pd.read_csv(T / "contrasts_all_genes.csv.gz").set_index("gene_id")
    lrt = pd.read_csv(T / "lrt_all_genes.csv.gz").set_index("gene_id")
    E = np.column_stack([c[f"{n}_log2"] for n in NAMES])
    S = np.column_stack([c[f"{n}_se"] for n in NAMES])
    Q = np.column_stack([c[f"{n}_q"] for n in NAMES])

    selected = (Q < 0.05) & (np.abs(E) >= 1)
    supported = selected & (S <= 1.0)
    unstable = (np.abs(E) > 10) & (S > 2)

    broad = (lrt["qvalue"] < 0.05) & (np.abs(E).max(axis=1) >= 1)
    stable_any = supported.any(axis=1) & (lrt["qvalue"] < 0.05).to_numpy()

    per_contrast = {}
    for j, name in enumerate(NAMES):
        per_contrast[name] = {
            "n_selected": int(selected[:, j].sum()),
            "n_supported_stable": int((supported[:, j] & (lrt["qvalue"] < 0.05).to_numpy()).sum()),
        }

    jmax = np.abs(E).argmax(axis=1)
    out = pd.DataFrame({"gene_id": c.index, "base_mean": c["base_mean"],
                        "lrt_q": lrt.loc[c.index, "qvalue"].to_numpy(),
                        "n_selected": selected.sum(axis=1),
                        "n_supported_stable": supported.sum(axis=1),
                        "max_selected_abs": np.where(selected, np.abs(E), 0).max(axis=1),
                        "max_selected_se": np.where(selected, S, 0).max(axis=1),
                        "max_abs_all": np.abs(E).max(axis=1),
                        "se_at_max_abs": S[np.arange(len(S)), jmax],
                        "unstable_extreme": unstable.any(axis=1)})
    for j, name in enumerate(NAMES):
        out[f"{name}_log2"] = E[:, j]
        out[f"{name}_se"] = S[:, j]
        out[f"{name}_q"] = Q[:, j]
    sym = gene_symbols()
    out.insert(1, "symbol", sym.reindex(out["gene_id"]).to_numpy())
    out.to_csv(T / "candidate_audit_all_genes.csv.gz", index=False, compression="gzip")
    out[stable_any].sort_values("lrt_q").to_csv(T / "candidates_stable.csv", index=False)

    summary = {
        "pre_registered_broad": {
            "rule": "LRT q<0.05 and max |seven contrasts| >=1 log2",
            "n": int(broad.sum()),
        },
        "post_hoc_stability": {
            "rule": ("at least one contrast with q<0.05, |log2|>=1 and Wald SE<=1; "
                     "LRT q<0.05"),
            "n": int(stable_any.sum()),
        },
        "unstable_extreme": {
            "rule": "any contrast with |log2|>10 and Wald SE>2 (separation artifact)",
            "n_genes": int(unstable.any(axis=1).sum()),
            "n_among_broad": int((unstable.any(axis=1) & broad).sum()),
        },
        "per_contrast": per_contrast,
        "max_selected_se_quantiles": [float(x) for x in
                                      np.quantile(out.loc[broad, "max_selected_se"],
                                                  [0.5, 0.9, 0.99, 1.0])],
    }
    (T / "candidate_audit_summary.json").write_text(json.dumps(summary, indent=2),
                                                    encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
