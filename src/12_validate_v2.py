"""External validation V2: GSE5634 AtGenExpress seed development, stages 6-10.

Implements the frozen rules in results/external/frozen_external.json:
GPL198 probe mapping, the frozen 500-up/500-down maturity signature, and a
stage-level ordinal test.  Values in the series matrix are linear MAS5
signal, so log2 is applied once (the frozen rule forbids only a *second*
log).

GPL198.annot.gz carries Entrez Gene IDs but no TAIR AGI column, so probes are
first mapped to Entrez IDs by GPL198 and then to AGI through the LocusTag,
Synonyms and dbXrefs fields of NCBI Arabidopsis gene_info.  Organellar probes
without an AGI are dropped; this bridge is fixed before any V2 expression
value is used and is reported in the summary.
"""

import gzip
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "data" / "raw" / "external" / "GSE5634" / "GSE5634_series_matrix.txt.gz"
ANNOT = ROOT / "data" / "raw" / "external" / "GPL198" / "GPL198.annot.gz"
GENE_INFO = ROOT / "data" / "raw" / "external" / "GPL198" / "Arabidopsis_thaliana.gene_info.gz"
FROZEN = ROOT / "results" / "external" / "frozen_external.json"
OUT = ROOT / "results" / "external"
STAGES = [6, 7, 8, 9, 10]
AGI = re.compile(r"AT[1-5]G\d{5}")


def load_series():
    rows = [ln for ln in gzip.open(MATRIX, "rt") if not ln.startswith("!")]
    df = pd.read_csv(io.StringIO("".join(rows)), sep="\t")
    acc, chars = [], []
    for ln in gzip.open(MATRIX, "rt"):
        if ln.startswith("!Sample_geo_accession"):
            acc = [x.strip('"') for x in ln.rstrip().split("\t")[1:]]
        elif ln.startswith("!Sample_characteristics_ch1"):
            chars.append([x.strip('"') for x in ln.rstrip().split("\t")[1:]])
    stage_of = {}
    for gsm, txt in zip(acc, chars[0]):
        m = re.search(r"stage (\d+)", txt)
        if m:
            stage_of[gsm] = int(m.group(1))
    keep = [g for g in acc if stage_of.get(g) in STAGES]
    assert len(keep) == 15, len(keep)
    day = np.array([stage_of[g] for g in keep], float)
    return df.set_index("ID_REF")[keep], day, keep


def load_annotation():
    with gzip.open(ANNOT, "rt") as fh:
        lines = fh.readlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("!platform_table_begin")) + 1
    annot = pd.read_csv(io.StringIO("".join(lines[start:])), sep="\t", dtype=str)
    probe_entrez = annot[["ID", "Gene ID"]].dropna()
    probe_entrez.columns = ["probe", "GeneID"]
    probe_entrez["GeneID"] = probe_entrez["GeneID"].str.strip()

    gi_cols = ["tax_id", "GeneID", "Symbol", "LocusTag", "Synonyms", "dbXrefs",
               "chromosome", "map_location", "description", "type_of_gene",
               "symbol_auth", "full_auth", "status", "other", "mod", "feature"]
    gene_info = pd.read_csv(GENE_INFO, sep="\t", header=None, names=gi_cols, dtype=str)

    def to_agi(row):
        for field in ("LocusTag", "Synonyms", "dbXrefs"):
            val = row[field]
            if isinstance(val, str):
                hit = AGI.search(val.upper())
                if hit:
                    return hit.group(0)
        return None

    gene_info["gene_id"] = gene_info.apply(to_agi, axis=1)
    bridge = gene_info[["GeneID", "gene_id"]].dropna().drop_duplicates()
    pairs = probe_entrez.merge(bridge, on="GeneID", how="inner")
    return pairs[["probe", "gene_id"]].drop_duplicates()


def zscore(mat):
    return (mat - mat.mean(axis=1, keepdims=True)) / np.maximum(
        mat.std(axis=1, keepdims=True), 1e-9)


def signature_score(z, probe_map, genes):
    idx = [probe_map[g] for g in genes if g in probe_map]
    return z.loc[idx].mean(axis=0), len(idx)


def main() -> None:
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    up = frozen["S_maturity"]["up"]["genes"]
    down = frozen["S_maturity"]["down"]["genes"]

    expr, day, gsms = load_series()
    pairs = load_annotation()
    print(f"arrays {expr.shape[1]}, probes {expr.shape[0]}, probe-gene pairs {len(pairs)}")

    log2expr = np.log2(np.maximum(expr.to_numpy(dtype=np.float64), 1e-6))
    log2df = pd.DataFrame(log2expr, index=expr.index, columns=expr.columns)
    mean_expr = log2df.mean(axis=1)
    pairs["mean_expr"] = mean_expr.reindex(pairs["probe"]).to_numpy()
    best = pairs.sort_values("mean_expr", ascending=False).drop_duplicates("gene_id")
    probe_map = dict(zip(best["gene_id"], best["probe"]))
    mapped_genes = set(probe_map)

    up_hit = [g for g in up if g in mapped_genes]
    down_hit = [g for g in down if g in mapped_genes]
    coverage = {
        "up_total": len(up), "up_mapped": len(up_hit),
        "down_total": len(down), "down_mapped": len(down_hit),
        "combined_fraction": (len(up_hit) + len(down_hit)) / (len(up) + len(down)),
    }
    gate = coverage["combined_fraction"] >= 0.5

    z = zscore(log2expr)
    zdf = pd.DataFrame(z, index=expr.index, columns=expr.columns)
    up_score, n_up = signature_score(zdf, probe_map, up_hit)
    down_score, n_down = signature_score(zdf, probe_map, down_hit)
    score = up_score - down_score

    stage_tab = pd.DataFrame({"stage": day, "score": score, "gsm": gsms})
    stage_means = stage_tab.groupby("stage")["score"].agg(["mean", "std", "count"])
    rho_stage = stats.spearmanr(stage_means.index, stage_means["mean"])
    rho_array = stats.spearmanr(day, score)

    # exact permutation over the 5 stage means: asymptotic p is invalid at n=5
    from itertools import permutations
    vals = stage_means["mean"].to_numpy()
    obs = rho_stage.statistic
    null = np.array([stats.spearmanr(np.arange(5), vals[list(p)]).statistic
                     for p in permutations(range(5))])
    p_exact = float((np.abs(null) >= abs(obs) - 1e-12).mean())

    marker_rows = []
    for agi, name in frozen["markers"].items():
        if agi not in probe_map:
            marker_rows.append({"gene_id": agi, "name": name, "mapped": False})
            continue
        probe = probe_map[agi]
        vals = log2df.loc[probe]
        row = {"gene_id": agi, "name": name, "mapped": True, "probe": probe}
        for s in STAGES:
            row[f"stage{s}"] = float(vals[day == s].mean())
        row["delta_10_minus_6"] = row["stage10"] - row["stage6"]
        marker_rows.append(row)
    markers = pd.DataFrame(marker_rows)

    summary = {
        "data": "GSE5634 AtGenExpress, GSM131694-131708, stages 6-10, 3 arrays each",
        "n_arrays": int(len(gsms)),
        "value_scale": "linear MAS5 signal as deposited; log2 applied once before z-scoring",
        "mapping": {
            "platform": "GPL198 ATH1",
            "probe_to_agi_pairs": int(len(pairs)),
            "unique_agi": int(pairs["gene_id"].nunique()),
            "bridge": ("GPL198 Entrez Gene ID -> NCBI Arabidopsis gene_info "
                       "(LocusTag/Synonyms/dbXrefs); organellar probes without AGI dropped"),
        },
        "coverage": coverage,
        "coverage_gate_pass": bool(gate),
        "maturity_score": {
            "n_up_used": int(n_up), "n_down_used": int(n_down),
            "stage_means": {int(k): float(v) for k, v in stage_means["mean"].items()},
            "stage_sd": {int(k): float(v) for k, v in stage_means["std"].items()},
            "spearman_stage_means_n5": float(rho_stage.statistic),
            "spearman_stage_means_p_asymptotic_not_used": float(rho_stage.pvalue),
            "spearman_stage_means_p_exact_two_sided": p_exact,
            "n_stage_permutations": 120,
            "spearman_arrays_n15_descriptive": float(rho_array.statistic),
            "spearman_arrays_p_descriptive": float(rho_array.pvalue),
            "monotone_stage_means": bool(np.all(np.diff(stage_means["mean"]) > 0)),
            "delta_stage10_minus_stage6": float(
                stage_means.loc[10, "mean"] - stage_means.loc[6, "mean"]),
        },
        "markers": markers.to_dict(orient="records"),
    }
    if not gate:
        summary["status"] = "insufficient coverage of the frozen maturity signature on GPL198"
    (OUT / "v2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    stage_tab.to_csv(OUT / "v2_stage_scores.csv", index=False)
    markers.to_csv(OUT / "v2_marker_stages.csv", index=False)
    hit = sorted(mapped_genes & set(up + down))
    pd.DataFrame({"gene_id": hit, "probe": [probe_map[g] for g in hit],
                  "direction": ["up" if g in up else "down" for g in hit],
                  "mean_log2": [float(log2df.loc[probe_map[g]].mean()) for g in hit]}
                 ).to_csv(OUT / "v2_signature_genes.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
