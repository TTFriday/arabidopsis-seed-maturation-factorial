"""Genome-wide factorial analysis of the 24 dry-seed samples.

Steps: poscounts size factors -> pre-registered CPM filter -> Cox-Reid
dispersion grid -> trend and empirical-Bayes shrinkage -> 4-df interaction
LRT (full ~A*D*B vs ~A+D+B) -> seven planned Wald contrasts -> BH.

All genes passing the filter are reported, not only significant ones.
"""

import json
import platform
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import nb_glm as nb

ROOT = Path(__file__).resolve().parents[1]
COUNTS = ROOT / "data" / "raw" / "GSE270988_raw_counts.csv"
DESIGN = ROOT / "data" / "processed" / "main_design_24.csv"
OUT_T = ROOT / "results" / "tables"
OUT_F = ROOT / "results" / "figures"
OUT_T.mkdir(parents=True, exist_ok=True)
OUT_F.mkdir(parents=True, exist_ok=True)

CONTRASTS = {
    "AD_given_B0": [0, 0, 0, 0, 1, 0, 0, 0],
    "AD_given_B1": [0, 0, 0, 0, 1, 0, 0, 1],
    "AB_given_D0": [0, 0, 0, 0, 0, 1, 0, 0],
    "AB_given_D1": [0, 0, 0, 0, 0, 1, 0, 1],
    "DB_given_A0": [0, 0, 0, 0, 0, 0, 1, 0],
    "DB_given_A1": [0, 0, 0, 0, 0, 0, 1, 1],
    "ADB_three_way": [0, 0, 0, 0, 0, 0, 0, 1],
}


def bh_qvalues(p):
    p = np.asarray(p, dtype=np.float64)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def main() -> None:
    t_start = time.time()
    df = pd.read_csv(COUNTS)
    genes = df["gene_id"].astype(str).to_numpy()
    mat = df.drop(columns="gene_id").to_numpy(dtype=np.int64)
    cols = [c for c in df.columns if c != "gene_id"]
    design = pd.read_csv(DESIGN)

    idx = [cols.index(s) for s in design["sample_id"]]
    y_genes = mat[:, idx].astype(np.float64)
    libs = y_genes.sum(axis=0)
    cpm = y_genes / libs * 1e6
    keep = (cpm >= 1.0).sum(axis=1) >= 3
    y = y_genes[keep].T
    genes = genes[keep]
    print(f"filtered genes: {len(genes):,} of {len(mat):,}")

    sf = nb.poscounts_size_factors(y)
    offset = np.log(sf)
    x_full, names = nb.factorial_matrix(design["A"], design["D"], design["B"])
    x_red = x_full[:, :4]

    t0 = time.time()
    alpha_gene, _ = nb.estimate_dispersions(y, x_full, offset)
    base_mean = (y / sf[:, None]).mean(axis=0)
    trend = nb.dispersion_trend(base_mean, alpha_gene)
    alpha, df_prior = nb.shrink_dispersions(alpha_gene, trend, df_resid=16)
    t_disp = time.time() - t0

    t0 = time.time()
    res = nb.lrt(y, x_full, x_red, offset, alpha)
    cov = nb.nb_cov(x_full, res["fit_full"]["mu"], alpha)
    t_lrt = time.time() - t0

    p_lrt = res["pvalue"]
    q_lrt = bh_qvalues(p_lrt)

    table = pd.DataFrame({
        "gene_id": genes,
        "base_mean": base_mean,
        "dispersion_gene": alpha_gene,
        "dispersion_trend": trend,
        "dispersion": alpha,
        "lrt_stat": res["stat"],
        "pvalue": p_lrt,
        "qvalue": q_lrt,
    })
    for k, name in enumerate(names):
        table[f"coef_{name}"] = res["fit_full"]["beta"][k, :]

    c_names, c_est, c_se, c_p = [], [], [], []
    for name, c in CONTRASTS.items():
        w = nb.wald_contrast(res["fit_full"]["beta"], cov, c)
        c_names.append(name)
        c_est.append(w["estimate"] / np.log(2))
        c_se.append(w["se"] / np.log(2))
        c_p.append(w["pvalue"])
    c_est = np.vstack(c_est).T
    c_se = np.vstack(c_se).T
    c_p = np.vstack(c_p).T
    q_all = bh_qvalues(c_p.ravel()).reshape(c_p.shape)

    contrast_table = pd.DataFrame({"gene_id": genes, "base_mean": base_mean})
    for j, name in enumerate(c_names):
        contrast_table[f"{name}_log2"] = c_est[:, j]
        contrast_table[f"{name}_se"] = c_se[:, j]
        contrast_table[f"{name}_p"] = c_p[:, j]
        contrast_table[f"{name}_q"] = q_all[:, j]

    max_abs = np.abs(c_est).max(axis=1)
    table["max_abs_contrast_log2"] = max_abs
    table["n_contrasts_q05_abs1"] = ((q_all < 0.05) & (np.abs(c_est) >= 1.0)).sum(axis=1)

    table.to_csv(OUT_T / "lrt_all_genes.csv.gz", index=False)
    contrast_table.to_csv(OUT_T / "contrasts_all_genes.csv.gz", index=False)
    table.nlargest(500, "lrt_stat").to_csv(OUT_T / "top500_lrt.csv", index=False)

    cand = table[(table["qvalue"] < 0.05) & (table["max_abs_contrast_log2"] >= 1.0)]
    cand = cand.sort_values("qvalue")
    cand.to_csv(OUT_T / "candidates_q05_effect1.csv", index=False)

    # group means on normalized counts for figures and candidate review
    norm = y / sf[:, None]
    group = (design["A"] * 4 + design["D"] * 2 + design["B"]).astype(int).to_numpy()
    group_means = np.vstack([norm[group == k].mean(axis=0) for k in range(8)])
    gm = pd.DataFrame(group_means.T, columns=[f"g{k}" for k in range(8)])
    gm.insert(0, "gene_id", genes)
    gm.to_csv(OUT_T / "group_means_normalized.csv.gz", index=False)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(p_lrt, bins=60, color="#4c72b0", edgecolor="white", linewidth=0.3)
    ax.axhline(len(p_lrt) / 60, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel("LRT p-value (4-df interaction)")
    ax.set_ylabel("genes")
    ax.set_title(f"{len(genes):,} genes; {(q_lrt < 0.05).sum():,} with q<0.05")
    fig.tight_layout()
    fig.savefig(OUT_F / "lrt_pvalue_hist.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.hist(max_abs, bins=80, color="#55a868", edgecolor="white", linewidth=0.3)
    ax.axvline(1.0, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel("max |contrast| across seven planned contrasts (log2)")
    ax.set_ylabel("genes")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(OUT_F / "effect_size_distribution.png", dpi=300)
    plt.close(fig)

    top = table.nlargest(40, "lrt_stat")["gene_id"].tolist()
    top_idx = [int(np.where(genes == g)[0][0]) for g in top]
    z = np.log2(group_means[:, top_idx] + 1.0)
    z = (z - z.mean(axis=0)) / (z.std(axis=0) + 1e-9)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(z, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(top, rotation=90, fontsize=6)
    ax.set_yticks(range(8))
    ax.set_yticklabels([f"A{a}D{d}B{b}" for a in (0, 1) for d in (0, 1) for b in (0, 1)])
    fig.colorbar(im, ax=ax, shrink=0.6, label="row z-score (log2 normalized)")
    fig.tight_layout()
    fig.savefig(OUT_F / "top40_heatmap.png", dpi=300)
    plt.close(fig)

    summary = {
        "n_genes_tested": int(len(genes)),
        "filter": "CPM>=1 in >=3 of 24 samples",
        "size_factor_range": [float(sf.min()), float(sf.max())],
        "dispersion_median": float(np.median(alpha)),
        "dispersion_df_prior": float(df_prior),
        "lrt_df": 4,
        "n_q05": int((q_lrt < 0.05).sum()),
        "n_q01": int((q_lrt < 0.01).sum()),
        "n_contrast_q05_effect1": int(len(cand)),
        "n_three_way_q05_effect1": int(((contrast_table["ADB_three_way_q"] < 0.05) &
                                         (np.abs(contrast_table["ADB_three_way_log2"]) >= 1)).sum()),
        "max_abs_contrast_median": float(np.median(max_abs)),
        "seconds_dispersion": t_disp,
        "seconds_lrt": t_lrt,
        "seconds_total": time.time() - t_start,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    with open(OUT_T / "main_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
