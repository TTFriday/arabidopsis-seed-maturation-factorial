"""Quality control of the count matrix, with emphasis on the 24-sample design.

Outputs land in results/qc: sample-level statistics, low-expression filter
counts (pre-registered CPM>=1 in >=3 samples), PCA and correlation figures.
No sample is removed here; flags are recorded for the sensitivity analysis.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COUNTS = ROOT / "data" / "raw" / "GSE270988_raw_counts.csv"
SAMPLES = ROOT / "data" / "processed" / "samples.csv"
DESIGN = ROOT / "data" / "processed" / "main_design_24.csv"
OUT = ROOT / "results" / "qc"
OUT.mkdir(parents=True, exist_ok=True)

FILTER_CPM = [0.5, 1.0, 2.0]
MIN_SAMPLES = 3


def load_counts():
    df = pd.read_csv(COUNTS)
    genes = df["gene_id"].astype(str).to_numpy()
    mat = df.drop(columns="gene_id").to_numpy(dtype=np.int64)
    cols = [c for c in df.columns if c != "gene_id"]
    return genes, mat, cols


def main() -> None:
    genes, mat, cols = load_counts()
    samples = pd.read_csv(SAMPLES).set_index("sample_id")
    design = pd.read_csv(DESIGN)

    assert mat.min() >= 0
    assert np.isfinite(mat).all()
    dup = len(genes) - len(pd.unique(genes))
    libs = mat.sum(axis=0)

    qc = pd.DataFrame({
        "sample_id": cols,
        "library_size": libs,
        "genes_detected": (mat > 0).sum(axis=0),
    }).set_index("sample_id")
    qc = qc.join(samples[["genotype", "stage", "round", "rep"]])
    qc["library_size_m"] = qc["library_size"] / 1e6
    qc.to_csv(OUT / "sample_qc_all123.csv")

    print(f"genes: {len(genes):,}  duplicates: {dup}  samples: {len(cols)}")
    print(f"gene id example: {genes[:3].tolist()}")
    print(f"library size (M reads): min {libs.min()/1e6:.2f}, "
          f"median {np.median(libs)/1e6:.2f}, max {libs.max()/1e6:.2f}")

    cpm = mat / libs * 1e6
    filter_counts_all = {}
    for thr in FILTER_CPM:
        keep_all = (cpm >= thr).sum(axis=1) >= MIN_SAMPLES
        filter_counts_all[f"cpm{thr}"] = int(keep_all.sum())

    # main 24-sample subset, in frozen design order
    idx = [cols.index(s) for s in design["sample_id"]]
    mat24 = mat[:, idx]
    libs24 = mat24.sum(axis=0)
    cpm24 = mat24 / libs24 * 1e6
    filter_counts_main = {}
    for thr in FILTER_CPM:
        keep24 = (cpm24 >= thr).sum(axis=1) >= MIN_SAMPLES
        filter_counts_main[f"cpm{thr}"] = int(keep24.sum())
    keep = (cpm24 >= 1.0).sum(axis=1) >= MIN_SAMPLES

    logcpm = np.log2(cpm24[keep] + 1.0)
    x = logcpm.T - logcpm.T.mean(axis=0)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    pcs = u[:, :2] * s[:2]
    var_exp = (s ** 2) / (s ** 2).sum()

    corr = np.corrcoef(logcpm.T)
    labels = [f"{g}_r{r}" for g, r in zip(design["genotype"], design["rep"])]

    within = []
    for i, gi in enumerate(design["genotype"]):
        same = [j for j, gj in enumerate(design["genotype"]) if gj == gi and j != i]
        other = [j for j, gj in enumerate(design["genotype"]) if gj != gi]
        within.append({
            "max_within_genotype_r": float(corr[i, same].max()),
            "median_between_genotype_r": float(np.median(corr[i, other])),
        })
    main_qc = pd.DataFrame(within, index=design["sample_id"])
    main_qc["library_size_m"] = libs24 / 1e6
    main_qc["genes_detected"] = (mat24 > 0).sum(axis=0)
    main_qc["pc1"] = pcs[:, 0]
    main_qc["pc2"] = pcs[:, 1]
    main_qc.to_csv(OUT / "sample_qc_main24.csv")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    cmap = plt.get_cmap("tab10")
    for k, gi in enumerate(sorted(design["genotype"].unique())):
        m = (design["genotype"] == gi).to_numpy()
        ax.scatter(pcs[m, 0], pcs[m, 1], s=42, color=cmap(k), label=gi)
    ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)")
    ax.legend(fontsize=7, frameon=False, loc="best")
    ax.set_title("24-sample factorial design (dry seed, round 3)")
    fig.tight_layout()
    fig.savefig(OUT / "pca_main24.png", dpi=300)
    fig.savefig(OUT / "pca_main24.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 6))
    im = ax.imshow(corr, cmap="viridis", vmin=0.7, vmax=1)
    ax.set_xticks(range(24))
    ax.set_yticks(range(24))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman r (log2 CPM)")
    fig.tight_layout()
    fig.savefig(OUT / "corr_main24.png", dpi=300)
    plt.close(fig)

    summary = {
        "n_genes": int(len(genes)),
        "duplicate_gene_ids": int(dup),
        "n_samples": int(len(cols)),
        "genes_passing_filter_all123": filter_counts_all,
        "genes_passing_filter_main24": filter_counts_main,
        "min_samples_for_filter": MIN_SAMPLES,
        "pc1_variance": float(var_exp[0]),
        "pc2_variance": float(var_exp[1]),
        "min_within_genotype_corr": float(main_qc["max_within_genotype_r"].min()),
        "max_between_genotype_corr": float(main_qc["median_between_genotype_r"].max()),
        "library_size_min_m": float(libs.min() / 1e6),
        "library_size_median_m": float(np.median(libs) / 1e6),
    }
    with open(OUT / "qc_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2))
    print("\nlowest within-genotype correlations:")
    print(main_qc["max_within_genotype_r"].sort_values().head(5).to_string())


if __name__ == "__main__":
    main()
