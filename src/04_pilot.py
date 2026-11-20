"""Pilot fit of the 24-sample factorial model on a gene subset.

Checks timing, convergence, dispersion trend and p-value behaviour before
committing to the full genome-wide run.  No results are interpreted here.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

import nb_glm as nb

ROOT = Path(__file__).resolve().parents[1]
COUNTS = ROOT / "data" / "raw" / "GSE270988_raw_counts.csv"
DESIGN = ROOT / "data" / "processed" / "main_design_24.csv"
OUT = ROOT / "results" / "pilot"
OUT.mkdir(parents=True, exist_ok=True)

N_GENES = 1200
SEED = 20261030


def main() -> None:
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
    print(f"genes passing CPM>=1 in >=3 samples: {len(genes):,}")
    print(f"matrix orientation (samples x genes): {y.shape}")

    sf = nb.poscounts_size_factors(y)
    offset = np.log(sf)
    print("size factors:", np.round(sf, 3))

    x_full, names = nb.factorial_matrix(design["A"], design["D"], design["B"])
    x_red = x_full[:, :4]
    print("design rank:", np.linalg.matrix_rank(x_full), "columns:", names)

    rng = np.random.default_rng(SEED)
    sub = np.sort(rng.choice(y.shape[1], N_GENES, replace=False))
    ys = y[:, sub]

    t0 = time.time()
    alpha_gene, ll = nb.estimate_dispersions(ys, x_full, offset)
    t_disp = time.time() - t0
    print(f"dispersion grid fit: {t_disp:.1f}s for {N_GENES} genes "
          f"({t_disp/N_GENES*1000:.1f} ms/gene)")

    base_mean = (ys / sf[:, None]).mean(axis=0)
    trend = nb.dispersion_trend(base_mean, alpha_gene)
    alpha, df_prior = nb.shrink_dispersions(alpha_gene, trend, df_resid=16)
    print(f"df_prior: {df_prior:.1f}")
    print("gene-wise alpha quantiles:",
          np.round(np.quantile(alpha_gene, [0.05, 0.25, 0.5, 0.75, 0.95]), 4))
    print("shrunk alpha quantiles:",
          np.round(np.quantile(alpha, [0.05, 0.25, 0.5, 0.75, 0.95]), 4))

    t0 = time.time()
    res = nb.lrt(ys, x_full, x_red, offset, alpha)
    t_lrt = time.time() - t0
    print(f"LRT: {t_lrt:.1f}s ({t_lrt/N_GENES*1000:.1f} ms/gene), "
          f"full-model iterations {res['fit_full']['iterations']}")

    cov = nb.nb_cov(x_full, res["fit_full"]["mu"], alpha)
    contrasts = {
        "AD|B=0": [0, 0, 0, 0, 1, 0, 0, 0],
        "AD|B=1": [0, 0, 0, 0, 1, 0, 0, 1],
        "ADB": [0, 0, 0, 0, 0, 0, 0, 1],
    }
    for name, c in contrasts.items():
        w = nb.wald_contrast(res["fit_full"]["beta"], cov, c)
        print(f"{name:7s} estimate (log2) quantiles:",
              np.round(np.quantile(w["estimate"] / np.log(2), [0.01, 0.5, 0.99]), 2))

    p = res["pvalue"]
    print(f"LRT p<0.05 fraction: {(p < 0.05).mean():.3f}  p<1e-4: {(p < 1e-4).mean():.3f}")
    print(f"non-finite p: {np.isnan(p).sum()}")

    pd.DataFrame({
        "gene_id": genes[sub],
        "base_mean": base_mean,
        "alpha_gene": alpha_gene,
        "alpha_trend": trend,
        "alpha_shrunk": alpha,
        "lrt_stat": res["stat"],
        "pvalue": p,
        "beta_AD": res["fit_full"]["beta"][4, :],
        "beta_ADB": res["fit_full"]["beta"][7, :],
    }).to_csv(OUT / "pilot_1200genes.csv", index=False)

    print("\nprojected full-genome time (20,794 genes, single core): "
          f"dispersion {t_disp/N_GENES*20794/60:.1f} min, "
          f"LRT {t_lrt/N_GENES*20794/60:.1f} min")


if __name__ == "__main__":
    main()
