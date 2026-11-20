"""Sensitivity analyses for the 24-sample factorial interaction model.

Each analysis re-runs the full pipeline under one deliberate change and
compares the AD-related interaction signal with the primary analysis.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed

import nb_glm as nb

ROOT = Path(__file__).resolve().parents[1]
COUNTS = ROOT / "data" / "raw" / "GSE270988_raw_counts.csv"
DESIGN = ROOT / "data" / "processed" / "main_design_24.csv"
OUT = ROOT / "results" / "sensitivity"
OUT.mkdir(parents=True, exist_ok=True)

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


def load_data():
    df = pd.read_csv(COUNTS)
    genes = df["gene_id"].astype(str).to_numpy()
    mat = df.drop(columns="gene_id").to_numpy(dtype=np.int64)
    cols = [c for c in df.columns if c != "gene_id"]
    design = pd.read_csv(DESIGN)
    idx = [cols.index(s) for s in design["sample_id"]]
    return genes, mat, cols, design, idx


def subset(y_genes, cpm, genes, threshold):
    keep = (cpm >= threshold).sum(axis=1) >= 3
    return y_genes[keep].T, genes[keep]


def fit_variant(y, x_full, x_red, offset, dispersion="shrink", safety=1.2):
    alpha_gene, _ = nb.estimate_dispersions(y, x_full, offset)
    base_mean = (y / np.exp(offset)[:, None]).mean(axis=0)
    trend = nb.dispersion_trend(base_mean, alpha_gene)
    df_resid = y.shape[0] - x_full.shape[1]
    if dispersion == "shrink":
        alpha, _ = nb.shrink_dispersions(alpha_gene, trend, df_resid, safety=safety)
    elif dispersion == "gene":
        alpha = np.clip(alpha_gene, 1e-6, 50.0)
    elif dispersion == "trend":
        alpha = np.clip(trend, 1e-6, 50.0)
    else:
        raise ValueError(dispersion)
    res = nb.lrt(y, x_full, x_red, offset, alpha)
    cov = nb.nb_cov(x_full, res["fit_full"]["mu"], alpha)
    est = np.vstack([nb.wald_contrast(res["fit_full"]["beta"], cov, c)["estimate"]
                     for c in CONTRASTS.values()]).T / np.log(2)
    return {
        "pvalue": res["pvalue"],
        "stat": res["stat"],
        "ad_effect": est[:, 0],
        "max_abs": np.abs(est).max(axis=1),
        "alpha": alpha,
        "beta": res["fit_full"]["beta"],
        "mu": res["fit_full"]["mu"],
        "base_mean": base_mean,
    }


def tmm_factors(counts):
    """Trimmed-mean-of-M factors (30% M trim, 5% A trim, unweighted)."""
    counts = np.asarray(counts, dtype=np.float64)
    libs = counts.sum(axis=1)
    uq = np.array([np.quantile(row[row > 0], 0.75) for row in counts])
    ref = counts[np.argmin(np.abs(uq - uq.mean()))]
    f = np.ones(counts.shape[0])
    for i, row in enumerate(counts):
        ok = (row > 0) & (ref > 0)
        m = np.log2(row[ok] / ref[ok])
        a = 0.5 * np.log2(row[ok] * ref[ok])
        lo, hi = np.quantile(m, [0.30, 0.70])
        alo, ahi = np.quantile(a, [0.05, 0.95])
        sel = (m >= lo) & (m <= hi) & (a >= alo) & (a <= ahi)
        f[i] = 2 ** np.mean(m[sel])
    f = f / np.exp(np.mean(np.log(f)))
    # effective library size = library size times the TMM factor
    sf = (libs / np.exp(np.mean(np.log(libs)))) * f
    return sf / np.exp(np.mean(np.log(sf)))


def ql_test(y, x_full, x_red, offset, alpha_trend):
    """edgeR-style quasi-likelihood F-test with trended dispersion."""
    fit_full = nb.irls(y, x_full, offset, alpha_trend)
    fit_red = nb.irls(y, x_red, offset, alpha_trend)
    a = alpha_trend[None, :]
    x2_full = (((y - fit_full["mu"]) ** 2) / (fit_full["mu"] * (1 + a * fit_full["mu"]))).sum(0)
    x2_red = (((y - fit_red["mu"]) ** 2) / (fit_red["mu"] * (1 + a * fit_red["mu"]))).sum(0)
    phi_raw = x2_full / 16.0
    phi = (16.0 * phi_raw + 10.0) / 26.0
    f_stat = ((x2_red - x2_full) / 4.0) / np.maximum(phi, 1e-8)
    p = stats.f.sf(np.maximum(f_stat, 0), 4, 26)
    return p, f_stat


def loo_worker(drop, y, x_full, x_red, offset):
    keep = np.ones(y.shape[0], bool)
    keep[drop] = False
    yl = y[keep]
    offl = offset[keep] - offset[keep].mean()
    out = fit_variant(yl, x_full[keep][:, :], x_red[keep], offl)
    q = bh_qvalues(out["pvalue"])
    return drop, out["stat"], out["ad_effect"], q, out["max_abs"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", required=True,
                    choices=["filter", "norm", "dispersion", "ql", "loo"])
    args = ap.parse_args()

    genes, mat, cols, design, idx = load_data()
    y_genes = mat[:, idx].astype(np.float64)
    libs = y_genes.sum(axis=0)
    cpm = y_genes / libs * 1e6
    x_full, _ = nb.factorial_matrix(design["A"], design["D"], design["B"])
    x_red = x_full[:, :4]

    primary = pd.read_csv(ROOT / "results" / "tables" / "lrt_all_genes.csv.gz")
    primary = primary.set_index("gene_id")

    if args.analysis == "filter":
        rows = []
        for thr in [0.5, 2.0]:
            y, g = subset(y_genes, cpm, genes, thr)
            sf = nb.poscounts_size_factors(y)
            out = fit_variant(y, x_full, x_red, np.log(sf))
            q = bh_qvalues(out["pvalue"])
            df = pd.DataFrame({"gene_id": g, "pvalue": out["pvalue"], "qvalue": q,
                               "ad_effect": out["ad_effect"], "max_abs": out["max_abs"]})
            df.to_csv(OUT / f"filter_cpm{thr}.csv.gz", index=False)
            common = df.set_index("gene_id").join(primary[["qvalue", "max_abs_contrast_log2"]],
                                                  rsuffix="_main", how="inner")
            rho = common[["ad_effect"]].assign(main=primary.loc[common.index, "coef_A:D"] / np.log(2))
            rho = stats.spearmanr(rho["ad_effect"], rho["main"]).statistic
            cand_v = ((df["qvalue"] < 0.05) & (df["max_abs"] >= 1))
            cand_m = set(primary[(primary["qvalue"] < 0.05) &
                                 (primary["max_abs_contrast_log2"] >= 1)].index)
            cand_v_set = set(df.loc[cand_v, "gene_id"])
            rows.append({
                "filter": f"cpm{thr}",
                "n_genes": int(len(df)),
                "n_q05": int((q < 0.05).sum()),
                "n_candidates": int(cand_v.sum()),
                "spearman_ad_vs_primary": float(rho),
                "overlap_primary": len(cand_v_set & cand_m),
                "jaccard_primary": len(cand_v_set & cand_m) /
                                   max(len(cand_v_set | cand_m), 1),
            })
        pd.DataFrame(rows).to_csv(OUT / "filter_summary.csv", index=False)
        print(pd.DataFrame(rows).to_string(index=False))

    elif args.analysis == "norm":
        keep = (cpm >= 1.0).sum(axis=1) >= 3
        y = y_genes[keep].T
        g = genes[keep]
        sf_lib = (y.sum(axis=1) / np.exp(np.mean(np.log(y.sum(axis=1)))))
        sf_tmm = tmm_factors(y)
        # poscounts over the filtered subset (primary)
        sf_pos = nb.poscounts_size_factors(y)
        variants = {"poscounts": sf_pos, "library": sf_lib, "tmm_like": sf_tmm}
        rows = []
        for name, sf in variants.items():
            out = fit_variant(y, x_full, x_red, np.log(sf))
            q = bh_qvalues(out["pvalue"])
            df = pd.DataFrame({"gene_id": g, "qvalue": q, "stat": out["stat"],
                               "ad_effect": out["ad_effect"], "max_abs": out["max_abs"]})
            df.to_csv(OUT / f"norm_{name}.csv.gz", index=False)
            cand = set(df.loc[(df["qvalue"] < 0.05) & (df["max_abs"] >= 1), "gene_id"])
            cand_m = set(primary[(primary["qvalue"] < 0.05) &
                                 (primary["max_abs_contrast_log2"] >= 1)].index)
            common = df.set_index("gene_id").reindex(primary.index)
            rho = stats.spearmanr(common["stat"], primary["lrt_stat"], nan_policy="omit").statistic
            rows.append({
                "normalization": name,
                "n_q05": int((q < 0.05).sum()),
                "n_candidates": len(cand),
                "spearman_stat_vs_primary": float(rho),
                "overlap_primary": len(cand & cand_m),
                "jaccard_primary": len(cand & cand_m) / max(len(cand | cand_m), 1),
            })
        pd.DataFrame(rows).to_csv(OUT / "norm_summary.csv", index=False)
        print(pd.DataFrame(rows).to_string(index=False))

    elif args.analysis == "dispersion":
        keep = (cpm >= 1.0).sum(axis=1) >= 3
        y = y_genes[keep].T
        g = genes[keep]
        sf = nb.poscounts_size_factors(y)
        rows = []
        for name, kwargs in {"shrink_1.0": dict(dispersion="shrink", safety=1.0),
                             "shrink_1.2": dict(dispersion="shrink", safety=1.2),
                             "shrink_1.3": dict(dispersion="shrink", safety=1.3),
                             "trend_only": dict(dispersion="trend"),
                             "gene_wise": dict(dispersion="gene")}.items():
            out = fit_variant(y, x_full, x_red, np.log(sf), **kwargs)
            q = bh_qvalues(out["pvalue"])
            df = pd.DataFrame({"gene_id": g, "qvalue": q, "stat": out["stat"],
                               "ad_effect": out["ad_effect"], "max_abs": out["max_abs"]})
            df.to_csv(OUT / f"dispersion_{name}.csv.gz", index=False)
            cand = set(df.loc[(df["qvalue"] < 0.05) & (df["max_abs"] >= 1), "gene_id"])
            cand_m = set(primary[(primary["qvalue"] < 0.05) &
                                 (primary["max_abs_contrast_log2"] >= 1)].index)
            common = df.set_index("gene_id").reindex(primary.index)
            rho = stats.spearmanr(common["ad_effect"],
                                  primary.loc[common.index, "coef_A:D"] / np.log(2),
                                  nan_policy="omit").statistic
            rows.append({
                "variant": name,
                "n_q05": int((q < 0.05).sum()),
                "n_candidates": len(cand),
                "spearman_effect_vs_primary": float(rho),
                "overlap_primary": len(cand & cand_m),
                "jaccard_primary": len(cand & cand_m) / max(len(cand | cand_m), 1),
            })
        pd.DataFrame(rows).to_csv(OUT / "dispersion_summary.csv", index=False)
        print(pd.DataFrame(rows).to_string(index=False))

    elif args.analysis == "ql":
        keep = (cpm >= 1.0).sum(axis=1) >= 3
        y = y_genes[keep].T
        g = genes[keep]
        sf = nb.poscounts_size_factors(y)
        alpha_gene, _ = nb.estimate_dispersions(y, x_full, np.log(sf))
        trend = nb.dispersion_trend((y / sf[:, None]).mean(axis=0), alpha_gene)
        p, f_stat = ql_test(y, x_full, x_red, np.log(sf), trend)
        q = bh_qvalues(p)
        df = pd.DataFrame({"gene_id": g, "pvalue": p, "qvalue": q, "F_stat": f_stat})
        df.to_csv(OUT / "ql_all_genes.csv.gz", index=False)
        cand = set(df.loc[q < 0.05, "gene_id"])
        cand_m = set(primary.index[primary["qvalue"] < 0.05])
        common = df.set_index("gene_id").reindex(primary.index)
        rho = stats.spearmanr(common["F_stat"], primary["lrt_stat"],
                              nan_policy="omit").statistic
        summary = {
            "n_q05": int((q < 0.05).sum()),
            "overlap_primary_q05": len(cand & cand_m),
            "jaccard_primary_q05": len(cand & cand_m) / max(len(cand | cand_m), 1),
            "spearman_stat_vs_primary": float(rho),
        }
        with open(OUT / "ql_summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(json.dumps(summary, indent=2))

    elif args.analysis == "loo":
        keep = (cpm >= 1.0).sum(axis=1) >= 3
        y = y_genes[keep].T
        sf = nb.poscounts_size_factors(y)
        t0 = time.time()
        results = Parallel(n_jobs=12)(
            delayed(loo_worker)(i, y, x_full, x_red, np.log(sf)) for i in range(y.shape[0])
        )
        stats_mat = np.vstack([r[1] for r in results])
        ad_mat = np.vstack([r[2] for r in results])
        q_mat = np.vstack([r[3] for r in results])
        max_mat = np.vstack([r[4] for r in results])
        np.save(OUT / "loo_lrt_stats.npy", stats_mat.astype(np.float32))
        np.save(OUT / "loo_ad_effects.npy", ad_mat.astype(np.float32))
        np.save(OUT / "loo_maxabs.npy", max_mat.astype(np.float32))
        cand = ((primary["qvalue"] < 0.05) &
                (primary["max_abs_contrast_log2"] >= 1)).to_numpy()
        persist = ((q_mat[:, cand] < 0.05) & (max_mat[:, cand] >= 1)).mean(axis=1)
        rho = [stats.spearmanr(stats_mat[i], primary["lrt_stat"]).statistic
               for i in range(stats_mat.shape[0])]
        summary = {
            "n_candidates": int(cand.sum()),
            "mean_candidate_persistence": float(persist.mean()),
            "min_candidate_persistence": float(persist.min()),
            "mean_spearman_stat": float(np.mean(rho)),
            "seconds": time.time() - t0,
        }
        pd.DataFrame({"drop_sample": design["sample_id"],
                      "candidate_persistence": persist,
                      "spearman_stat": rho}).to_csv(OUT / "loo_summary.csv", index=False)
        with open(OUT / "loo_summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
