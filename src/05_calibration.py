"""Calibration of the interaction LRT by parametric simulation.

Data are generated from the additive (main-effects-only) model with the
real design, size factors and empirically estimated dispersions.  The full
pipeline (dispersion grid + Cox-Reid, trend, shrinkage, LRT) is re-run on
each simulated data set, so false-positive rates reflect the analysis as
implemented rather than an idealised test.

Modes:
  null  -- no interaction anywhere (type I error calibration)
  power -- known AD / ADB effects in a subset (recovery and detection)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import nb_glm as nb

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "data" / "processed" / "main_design_24.csv"
PILOT = ROOT / "results" / "pilot" / "pilot_1200genes.csv"


def bh_qvalues(p):
    p = np.asarray(p, dtype=np.float64)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def build_pools():
    pilot = pd.read_csv(PILOT)
    base = pilot["base_mean"].to_numpy()
    base = base[base > 0]
    alpha = pilot["alpha_gene"].to_numpy()
    alpha = alpha[np.isfinite(alpha) & (alpha > 0)]
    return np.log(base), np.log(alpha)


def simulate(seed, n_genes, log_base, log_alpha, x_main, offset,
             ad_effect=None, adb_effect=None, effect_frac=0.1):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(log_base), n_genes)
    b0 = log_base[idx] + rng.normal(0.0, 0.3, n_genes)
    bmain = rng.normal(0.0, 0.5, (3, n_genes))
    a_idx = rng.integers(0, len(log_alpha), n_genes)
    alpha = np.exp(log_alpha[a_idx] + rng.normal(0.0, 0.7, n_genes))
    alpha = np.clip(alpha, 1e-5, 5.0)

    eta = b0[None, :] + x_main[:, 1:4] @ bmain + offset[:, None]
    true_ad = np.zeros(n_genes)
    true_adb = np.zeros(n_genes)
    n_eff = int(effect_frac * n_genes)
    eff_idx = rng.choice(n_genes, n_eff, replace=False)
    a, d, b = x_main[:, 1], x_main[:, 2], x_main[:, 3]
    if ad_effect is not None:
        sign = rng.choice([-1.0, 1.0], n_eff)
        true_ad[eff_idx] = ad_effect * sign
        eta[:, eff_idx] += (a * d)[:, None] * (true_ad * np.log(2))[None, eff_idx]
    if adb_effect is not None:
        sign3 = rng.choice([-1.0, 1.0], n_eff)
        true_adb[eff_idx] = adb_effect * sign3
        eta[:, eff_idx] += (a * d * b)[:, None] * (true_adb * np.log(2))[None, eff_idx]

    mu = np.exp(eta)
    r = (1.0 / alpha)[None, :]
    y = rng.negative_binomial(r, r / (r + mu)).astype(np.float64)
    return y, true_ad, true_adb


def analyse(y, x_full, x_red, offset):
    alpha_gene, _ = nb.estimate_dispersions(y, x_full, offset)
    base_mean = (y / np.exp(offset)[:, None]).mean(axis=0)
    trend = nb.dispersion_trend(base_mean, alpha_gene)
    alpha, df_prior = nb.shrink_dispersions(alpha_gene, trend, df_resid=16)
    res = nb.lrt(y, x_full, x_red, offset, alpha)
    return {
        "pvalue": res["pvalue"],
        "stat": res["stat"],
        "beta": res["fit_full"]["beta"],
        "mu": res["fit_full"]["mu"],
        "alpha": alpha,
        "df_prior": df_prior,
    }


def null_run(seed, n_genes, log_base, log_alpha, x_full, x_red, offset):
    t0 = time.time()
    y, _, _ = simulate(seed, n_genes, log_base, log_alpha, x_full[:, :4], offset)
    out = analyse(y, x_full, x_red, offset)
    p = out["pvalue"]
    q = bh_qvalues(p)
    return {
        "seed": seed,
        "n": int(n_genes),
        "fpr05": float((p < 0.05).mean()),
        "fpr01": float((p < 0.01).mean()),
        "fpr001": float((p < 0.001).mean()),
        "bh05_frac": float((q < 0.05).mean()),
        "median_stat": float(np.median(out["stat"])),
        "df_prior": float(out["df_prior"]),
        "seconds": time.time() - t0,
    }, p.astype(np.float32)


def power_run(seed, n_genes, log_base, log_alpha, x_full, x_red, offset,
              ad_effect, adb_effect):
    y, true_ad, true_adb = simulate(seed, n_genes, log_base, log_alpha,
                                    x_full[:, :4], offset,
                                    ad_effect=ad_effect, adb_effect=adb_effect)
    out = analyse(y, x_full, x_red, offset)
    q = bh_qvalues(out["pvalue"])
    cov = nb.nb_cov(x_full, out["mu"], out["alpha"])
    c_ad = np.array([0, 0, 0, 0, 1, 0, 0, 0], float)
    c_adb = np.array([0, 0, 0, 0, 0, 0, 0, 1], float)
    w_ad = nb.wald_contrast(out["beta"], cov, c_ad)
    w_adb = nb.wald_contrast(out["beta"], cov, c_adb)

    mask = true_ad != 0
    mask3 = true_adb != 0
    est_ad_log2 = w_ad["estimate"] / np.log(2)
    est_adb_log2 = w_adb["estimate"] / np.log(2)
    ci_lo = est_ad_log2 - 1.96 * w_ad["se"] / np.log(2)
    ci_hi = est_ad_log2 + 1.96 * w_ad["se"] / np.log(2)
    q_ad = bh_qvalues(w_ad["pvalue"])
    q_adb = bh_qvalues(w_adb["pvalue"])

    return {
        "seed": seed,
        "ad_effect": ad_effect,
        "adb_effect": adb_effect,
        "power_ad": float((q[mask] < 0.05).mean()) if mask.any() else np.nan,
        "power_adb": float((q[mask3] < 0.05).mean()) if mask3.any() else np.nan,
        "power_ad_contrast": float((q_ad[mask] < 0.05).mean()) if mask.any() else np.nan,
        "power_adb_contrast": float((q_adb[mask3] < 0.05).mean()) if mask3.any() else np.nan,
        "contrast_fpr05_ad": float((q_ad[~mask & ~mask3] < 0.05).mean()),
        "ci_width_med_ad": float(np.median(ci_hi[mask] - ci_lo[mask])) if mask.any() else np.nan,
        "bias_ad": float(np.mean(est_ad_log2[mask] - true_ad[mask])) if mask.any() else np.nan,
        "coverage_ad": float(np.mean((ci_lo[mask] <= true_ad[mask]) &
                                     (true_ad[mask] <= ci_hi[mask]))) if mask.any() else np.nan,
        "fpr05": float((out["pvalue"][~mask & ~mask3] < 0.05).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["null", "power"], default="null")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--genes", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--effect", type=float, default=None)
    ap.add_argument("--effect3", type=float, default=None)
    ap.add_argument("--tag", default="pilot")
    args = ap.parse_args()

    design = pd.read_csv(DESIGN)
    x_full, _ = nb.factorial_matrix(design["A"], design["D"], design["B"])
    x_red = x_full[:, :4]
    sf = np.array([0.788, 0.856, 0.807, 0.807, 0.783, 0.734, 0.951, 0.974,
                   1.013, 1.058, 0.943, 0.775, 1.125, 0.894, 0.881, 1.144,
                   1.033, 1.026, 1.431, 1.369, 1.704, 0.997, 1.306, 1.170])
    offset = np.log(sf)
    log_base, log_alpha = build_pools()

    out_dir = ROOT / "results" / ("calibration" if args.mode == "null" else "power")
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = np.arange(20261030, 20261030 + args.runs)
    t0 = time.time()
    if args.mode == "null":
        results = Parallel(n_jobs=args.jobs)(
            delayed(null_run)(int(s), args.genes, log_base, log_alpha,
                              x_full, x_red, offset) for s in seeds
        )
        runs = pd.DataFrame([r for r, _ in results])
        pvals = np.concatenate([p for _, p in results])
        np.save(out_dir / f"null_pvalues_{args.tag}.npy", pvals)
        runs.to_csv(out_dir / f"null_runs_{args.tag}.csv", index=False)
        summary = {
            "runs": args.runs,
            "genes_per_run": args.genes,
            "total_tests": int(len(pvals)),
            "mean_fpr05": float(runs["fpr05"].mean()),
            "mean_fpr01": float(runs["fpr01"].mean()),
            "mean_fpr001": float(runs["fpr001"].mean()),
            "mean_bh05_frac": float(runs["bh05_frac"].mean()),
            "all_null_fdr": float((runs["bh05_frac"] > 0).mean()),
            "bh05_frac_definition": "Rejected-gene proportion, not false discovery rate",
            "all_null_fdr_definition": "Fraction of all-null simulations with any BH rejection",
            "se_fpr05": float(runs["fpr05"].std() / np.sqrt(len(runs))),
            "median_stat": float(runs["median_stat"].median()),
            "seconds": time.time() - t0,
        }
    else:
        results = Parallel(n_jobs=args.jobs)(
            delayed(power_run)(int(s), args.genes, log_base, log_alpha,
                               x_full, x_red, offset, args.effect, args.effect3)
            for s in seeds
        )
        runs = pd.DataFrame(results)
        runs.to_csv(out_dir / f"power_runs_{args.tag}.csv", index=False)
        summary = {
            "runs": args.runs,
            "ad_effect_log2": args.effect,
            "adb_effect_log2": args.effect3,
            "mean_power_ad": float(runs["power_ad"].mean()),
            "mean_power_adb": float(runs["power_adb"].mean()),
            "mean_power_ad_contrast": float(runs["power_ad_contrast"].mean()),
            "mean_power_adb_contrast": float(runs["power_adb_contrast"].mean()),
            "mean_contrast_fpr05_ad": float(runs["contrast_fpr05_ad"].mean()),
            "median_ci_width_ad": float(runs["ci_width_med_ad"].median()),
            "mean_bias_ad": float(runs["bias_ad"].mean()),
            "mean_coverage_ad": float(runs["coverage_ad"].mean()),
            "mean_fpr05": float(runs["fpr05"].mean()),
            "seconds": time.time() - t0,
        }

    with open(out_dir / f"summary_{args.tag}.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
