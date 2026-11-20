"""Exploratory empirical calibration with independent simulation evaluation.

The original all-null rejection proportion is not the false discovery rate:
under an all-null model FDR equals the probability of any rejection. We retain
the original statistics and report that distinction explicitly.

Simulations use the original pilot-derived count/dispersion pools and the
observed size factors. The first 300 simulations calibrate the LRT and each
of seven absolute Wald statistics; 100 different seeds evaluate BH on wholly
null datasets. Calibration is pooled across simulated genes, not a guarantee
of gene-conditional or biological FDR control. The same mapping is applied
to the stored primary results without changing their effects or frozen sets.
"""

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import nb_glm as nb

sim = importlib.import_module("05_calibration")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "empirical_calibration"
OUT.mkdir(parents=True, exist_ok=True)
C = {
    "AD_given_B0": [0, 0, 0, 0, 1, 0, 0, 0],
    "AD_given_B1": [0, 0, 0, 0, 1, 0, 0, 1],
    "AB_given_D0": [0, 0, 0, 0, 0, 1, 0, 0],
    "AB_given_D1": [0, 0, 0, 0, 0, 1, 0, 1],
    "DB_given_A0": [0, 0, 0, 0, 0, 0, 1, 0],
    "DB_given_A1": [0, 0, 0, 0, 0, 0, 1, 1],
    "ADB_three_way": [0, 0, 0, 0, 0, 0, 0, 1],
}
N_CAL, N_EVAL, N_GENES = 300, 100, 2000
SEED = 202611200


def run(seed, pools, x, offset):
    y, _, _ = sim.simulate(seed, N_GENES, *pools, x[:, :4], offset)
    fit = sim.analyse(y, x, x[:, :4], offset)
    cov = nb.nb_cov(x, fit["mu"], fit["alpha"])
    z = np.column_stack([np.abs(nb.wald_contrast(fit["beta"], cov, c)["z"])
                         for c in C.values()])
    return fit["stat"].astype(np.float32), z.astype(np.float32)


def empirical_p(stat, null_sorted):
    return (1 + len(null_sorted) - np.searchsorted(null_sorted, stat, side="left")) / (len(null_sorted) + 1)


def main():
    design = pd.read_csv(ROOT / "data" / "processed" / "main_design_24.csv")
    counts = pd.read_csv(ROOT / "data" / "raw" / "GSE270988_raw_counts.csv")
    y0 = counts[design.sample_id].to_numpy(dtype=float)
    cpm = y0 / y0.sum(axis=0) * 1e6
    sf = nb.poscounts_size_factors(y0[(cpm >= 1).sum(axis=1) >= 3].T)
    x, _ = nb.factorial_matrix(design.A, design.D, design.B)
    cache = OUT / "simulation_statistics.npz"
    if cache.exists():
        data = np.load(cache)
        lrt, wald = data["lrt"], data["wald"]
        if lrt.shape != (N_CAL + N_EVAL, N_GENES):
            raise ValueError("simulation cache has a different protocol")
    else:
        pools = sim.build_pools()
        results = Parallel(n_jobs=8)(delayed(run)(SEED + i, pools, x, np.log(sf))
                                     for i in range(N_CAL + N_EVAL))
        lrt = np.stack([r[0] for r in results])
        wald = np.stack([r[1] for r in results])
        np.savez_compressed(cache, lrt=lrt, wald=wald)
    null_lrt = np.sort(lrt[:N_CAL].ravel())
    null_w = [np.sort(wald[:N_CAL, :, j].ravel()) for j in range(len(C))]
    rows = []
    for i in range(N_CAL, N_CAL + N_EVAL):
        lp = empirical_p(lrt[i], null_lrt)
        wp = np.column_stack([empirical_p(wald[i, :, j], null_w[j]) for j in range(len(C))])
        lq = sim.bh_qvalues(lp)
        wq = sim.bh_qvalues(wp.ravel()).reshape(wp.shape)
        rows.append({"seed": SEED + i, "lrt_rejections": int((lq < 0.05).sum()),
                     "wald_rejections": int((wq < 0.05).sum()),
                     "lrt_fpr05": float((lp < 0.05).mean()),
                     "lrt_fpr01": float((lp < 0.01).mean()),
                     "lrt_fpr001": float((lp < 0.001).mean())})
    evaluation = pd.DataFrame(rows)
    evaluation.to_csv(OUT / "heldout_null_evaluation.csv", index=False)
    old = pd.read_csv(ROOT / "results" / "calibration" / "null_runs_full500.csv")
    lt = pd.read_csv(ROOT / "results" / "tables" / "lrt_all_genes.csv.gz")
    ct = pd.read_csv(ROOT / "results" / "tables" / "contrasts_all_genes.csv.gz")
    assert list(lt.gene_id) == list(ct.gene_id)
    lq = sim.bh_qvalues(empirical_p(lt.lrt_stat.to_numpy(), null_lrt))
    wp = np.column_stack([empirical_p(np.abs(ct[name + "_log2"] / ct[name + "_se"]), null_w[j])
                          for j, name in enumerate(C)])
    wq = sim.bh_qvalues(wp.ravel()).reshape(wp.shape)
    tab = pd.DataFrame({"gene_id": lt.gene_id, "nominal_lrt_q": lt.qvalue,
                        "empirical_lrt_p": empirical_p(lt.lrt_stat.to_numpy(), null_lrt),
                        "empirical_lrt_q": lq})
    per = {}
    stable = np.zeros(len(tab), dtype=bool)
    for j, name in enumerate(C):
        tab[name + "_empirical_p"] = wp[:, j]
        tab[name + "_empirical_q"] = wq[:, j]
        sel = (wq[:, j] < 0.05) & (ct[name + "_log2"].abs() >= 1) & (ct[name + "_se"] <= 1) & (lq < 0.05)
        stable |= sel
        per[name] = int(sel.sum())
    tab["empirical_stable"] = stable
    tab.to_csv(OUT / "primary_empirical_results.csv.gz", index=False)
    summary = {
        "n_calibration_runs": N_CAL, "n_evaluation_runs": N_EVAL, "genes_per_run": N_GENES,
        "seed_start": SEED, "size_factors": sf.tolist(),
        "original_all_null_rejection_fraction": float(old.bh05_frac.mean()),
        "original_all_null_fdr": float((old.bh05_frac > 0).mean()),
        "heldout_lrt_all_null_fdr": float((evaluation.lrt_rejections > 0).mean()),
        "heldout_wald_all_null_fdr": float((evaluation.wald_rejections > 0).mean()),
        "heldout_lrt_mean_fpr05": float(evaluation.lrt_fpr05.mean()),
        "heldout_lrt_mean_fpr01": float(evaluation.lrt_fpr01.mean()),
        "heldout_lrt_mean_fpr001": float(evaluation.lrt_fpr001.mean()),
        "empirical_n_lrt_q05": int((lq < 0.05).sum()),
        "empirical_n_lrt_q01": int((lq < 0.01).sum()),
        "nominal_empirical_lrt_overlap": int(((lq < 0.05) & (lt.qvalue < 0.05)).sum()),
        "empirical_n_stable": int(stable.sum()), "empirical_stable_per_contrast": per,
        "caution": "Pooled simulation-based calibration is exploratory, not gene-conditional or external validation.",
    }
    (OUT / "empirical_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
