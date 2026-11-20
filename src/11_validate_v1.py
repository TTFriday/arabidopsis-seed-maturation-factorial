"""External validation V1: GSE251921 Col-0 vs dog1-4 dry seeds.

Implements exactly the rules frozen in results/external/frozen_external.json:
two-group NB GLM on processed counts, log2(dog1-4 / Col-0), and the S_D
signature.  The frozen file must exist before this script runs.
"""

import gzip
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import nb_glm as nb

ROOT = Path(__file__).resolve().parents[1]
TAR = ROOT / "data" / "raw" / "external" / "GSE251921" / "GSE251921_RAW.tar"
FROZEN = ROOT / "results" / "external" / "frozen_external.json"
OUT = ROOT / "results" / "external"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261030
COL = [f"GSM{n}" for n in range(7989996, 7990000)]
DOG = [f"GSM{n}" for n in range(7990000, 7990004)]
AGI = __import__("re").compile(r"^AT[1-5]G\d{5}$")


def bh_qvalues(p):
    p = np.asarray(p, dtype=np.float64)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def load_counts():
    frames = {}
    with tarfile.open(TAR) as t:
        for m in t.getmembers():
            if not m.name.endswith(".txt.gz"):
                continue
            gsm = m.name.split("_")[0]
            if gsm not in COL + DOG:
                continue
            with gzip.open(t.extractfile(m), "rt") as fh:
                df = pd.read_csv(fh, sep="\t", header=None, names=["gene_id", "count"])
            frames[gsm] = df.set_index("gene_id")["count"]
    mat = pd.DataFrame(frames)
    mat.index = mat.index.astype(str).str.upper().str.replace(r"\.\d+$", "", regex=True)
    mat = mat.loc[mat.index.map(lambda g: bool(AGI.match(g)))]
    mat = mat.groupby(level=0).sum()
    return mat[COL + DOG]


def fit_two_group(y, sf):
    x = np.column_stack([np.ones(len(sf)), np.r_[np.zeros(len(COL)), np.ones(len(DOG))]])
    offset = np.log(sf)
    alpha_gene, _ = nb.estimate_dispersions(y, x, offset)
    base_mean = (y / np.exp(offset)[:, None]).mean(axis=0)
    trend = nb.dispersion_trend(base_mean, alpha_gene)
    df_resid = y.shape[0] - x.shape[1]
    alpha, _ = nb.shrink_dispersions(alpha_gene, trend, df_resid, safety=1.2)
    fit = nb.irls(y, x, offset, alpha)
    cov = nb.nb_cov(x, fit["mu"], alpha)
    w = nb.wald_contrast(fit["beta"], cov, [0, 1])
    return {
        "log2fc": w["estimate"] / np.log(2),
        "se": w["se"] / np.log(2),
        "z": w["z"],
        "p": w["pvalue"],
        "alpha": alpha,
        "base_mean": base_mean,
    }


def signature_scores(logc, genes, sign_source):
    idx = [i for i, g in enumerate(genes) if g in sign_source]
    z = (logc[idx] - logc[idx].mean(axis=1, keepdims=True)) / np.maximum(
        logc[idx].std(axis=1, keepdims=True), 1e-9)
    signs = np.sign([sign_source[genes[i]] for i in idx])
    return (z * signs[:, None]).mean(axis=0), len(idx)


def permutation_p(score):
    obs = score[len(COL):].mean() - score[:len(COL)].mean()
    n = len(score)
    combos = []
    from itertools import combinations
    for pick in combinations(range(n), len(COL)):
        mask = np.zeros(n, bool)
        mask[list(pick)] = True
        combos.append(score[~mask].mean() - score[mask].mean())
    combos = np.asarray(combos)
    p = float((np.abs(combos) >= abs(obs) - 1e-12).mean())
    return float(obs), p, len(combos)


def subset_metrics(gs, tab, main_log2):
    sub = tab.loc[tab.index.intersection(gs)]
    if len(sub) == 0:
        return {"n": 0}
    conc = float((np.sign(sub["log2fc"]) == np.sign(main_log2.loc[sub.index])).mean())
    rho = stats.spearmanr(sub["log2fc"], main_log2.loc[sub.index]).statistic
    lr = stats.linregress(main_log2.loc[sub.index], sub["log2fc"])
    rng = np.random.default_rng(SEED)
    slopes = np.empty(2000)
    xv = main_log2.loc[sub.index].to_numpy()
    yv = sub["log2fc"].to_numpy()
    for b in range(len(slopes)):
        pick = rng.integers(0, len(xv), len(xv))
        slopes[b] = np.polyfit(xv[pick], yv[pick], 1)[0]
    return {
        "n": int(len(sub)),
        "coverage_of_frozen_set": float(len(sub) / len(gs)),
        "sign_concordance": conc,
        "spearman": float(rho),
        "slope": float(lr.slope),
        "slope_ci95": [float(np.quantile(slopes, 0.025)), float(np.quantile(slopes, 0.975))],
        "median_abs_log2fc": float(sub["log2fc"].abs().median()),
    }


def main() -> None:
    if not FROZEN.exists():
        raise SystemExit("frozen_external.json missing; run 10_freeze_external.py first")
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    s_d = frozen["S_D"]["genes"]
    s_dr = frozen["S_D_robust"]["genes"]
    print(f"frozen S_D {len(s_d)}, S_D_robust {len(s_dr)}")

    mat = load_counts()
    print("samples", mat.shape[1], "genes", mat.shape[0])
    libs = mat.sum(axis=0)
    cpm = mat / libs * 1e6
    keep = (cpm >= 1).sum(axis=1) >= 3
    sub = mat.loc[keep]
    print(f"filtered genes {sub.shape[0]}, library sizes {libs.min():.0f}-{libs.max():.0f}")

    y = sub.to_numpy(dtype=np.float64).T
    sf = nb.poscounts_size_factors(y)
    out = fit_two_group(y, sf)
    tab = pd.DataFrame({
        "gene_id": sub.index,
        "base_mean_cpm": out["base_mean"] / 1e6,
        "log2fc": out["log2fc"],
        "se_log2": out["se"],
        "z": out["z"],
        "pvalue": out["p"],
        "dispersion": out["alpha"],
    }).set_index("gene_id")
    tab["qvalue"] = bh_qvalues(tab["pvalue"])

    lrt = pd.read_csv(ROOT / "results" / "tables" / "lrt_all_genes.csv.gz").set_index("gene_id")
    main_log2 = lrt["coef_D"] / np.log(2)
    tab["main_log2"] = main_log2.reindex(tab.index)
    tab["in_S_D"] = tab.index.isin(s_d)
    tab.to_csv(OUT / "v1_effects.csv.gz", compression="gzip")

    mapped = [g for g in s_d if g in mat.index]
    summary = {
        "data": "GSE251921, Col-0 GSM7989996-7989999 vs dog1-4 GSM7990000-7990003",
        "n_genes_filtered": int(sub.shape[0]),
        "library_size_range": [float(libs.min()), float(libs.max())],
        "size_factors": dict(zip(mat.columns, map(float, sf))),
        "coverage": {
            "S_D_total": len(s_d),
            "mapped_to_V1": len(mapped),
            "mapped_fraction": len(mapped) / len(s_d),
            "mapped_and_filtered": int(tab.index.isin(s_d).sum()),
            "mapped_and_filtered_fraction": int(tab.index.isin(s_d).sum()) / len(s_d),
        },
        "gate_rule": frozen["V1_rules"]["gate"],
        "S_D": subset_metrics(s_d, tab, main_log2),
        "S_D_robust": subset_metrics(s_dr, tab, main_log2),
    }

    # sample-level signature scores of the frozen S_D set
    logc = np.log2(cpm.loc[sub.index] + 1).to_numpy()
    genes = list(sub.index)
    sign_source = {g: float(main_log2[g]) for g in s_d if g in main_log2.index}
    score, n_sig = signature_scores(logc, genes, sign_source)
    obs, perm_p, n_perm = permutation_p(score)
    welch = stats.ttest_ind(score[len(COL):], score[:len(COL)], equal_var=False)
    summary["signature_score"] = {
        "n_genes": int(n_sig),
        "mean_col0": float(score[:len(COL)].mean()),
        "mean_dog1": float(score[len(COL):].mean()),
        "difference": obs,
        "welch_p": float(welch.pvalue),
        "exact_permutation_p": perm_p,
        "n_permutations": n_perm,
        "per_sample": dict(zip(mat.columns, map(float, score))),
    }

    # DOG1 itself, descriptive
    dog1 = "AT5G45830"
    if dog1 in tab.index:
        summary["DOG1_gene"] = {
            "log2fc": float(tab.loc[dog1, "log2fc"]),
            "pvalue": float(tab.loc[dog1, "pvalue"]),
            "base_mean_cpm": float(tab.loc[dog1, "base_mean_cpm"]),
            "main_log2": float(main_log2.get(dog1, np.nan)),
        }

    gate = summary["coverage"]["mapped_and_filtered_fraction"] >= 0.5
    summary["coverage_gate_pass"] = bool(gate)
    if not gate:
        summary["status"] = ("insufficient coverage: the frozen gate requires >=50% of S_D "
                             "mapped and passing the filter; subset results are descriptive")
    (OUT / "v1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame({"sample": mat.columns, "group": ["Col-0"] * 4 + ["dog1-4"] * 4,
                  "score": score}).to_csv(OUT / "v1_signature_scores.csv", index=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "size_factors"}, indent=2))


if __name__ == "__main__":
    main()
