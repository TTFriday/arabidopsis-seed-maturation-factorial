"""E3: same-study internal evidence from three additional genotypes.

CYP707A2 combination (abi3-12 dog1-4 cyp707a2), the strong abi3-6 allele and
the DOG1-Cvi construct line are used only as bounded internal evidence on
gene sets frozen from the 24-sample dry-seed analysis.  None of these is an
independent external replication and none supports a "rescue" claim.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import nb_glm as nb

ROOT = Path(__file__).resolve().parents[1]
COUNTS = ROOT / "data" / "raw" / "GSE270988_raw_counts.csv"
SAMPLES = ROOT / "data" / "processed" / "samples.csv"
OUT = ROOT / "results" / "internal"
FIG = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def bh_qvalues(p):
    p = np.asarray(p, dtype=np.float64)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def load():
    df = pd.read_csv(COUNTS)
    genes = df["gene_id"].astype(str).to_numpy()
    cols = [c for c in df.columns if c != "gene_id"]
    samples = pd.read_csv(SAMPLES)
    return df, genes, cols, samples


def fetch(df, cols, samples, genotype, stage="DS", rnd=3):
    sub = samples[(samples["genotype"] == genotype) & (samples["stage"] == stage) &
                  (samples["round"] == rnd)].sort_values("rep")
    idx = [cols.index(s) for s in sub["sample_id"]]
    return df.iloc[:, 1:].to_numpy(dtype=np.float64)[:, idx], sub["sample_id"].tolist()


def fit_groups(mats, genes, group_sizes):
    """NB GLM with one intercept per group (no other covariates)."""
    mat = np.hstack(mats)
    libs = mat.sum(axis=0)
    cpm = mat / libs * 1e6
    keep = (cpm >= 1.0).sum(axis=1) >= 3
    y = mat[keep].T
    sf = nb.poscounts_size_factors(y)
    offset = np.log(sf)
    n_groups = len(group_sizes)
    x = np.zeros((y.shape[0], n_groups))
    start = 0
    for j, k in enumerate(group_sizes):
        x[start:start + k, j] = 1.0
        start += k
    alpha_gene, _ = nb.estimate_dispersions(y, x, offset)
    base_mean = (y / sf[:, None]).mean(axis=0)
    trend = nb.dispersion_trend(base_mean, alpha_gene)
    alpha, df_prior = nb.shrink_dispersions(alpha_gene, trend, y.shape[0] - n_groups)
    fit = nb.irls(y, x, offset, alpha)
    cov = nb.nb_cov(x, fit["mu"], alpha)
    norm = y / sf[:, None]
    group_means = np.vstack([norm[start:start + k].mean(axis=0)
                             for start, k in zip(np.cumsum([0] + group_sizes[:-1]), group_sizes)])
    return {"genes": genes[keep], "y": y, "sf": sf, "alpha": alpha,
            "df_prior": df_prior, "base_mean": base_mean, "fit": fit, "cov": cov,
            "group_means": group_means}


def wald_tab(res, contrasts):
    out = {}
    for name, c in contrasts.items():
        w = nb.wald_contrast(res["fit"]["beta"], res["cov"], np.asarray(c, float))
        out[f"{name}_log2"] = w["estimate"] / np.log(2)
        out[f"{name}_se"] = w["se"] / np.log(2)
        out[f"{name}_p"] = w["pvalue"]
    df = pd.DataFrame({"gene_id": res["genes"], "base_mean": res["base_mean"], **out})
    for name in contrasts:
        df[f"{name}_q"] = bh_qvalues(df[f"{name}_p"].to_numpy())
    return df


def freeze_set(name, rule, genes):
    genes = np.sort(np.asarray(genes, dtype=str))
    digest = hashlib.sha256("\n".join(genes).encode()).hexdigest()
    path = OUT / f"{name}.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["sha256"] != digest:
            raise SystemExit(f"{name} changed; refusing to redefine")
        return saved
    payload = {"rule": rule, "n_genes": int(len(genes)), "sha256": digest,
               "genes": list(genes)}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def section_cyp(df, genes, cols, samples, sig_genes):
    wt, _ = fetch(df, cols, samples, "Col-0")
    ad, _ = fetch(df, cols, samples, "abi3-12_dog1-4")
    adc, _ = fetch(df, cols, samples, "abi3-12_dog1-4_cyp707a2")
    res = fit_groups([wt, ad, adc], genes, [3, 3, 3])
    tab = wald_tab(res, {"AD_vs_WT": [-1, 1, 0],
                         "CYP_vs_AD": [0, -1, 1],
                         "NET_vs_WT": [-1, 0, 1]})
    tab["in_signature"] = tab["gene_id"].isin(sig_genes)
    gm = res["group_means"]
    expressed = (gm[0] >= 10) & (gm[1] >= 10)
    tab["expressed_WT_AD"] = expressed
    tab.to_csv(OUT / "cyp_gene_level.csv.gz", index=False)
    sig = tab[tab["in_signature"]].copy()
    sigx = sig[sig["expressed_WT_AD"]]
    opposite = np.sign(sig["CYP_vs_AD_log2"]) == -np.sign(sig["AD_vs_WT_log2"])
    closer = np.abs(sig["NET_vs_WT_log2"]) < np.abs(sig["AD_vs_WT_log2"])
    still_diff = (sig["NET_vs_WT_q"] < 0.05) & (np.abs(sig["NET_vs_WT_log2"]) >= 0.5)
    summary = {
        "n_tested": int(len(tab)),
        "signature_tested": int(len(sig)),
        "signature_expressed_WT_AD": int(len(sigx)),
        "signature_median_AD_vs_WT": float(sig["AD_vs_WT_log2"].median()),
        "signature_median_CYP_vs_AD": float(sig["CYP_vs_AD_log2"].median()),
        "signature_median_NET_vs_WT": float(sig["NET_vs_WT_log2"].median()),
        "expressed_median_AD_vs_WT": float(sigx["AD_vs_WT_log2"].median()),
        "expressed_median_CYP_vs_AD": float(sigx["CYP_vs_AD_log2"].median()),
        "expressed_median_NET_vs_WT": float(sigx["NET_vs_WT_log2"].median()),
        "frac_CYP_opposite_AD": float(opposite.mean()),
        "frac_closer_to_WT": float(closer.mean()),
        "frac_closer_to_WT_expressed": float(
            (np.abs(sigx["NET_vs_WT_log2"]) < np.abs(sigx["AD_vs_WT_log2"])).mean()),
        "frac_still_different_from_WT_q05_abs05": float(still_diff.mean()),
        "n_still_different": int(still_diff.sum()),
        "note": ("AD and CYP contrasts share the AD group; direction opposition is partly "
                 "mechanical and is not reported as rescue. NET_vs_WT does not share a "
                 "control group of its own. 'expressed' requires >=10 normalized counts "
                 "in both WT and AD group means."),
    }
    return tab, summary


def section_two_group(df, genes, cols, samples, mutant, frozen, dry_col, label):
    wt, _ = fetch(df, cols, samples, "Col-0")
    mut, _ = fetch(df, cols, samples, mutant)
    res = fit_groups([wt, mut], genes, [3, 3])
    tab = wald_tab(res, {"MUT_vs_WT": [-1, 1]})
    tab["in_set"] = tab["gene_id"].isin(frozen["genes"])
    gm = res["group_means"]
    tab["expressed_both"] = (gm[0] >= 10) & (gm[1] >= 10)
    tab.to_csv(OUT / f"{label}_gene_level.csv.gz", index=False)
    lrt = pd.read_csv(ROOT / "results" / "tables" / "lrt_all_genes.csv.gz")
    lrt = lrt.set_index("gene_id")
    sub = tab[tab["in_set"]].set_index("gene_id")
    ref = lrt.loc[sub.index, dry_col]
    ok = np.isfinite(ref) & np.isfinite(sub["MUT_vs_WT_log2"])
    rho = stats.spearmanr(sub["MUT_vs_WT_log2"][ok], ref[ok]).statistic
    slope = np.polyfit(ref[ok], sub["MUT_vs_WT_log2"][ok], 1)[0]
    same = np.sign(sub["MUT_vs_WT_log2"][ok]) == np.sign(ref[ok])
    okx = ok & sub["expressed_both"].to_numpy()
    rho_x = stats.spearmanr(sub["MUT_vs_WT_log2"][okx], ref[okx]).statistic
    slope_x = np.polyfit(ref[okx], sub["MUT_vs_WT_log2"][okx], 1)[0]
    same_x = np.sign(sub["MUT_vs_WT_log2"][okx]) == np.sign(ref[okx])
    summary = {
        "mutant": mutant,
        "frozen_set": frozen["rule"],
        "allele_note": ("abi3-6 is a strong loss-of-function allele with severe maturation "
                        "defects; abi3-12 is milder and partially divergent. The comparison "
                        "is not a replicate of the same allele.") if label == "abi3_6" else
                       ("DOG1-Cvi is a Col-0 line carrying a DOG1-Cvi construct, not a "
                        "loss-of-function dog1 allele; opposite direction to dog1-4 is the "
                        "expected pattern for increased DOG1 dosage."),
        "n_tested": int(len(tab)),
        "set_tested": int(len(sub)),
        "set_expressed_both": int(okx.sum()),
        "set_median_effect": float(sub["MUT_vs_WT_log2"].median()),
        "expressed_median_effect": float(sub.loc[sub["expressed_both"], "MUT_vs_WT_log2"].median()),
        "spearman_vs_primary_effect": float(rho),
        "slope_vs_primary_effect": float(slope),
        "sign_concordance": float(same.mean()),
        "binom_p": float(stats.binomtest(int(same.sum()), int(len(same)), 0.5).pvalue),
        "expressed_spearman": float(rho_x),
        "expressed_slope": float(slope_x),
        "expressed_sign_concordance": float(same_x.mean()),
        "n_set_changed_q05_abs05": int(((sub["MUT_vs_WT_q"] < 0.05) &
                                        (np.abs(sub["MUT_vs_WT_log2"]) >= 0.5)).sum()),
        "n_set_changed_q05_abs05_expressed": int((
            (sub["MUT_vs_WT_q"] < 0.05) & (np.abs(sub["MUT_vs_WT_log2"]) >= 0.5) &
            sub["expressed_both"]).sum()),
    }
    return tab, summary


def main() -> None:
    t0 = time.time()
    df, genes, cols, samples = load()
    lrt = pd.read_csv(ROOT / "results" / "tables" / "lrt_all_genes.csv.gz").set_index("gene_id")

    frozen_ad = json.loads((ROOT / "results" / "stages" /
                            "frozen_signature.json").read_text(encoding="utf-8"))
    sig_ad = frozen_ad["genes"]

    rule_a = "LRT q<0.05 and |coef_A|>=1 (ABI3 main effect at D=0, B=0)"
    rule_d = "LRT q<0.05 and |coef_D|>=1 (DOG1 main effect at A=0, B=0)"
    set_a = lrt[(lrt["qvalue"] < 0.05) & (lrt["coef_A"].abs() >= 1)]
    set_d = lrt[(lrt["qvalue"] < 0.05) & (lrt["coef_D"].abs() >= 1)]
    frozen_a = freeze_set("frozen_A_main", rule_a, set_a.index)
    frozen_d = freeze_set("frozen_D_main", rule_d, set_d.index)
    print(f"S_AD {len(sig_ad)}, S_A {frozen_a['n_genes']}, S_D {frozen_d['n_genes']}")

    cyp_tab, cyp_sum = section_cyp(df, genes, cols, samples, sig_ad)
    print("CYP:", json.dumps(cyp_sum, indent=2))

    a6_tab, a6_sum = section_two_group(df, genes, cols, samples, "abi3-6",
                                       frozen_a, "coef_A", "abi3_6")
    print("abi3-6:", json.dumps(a6_sum, indent=2))

    cvi_tab, cvi_sum = section_two_group(df, genes, cols, samples,
                                         "L2-1_Col-0_+_DOG1-Cvi",
                                         frozen_d, "coef_D", "dog1cvi")
    print("DOG1-Cvi:", json.dumps(cvi_sum, indent=2))

    summary = {"cyp_combination": cyp_sum, "abi3_6": a6_sum, "dog1_cvi": cvi_sum,
               "signature_AD": {"n": frozen_ad["n_genes"], "sha256": frozen_ad["sha256"]},
               "runtime_seconds": time.time() - t0}
    (OUT / "internal_summary.json").write_text(json.dumps(summary, indent=2),
                                              encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4))
    # CYP: AD effect vs net effect
    sub = cyp_tab[cyp_tab["in_signature"]]
    axes[0].scatter(sub["AD_vs_WT_log2"], sub["NET_vs_WT_log2"], s=5, alpha=0.3,
                    color="#2c3e50", rasterized=True)
    lim = 4
    axes[0].plot([-lim, lim], [-lim, lim], ls="--", color="black", lw=0.8)
    axes[0].axhline(0, color="grey", lw=0.6)
    axes[0].axvline(0, color="grey", lw=0.6)
    axes[0].set_xlabel("AD vs WT (log2), frozen signature")
    axes[0].set_ylabel("ADC vs WT (log2)")
    axes[0].set_title(f"CYP707A2 combination\n{cyp_sum['frac_closer_to_WT']:.0%} closer to WT")
    axes[0].set_xlim(-lim, lim)
    axes[0].set_ylim(-lim, lim)
    # abi3-6 vs abi3-12
    sub = a6_tab[a6_tab["in_set"]]
    ref = lrt.loc[sub["gene_id"], "coef_A"].to_numpy()
    axes[1].scatter(ref, sub["MUT_vs_WT_log2"], s=5, alpha=0.3, color="#c0392b",
                    rasterized=True)
    axes[1].plot([-lim, lim], [-lim, lim], ls="--", color="black", lw=0.8)
    axes[1].axhline(0, color="grey", lw=0.6)
    axes[1].axvline(0, color="grey", lw=0.6)
    axes[1].set_xlabel("abi3-12 effect (log2)")
    axes[1].set_ylabel("abi3-6 effect (log2)")
    axes[1].set_title(f"abi3-6 vs abi3-12\nrho={a6_sum['spearman_vs_primary_effect']:.2f}, "
                      f"slope={a6_sum['slope_vs_primary_effect']:.2f}")
    axes[1].set_xlim(-lim, lim)
    axes[1].set_ylim(-lim, lim)
    # DOG1-Cvi vs dog1-4
    sub = cvi_tab[cvi_tab["in_set"]]
    ref = lrt.loc[sub["gene_id"], "coef_D"].to_numpy()
    axes[2].scatter(ref, sub["MUT_vs_WT_log2"], s=5, alpha=0.3, color="#16a085",
                    rasterized=True)
    axes[2].plot([-lim, lim], [-lim, lim], ls="--", color="black", lw=0.8)
    axes[2].axhline(0, color="grey", lw=0.6)
    axes[2].axvline(0, color="grey", lw=0.6)
    axes[2].set_xlabel("dog1-4 effect (log2)")
    axes[2].set_ylabel("DOG1-Cvi effect (log2)")
    axes[2].set_title(f"DOG1-Cvi vs dog1-4\nrho={cvi_sum['spearman_vs_primary_effect']:.2f}, "
                      f"slope={cvi_sum['slope_vs_primary_effect']:.2f}")
    axes[2].set_xlim(-lim, lim)
    axes[2].set_ylim(-lim, lim)
    fig.tight_layout()
    fig.savefig(FIG / "internal_support.png", dpi=300)
    fig.savefig(FIG / "internal_support.pdf")
    plt.close(fig)
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
