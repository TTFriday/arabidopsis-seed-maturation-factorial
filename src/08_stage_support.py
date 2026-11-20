"""E2: within-stage ABI3 x DOG1 interaction at 20/22/24 DAP (round 3).

The 36 samples are four genotypes (Col-0, abi3-12, dog1-4, abi3-12 dog1-4)
at three developmental stages with three replicates each.  ABA2 is wild type
in all of them, so the stage contrasts are comparable to the dry-seed
AD|B=0 contrast, not to AD|B=1.

Samples from different stages may come from the same plant pools, so no
stage x genotype model is fitted across the 36 rows.  Each stage is analysed
on its own 12-sample 2x2 factorial, and cross-stage agreement is described
rather than tested as an interaction.
"""

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import nb_glm as nb

ROOT = Path(__file__).resolve().parents[1]
COUNTS = ROOT / "data" / "raw" / "GSE270988_raw_counts.csv"
SAMPLES = ROOT / "data" / "processed" / "samples.csv"
OUT = ROOT / "results" / "stages"
FIG = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

GENOTYPES = {
    "Col-0": (0, 0),
    "abi3-12": (1, 0),
    "dog1-4": (0, 1),
    "abi3-12_dog1-4": (1, 1),
}
STAGES = ["20d", "22d", "24d"]
CONTRAST_AD = np.array([0.0, 0.0, 0.0, 1.0])

SIG_RULE = (
    "S_AD: AD|B=0 q<0.05, |AD|B=0 log2|>=1, AD|B=1 q<0.05, and the two AD "
    "contrasts share sign. Defined from the 24-sample dry-seed results only, "
    "before any stage-based computation."
)


def bh_qvalues(p):
    p = np.asarray(p, dtype=np.float64)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def frozen_signature():
    path = OUT / "frozen_signature.json"
    contr = pd.read_csv(ROOT / "results" / "tables" / "contrasts_all_genes.csv.gz")
    a0 = contr["AD_given_B0_log2"].to_numpy()
    a1 = contr["AD_given_B1_log2"].to_numpy()
    q0 = contr["AD_given_B0_q"].to_numpy()
    q1 = contr["AD_given_B1_q"].to_numpy()
    mask = (q0 < 0.05) & (np.abs(a0) >= 1.0) & (q1 < 0.05) & (np.sign(a0) == np.sign(a1))
    genes = np.sort(contr.loc[mask, "gene_id"].astype(str).to_numpy())
    digest = hashlib.sha256("\n".join(genes).encode()).hexdigest()
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["sha256"] != digest:
            raise SystemExit("frozen signature changed; refusing to redefine it")
        print(f"loaded frozen signature: {len(genes)} genes, sha256 {digest[:12]}")
        return genes, saved
    payload = {
        "rule": SIG_RULE,
        "n_genes": int(len(genes)),
        "sha256": digest,
        "created": datetime.now().isoformat(timespec="seconds"),
        "source": "results/tables/contrasts_all_genes.csv.gz",
        "genes": list(genes),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"froze signature: {len(genes)} genes, sha256 {digest[:12]}")
    return genes, payload


def load_stage_data():
    df = pd.read_csv(COUNTS)
    genes = df["gene_id"].astype(str).to_numpy()
    cols = [c for c in df.columns if c != "gene_id"]
    samples = pd.read_csv(SAMPLES)
    sel = samples[(samples["round"] == 3) & (samples["stage"].isin(STAGES)) &
                  (samples["genotype"].isin(GENOTYPES))]
    assert len(sel) == 36, len(sel)
    counts = {g: sel[sel["genotype"] == g].sort_values(["stage", "rep"]) for g in GENOTYPES}
    for g, sub in counts.items():
        assert len(sub) == 9, (g, len(sub))
        assert sorted(sub["stage"].unique()) == STAGES
    return df, genes, cols, sel


def stage_matrix(df, cols, sel, stage):
    sub = sel[sel["stage"] == stage].sort_values(["genotype", "rep"])
    idx = [cols.index(s) for s in sub["sample_id"]]
    mat = df.iloc[:, 1:].to_numpy(dtype=np.float64)[:, idx]
    a = np.array([GENOTYPES[g][0] for g in sub["genotype"]], float)
    d = np.array([GENOTYPES[g][1] for g in sub["genotype"]], float)
    x = np.column_stack([np.ones_like(a), a, d, a * d])
    return mat, x, sub


def fit_stage(mat, genes, x):
    libs = mat.sum(axis=0)
    cpm = mat / libs * 1e6
    keep = (cpm >= 1.0).sum(axis=1) >= 3
    y = mat[keep].T
    sf = nb.poscounts_size_factors(y)
    offset = np.log(sf)
    alpha_gene, _ = nb.estimate_dispersions(y, x, offset)
    base_mean = (y / sf[:, None]).mean(axis=0)
    trend = nb.dispersion_trend(base_mean, alpha_gene)
    alpha, df_prior = nb.shrink_dispersions(alpha_gene, trend, y.shape[0] - x.shape[1])
    fit = nb.irls(y, x, offset, alpha)
    cov = nb.nb_cov(x, fit["mu"], alpha)
    w = nb.wald_contrast(fit["beta"], cov, CONTRAST_AD)
    return {
        "genes": genes[keep],
        "y": y,
        "sf": sf,
        "alpha": alpha,
        "df_prior": df_prior,
        "base_mean": base_mean,
        "estimate": w["estimate"] / np.log(2),
        "se": w["se"] / np.log(2),
        "z": w["z"],
        "p": w["pvalue"],
    }


def matched_null(sig_index, tab, dry_eff, n_draw=2000, seed=20261030):
    """Empirical null for the dry-sign concordance score of a gene set.

    Random sets are drawn from the tested genes, matched on expression decile,
    and scored with the same dry-seed sign convention.  Genes, not samples,
    are resampled, so this accounts for expression matching but not for
    co-expression between genes.
    """
    rng = np.random.default_rng(seed)
    tab = tab.loc[tab.index.intersection(dry_eff.dropna().index)]
    dec = pd.qcut(tab["base_mean"], 10, labels=False, duplicates="drop")
    sign_dry = np.sign(dry_eff.loc[tab.index])
    score = tab["z"] * sign_dry
    obs = float(score.loc[sig_index].mean())
    k_by_dec = dec.loc[sig_index].value_counts()
    draws = np.empty(n_draw)
    for i in range(n_draw):
        pick = []
        for d, k in k_by_dec.items():
            pool = dec.index[dec == d]
            pick.append(rng.choice(pool, size=int(k), replace=False))
        draws[i] = score.loc[np.concatenate(pick)].mean()
    p = 2 * min((draws >= obs).mean(), (draws <= obs).mean())
    return float(min(p, 1.0)), float(draws.std()), obs


def main() -> None:
    t0 = time.time()
    sig_genes, sig_meta = frozen_signature()
    df, genes_all, cols, sel = load_stage_data()
    print(sel.groupby(["stage", "genotype"]).size().unstack().to_string())

    results = {}
    tables = []
    for stage in STAGES:
        mat, x, sub = stage_matrix(df, cols, sel, stage)
        out = fit_stage(mat, genes_all, x)
        results[stage] = out
        tab = pd.DataFrame({
            "gene_id": out["genes"],
            "stage": stage,
            "base_mean": out["base_mean"],
            "dispersion": out["alpha"],
            "ad_log2": out["estimate"],
            "ad_se": out["se"],
            "ad_z": out["z"],
            "pvalue": out["p"],
        })
        tables.append(tab)
        print(f"{stage}: {len(tab)} genes tested, median dispersion "
              f"{np.median(out['alpha']):.4g}, median se {np.median(out['se']):.3f}")

    all_tab = pd.concat(tables, ignore_index=True)
    all_tab["qvalue_unified"] = bh_qvalues(all_tab["pvalue"].to_numpy())
    all_tab["qvalue_stage"] = np.nan
    for stage in STAGES:
        m = all_tab["stage"] == stage
        all_tab.loc[m, "qvalue_stage"] = bh_qvalues(all_tab.loc[m, "pvalue"].to_numpy())
    all_tab.to_csv(OUT / "stage_ad_all.csv.gz", index=False)

    dry = pd.read_csv(ROOT / "results" / "tables" / "contrasts_all_genes.csv.gz")
    dry = dry.set_index("gene_id")
    dry_eff = dry["AD_given_B0_log2"]

    summary = {"signature": {"n_genes": sig_meta["n_genes"], "sha256": sig_meta["sha256"]},
               "stages": {}, "concordance": {}}
    sig_set = set(sig_genes)
    rows = []
    for stage in STAGES:
        out = results[stage]
        tab = pd.DataFrame({"gene_id": out["genes"], "z": out["z"],
                            "effect": out["estimate"], "se": out["se"],
                            "base_mean": out["base_mean"]}).set_index("gene_id")
        in_sig = tab.index.intersection(sig_genes)
        sign_dry = np.sign(dry_eff.loc[in_sig])
        signed_effect = tab.loc[in_sig, "effect"] * sign_dry
        same = sign_dry == np.sign(tab.loc[in_sig, "effect"])
        z_conc = tab.loc[in_sig, "z"] * sign_dry
        stouffer = float(z_conc.sum() / np.sqrt(len(z_conc)))
        p_stouffer = float(2 * stats.norm.sf(abs(stouffer)))
        null_p, null_sd, _ = matched_null(in_sig, tab, dry_eff, n_draw=2000)
        qmap = all_tab[all_tab["stage"] == stage].set_index("gene_id")["qvalue_unified"]
        rep = ((qmap.loc[in_sig] < 0.05) & same).sum()
        common = tab.index.intersection(dry_eff.dropna().index)
        rho = stats.spearmanr(tab.loc[common, "effect"],
                              dry_eff.loc[common]).statistic if len(common) > 10 else np.nan
        slope = np.polyfit(dry_eff.loc[common].to_numpy(), tab.loc[common, "effect"].to_numpy(), 1)
        summary["stages"][stage] = {
            "n_tested": int(len(tab)),
            "n_q05_unified": int((qmap < 0.05).sum()),
            "signature_tested": int(len(in_sig)),
            "signature_mean_effect_raw": float(tab.loc[in_sig, "effect"].mean()),
            "signature_median_effect_raw": float(tab.loc[in_sig, "effect"].median()),
            "signature_mean_effect_dry_signed": float(signed_effect.mean()),
            "signature_median_effect_dry_signed": float(signed_effect.median()),
            "signature_sign_concordance": float(same.mean()),
            "signature_rep_rate_q05_same_sign": float(rep / len(in_sig)),
            "signature_concordance_z": stouffer,
            "signature_concordance_p": p_stouffer,
            "signature_matched_null_p": null_p,
            "signature_matched_null_sd": null_sd,
            "all_gene_effect_spearman_vs_dry": float(rho),
            "all_gene_slope_vs_dry": [float(slope[0]), float(slope[1])],
            "binom_p_sign_concordance": float(stats.binomtest(int(same.sum()), len(same), 0.5).pvalue),
        }
        rows.append(summary["stages"][stage])
        print(f"{stage}: signature n={len(in_sig)}, dry-signed mean effect "
              f"{signed_effect.mean():+.3f}, concordance {same.mean():.3f}, "
              f"concordance z {stouffer:+.1f}, matched-null p {null_p:.4g}")

    summary["signature"]["bonferroni_p_concordance"] = float(min(
        1.0, 3 * min(s["signature_concordance_p"] for s in rows)))

    # per-gene stage consistency among signature genes present in all three stages
    wide = all_tab.pivot_table(index="gene_id", columns="stage", values="ad_log2")
    wide_q = all_tab.pivot_table(index="gene_id", columns="stage", values="qvalue_unified")
    present = wide.dropna()
    sig_present = present.index.intersection(sig_genes)
    consistent = (np.sign(present.loc[sig_present]) ==
                  np.sign(dry_eff.loc[sig_present]).to_numpy()[:, None]).sum(axis=1)
    summary["signature"]["n_present_all_stages"] = int(len(sig_present))
    summary["signature"]["stage_sign_count_distribution"] = {
        int(k): int(v) for k, v in consistent.value_counts().sort_index().items()}

    summary["runtime_seconds"] = time.time() - t0
    (OUT / "stage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ---- figures -----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    for ax, stage in zip(axes, STAGES):
        tab = all_tab[all_tab["stage"] == stage].set_index("gene_id")
        common = tab.index.intersection(dry_eff.dropna().index)
        ax.scatter(dry_eff.loc[common], tab.loc[common, "ad_log2"], s=2, alpha=0.15,
                   color="#777777", rasterized=True)
        sigc = common.intersection(sig_genes)
        ax.scatter(dry_eff.loc[sigc], tab.loc[sigc, "ad_log2"], s=3, alpha=0.45,
                   color="#c0392b", rasterized=True)
        lim = 4
        ax.plot([-lim, lim], [-lim, lim], color="black", lw=0.8, ls="--")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        st = summary["stages"][stage]
        ax.set_title(f"{stage}: rho={st['all_gene_effect_spearman_vs_dry']:.2f}, "
                     f"slope={st['all_gene_slope_vs_dry'][0]:.2f}")
        ax.set_xlabel("dry seed AD|B=0 (log2)")
        ax.axhline(0, color="grey", lw=0.5)
        ax.axvline(0, color="grey", lw=0.5)
    axes[0].set_ylabel("stage AD (log2)")
    fig.tight_layout()
    fig.savefig(FIG / "stage_ad_concordance.png", dpi=300)
    fig.savefig(FIG / "stage_ad_concordance.pdf")
    plt.close(fig)

    # signature effect distribution per stage
    fig, ax = plt.subplots(figsize=(5.2, 4))
    data, labels = [], []
    for stage in STAGES:
        tab = all_tab[all_tab["stage"] == stage].set_index("gene_id")
        genes_here = tab.index.intersection(sig_genes)
        vals = tab.loc[genes_here, "ad_log2"] * np.sign(dry_eff.loc[genes_here])
        data.append(vals.to_numpy())
        labels.append(f"{stage}\n(n={len(vals)})")
    bp = ax.boxplot(data, labels=labels, showfliers=False, widths=0.55,
                    medianprops=dict(color="#c0392b", lw=1.5))
    for i, vals in enumerate(data, start=1):
        ax.scatter(np.full(len(vals), i) + np.random.default_rng(1).normal(0, 0.05, len(vals)),
                   vals, s=1, alpha=0.15, color="#2c3e50", rasterized=True)
        ax.scatter([i], [np.mean(vals)], marker="D", color="#e67e22", zorder=5, s=30)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("AD interaction aligned to dry-seed direction (log2)")
    ax.set_title("Dry-seed AD signature across development")
    fig.tight_layout()
    fig.savefig(FIG / "stage_signature_effects.png", dpi=300)
    fig.savefig(FIG / "stage_signature_effects.pdf")
    plt.close(fig)

    print(json.dumps(summary["signature"], indent=2))
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
