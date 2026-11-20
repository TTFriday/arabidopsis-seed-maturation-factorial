"""Exploratory, genotype-level analysis of deposited dry-seed traits.

Table S10 of Artur et al. (2026) contains four plant-pool replicates of
chlorophyll fluorescence, germination and dormancy. Table S11 contains
four replicate EPPO ageing curves per genotype. Trait replicates are not
paired with individual RNA-seq libraries.

The primary longevity summary is trapezoidal area under the complete
0-160-day germination curve (AUC). A secondary P50 uses equal-weight
isotonic regression and interpolation at 50% germination. A curve whose
initial viability is already below 50% has an undefined P50, not zero.
Probit fits are retained only as diagnostics, including their lack of fit.

Contrasts use separate cell variances and Welch-Satterthwaite degrees of
freedom, not a common variance across genotypes. The 42 interaction tests
(six outcomes x seven contrasts) share one BH family. All unique pairwise
cell comparisons share a separate BH family. Non-significance is never
interpreted as equivalence or proof of a common pathway.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[1]
XLSX = ROOT / "data" / "raw" / "supplemental_tables.xlsx"
OUT = ROOT / "results" / "traits"
OUT.mkdir(parents=True, exist_ok=True)

FACTORIAL = {
    "Col-0": (0, 0, 0), "aba2-1": (0, 0, 1),
    "dog1-4": (0, 1, 0), "dog1-4 aba2-1": (0, 1, 1),
    "abi3-12": (1, 0, 0), "abi3-12 aba2-1": (1, 0, 1),
    "abi3-12 dog1-4": (1, 1, 0), "abi3-12 dog1-4 aba2-1": (1, 1, 1),
}
EXTRA = ["abi3-12 dog1-4 cyp707a2", "abi3-6", "L2-1 Col-0 + DOG1-Cvi"]
CELL_CONTRASTS = {
    "AD_given_B0": {6: 1, 4: -1, 2: -1, 0: 1},
    "AD_given_B1": {7: 1, 5: -1, 3: -1, 1: 1},
    "AB_given_D0": {5: 1, 4: -1, 1: -1, 0: 1},
    "AB_given_D1": {7: 1, 6: -1, 3: -1, 2: 1},
    "DB_given_A0": {3: 1, 2: -1, 1: -1, 0: 1},
    "DB_given_A1": {7: 1, 6: -1, 5: -1, 4: 1},
    "ADB_three_way": {7: 1, 6: -1, 5: -1, 3: -1, 4: 1, 2: 1, 1: 1, 0: -1},
}
PAIRS = {
    "AD_given_B0": ("A", "D", 0, 4, 2, 6),
    "AD_given_B1": ("A", "D", 1, 5, 3, 7),
    "AB_given_D0": ("A", "B", 0, 4, 1, 5),
    "AB_given_D1": ("A", "B", 2, 6, 3, 7),
    "DB_given_A0": ("D", "B", 0, 2, 1, 3),
    "DB_given_A1": ("D", "B", 4, 6, 5, 7),
}
ALLELE = {"A": "abi3-12", "D": "dog1-4", "B": "aba2-1"}


def bh_qvalues(p):
    p = np.nan_to_num(np.asarray(p, dtype=float), nan=1.0)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    out = np.empty(len(p))
    out[order] = np.minimum.accumulate(ranked[::-1])[::-1].clip(0, 1)
    return out


def genotype_from_label(label):
    label = str(label).strip()
    for g in sorted(list(FACTORIAL) + EXTRA, key=len, reverse=True):
        if label.startswith(g + " "):
            return g
    return None


def read_dry_traits():
    t = pd.read_excel(XLSX, sheet_name="Table S10", header=2)
    t = t.rename(columns={t.columns[0]: "label"})
    t = t[t["DAP"].astype(str) == "Dry"].copy()
    t["genotype"] = t["label"].map(genotype_from_label)
    if t["genotype"].isna().any():
        raise ValueError("unrecognised genotype label in Table S10")
    t = t.rename(columns={
        "Chlorophyll Fluorescence": "chlorophyll_fluorescence",
        "Total Germination (%)": "germination_pct",
        "Normal Seedlings (%)": "normal_seedlings_pct",
        "Dormancy (%)": "dormancy_pct", "Replicate": "rep",
    })
    return t[["genotype", "rep", "chlorophyll_fluorescence", "germination_pct",
              "normal_seedlings_pct", "dormancy_pct"]]


def probit_fit(days, frac):
    """Equal-weight fractional-binomial objective; diagnostic fit only."""
    frac = np.clip(frac, 0.0, 1.0)

    def nll(theta):
        pi = stats.norm.cdf(theta[0] - np.exp(theta[1]) * days).clip(1e-9, 1 - 1e-9)
        return -np.sum(frac * np.log(pi) + (1 - frac) * np.log(1 - pi))

    best = None
    for a0 in (1.0, 2.0, 3.0):
        for b0 in (0.005, 0.02, 0.08):
            r = optimize.minimize(nll, [a0, np.log(b0)], method="Nelder-Mead",
                                  options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 4000})
            if best is None or r.fun < best.fun:
                best = r
    return float(best.x[0]), float(np.exp(best.x[1]))


def empirical_p50(days, frac):
    """Interpolate a non-increasing fitted curve; never clamp a negative P50."""
    if frac[0] < 0.5:
        return np.nan
    fitted = optimize.isotonic_regression(frac, increasing=False).x
    if fitted[0] < 0.5 or fitted[-1] > 0.5:
        return np.nan
    j = int(np.flatnonzero(fitted <= 0.5)[0])
    if j == 0:
        return float(days[0])
    return float(days[j - 1] + (fitted[j - 1] - 0.5) / (fitted[j - 1] - fitted[j])
                 * (days[j] - days[j - 1]))


def read_longevity():
    t = pd.read_excel(XLSX, sheet_name="Table S11", header=2).rename(columns={
        "Genotype": "genotype", "Replicate": "rep", "Days of EPPO": "days",
        "EPPO - Germination (%)": "germination_pct",
        "EPPO - Seedlings (%)": "seedlings_pct",
    })
    rows = []
    for (g, rep), sub in t.groupby(["genotype", "rep"]):
        sub = sub.sort_values("days")
        days = sub["days"].to_numpy(dtype=float)
        row = {"genotype": g, "rep": rep}
        for col, tag in (("germination_pct", "germ"), ("seedlings_pct", "seedl")):
            frac = sub[col].to_numpy(dtype=float) / 100.0
            a, b = probit_fit(days, frac)
            row[f"{tag}_probit_a"] = a
            row[f"{tag}_probit_b"] = b
            row[f"{tag}_probit_p50_days"] = a / b
            row[f"{tag}_probit_rmse_pct"] = float(100 * np.sqrt(
                np.mean((stats.norm.cdf(a - b * days) - frac) ** 2)))
            row[f"{tag}_p50_days"] = empirical_p50(days, frac)
            row[f"{tag}_auc_days"] = float(np.trapezoid(frac, days))
            row[f"{tag}_initial_pct"] = float(sub[col].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows), t


def cell_index(genotype):
    a, d, b = FACTORIAL[genotype]
    return 4 * a + 2 * d + b


def heteroscedastic_contrast(values, weights):
    """Independent-cell Welch contrast, allowing zero variance in some cells."""
    estimate = sum(w * np.mean(values[g]) for g, w in weights.items())
    pieces = np.array([w * w * np.var(values[g], ddof=1) / len(values[g])
                       for g, w in weights.items()])
    n = np.array([len(values[g]) for g in weights])
    variance = pieces.sum()
    if variance <= 1e-24:
        return estimate, 0.0, 1.0, np.nan, estimate, estimate
    se = np.sqrt(variance)
    df = variance ** 2 / np.sum(pieces ** 2 / (n - 1))
    p = 2 * stats.t.sf(abs(estimate / se), df)
    half = stats.t.ppf(0.975, df) * se
    return estimate, se, p, df, estimate - half, estimate + half


def factorial_table(data, col, transform, label):
    sub = data[data.genotype.isin(FACTORIAL)].copy()
    values = {cell_index(g): transform(grp[col].to_numpy(dtype=float))
              for g, grp in sub.groupby("genotype")}
    if any(len(v) != 4 or not np.isfinite(v).all() for v in values.values()):
        raise ValueError(f"{label}: incomplete factorial outcome")
    contrasts, pairs, comparisons = [], [], []
    for name, w in CELL_CONTRASTS.items():
        est, se, p, df, lo, hi = heteroscedastic_contrast(values, w)
        contrasts.append({"trait": label, "contrast": name, "estimate": est, "se": se,
                          "ci_low": lo, "ci_high": hi, "pvalue": p, "df": df})
    for g in range(8):
        for h in range(g):
            est, se, p, df, lo, hi = heteroscedastic_contrast(values, {g: 1, h: -1})
            comparisons.append({"trait": label, "cell_first": g, "cell_second": h,
                                "estimate": est, "se": se, "pvalue": p,
                                "df": df, "ci_low": lo, "ci_high": hi})
    for name, (pn, qn, ref, gp, gq, gpq) in PAIRS.items():
        row = {"trait": label, "pair": name, "first": ALLELE[pn], "second": ALLELE[qn]}
        for key, w in [("single_first", {gp: 1, ref: -1}),
                       ("single_second", {gq: 1, ref: -1}),
                       ("double", {gpq: 1, ref: -1}),
                       ("interaction", CELL_CONTRASTS[name]),
                       ("double_vs_first", {gpq: 1, gp: -1}),
                       ("double_vs_second", {gpq: 1, gq: -1})]:
            est, se, p, df, lo, hi = heteroscedastic_contrast(values, w)
            row.update({key: est, key + "_se": se, key + "_p": p,
                        key + "_ci_low": lo, key + "_ci_high": hi})
        pairs.append(row)
    cells = [{"trait": label, "genotype": g, "cell": cell_index(g),
              "mean_transformed": float(values[cell_index(g)].mean()), "n": 4}
             for g in FACTORIAL]
    return contrasts, pairs, comparisons, cells


def descriptive(data, cols):
    rows = []
    for g, sub in data.groupby("genotype"):
        row = {"genotype": g, "n": len(sub)}
        for c in cols:
            row[c + "_mean"] = float(sub[c].mean())
            row[c + "_sd"] = float(sub[c].std(ddof=1))
        rows.append(row)
    out = pd.DataFrame(rows)
    order = {g: i for i, g in enumerate(list(FACTORIAL) + EXTRA)}
    return out.assign(order=out.genotype.map(order)).sort_values("order").drop(columns="order")


def main():
    dry = read_dry_traits()
    longevity, eppo = read_longevity()
    dry.to_csv(OUT / "dry_seed_traits_replicates.csv", index=False)
    longevity.to_csv(OUT / "eppo_replicate_fits.csv", index=False)
    curves = eppo.groupby(["genotype", "days"]).germination_pct.agg(
        ["mean", "std", "count"]).reset_index()
    curves.to_csv(OUT / "eppo_germination_curves.csv", index=False)
    desc = descriptive(dry, ["chlorophyll_fluorescence", "germination_pct",
                             "normal_seedlings_pct", "dormancy_pct"]).merge(
        descriptive(longevity, ["germ_p50_days", "germ_auc_days", "seedl_auc_days",
                                 "germ_initial_pct"]), on="genotype", how="outer",
        suffixes=("", "_eppo"))
    desc.to_csv(OUT / "trait_genotype_summary.csv", index=False)
    log2p = lambda v: np.log2(v + 1.0)
    outcomes = [
        (dry, "chlorophyll_fluorescence", np.log2, "chlorophyll fluorescence (log2)"),
        (longevity, "germ_auc_days", log2p, "longevity AUC (log2 days+1)"),
        (longevity, "germ_p50_days", log2p, "longevity P50 (log2 days+1)"),
        (longevity, "seedl_auc_days", log2p, "seedling AUC (log2 days+1)"),
        (dry, "dormancy_pct", lambda v: v, "dormancy (%)"),
        (dry, "germination_pct", lambda v: v, "germination of fresh seeds (%)"),
    ]
    allc, allp, alltests, allcells = [], [], [], []
    for data, col, fn, label in outcomes:
        c, p, comparisons, cells = factorial_table(data, col, fn, label)
        allc.extend(c); allp.extend(p); alltests.extend(comparisons); allcells.extend(cells)
    c, pairs, comparisons = map(pd.DataFrame, (allc, allp, alltests))
    c["qvalue"] = bh_qvalues(c.pvalue)
    comparisons["qvalue"] = bh_qvalues(comparisons.pvalue)
    qmap = c.set_index(["trait", "contrast"]).qvalue
    pairs["interaction_q"] = [qmap.loc[(r.trait, r.pair)] for r in pairs.itertuples()]
    pairs["class"] = "not detected"
    detected = pairs.interaction_q < 0.05
    beyond = ((np.sign(pairs.interaction) == np.sign(pairs.double)) &
              (pairs.double.abs() > (pairs.single_first + pairs.single_second).abs()))
    pairs.loc[detected & beyond, "class"] = "synergistic"
    pairs.loc[detected & ~beyond, "class"] = "opposing interaction"
    c.to_csv(OUT / "trait_interaction_contrasts.csv", index=False)
    pairs.to_csv(OUT / "trait_pairwise_epistasis.csv", index=False)
    comparisons.to_csv(OUT / "trait_cell_comparisons.csv", index=False)
    pd.DataFrame(allcells).to_csv(OUT / "trait_cell_means.csv", index=False)

    extra_rows = []
    for data, col, fn, label, tests in [
        (longevity, "germ_auc_days", log2p, "longevity AUC (log2 days+1)",
         [(EXTRA[0], "abi3-12 dog1-4"), (EXTRA[0], "Col-0"),
          ("abi3-6", "Col-0"), ("abi3-6", "abi3-12")]),
        (dry, "chlorophyll_fluorescence", np.log2, "chlorophyll fluorescence (log2)",
         [(EXTRA[0], "abi3-12 dog1-4"), (EXTRA[0], "Col-0"),
          ("abi3-6", "Col-0"), (EXTRA[2], "Col-0")])]:
        values = {g: fn(s[col].to_numpy(dtype=float)) for g, s in data.groupby("genotype")}
        for a, b in tests:
            est, se, p, df, lo, hi = heteroscedastic_contrast(values, {a: 1, b: -1})
            extra_rows.append({"trait": label, "comparison": f"{a} vs {b}",
                               "difference": est, "se": se, "welch_p": p,
                               "ci_low": lo, "ci_high": hi, "df": df})
    extra = pd.DataFrame(extra_rows)
    extra["qvalue"] = bh_qvalues(extra.welch_p)
    extra.to_csv(OUT / "trait_internal_comparisons.csv", index=False)
    summary = {
        "source": "Artur et al. 2026, Tables S10/S11",
        "replicates_per_genotype": 4,
        "primary_longevity": "Trapezoidal germination AUC, 0-160 days",
        "secondary_p50": "Isotonic regression plus interpolation; undefined below 50% at harvest",
        "inference": "Cell-specific variance, Welch-Satterthwaite df, BH across 42 interactions",
        "n_interaction_tests": len(c), "n_cell_comparisons": len(comparisons),
        "n_probit_p50_below_zero": int((longevity.germ_probit_p50_days < 0).sum()),
        "n_empirical_p50_undefined": int(longevity.germ_p50_days.isna().sum()),
        "eppo_days": sorted(eppo.days.unique().tolist()),
    }
    (OUT / "trait_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(c.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
