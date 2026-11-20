"""Figure for the two external validations (V1 GSE251921, V2 GSE5634)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "results" / "external"
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.5,
    "axes.linewidth": 0.7,
    "axes.labelsize": 7.5,
    "axes.titlesize": 7.8,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def panel_a(ax):
    v1 = pd.read_csv(EXT / "v1_effects.csv.gz").set_index("gene_id")
    sub = v1[v1["in_S_D"]].dropna(subset=["main_log2"])
    s = json.loads((EXT / "v1_summary.json").read_text())["S_D"]
    lim = 10
    inside = (sub["main_log2"].abs() <= lim) & (sub["log2fc"].abs() <= lim)
    ax.scatter(sub.loc[inside, "main_log2"], sub.loc[inside, "log2fc"],
               s=6, alpha=0.35, color=OKABE[0], edgecolors="none", rasterized=True)
    ax.plot([-lim, lim], [-lim, lim], ls="--", color="grey", lw=0.8)
    from scipy import stats
    lr = stats.linregress(sub["main_log2"], sub["log2fc"])
    xx = np.linspace(-lim, lim, 50)
    ax.plot(xx, lr.intercept + lr.slope * xx, color=OKABE[1], lw=1.2)
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Main study dog1-4 effect (log$_2$)")
    ax.set_ylabel("V1 dog1-4 vs Col-0 (log$_2$)")
    ax.text(0.03, 0.97,
            f"n = {s['n']}/1,989 S$_{{D}}$ genes testable\ncoverage gate not met (24%)\n"
            f"sign agreement {s['sign_concordance']:.1%}\n"
            f"Spearman rho = {s['spearman']:.2f}\nslope = {s['slope']:.2f} "
            f"[{s['slope_ci95'][0]:.2f}, {s['slope_ci95'][1]:.2f}]",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.5))
    ax.text(0.03, 0.05,
            f"{int((~inside).sum())} of {len(sub)} genes outside axis limits",
            transform=ax.transAxes, fontsize=6.5, color="0.35")


def panel_b(ax):
    sc = pd.read_csv(EXT / "v1_signature_scores.csv")
    rng = np.random.default_rng(20261030)
    groups = ["Col-0", "dog1-4"]
    colors = [OKABE[0], OKABE[1]]
    for i, (g, c) in enumerate(zip(groups, colors)):
        vals = sc.loc[sc["group"] == g, "score"].to_numpy()
        x = i + rng.uniform(-0.06, 0.06, len(vals))
        ax.scatter(x, vals, s=26, color=c, alpha=0.85, edgecolors="none", zorder=3)
        ax.hlines(vals.mean(), i - 0.18, i + 0.18, color="black", lw=1.2, zorder=4)
    ax.set_xticks([0, 1], groups)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("S$_{D}$ signature score (z units)")
    ax.text(0.03, 0.97,
            "4 vs 4 arrays\nexact permutation p = 0.029",
            transform=ax.transAxes, va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.5))


def panel_c(ax):
    st = pd.read_csv(EXT / "v2_stage_scores.csv")
    rng = np.random.default_rng(20261030)
    for s in sorted(st["stage"].unique()):
        vals = st.loc[st["stage"] == s, "score"].to_numpy()
        x = s + rng.uniform(-0.10, 0.10, len(vals))
        ax.scatter(x, vals, s=20, color=OKABE[2], alpha=0.8, edgecolors="none", zorder=3)
    means = st.groupby("stage")["score"].mean()
    sds = st.groupby("stage")["score"].std()
    ax.errorbar(means.index, means.values, yerr=sds.values, color="black", lw=1.2,
                capsize=3, marker="o", ms=3.5, zorder=4)
    ax.set_xticks([6, 7, 8, 9, 10])
    ax.set_xlim(5.5, 10.5)
    ax.set_xlabel("Seed developmental stage")
    ax.set_ylabel("Frozen maturity score (z units)")
    ax.text(0.03, 0.97,
            "5 stage means, 3 arrays each\nSpearman rho = 1.00, exact p = 0.017\n"
            "array-level rho = 0.97 (descriptive)",
            transform=ax.transAxes, va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.5))


def panel_d(ax):
    m = pd.read_csv(EXT / "v2_marker_stages.csv")
    stages = np.arange(6, 11)
    offsets = {"OLEOSIN1": -0.05, "CRUCIFERIN2": 0.05, "LEA": 0.0,
               "DOG1": 0.0, "ABI3": 0.0, "FUS3": 0.0}
    for (_, row), color in zip(m.iterrows(), OKABE):
        if not bool(row["mapped"]):
            continue
        vals = [row[f"stage{s}"] for s in stages]
        ax.plot(stages, vals, marker="o", ms=3, lw=1.1, color=color)
        ax.text(10.12, vals[-1] + offsets.get(row["name"], 0), row["name"],
                color=color, fontsize=7, va="center")
    ax.set_xticks(stages)
    ax.set_xlim(5.7, 11.1)
    ax.set_xlabel("Seed developmental stage")
    ax.set_ylabel("MAS5 signal (log$_2$)")
    ax.text(0.03, 0.97, "Frozen marker genes, V2 arrays",
            transform=ax.transAxes, va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.5))


def main():
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4))
    panel_a(axes[0, 0])
    panel_b(axes[0, 1])
    panel_c(axes[1, 0])
    panel_d(axes[1, 1])
    for ax, letter in zip(axes.ravel(), "abcd"):
        ax.text(-0.14, 1.03, letter, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="bottom", ha="left")
    fig.tight_layout(w_pad=2.0, h_pad=2.0)
    fig.savefig(FIG / "external_validation.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG / "external_validation.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved external_validation.png/pdf")


if __name__ == "__main__":
    main()
