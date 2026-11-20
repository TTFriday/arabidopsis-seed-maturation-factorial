"""Figure: numerical stability of the broad interaction candidates."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.5,
    "pdf.fonttype": 42,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
BLUE, RED = "#0072B2", "#D55E00"


def main():
    a = pd.read_csv(T / "candidate_audit_all_genes.csv.gz")
    s = json.loads((T / "candidate_audit_summary.json").read_text())
    broad = a[(a["lrt_q"] < 0.05) & (a["max_abs_all"] >= 1)].copy()
    broad["y"] = np.maximum(broad["se_at_max_abs"], 1e-3)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    stable = ~broad["unstable_extreme"]
    ax.scatter(broad.loc[stable, "max_abs_all"], broad.loc[stable, "y"],
               s=5, alpha=0.30, color=BLUE, edgecolors="none", rasterized=True,
               label=f"not separation-like ({int(stable.sum()):,})")
    ax.scatter(broad.loc[~stable, "max_abs_all"], broad.loc[~stable, "y"],
               s=6, alpha=0.55, color=RED, edgecolors="none", rasterized=True,
               label=f"separation-like ({int((~stable).sum()):,})")
    ax.axvline(10, color="grey", ls="--", lw=0.8)
    ax.axhline(2, color="grey", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Largest |interaction contrast| (log2)")
    ax.set_ylabel("Wald SE of that contrast")
    ax.set_xlim(1, 120)
    ax.set_ylim(1e-2, 1e4)
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    top = broad.nlargest(1, "max_abs_all").iloc[0]
    ax.annotate(top["gene_id"], (top["max_abs_all"], top["y"]),
                xytext=(-28, -16), textcoords="offset points", fontsize=6.5,
                arrowprops=dict(arrowstyle="-", lw=0.5, color="0.4"))
    ax.text(0.04, 0.96,
            f"broad candidates: {len(broad):,}\n"
            f"not separation-like: {int(stable.sum()):,}\n"
            f"separation-like: {int((~stable).sum()):,}",
            transform=ax.transAxes, va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.5))

    ax = axes[1]
    labels = {"AD_given_B0": "AD | B=0", "AD_given_B1": "AD | B=1",
              "AB_given_D0": "AB | D=0", "AB_given_D1": "AB | D=1",
              "DB_given_A0": "DB | A=0", "DB_given_A1": "DB | A=1",
              "ADB_three_way": "ADB"}
    names = list(labels)
    vals = [s["per_contrast"][n]["n_supported_stable"] for n in names]
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=BLUE, height=0.62)
    ax.set_yticks(y, [labels[n] for n in names])
    ax.set_xlabel("Genes with stable support")
    for yi, v in zip(y, vals):
        ax.text(v + 60, yi, f"{v:,}", va="center", fontsize=7)
    ax.set_xlim(0, max(vals) * 1.22)
    ax.text(0.04, 0.96,
            "q < 0.05, |log2| \u2265 1, SE \u2264 1\nand LRT q < 0.05",
            transform=ax.transAxes, va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.5))

    for ax, letter in zip(axes, "ab"):
        ax.text(-0.15, 1.03, letter, transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="bottom", ha="left")
    fig.tight_layout(w_pad=2.2)
    fig.savefig(FIG / "candidate_stability.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "candidate_stability.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved candidate_stability.png/pdf")


if __name__ == "__main__":
    main()
