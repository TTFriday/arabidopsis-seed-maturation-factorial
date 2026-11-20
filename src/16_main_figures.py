"""Design, quality-control, background-dependence and stage-support figures.

All panels are drawn directly from the derived result tables.

  design_overview         sample landscape, the 2x2x2 design and analysis layers
  qc_nonadditivity        library QC, interaction tests and representative profiles
  background_dependence   pairwise interactions compared across backgrounds
  stage_support           developmental and same-study genetic support

The external datasets are drawn by 13_external_figures.py and the trait,
epistasis and annotation figures by 21_interaction_figures.py.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nb_glm as nb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RES = ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)

CONTRASTS = ["AD_given_B0", "AD_given_B1", "AB_given_D0", "AB_given_D1",
             "DB_given_A0", "DB_given_A1", "ADB_three_way"]
CONTRAST_LABEL = {
    "AD_given_B0": "AD | B=0", "AD_given_B1": "AD | B=1",
    "AB_given_D0": "AB | D=0", "AB_given_D1": "AB | D=1",
    "DB_given_A0": "DB | A=0", "DB_given_A1": "DB | A=1",
    "ADB_three_way": "ADB",
}
# group index follows g = 4*A + 2*D + B
GROUPS = ["000", "001", "010", "011", "100", "101", "110", "111"]
GENOTYPE = {
    "000": "Col-0", "001": "aba2-1", "010": "dog1-4", "011": "dog1-4 aba2-1",
    "100": "abi3-12", "101": "abi3-12 aba2-1", "110": "abi3-12 dog1-4",
    "111": "abi3-12 dog1-4 aba2-1",
}
GROUP_COLOR = {
    "000": "#333333", "001": "#E69F00", "010": "#56B4E9", "011": "#0072B2",
    "100": "#009E73", "101": "#66A61E", "110": "#D55E00", "111": "#CC79A7",
}
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.5,
    "axes.titlesize": 8.5,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.8,
    "ytick.major.size": 2.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
})


def panel_label(ax, letter, x=-0.09, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontweight="bold",
            fontsize=11, va="bottom", ha="left")


def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# design overview
# ---------------------------------------------------------------------------
def figure1():
    samples = pd.read_csv(DATA / "processed" / "samples.csv")
    design = pd.read_csv(DATA / "processed" / "main_design_24.csv")

    order = ["Col-0", "aba2-1", "dog1-4", "dog1-4_aba2-1",
             "abi3-12", "abi3-12_aba2-1", "abi3-12_dog1-4",
             "abi3-12_dog1-4_aba2-1", "abi3-12_dog1-4_cyp707a2",
             "abi3-6", "L2-1_Col-0_+_DOG1-Cvi"]
    display = ["Col-0", "aba2-1", "dog1-4", "dog1-4 aba2-1",
               "abi3-12", "abi3-12 aba2-1", "abi3-12 dog1-4",
               "abi3-12 dog1-4 aba2-1", "abi3-12 dog1-4 cyp707a2",
               "abi3-6", "DOG1-Cvi (Col-0)"]
    stages = ["12d", "14d", "16d", "18d", "20d", "22d", "24d", "DS"]
    stage_disp = ["12", "14", "16", "18", "20", "22", "24", "DS"]
    counts = samples.groupby(["genotype", "stage", "round"]).size().unstack(fill_value=0)

    fig = plt.figure(figsize=(7.2, 7.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], width_ratios=[1.18, 1.0],
                          hspace=0.42, wspace=0.26,
                          left=0.115, right=0.985, top=0.975, bottom=0.045)

    # ---- (a) sample landscape -------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    nrow, ncol = len(order), len(stages)
    ax.set_xlim(-0.6, ncol - 0.4)
    ax.set_ylim(nrow - 0.4, -0.6)
    ax.set_xticks(range(ncol))
    ax.set_xticklabels(stage_disp)
    ax.set_yticks(range(nrow))
    ax.set_yticklabels(display, fontsize=6.4)
    for t in ax.get_yticklabels():
        if t.get_text() != "Col-0":
            t.set_fontstyle("italic")
    ax.set_xlabel("Developmental stage (DAP; DS, dry seed)")
    for i in range(nrow):
        for j in range(ncol):
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="white",
                                   edgecolor="#E4E4E4", linewidth=0.5, zorder=0))
            try:
                r2 = int(counts.loc[(order[i], stages[j]), 2])
            except KeyError:
                r2 = 0
            try:
                r3 = int(counts.loc[(order[i], stages[j]), 3])
            except KeyError:
                r3 = 0
            total = r2 + r3
            if total == 0:
                continue
            # stacked bar, height scaled to at most 0.62 of a cell
            hmax = 0.62
            h2 = hmax * r2 / 6.0
            h3 = hmax * r3 / 6.0
            base = i + 0.30
            if r2:
                ax.add_patch(Rectangle((j - 0.25, base - h2), 0.5, h2,
                                       facecolor="#BDBDBD", edgecolor="none", zorder=2))
            if r3:
                ax.add_patch(Rectangle((j - 0.25, base - h2 - h3), 0.5, h3,
                                       facecolor="#4D4D4D", edgecolor="none", zorder=3))
            ax.text(j, i - 0.33, str(total), ha="center", va="center",
                    fontsize=5.8, color="#222222", zorder=4)
    # outlines: main 24-sample design, stage panel, internal perturbations
    for i in range(8):
        ax.add_patch(Rectangle((ncol - 1 - 0.5, i - 0.5), 1, 1, fill=False,
                               edgecolor="#E69F00", linewidth=1.4, zorder=6))
    for i in (0, 2, 4, 6):
        for j in (4, 5, 6):
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="#009E73", linewidth=0.9,
                                   linestyle=(0, (2, 1.2)), zorder=5))
    for i in (8, 9, 10):
        ax.add_patch(Rectangle((ncol - 1 - 0.5, i - 0.5), 1, 1, fill=False,
                               edgecolor="#CC79A7", linewidth=0.9,
                               linestyle=(0, (2, 1.2)), zorder=5))
    handles = [
        Rectangle((0, 0), 1, 1, facecolor="#BDBDBD", edgecolor="none"),
        Rectangle((0, 0), 1, 1, facecolor="#4D4D4D", edgecolor="none"),
        Line2D([0], [0], color="#E69F00", lw=1.4),
        Line2D([0], [0], color="#009E73", lw=0.9, ls=(0, (2, 1.2))),
        Line2D([0], [0], color="#CC79A7", lw=0.9, ls=(0, (2, 1.2))),
    ]
    ax.legend(handles, ["Round 2", "Round 3", "24-sample design",
                        "Stage panel", "Internal lines"],
              loc="upper left", bbox_to_anchor=(0.0, -0.16), ncol=3,
              frameon=False, handlelength=1.2, columnspacing=1.0,
              handletextpad=0.5)
    ax.set_title("Sample landscape of GSE270988 (123 RNA-seq samples)")
    panel_label(ax, "a", x=-0.42, y=1.03)

    # ---- (b) 2x2x2 design -----------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    ax.set_xlim(0, 2)
    ax.set_ylim(4, 0)
    ax.axis("off")
    cells = [
        (0, 0, "000"), (0, 1, "001"),
        (1, 0, "010"), (1, 1, "011"),
        (2, 0, "100"), (2, 1, "101"),
        (3, 0, "110"), (3, 1, "111"),
    ]
    for r, c, code in cells:
        ax.add_patch(FancyBboxPatch((c + 0.04, r + 0.08), 0.92, 0.84,
                                    boxstyle="round,pad=0.01,rounding_size=0.05",
                                    facecolor=GROUP_COLOR[code], alpha=0.12,
                                    edgecolor=GROUP_COLOR[code], linewidth=1.0))
        ax.text(c + 0.5, r + 0.36, GENOTYPE[code], ha="center", va="center",
                fontsize=6.5, color="#111111",
                style="normal" if code == "000" else "italic")
        ax.text(c + 0.5, r + 0.68, "n = 3", ha="center", va="center",
                fontsize=6.2, color="#444444")
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["B = 0\n(ABA2)", "B = 1\n(aba2-1)"], fontsize=7)
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_yticks([0.5, 1.5, 2.5, 3.5])
    ax.set_yticklabels(["A=0, D=0", "A=0, D=1", "A=1, D=0", "A=1, D=1"],
                       fontsize=6.8)
    ax.tick_params(axis="y", length=0)
    for y in (1, 2, 3):
        ax.plot([0, 2], [y, y], color="#DDDDDD", lw=0.6)
    ax.text(-0.12, -0.30, "A = $ABI3$ ($abi3$-$12$)   D = $DOG1$ ($dog1$-$4$)   B = $ABA2$ ($aba2$-$1$)",
            transform=ax.transAxes, fontsize=6.4, color="#333333", ha="left")
    ax.set_title("Balanced dry-seed factorial (round 3)", pad=20)
    panel_label(ax, "b", x=-0.12, y=1.02)

    # ---- (c) evidence boundaries ---------------------------------------------
    ax = fig.add_subplot(gs[1, :])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, title, body, edge, face, title_size=7.4, body_size=6.4):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.6,rounding_size=1.2",
                                    facecolor=face, edgecolor=edge, linewidth=0.9))
        ax.text(x + 1.6, y + h - 2.2, title, fontsize=title_size,
                fontweight="bold", va="top", ha="left", color="#111111")
        ax.text(x + 1.6, y + h - 7.0, body, fontsize=body_size, va="top",
                ha="left", color="#222222", linespacing=1.3)

    box(2, 66, 29, 25, "Discovery factorial",
        "24 dry-seed libraries\n8 genotypes × 3 replicates\nNB GLM, 4-df LRT, 7 contrasts", "#0072B2", "#EAF2F8")
    box(35.5, 66, 29, 25, "Epistasis layer",
        "coefficient s vs simulated additive,\nmasking and suppression models;\neight gene-level interaction types", "#7B3294", "#F3EAF7")
    box(69, 66, 29, 25, "Functional layer",
        "GO biological process\nABI3 targets (Tian et al. 2020)\nRY and ABRE promoter elements", "#1B7837", "#EAF5EC")
    box(2, 36, 29, 25, "Trait layer (same study)",
        "dormancy, chlorophyll, EPPO longevity\n4 replicates per genotype\ngenotype level; no library pairing", "#8C564B", "#F6EEEA")
    box(35.5, 36, 29, 25, "Same-study stage support",
        "36 libraries, 20/22/24 DAP\nfrozen 1,396-gene signature\nconsistency, not replication", "#009E73", "#E9F5EF")
    box(69, 36, 29, 25, "Same-study genetic support",
        "CYP707A2 combination\nabi3-6; DOG1-Cvi\nno rescue claim", "#666666", "#F2F2F2")
    box(2, 6, 29, 25, "External data",
        "V1 GSE251921: 24% coverage,\ngate not met\nV2 GSE5634: maturation axis", "#E69F00", "#FFF6E5")
    box(35.5, 6, 29, 25, "Interpretation boundary",
        "log-scale non-additivity\n≠ physical interaction or\n≠ direct transcriptional control", "#B03A2E", "#F9ECEA")
    box(69, 6, 29, 25, "Reproducibility",
        "gene sets and external rules frozen\nbefore external data were opened;\nall code and result tables shared", "#555555", "#F2F2F2")

    arrow = dict(arrowstyle="-|>", mutation_scale=8, lw=0.9, color="#555555")
    ax.add_patch(FancyArrowPatch((31.6, 78.5), (35.0, 78.5), **arrow))
    ax.add_patch(FancyArrowPatch((65.1, 78.5), (68.5, 78.5), **arrow))
    ax.add_patch(FancyArrowPatch((24, 61.6), (37, 65.5), linestyle="--", **arrow))
    ax.text(50, 98, "Analysis layers and evidence boundaries",
            ha="center", va="top", fontsize=8.2, fontweight="bold")
    panel_label(ax, "c", x=-0.01, y=1.02)

    fig.savefig(FIG / "design_overview.png", dpi=600)
    fig.savefig(FIG / "design_overview.pdf")
    plt.close(fig)
    print("design_overview done")


# ---------------------------------------------------------------------------
# quality control and non-additivity
# ---------------------------------------------------------------------------
def load_counts_main():
    design = pd.read_csv(DATA / "processed" / "main_design_24.csv")
    counts = pd.read_csv(DATA / "raw" / "GSE270988_raw_counts.csv").set_index("gene_id")
    y_all = counts[design["sample_id"]].to_numpy(dtype=np.float64)
    cpm = y_all / y_all.sum(axis=0) * 1e6
    keep = (cpm >= 1).sum(axis=1) >= 3
    sf = nb.poscounts_size_factors(y_all[keep].T)
    return design, counts, y_all, keep, sf


def figure2():
    design, counts, y_all, keep, sf = load_counts_main()
    qc = pd.read_csv(RES / "qc" / "sample_qc_main24.csv").merge(design, on="sample_id")
    lrt = pd.read_csv(RES / "tables" / "lrt_all_genes.csv.gz")
    audit = pd.read_csv(RES / "tables" / "candidate_audit_all_genes.csv.gz")

    fig = plt.figure(figsize=(7.2, 6.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], height_ratios=[1, 1],
                          hspace=0.48, wspace=0.30,
                          left=0.085, right=0.985, top=0.965, bottom=0.07)

    # ---- (a) PCA -------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    for gi, code in enumerate(GROUPS):
        sub = qc[qc["A"] * 4 + qc["D"] * 2 + qc["B"] == gi]
        ax.scatter(sub["pc1"], sub["pc2"], s=24, marker=MARKERS[gi],
                   color=GROUP_COLOR[code], edgecolor="white", linewidth=0.4,
                   zorder=3)
        ax.text(sub["pc1"].mean(), sub["pc2"].mean() + 2.6, code,
                ha="center", va="bottom", fontsize=6.2, color=GROUP_COLOR[code],
                fontweight="bold")
    ax.axhline(0, color="#CCCCCC", lw=0.6, zorder=0)
    ax.axvline(0, color="#CCCCCC", lw=0.6, zorder=0)
    ax.set_xlabel("PC1 (79.4% variance)")
    ax.set_ylabel("PC2 (6.9% variance)")
    ax.set_title("Global structure of the 24 dry-seed samples")
    clean(ax)
    panel_label(ax, "a")

    # ---- (b) library size -----------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    data, colors = [], []
    for gi, code in enumerate(GROUPS):
        sub = qc[qc["A"] * 4 + qc["D"] * 2 + qc["B"] == gi]
        data.append(sub["library_size_m"].to_numpy())
        colors.append(GROUP_COLOR[code])
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color="#111111", lw=0.9),
                    whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7))
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.25)
        patch.set_edgecolor(col)
    rng = np.random.default_rng(7)
    for i, (vals, col) in enumerate(zip(data, colors), start=1):
        ax.scatter(np.full(len(vals), i) + rng.normal(0, 0.045, len(vals)),
                   vals, s=9, color=col, edgecolor="white", linewidth=0.3, zorder=3)
    ax.set_xticks(range(1, 9))
    ax.set_xticklabels(GROUPS, fontsize=6.3)
    ax.set_ylabel("Library size (million reads)")
    ax.set_title("Sequencing depth by genotype combination")
    clean(ax)
    panel_label(ax, "b")

    # ---- (c) effect-size distribution ---------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    x = lrt["max_abs_contrast_log2"].to_numpy()
    bins = np.linspace(0, 30, 91)
    ax.hist(x, bins=bins, color="#B8C4CE", edgecolor="white", linewidth=0.25)
    ax.axvline(1.0, color="#B03A2E", lw=0.9, ls="--")
    ax.set_yscale("log")
    ax.set_xlim(0, 30)
    ax.set_xlabel("Maximum |log$_2$ contrast| across seven tests")
    ax.set_ylabel("Genes (log scale)")
    ax.text(11.0, ax.get_ylim()[1] * 0.80,
            "6,305 broad candidates\n5,571 numerically stable\n921 separation-like",
            fontsize=6.4, va="top", ha="left", color="#222222")
    ax.set_title("Interaction effects span orders of magnitude")
    clean(ax)
    panel_label(ax, "c")

    # ---- (d) representative profiles ----------------------------------------
    gs_d = gs[1, 1].subgridspec(2, 2, hspace=0.55, wspace=0.38)
    genes = [("AT5G45830", "DOG1", "ADB q = 2.7 × 10$^{-5}$"),
             ("AT1G52340", "ABA2", "AD|B=0 q = 3.1 × 10$^{-5}$"),
             ("AT5G15970", "KIN2", "ADB q = 1.1 × 10$^{-7}$"),
             ("AT1G29930", "CAB1", "AD|B=0 q = 7.7 × 10$^{-32}$")]
    group_idx = (design["A"] * 4 + design["D"] * 2 + design["B"]).astype(int).to_numpy()
    libs = y_all.sum(axis=0)
    all_index = {s: i for i, s in enumerate(design["sample_id"])}
    for k, (gene, name, annotation) in enumerate(genes):
        ax = fig.add_subplot(gs_d[k // 2, k % 2])
        y = counts.loc[gene, design["sample_id"]].to_numpy(dtype=np.float64)
        cpm_g = (y / sf) / (libs / sf) * 1e6
        means, sems = [], []
        for gi in range(8):
            vals = np.log2(cpm_g[group_idx == gi] + 1)
            means.append(vals.mean())
            sems.append(vals.std(ddof=1) / np.sqrt(len(vals)))
        means = np.array(means)
        sems = np.array(sems)
        ax.plot(range(8), means, color="#333333", lw=0.8, zorder=2)
        ax.fill_between(range(8), means - sems, means + sems,
                        color="#999999", alpha=0.22, linewidth=0)
        for gi, code in enumerate(GROUPS):
            ax.scatter(gi, means[gi], s=14, color=GROUP_COLOR[code],
                       edgecolor="white", linewidth=0.35, zorder=3)
        ax.set_xlim(-0.4, 7.4)
        ax.set_xticks(range(8))
        if k >= 2:
            ax.set_xticklabels(GROUPS, fontsize=4.8, rotation=45, ha="right",
                               rotation_mode="anchor")
        else:
            ax.set_xticklabels([])
        ax.set_title(f"{name} ({gene})\n{annotation}", fontsize=6.4, pad=3)
        if k % 2 == 0:
            ax.set_ylabel("log$_2$(CPM + 1)", fontsize=6.4)
        ax.tick_params(labelsize=5.8)
        clean(ax)
    # place the panel letter at the upper-left of the representative block
    first_d = fig.axes[-4]
    first_d.text(-0.42, 1.28, "d", transform=first_d.transAxes, fontweight="bold",
                 fontsize=11, va="bottom", ha="left")

    fig.savefig(FIG / "qc_nonadditivity.png", dpi=600)
    fig.savefig(FIG / "qc_nonadditivity.pdf")
    plt.close(fig)
    print("qc_nonadditivity done")


# ---------------------------------------------------------------------------
# background dependence
# ---------------------------------------------------------------------------
def figure3():
    a = pd.read_csv(RES / "tables" / "candidate_audit_all_genes.csv.gz")
    stable = (a["n_supported_stable"] > 0) & (a["lrt_q"] < 0.05)
    s = a[stable].copy()

    fig = plt.figure(figsize=(7.2, 6.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1.05],
                          hspace=0.46, wspace=0.30,
                          left=0.085, right=0.985, top=0.965, bottom=0.075)

    def conditional_panel(ax, xname, yname, letter):
        x = s[xname + "_log2"]
        y = s[yname + "_log2"]
        ok = (s[xname + "_se"] <= 1) & (s[yname + "_se"] <= 1) & \
             (x.abs() <= 10) & (y.abs() <= 10) & x.notna() & y.notna()
        x, y = x[ok], y[ok]
        both = ((s[xname + "_q"] < 0.05) & (s[xname + "_log2"].abs() >= 1) &
                (s[xname + "_se"] <= 1) & (s[yname + "_q"] < 0.05) &
                (s[yname + "_log2"].abs() >= 1) & (s[yname + "_se"] <= 1))[ok]
        same = np.sign(x) == np.sign(y)
        ax.scatter(x[~both], y[~both], s=2.2, color="#B0B0B0", alpha=0.35,
                   linewidths=0, rasterized=True, zorder=1)
        ax.scatter(x[both], y[both], s=3.0, color="#D55E00", alpha=0.55,
                   linewidths=0, rasterized=True, zorder=2)
        lim = 8
        ax.plot([-lim, lim], [-lim, lim], color="#333333", lw=0.8, ls="--", zorder=3)
        slope, intercept = np.polyfit(x, y, 1)
        xx = np.array([-lim, lim])
        ax.plot(xx, slope * xx + intercept, color="#0072B2", lw=1.0, zorder=4)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.axhline(0, color="#DDDDDD", lw=0.6, zorder=0)
        ax.axvline(0, color="#DDDDDD", lw=0.6, zorder=0)
        ax.set_xlabel(f"{CONTRAST_LABEL[xname]} (log$_2$)")
        ax.set_ylabel(f"{CONTRAST_LABEL[yname]} (log$_2$)")
        rho = stats.spearmanr(x, y).statistic
        ax.text(0.03, 0.97, f"n = {len(x):,}\nSpearman ρ = {rho:.2f}\n"
                           f"same sign = {same.mean() * 100:.1f}%\n"
                           f"both supported = {int(both.sum()):,}",
                transform=ax.transAxes, fontsize=6.3, va="top", ha="left")
        handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#D55E00",
                   markeredgewidth=0, markersize=3.2, alpha=0.7),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#B0B0B0",
                   markeredgewidth=0, markersize=3.2, alpha=0.7),
            Line2D([0], [0], color="#333333", lw=0.8, ls="--"),
            Line2D([0], [0], color="#0072B2", lw=1.0),
        ]
        ax.legend(handles, ["Both contrasts supported", "Other comparable genes",
                            "Identity", "OLS fit"], loc="lower right",
                  frameon=False, fontsize=5.4, handlelength=1.2,
                  handletextpad=0.4, borderpad=0.2)
        ax.set_title("Background dependence of the AD interaction"
                     if xname.startswith("AD") else
                     "Background dependence of the DB interaction", fontsize=8.0)
        clean(ax)
        panel_label(ax, letter)

    ax = fig.add_subplot(gs[0, 0])
    conditional_panel(ax, "AD_given_B0", "AD_given_B1", "a")
    ax = fig.add_subplot(gs[0, 1])
    conditional_panel(ax, "DB_given_A0", "DB_given_A1", "b")

    # ---- (c) per-contrast effect distribution -------------------------------
    ax = fig.add_subplot(gs[1, 0])
    data, labels, colors = [], [], []
    for name in CONTRASTS:
        m = ((s[name + "_q"] < 0.05) & (s[name + "_log2"].abs() >= 1) &
             (s[name + "_se"] <= 1))
        vals = s.loc[m, name + "_log2"].clip(-10, 10)
        data.append(vals.to_numpy())
        labels.append(f"{CONTRAST_LABEL[name]}\n{len(vals):,}")
        colors.append("#0072B2" if name.startswith("AD") else
                      "#009E73" if name.startswith("AB") else
                      "#E69F00" if name.startswith("DB") else "#CC79A7")
    parts = ax.violinplot(data, showextrema=False, widths=0.72)
    for body, col in zip(parts["bodies"], colors):
        body.set_facecolor(col)
        body.set_edgecolor(col)
        body.set_alpha(0.30)
        body.set_linewidth(0.6)
    for i, vals in enumerate(data, start=1):
        ax.scatter(i, np.median(vals), s=10, color="white", edgecolor="#111111",
                   linewidth=0.5, zorder=4)
    ax.axhline(0, color="#333333", lw=0.7)
    ax.axhline(1, color="#999999", lw=0.5, ls=":")
    ax.axhline(-1, color="#999999", lw=0.5, ls=":")
    ax.set_xticks(range(1, 8))
    ax.set_xticklabels(labels, fontsize=5.6)
    ax.set_ylabel("Stable interaction effect (log$_2$)")
    ax.set_ylim(-10, 10)
    ax.set_title("Effect sizes among numerically stable support")
    clean(ax)
    panel_label(ax, "c")

    # ---- (d) stable support per contrast ------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    counts = [3172, 1377, 1216, 1095, 530, 2854, 2235]
    cols = ["#0072B2", "#56B4E9", "#009E73", "#66A61E",
            "#E69F00", "#D55E00", "#CC79A7"]
    y = np.arange(len(CONTRASTS))[::-1]
    ax.barh(y, counts, height=0.62, color=cols, alpha=0.85, edgecolor="white")
    for yy, n in zip(y, counts):
        ax.text(n + 60, yy, f"{n:,}", va="center", fontsize=6.4)
    ax.set_yticks(y)
    ax.set_yticklabels([CONTRAST_LABEL[n] for n in CONTRASTS], fontsize=6.6)
    ax.set_xlabel("Genes with stable support (n)")
    ax.set_xlim(0, 3700)
    ax.set_title("Stable support by genetic background")
    clean(ax)
    panel_label(ax, "d")

    fig.savefig(FIG / "background_dependence.png", dpi=600)
    fig.savefig(FIG / "background_dependence.pdf")
    plt.close(fig)
    print("background_dependence done")


# ---------------------------------------------------------------------------
# stage and same-study support
# ---------------------------------------------------------------------------
def figure4():
    stage_sum = json.loads((RES / "stages" / "stage_summary.json").read_text())
    stage_tab = pd.read_csv(RES / "stages" / "stage_ad_all.csv.gz")
    frozen = json.loads((RES / "stages" / "frozen_signature.json").read_text())
    sig = frozen["genes"]
    dry = pd.read_csv(RES / "tables" / "contrasts_all_genes.csv.gz").set_index("gene_id")
    internal = json.loads((RES / "internal" / "internal_summary.json").read_text())
    stages = ["20d", "22d", "24d"]

    fig = plt.figure(figsize=(7.2, 6.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.1], height_ratios=[1, 1],
                          hspace=0.48, wspace=0.30,
                          left=0.09, right=0.985, top=0.96, bottom=0.075)

    # ---- (a) concordance and replication -----------------------------------
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(3)
    conc = [stage_sum["stages"][s]["signature_sign_concordance"] * 100 for s in stages]
    rep = [stage_sum["stages"][s]["signature_rep_rate_q05_same_sign"] * 100 for s in stages]
    ax.bar(x - 0.19, conc, width=0.34, color="#0072B2", alpha=0.9, label="Same direction")
    ax.bar(x + 0.19, rep, width=0.34, color="#E69F00", alpha=0.9,
           label="Same direction, q < 0.05")
    for i in range(3):
        ax.text(x[i] - 0.19, conc[i] + 1.5, f"{conc[i]:.1f}", ha="center",
                fontsize=6.2)
        ax.text(x[i] + 0.19, rep[i] + 1.5, f"{rep[i]:.1f}", ha="center",
                fontsize=6.2)
    ax.axhline(50, color="#777777", lw=0.7, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n(n={stage_sum['stages'][s]['signature_tested']:,})"
                        for s in stages], fontsize=6.6)
    ax.set_ylim(0, 122)
    ax.set_ylabel("Signature genes (%)")
    ax.set_title("Dry-seed AD signature across maturation")
    ax.legend(frameon=False, loc="upper left", fontsize=5.8, ncol=1,
              handlelength=1.0, borderpad=0.1)
    clean(ax)
    panel_label(ax, "a")

    # ---- (b) effect concordance -------------------------------------------
    gs_b = gs[0, 1].subgridspec(1, 3, wspace=0.18)
    stage_tab = stage_tab.copy()
    for j, stage in enumerate(stages):
        ax = fig.add_subplot(gs_b[j])
        t = stage_tab[stage_tab["stage"] == stage].set_index("gene_id")
        common = t.index.intersection(dry.index)
        x = dry.loc[common, "AD_given_B0_log2"]
        y = t.loc[common, "ad_log2"]
        ok = x.notna() & y.notna() & (x.abs() <= 6) & (y.abs() <= 6)
        hb = ax.hexbin(x[ok], y[ok], gridsize=38, bins="log", cmap="Greys",
                       mincnt=1, linewidths=0)
        sig_common = common.intersection(sig)
        ax.scatter(dry.loc[sig_common, "AD_given_B0_log2"].clip(-6, 6),
                   t.loc[sig_common, "ad_log2"].clip(-6, 6), s=2.2,
                   color="#D55E00", alpha=0.55, linewidths=0, rasterized=True)
        ax.plot([-6, 6], [-6, 6], color="#333333", lw=0.8, ls="--")
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_xticks([-6, -3, 0, 3, 6])
        ax.set_yticks([-6, -3, 0, 3, 6])
        rho = stage_sum["stages"][stage]["all_gene_effect_spearman_vs_dry"]
        ax.set_title(f"{stage}\nρ = {rho:.2f}", fontsize=7.4)
        if j == 0:
            ax.set_ylabel("stage AD (log$_2$)")
        ax.set_xlabel("dry AD|B=0", fontsize=6.4)
        ax.tick_params(labelsize=5.8)
        clean(ax)
        if j == 0:
            b_ax = ax
    b_ax.text(-0.58, 1.22, "b", transform=b_ax.transAxes, fontweight="bold",
              fontsize=11, va="bottom", ha="left")

    # ---- (c) stage signature mean ------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    means, los, his = [], [], []
    for stage in stages:
        t = stage_tab[stage_tab["stage"] == stage].set_index("gene_id")
        common = t.index.intersection(sig)
        vals = (t.loc[common, "ad_log2"] *
                np.sign(dry.loc[common, "AD_given_B0_log2"])).dropna()
        n = len(vals)
        mean = vals.mean()
        ci = stats.t.ppf(0.975, n - 1) * vals.std(ddof=1) / np.sqrt(n)
        means.append(mean)
        los.append(mean - ci)
        his.append(mean + ci)
    y = np.arange(3)
    ax.errorbar(means, y, xerr=[np.array(means) - los, np.array(his) - means],
                fmt="o", color="#0072B2", ecolor="#0072B2", elinewidth=1.0,
                capsize=2.5, markersize=4)
    for i, m in enumerate(means):
        ax.annotate(f"{m:.2f}", (los[i], y[i]), xytext=(-4, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=6.3, color="#0072B2")
    ax.axvline(0, color="#333333", lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(stages)
    ax.invert_yaxis()
    ax.set_xlim(-0.2, 2.5)
    ax.set_xlabel("Mean AD effect aligned to dry-seed direction (log$_2$)")
    ax.set_title("Signature effect remains directional")
    clean(ax)
    panel_label(ax, "c")

    # ---- (d) same-study genetic evidence -----------------------------------
    ax = fig.add_subplot(gs[1, 1])
    cyp = internal["cyp_combination"]
    abi = internal["abi3_6"]
    dog = internal["dog1_cvi"]
    labels = ["CYP707A2\ncloser to WT", "abi3-6\nsame sign", "DOG1-Cvi\nopposite to dog1-4"]
    vals = [cyp["frac_closer_to_WT_expressed"] * 100,
            abi["expressed_sign_concordance"] * 100,
            (1 - dog["expressed_sign_concordance"]) * 100]
    ns = [cyp["signature_expressed_WT_AD"], abi["set_expressed_both"],
          dog["set_expressed_both"]]
    cols = ["#009E73", "#0072B2", "#CC79A7"]
    xpos = np.arange(3)
    ax.bar(xpos, vals, width=0.58, color=cols, alpha=0.85, edgecolor="white")
    for i, (v, n) in enumerate(zip(vals, ns)):
        ax.text(i, v + 2.0, f"{v:.1f}%\n(n={n:,})", ha="center", fontsize=6.2)
    ax.axhline(50, color="#777777", lw=0.7, ls=":")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=6.4)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Expected-direction concordance (%)")
    ax.set_title("Same-study genetic support")
    ax.text(0.5, -0.34, "Same study; not independent. No rescue claim.",
            transform=ax.transAxes, ha="center", fontsize=6.0, color="#555555")
    clean(ax)
    panel_label(ax, "d")

    fig.savefig(FIG / "stage_support.png", dpi=600)
    fig.savefig(FIG / "stage_support.pdf")
    plt.close(fig)
    print("stage_support done")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "1"):
        figure1()
    if which in ("all", "2"):
        figure2()
    if which in ("all", "3"):
        figure3()
    if which in ("all", "4"):
        figure4()
