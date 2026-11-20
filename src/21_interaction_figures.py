"""Figures for the trait, epistasis and functional-annotation analyses.

  trait_interactions      dry-seed traits of the eight factorial genotypes
  epistasis_landscape     transcriptome-wide epistasis coefficients and types
  functional_annotation   GO, ABI3 targets, promoter elements, gene panel
  working_model           schematic summary of the inferred genetic architecture

Every panel is drawn from the tables written by scripts 17-20.  An optional
argument (traits, epistasis, function, model) draws a single figure.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)

ORDER = ["Col-0", "abi3-12", "dog1-4", "abi3-12 dog1-4",
         "aba2-1", "abi3-12 aba2-1", "dog1-4 aba2-1", "abi3-12 dog1-4 aba2-1"]
COLOR = {
    "Col-0": "#333333", "aba2-1": "#E69F00", "dog1-4": "#56B4E9",
    "dog1-4 aba2-1": "#0072B2", "abi3-12": "#009E73", "abi3-12 aba2-1": "#66A61E",
    "abi3-12 dog1-4": "#D55E00", "abi3-12 dog1-4 aba2-1": "#CC79A7",
    "abi3-12 dog1-4 cyp707a2": "#8C564B",
}
FACTORIAL_CODE = {
    "Col-0": (0, 0, 0), "abi3-12": (1, 0, 0), "dog1-4": (0, 1, 0), "abi3-12 dog1-4": (1, 1, 0),
    "aba2-1": (0, 0, 1), "abi3-12 aba2-1": (1, 0, 1), "dog1-4 aba2-1": (0, 1, 1),
    "abi3-12 dog1-4 aba2-1": (1, 1, 1),
}
PAIR_ORDER = ["AD_given_B0", "AD_given_B1", "DB_given_A0", "DB_given_A1",
              "AB_given_D0", "AB_given_D1"]
PAIR_LABEL = {
    "AD_given_B0": "abi3 × dog1\nABA2", "AD_given_B1": "abi3 × dog1\naba2-1",
    "DB_given_A0": "dog1 × aba2\nABI3", "DB_given_A1": "dog1 × aba2\nabi3-12",
    "AB_given_D0": "abi3 × aba2\nDOG1", "AB_given_D1": "abi3 × aba2\ndog1-4",
}
TYPE_COLOR = {
    "synthetic": "#7B3294", "synergistic": "#C2A5CF",
    "masking by first": "#1B7837", "masking by second": "#A6DBA0",
    "pathway-like": "#4D4D4D", "suppression": "#2166AC", "inversion": "#F4A582",
    "quantitative": "#D9D9D9", "additive": "#FFFFFF",
    "not detected": "#FFFFFF", "opposing interaction": "#A6DBA0",
}

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


def panel_label(ax, letter, x=-0.12, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontweight="bold",
            fontsize=11, va="bottom", ha="left")


def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)


def stars(p):
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def genotype_axis(ax, y0=-0.07, dy=0.075):
    """Factorial design matrix under the x axis instead of crowded labels."""
    ax.set_xticks(range(8))
    ax.set_xticklabels([])
    ax.set_xlim(-0.6, 7.6)
    ax.axvline(3.5, color="#999999", lw=0.6, ls=":")
    tr = ax.get_xaxis_transform()
    for r, (allele, idx) in enumerate((("abi3-12", 0), ("dog1-4", 1), ("aba2-1", 2))):
        y = y0 - r * dy
        ax.text(-0.8, y, allele, transform=tr, ha="right", va="center", fontsize=6,
                style="italic", clip_on=False)
        for i, g in enumerate(ORDER):
            code = FACTORIAL_CODE[g][idx]
            ax.text(i, y, "\u25CF" if code else "\u25CB", transform=tr, ha="center",
                    va="center", fontsize=6.5, color="#222222" if code else "#9A9A9A",
                    clip_on=False)
    ax.text(0.25, 0.985, "ABA2", transform=ax.transAxes, ha="center", va="top",
            fontsize=6.3, color="#555555", style="italic")
    ax.text(0.75, 0.985, "aba2-1", transform=ax.transAxes, ha="center", va="top",
            fontsize=6.3, color="#555555", style="italic")


def strip(ax, data, col, transform=None):
    rng = np.random.default_rng(3)
    for i, g in enumerate(ORDER):
        v = data.loc[data["genotype"] == g, col].to_numpy(dtype=float)
        if transform is not None:
            v = transform(v)
        ax.bar(i, v.mean(), width=0.62, color=COLOR[g], alpha=0.35, edgecolor=COLOR[g], lw=0.8)
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=9, color=COLOR[g],
                   edgecolor="white", linewidth=0.4, zorder=3)


def genotype_legend(fig, y=0.995, extra=True):
    names = ORDER + (["abi3-12 dog1-4 cyp707a2"] if extra else [])
    handles = [Line2D([0], [0], color=COLOR[g], lw=1.6, marker="o", ms=3,
                      ls="--" if "cyp707a2" in g else "-") for g in names]
    leg = fig.legend(handles, names, loc="upper center", bbox_to_anchor=(0.5, y),
                     ncol=5, frameon=False, fontsize=6.2, handlelength=2.0,
                     columnspacing=1.2)
    for t in leg.get_texts():
        t.set_fontstyle("italic" if t.get_text() != "Col-0" else "normal")


def trait_interactions():
    rep = pd.read_csv(RES / "traits" / "dry_seed_traits_replicates.csv")
    fits = pd.read_csv(RES / "traits" / "eppo_replicate_fits.csv")
    curves = pd.read_csv(RES / "traits" / "eppo_germination_curves.csv")
    pairs = pd.read_csv(RES / "traits" / "trait_pairwise_epistasis.csv")

    fig = plt.figure(figsize=(7.2, 6.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.08], hspace=0.78, wspace=0.5,
                          top=0.90, bottom=0.08, left=0.08, right=0.98)
    genotype_legend(fig)

    ax = fig.add_subplot(gs[0, 0])
    for g in ORDER + ["abi3-12 dog1-4 cyp707a2"]:
        sub = curves[curves["genotype"] == g].sort_values("days")
        ls = "--" if g == "abi3-12 dog1-4 cyp707a2" else "-"
        ax.errorbar(sub["days"], sub["mean"], yerr=sub["std"], color=COLOR[g], lw=1.1,
                    ls=ls, marker="o", ms=2.4, capsize=1.2, elinewidth=0.5)
    ax.axhline(50, color="#999999", lw=0.6, ls=":")
    ax.set_xlabel("EPPO storage (days)")
    ax.set_ylabel("Germination (%)")
    ax.set_title("Longevity under EPPO ageing", pad=8)
    ax.set_ylim(-3, 105)
    clean(ax)
    panel_label(ax, "a", x=-0.3, y=1.06)

    specs = [
        (gs[0, 1], fits, "germ_auc_days", None, "Integrated germination (days)",
         "Area under the germination curve\n(0-160 EPPO days)", "b", None),
        (gs[0, 2], rep, "chlorophyll_fluorescence", lambda v: v / 1000.0,
         "Chlorophyll fluorescence (×10³)", "Chlorophyll retained\nin dry seeds", "c", None),
        (gs[1, 0], rep, "dormancy_pct", None, "Dormancy (%)", "Primary dormancy\nof dry seeds", "d", (0, 112)),
    ]
    for spec, data, col, tf, ylab, title, letter, ylim in specs:
        ax = fig.add_subplot(spec)
        strip(ax, data, col, transform=tf)
        ax.set_ylabel(ylab)
        ax.set_title(title, pad=8)
        if ylim:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
        genotype_axis(ax)
        clean(ax)
        panel_label(ax, letter, x=-0.3, y=1.06)

    ax = fig.add_subplot(gs[1, 1:])
    traits = [("dormancy (%)", "Dormancy", "pp"),
              ("chlorophyll fluorescence (log2)", "Chlorophyll", "log2"),
              ("longevity AUC (log2 days+1)", "Ageing (AUC)", "log2")]
    nice = {"synthetic": "synthetic", "synergistic": "synergistic",
            "pathway-like": "pathway-like", "additive": "additive",
            "quantitative": "quantitative", "suppression": "suppression"}
    for r, (trait, label, unit) in enumerate(traits):
        for c, pair in enumerate(PAIR_ORDER):
            row = pairs[(pairs["trait"] == trait) & (pairs["pair"] == pair)].iloc[0]
            cls = row["class"]
            cls = "synthetic" if cls == "emergent" else cls
            face = TYPE_COLOR.get(cls, "#FFFFFF")
            ax.add_patch(plt.Rectangle((c, r), 0.96, 0.92, facecolor=face,
                                       edgecolor="#777777", lw=0.5))
            masking = cls.startswith("masking")
            if masking:
                who = row["first"] if cls.endswith("first") else row["second"]
                txt = f"{who}\nepistatic"
            else:
                txt = {"opposing interaction": "opposing\ninteraction", "not detected": "not\ndetected"}.get(cls, nice.get(cls, cls))
            val = row["interaction"]
            vtxt = f"{val:+.0f} pp" if unit == "pp" else f"{val:+.2f}"
            dark = cls in ("synthetic", "masking by first", "pathway-like", "suppression")
            ax.text(c + 0.48, r + 0.33, txt, ha="center", va="center", fontsize=5.8,
                    color="white" if dark else "black", style="italic" if masking else "normal")
            ax.text(c + 0.48, r + 0.70, f"{vtxt} {stars(row['interaction_q'])}",
                    ha="center", va="center", fontsize=5.6, color="white" if dark else "#333333")
    ax.set_xlim(-0.05, len(PAIR_ORDER))
    ax.set_ylim(len(traits), -0.05)
    ax.set_xticks(np.arange(len(PAIR_ORDER)) + 0.48)
    ax.set_xticklabels([PAIR_LABEL[p] for p in PAIR_ORDER], fontsize=6.2, style="italic")
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(traits)) + 0.46)
    ax.set_yticklabels([t[1] for t in traits])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Interaction direction by background (heteroscedastic contrasts; BH q)",
                 fontsize=7.5, pad=30)
    panel_label(ax, "e", x=-0.17, y=1.2)
    save(fig, "trait_interactions")


MODEL_STYLE = {
    "additive": ("#777777", "--", "additive"),
    "masking by first": ("#1B7837", "-.", "first mutant epistatic"),
    "masking by second": ("#5AAE61", ":", "second mutant epistatic"),
    "suppression": ("#2166AC", (0, (1, 1)), "suppression"),
}
TYPE_ORDER = ["synthetic", "synergistic", "masking by first", "masking by second",
              "pathway-like", "suppression", "inversion", "quantitative"]
TYPE_NICE = {"masking by first": "first mutant epistatic",
             "masking by second": "second mutant epistatic"}
PAIR_SHORT = {
    "AD_given_B0": "abi3 × dog1 | ABA2", "AD_given_B1": "abi3 × dog1 | aba2-1",
    "DB_given_A0": "dog1 × aba2 | ABI3", "DB_given_A1": "dog1 × aba2 | abi3-12",
    "AB_given_D0": "abi3 × aba2 | DOG1", "AB_given_D1": "abi3 × aba2 | dog1-4",
}


def epistasis_landscape():
    epi = pd.read_csv(RES / "epistasis" / "epistasis_gene_level.csv.gz")
    coefs = pd.read_csv(RES / "epistasis" / "epistasis_coefficients.csv").set_index("pair")
    counts = pd.read_csv(RES / "epistasis" / "interaction_type_counts.csv")

    fig = plt.figure(figsize=(7.2, 6.6))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.12], hspace=0.55,
                             top=0.94, bottom=0.11, left=0.08, right=0.985)
    top = outer[0].subgridspec(1, 4, wspace=0.42)
    bottom = outer[1].subgridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.62)
    shown = ["AD_given_B0", "AD_given_B1", "DB_given_A0", "DB_given_A1"]
    titles = {"AD_given_B0": "abi3-12 × dog1-4\n(ABA2)", "AD_given_B1": "abi3-12 × dog1-4\n(aba2-1)",
              "DB_given_A0": "dog1-4 × aba2-1\n(ABI3)", "DB_given_A1": "dog1-4 × aba2-1\n(abi3-12)"}
    lim = 9.0
    for k, pair in enumerate(shown):
        ax = fig.add_subplot(top[0, k])
        sub = epi[(epi["pair"] == pair) & (epi["shared_responder"])]
        x, y = sub["x"].clip(-lim, lim), sub["delta"].clip(-lim, lim)
        ax.hexbin(x, y, gridsize=34, extent=(-lim, lim, -lim, lim), cmap="Greys",
                  mincnt=1, bins="log", linewidths=0.1)
        xs = np.linspace(-lim, lim, 2)
        c = coefs.loc[pair]
        for m, (col, ls, _) in MODEL_STYLE.items():
            ax.plot(xs, c[f"sim_{m.replace(' ', '_')}_mean"] * xs, color=col, ls=ls, lw=0.9)
        ax.plot(xs, c["s_odr"] * xs, color="#D55E00", lw=1.6)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.axhline(0, color="#BBBBBB", lw=0.4, zorder=0)
        ax.axvline(0, color="#BBBBBB", lw=0.4, zorder=0)
        ax.set_title(titles[pair], fontsize=7.2, style="italic", pad=4)
        ax.text(0.04, 0.04, f"s = {c['s_odr']:.2f}\n[{c['s_ci_low']:.2f}, {c['s_ci_high']:.2f}]\n"
                f"n = {int(c['n_shared_responders']):,}", transform=ax.transAxes,
                fontsize=6, va="bottom", color="#D55E00")
        ax.set_xlabel("Additive expectation\n$\\delta_P + \\delta_Q$ (log2)", fontsize=6.6)
        if k == 0:
            ax.set_ylabel("Deviation from additivity\n$\\delta_{PQ} - \\delta_P - \\delta_Q$ (log2)",
                          fontsize=6.6)
        clean(ax)
        panel_label(ax, "abcd"[k], x=-0.3, y=1.1)
    handles = [Line2D([0], [0], color="#D55E00", lw=1.6, label="observed slope s")]
    handles += [Line2D([0], [0], color=col, ls=ls, lw=0.9, label=f"simulated: {lab}")
                for m, (col, ls, lab) in MODEL_STYLE.items()]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.53, 0.525), ncol=5,
               frameon=False, fontsize=5.9, handlelength=2.2, columnspacing=1.1)

    ax = fig.add_subplot(bottom[0, 0])
    order = ["AD_given_B0", "AD_given_B1", "DB_given_A0", "DB_given_A1", "AB_given_D0", "AB_given_D1"]
    offs = {"additive": -0.24, "masking by first": -0.12, "masking by second": 0.0, "suppression": 0.12}
    for i, pair in enumerate(order):
        c = coefs.loc[pair]
        for m, (col, ls, lab) in MODEL_STYLE.items():
            mu = c[f"sim_{m.replace(' ', '_')}_mean"]
            sd = c[f"sim_{m.replace(' ', '_')}_sd"]
            ax.errorbar(i + offs[m], mu, yerr=1.96 * sd, fmt="s", ms=3, color=col, mfc="white",
                        mew=0.9, elinewidth=0.7, capsize=0)
        ax.errorbar(i + 0.26, c["s_odr"], yerr=[[c["s_odr"] - c["s_ci_low"]],
                                                [c["s_ci_high"] - c["s_odr"]]],
                    fmt="o", ms=4.5, color="#D55E00", elinewidth=1.0, capsize=1.5)
    ax.axhline(0, color="#BBBBBB", lw=0.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([PAIR_SHORT[p] for p in order], fontsize=6.0, style="italic",
                       rotation=32, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.5, len(order) - 0.4)
    ax.set_ylim(-1.08, 0.12)
    ax.set_ylabel("Epistasis coefficient s")
    ax.set_title("Observed s versus simulated models", pad=6, fontsize=7.6)
    clean(ax)
    panel_label(ax, "e", x=-0.2, y=1.06)

    ax = fig.add_subplot(bottom[0, 1])
    piv = counts.pivot_table(index="pair", columns="class", values="n", fill_value=0).reindex(order)
    totals = piv.sum(axis=1).to_numpy(dtype=float)
    left = np.zeros(len(order))
    for t in TYPE_ORDER:
        if t not in piv.columns:
            continue
        frac = piv[t].to_numpy(dtype=float) / totals
        ax.barh(range(len(order)), frac, left=left, color=TYPE_COLOR[t], edgecolor="white",
                lw=0.4, label=TYPE_NICE.get(t, t), height=0.72)
        left += frac
    for i, n in enumerate(totals):
        ax.text(1.02, i, f"{int(n):,}", va="center", ha="left", fontsize=6)
    ax.text(1.02, -0.75, "genes", va="center", ha="left", fontsize=6, color="#555555")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([PAIR_SHORT[p] for p in order], fontsize=6, style="italic")
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of genes with stable interaction support")
    ax.set_title("Gene-level interaction types", pad=6, fontsize=7.6)
    ax.legend(frameon=False, fontsize=5.7, ncol=4, loc="upper center",
              bbox_to_anchor=(0.35, -0.2), handlelength=1.0, columnspacing=0.7)
    clean(ax)
    panel_label(ax, "f", x=-0.62, y=1.04)
    save(fig, "epistasis_landscape")


FUNC_SETS = [
    ("AD_given_B0 | synthetic | double up", "abi3 × dog1 | ABA2: synthetic, up"),
    ("AD_given_B0 | synergistic | double up", "abi3 × dog1 | ABA2: synergistic, up"),
    ("AD_given_B0 | synergistic | double down", "abi3 × dog1 | ABA2: synergistic, down"),
    ("AD_given_B1 | synergistic | double down", "abi3 × dog1 | aba2-1: synergistic, down"),
    ("AB_given_D0 | synergistic | double down", "abi3 × aba2 | DOG1: synergistic, down"),
    ("DB_given_A1 | masking by first | double down", "dog1 × aba2 | abi3-12: dog1 epistatic, down"),
    ("DB_given_A1 | pathway-like | double up", "dog1 × aba2 | abi3-12: pathway-like, up"),
    ("DOG1xABA2 in both ABI3 backgrounds", "dog1 × aba2 in both ABI3 backgrounds"),
    ("single abi3-12 down", "control: down in abi3-12"),
    ("single dog1-4 down", "control: down in dog1-4"),
]
FUNC_TERMS = ["photosynthesis", "photosynthesis, light harvesting", "chlorophyll biosynthetic process",
              "seed oilbody biogenesis", "lipid storage", "response to heat", "seed maturation",
              "seed dormancy process", "response to hypoxia", "DNA replication initiation",
              "plant-type cell wall loosening", "response to gibberellin"]
PANEL_GENES = [
    ("degreening", ["SGR1", "SGR2", "NYC1", "PPH"]),
    ("photosynthesis", ["CAB1"]),
    ("protection", ["HSFA9", "HSP17.4", "PER1", "EM6", "RAB18"]),
    ("storage", ["CRA1", "CRU3", "OLE1", "SESA1"]),
    ("regulators", ["FUS3", "ABI5", "DOG1", "AHG1", "MFT", "ABA2", "CYP707A1", "CYP707A2", "RGL2"]),
]
PANEL_COLS = [(4, "abi3-12"), (2, "dog1-4"), (6, "abi3 dog1"), (1, "aba2-1"),
              (5, "abi3 aba2"), (3, "dog1 aba2"), (7, "abi3 dog1 aba2")]


def functional_annotation():
    go = pd.read_csv(RES / "function" / "go_bp_enrichment_all.csv.gz")
    ov = pd.read_csv(RES / "function" / "abi3_target_overlap.csv").set_index("set")
    prom = pd.read_csv(RES / "promoters" / "promoter_element_enrichment.csv")
    panel = pd.read_csv(RES / "function" / "literature_panel.csv").set_index("gene")
    gm = pd.read_csv(RES / "tables" / "group_means_normalized.csv.gz").set_index("gene_id")
    keys = [k for k, _ in FUNC_SETS]
    labels = [lab for _, lab in FUNC_SETS]
    cmap = LinearSegmentedColormap.from_list("rb", ["#2166AC", "#F7F7F7", "#B2182B"])

    fig = plt.figure(figsize=(7.2, 7.6))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.25, 0.8], hspace=0.62,
                             top=0.85, bottom=0.07, left=0.27, right=0.95)
    top = outer[0].subgridspec(1, 3, width_ratios=[12, 3.2, 5], wspace=0.08)

    # (a) GO dot plot: rows = gene sets, columns = terms
    ax = fig.add_subplot(top[0, 0])
    sub = go[go["set"].isin(keys) & go["name"].isin(FUNC_TERMS)]
    for i, key in enumerate(keys):
        for j, term in enumerate(FUNC_TERMS):
            r = sub[(sub["set"] == key) & (sub["name"] == term)]
            if not len(r):
                continue
            r = r.iloc[0]
            if r["qvalue"] >= 0.05:
                ax.scatter(j, i, s=5, color="#DDDDDD", zorder=1)
                continue
            ax.scatter(j, i, s=10 + 7 * min(r["k"], 30), c=[min(-np.log10(r["qvalue"]), 10)],
                       cmap="viridis_r", vmin=1.3, vmax=10, edgecolor="#333333", lw=0.3, zorder=2)
    ax.set_xticks(range(len(FUNC_TERMS)))
    ax.set_xticklabels(FUNC_TERMS, rotation=50, ha="left", rotation_mode="anchor", fontsize=6.0)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(labels, fontsize=6.1, style="italic")
    ax.set_xlim(-0.6, len(FUNC_TERMS) - 0.4)
    ax.set_ylim(len(keys) - 0.5, -0.5)
    ax.grid(color="#F0F0F0", lw=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("GO biological process (q < 0.05 coloured)", fontsize=6.6)
    sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=plt.Normalize(1.3, 10))
    cax = ax.inset_axes([0.02, -0.2, 0.32, 0.035])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("\u2212log10 q", fontsize=5.8, labelpad=1)
    cb.ax.tick_params(labelsize=5.3, length=2)
    for kk, lab in ((5, "5"), (15, "15"), (30, "\u226530 genes")):
        ax.scatter([], [], s=10 + 7 * kk, color="#BBBBBB", edgecolor="#333333", lw=0.3, label=lab)
    ax.legend(fontsize=5.6, frameon=False, loc="upper left", bbox_to_anchor=(0.42, -0.13),
              ncol=3, columnspacing=1.2, handletextpad=0.3)
    panel_label(ax, "a", x=-0.68, y=1.36)

    # (b) ABI3 target overlap, same rows
    ax = fig.add_subplot(top[0, 1])
    vals = np.array([ov.loc[k, "odds_ratio"] if k in ov.index else np.nan for k in keys])
    qv = np.array([ov.loc[k, "qvalue"] if k in ov.index else np.nan for k in keys])
    ax.barh(range(len(keys)), np.log2(vals), color=np.where(qv < 0.05, "#1B7837", "#BBBBBB"),
            height=0.62)
    ax.axvline(0, color="#777777", lw=0.6)
    for i, (v, q) in enumerate(zip(vals, qv)):
        if np.isfinite(v):
            ax.text(max(np.log2(v), 0) + 0.1, i, stars(q), va="center", fontsize=5.3)
    ax.set_ylim(len(keys) - 0.5, -0.5)
    ax.set_yticks([])
    ax.set_xlim(-0.3, 3.6)
    ax.set_xlabel("log2 odds ratio", fontsize=6.4)
    ax.set_title("ABI3 targets\n(Tian et al. 2020)", fontsize=6.8, pad=4)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    panel_label(ax, "b", x=-0.12, y=1.13)

    # (c) promoter elements, same rows
    ax = fig.add_subplot(top[0, 2])
    elements = ["RY", "ABRE", "G-box", "RY+ABRE", "DRE"]
    p1 = prom[prom["window"] == 1000]
    mat = np.full((len(keys), len(elements)), np.nan)
    qmat = np.full_like(mat, np.nan)
    for i, k in enumerate(keys):
        for j, e in enumerate(elements):
            r = p1[(p1["set"] == k) & (p1["element"] == e)]
            if len(r):
                mat[i, j] = np.log2(r["mh_odds_ratio"].iloc[0])
                qmat[i, j] = r["mh_q"].iloc[0]
    im = ax.imshow(np.clip(mat, -2, 2), cmap=cmap, vmin=-2, vmax=2, aspect="auto")
    for i in range(len(keys)):
        for j in range(len(elements)):
            if np.isfinite(qmat[i, j]) and qmat[i, j] < 0.05:
                ax.text(j, i, stars(qmat[i, j]), ha="center", va="center", fontsize=5.3)
    ax.set_xticks(range(len(elements)))
    ax.set_xticklabels(elements, rotation=50, ha="right", rotation_mode="anchor", fontsize=6)
    ax.set_yticks([])
    ax.set_title("Promoter elements\n(1 kb; MH odds ratio)", fontsize=6.8, pad=4)
    cax = ax.inset_axes([1.05, 0.15, 0.07, 0.7])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("log2 OR", fontsize=5.8, labelpad=1)
    cb.ax.tick_params(labelsize=5.3, length=2)
    panel_label(ax, "c", x=-0.08, y=1.13)

    # (d) literature panel: rows = mutant genotypes, columns = genes
    ax = fig.add_subplot(outer[1])
    cols, groups = [], []
    for grp, genes in PANEL_GENES:
        for g in genes:
            if g in panel.index and bool(panel.loc[g, "tested"]):
                cols.append(g)
                groups.append(grp)
    mat = []
    for c, _ in PANEL_COLS:
        row = []
        for g in cols:
            m = gm.loc[panel.loc[g, "gene_id"]]
            row.append(np.log2(m[f"g{c}"] + 1.0) - np.log2(m["g0"] + 1.0))
        mat.append(row)
    mat = np.array(mat)
    im = ax.imshow(np.clip(mat, -6, 6), cmap=cmap, vmin=-6, vmax=6, aspect="auto")
    ax.set_yticks(range(len(PANEL_COLS)))
    ax.set_yticklabels([lab for _, lab in PANEL_COLS], fontsize=6.2, style="italic")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=60, ha="right", rotation_mode="anchor", fontsize=6.1,
                       style="italic")
    for j, g in enumerate(cols):
        t = panel.loc[g, "AD_given_B0_type"]
        t = t if isinstance(t, str) else ""
        face = TYPE_COLOR.get(t, "#FFFFFF") if t else "#FFFFFF"
        ax.add_patch(plt.Rectangle((j - 0.45, -1.35), 0.9, 0.6, facecolor=face,
                                   edgecolor="#999999", lw=0.3, clip_on=False))
    ax.text(-0.7, -1.05, "abi3 \u00d7 dog1 type (ABA2)", ha="right", va="center", fontsize=5.8,
            style="italic", clip_on=False)
    prev, start = None, 0
    for j, grp in enumerate(groups + [None]):
        if grp != prev:
            if prev is not None:
                ax.text((start + j - 1) / 2, -1.75, prev, ha="center", va="bottom", fontsize=6.2,
                        color="#333333", clip_on=False)
                if j < len(groups):
                    ax.axvline(j - 0.5, color="white", lw=1.6)
            prev, start = grp, j
    ax.set_ylim(len(PANEL_COLS) - 0.5, -0.5)
    cax = ax.inset_axes([1.012, 0.05, 0.012, 0.9])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("log2 change vs Col-0", fontsize=5.8)
    cb.ax.tick_params(labelsize=5.3, length=2)
    handles = [Patch(facecolor=TYPE_COLOR[t], edgecolor="#999999", lw=0.3, label=TYPE_NICE.get(t, t))
               for t in ("synthetic", "synergistic", "suppression", "inversion", "masking by second")]
    ax.legend(handles=handles, frameon=False, fontsize=5.6, ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, -0.42), handlelength=1.0, columnspacing=1.0)
    panel_label(ax, "d", x=-0.3, y=1.38)
    save(fig, "functional_annotation")


def working_model():
    """Schematic summary of the genetic architecture inferred from the data."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    coefs = pd.read_csv(RES / "epistasis" / "epistasis_coefficients.csv").set_index("pair")
    counts = pd.read_csv(RES / "epistasis" / "interaction_type_counts.csv")
    frac = counts.pivot_table(index="pair", columns="class", values="fraction", fill_value=0)
    aggr = lambda p: frac.loc[p, ["synthetic", "synergistic"]].sum()
    allev = lambda p: frac.loc[p, ["pathway-like", "masking by first", "masking by second",
                                    "suppression"]].sum()

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 66)
    ax.axis("off")

    def node(x, y, w, h, text, edge, face, size=7.0, style="normal", weight="bold"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    facecolor=face, edgecolor=edge, linewidth=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
                fontweight=weight, style=style, linespacing=1.3)

    def arrow(p0, p1, color="#555555", ls="-", lw=1.2, rad=0.0):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9, lw=lw,
                                     color=color, linestyle=ls,
                                     connectionstyle=f"arc3,rad={rad}"))

    node(8, 50, 26, 11, "ABI3 branch\nB3 factor; RY elements", "#009E73", "#E9F5EF", size=7.0)
    ax.add_patch(FancyBboxPatch((58, 50), 36, 11, boxstyle="round,pad=0.4,rounding_size=1.2",
                                facecolor="#F4F8FC", edgecolor="#0072B2", linewidth=1.0))
    ax.text(76, 59.2, "Candidate DOG1\u2013ABA response branch", ha="center", va="center", fontsize=7.0,
            fontweight="bold")
    ax.text(66, 53.2, "DOG1", ha="center", va="center", fontsize=7, style="italic")
    ax.text(87, 53.2, "ABA (ABA2)\nbZIPs; ABRE", ha="center", va="center", fontsize=6.4)
    ax.plot([70, 80.5], [53.2, 53.2], color="#0072B2", lw=1.3, linestyle="--")

    node(2, 18, 29, 14, "Degreening and plastid shutdown\nSGR1, SGR2, NYC1, PPH; CAB1\n"
         "trait: chlorophyll retention", "#7B3294", "#F3EAF7", size=6.3, weight="normal")
    node(35.5, 18, 29, 14, "Protection and storage\nHSFA9, HSP17.4, PER1, EM6, OLE1\n"
         "RY + ABRE promoters; ABI3 targets\ntrait: longevity", "#7B3294", "#F3EAF7",
         size=6.3, weight="normal")
    node(69, 18, 29, 14, "Dormancy and germination\nDOG1, RGL2; GA and\ncell-wall loosening genes\n"
         "trait: primary dormancy", "#1B7837", "#EAF5EC", size=6.3, weight="normal")

    arrow((17, 49.4), (16.5, 32.6), color="#009E73")
    arrow((26, 49.4), (44, 32.6), color="#009E73")
    arrow((64, 49.4), (55, 32.6), color="#0072B2")
    arrow((84, 49.4), (84, 32.6), color="#0072B2", lw=1.8)
    arrow((60, 49.4), (25, 32.6), color="#0072B2", ls="--", lw=0.9, rad=0.08)

    ax.text(28, 12.5, "Between branches: aggravating epistasis", ha="center", fontsize=6.6,
            fontweight="bold", color="#7B3294")
    ax.text(28, 6.4,
            f"abi3 \u00d7 dog1 (ABA2): s = {coefs.loc['AD_given_B0', 's_odr']:.2f}; "
            f"{100 * aggr('AD_given_B0'):.0f}% of interaction genes synthetic or synergistic\n"
            "AD synergy is supported for chlorophyll retention and ageing-curve area",
            ha="center", fontsize=5.8, color="#333333", linespacing=1.35)
    ax.text(81, 12.5, "Within the DOG1\u2013ABA branch: alleviating epistasis", ha="center",
            fontsize=6.6, fontweight="bold", color="#1B7837")
    ax.text(81, 6.4,
            f"dog1 \u00d7 aba2: s = {coefs.loc['DB_given_A0', 's_odr']:.2f} (ABI3) to "
            f"{coefs.loc['DB_given_A1', 's_odr']:.2f} (abi3-12)\n"
            "overlapping expression responses; not proof of pathway order",
            ha="center", fontsize=5.8, color="#333333", linespacing=1.35)
    ax.text(50, 0.4, "Loss of ABI3 exposes the DOG1\u2013ABA branch: dog1 \u00d7 aba2 interaction "
            f"genes rise from {int(coefs.loc['DB_given_A0', 'n_stable_interaction']):,} to "
            f"{int(coefs.loc['DB_given_A1', 'n_stable_interaction']):,} "
            f"({100 * allev('DB_given_A1'):.0f}% alleviating). Trait-level DB background effects "
            "remain inconclusive.", ha="center", fontsize=6.0, color="#333333",
            style="italic")
    save(fig, "working_model")


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "traits"):
        trait_interactions()
    if which in ("all", "epistasis"):
        epistasis_landscape()
    if which in ("all", "function"):
        functional_annotation()
    if which in ("all", "model"):
        working_model()


if __name__ == "__main__":
    main()
