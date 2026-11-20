"""Transcriptome-wide epistasis between each gene pair in each background.

For a pair (P, Q) with the third gene fixed, every gene has three log2 changes
relative to the background reference genotype: the P single mutant (p), the
Q single mutant (q) and the double mutant (pq).  The interaction contrast of
the main model is the deviation from additivity, delta = pq - p - q.

Two summaries are computed.

1. Transcriptome-wide epistasis coefficient s (Angeles-Albores et al. 2018):
   the slope of delta against the additive expectation x = p + q over genes
   changed by both single mutants.  s = 0 for independent action, s = -1/2 when
   the double mutant equals both single mutants (one shared pathway) and s = -1
   when the double mutant returns to the reference.  Because x and delta share
   estimation noise, the observed slope is compared with slopes computed in the
   same way on data simulated from explicit genetic models (additive, masking by
   either mutation, suppression) that use the observed single-mutant means and
   gene dispersions.

2. Gene-level interaction type for genes with stable interaction support:
   synthetic (only the double mutant changes), synergistic (double beyond
   additive in the same direction), masking (double equals one single mutant),
   pathway-like (double equals both), suppression (double returns to the
   reference), inversion (double opposite to the single mutants) and
   quantitative (anything else).  These calls are descriptive and post hoc.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import odr, stats

import nb_glm as nb

ROOT = Path(__file__).resolve().parents[1]
COUNTS = ROOT / "data" / "raw" / "GSE270988_raw_counts.csv"
DESIGN = ROOT / "data" / "processed" / "main_design_24.csv"
LRT = ROOT / "results" / "tables" / "lrt_all_genes.csv.gz"
CONTRASTS = ROOT / "results" / "tables" / "contrasts_all_genes.csv.gz"
OUT = ROOT / "results" / "epistasis"
OUT.mkdir(parents=True, exist_ok=True)

LN2 = np.log(2.0)
SEED = 20261119
N_BOOT = 1000
N_SIM = 50

# cell index g = 4A + 2D + B
PAIRS = {
    "AD_given_B0": ("abi3-12", "dog1-4", 0, 4, 2, 6),
    "AD_given_B1": ("abi3-12", "dog1-4", 1, 5, 3, 7),
    "AB_given_D0": ("abi3-12", "aba2-1", 0, 4, 1, 5),
    "AB_given_D1": ("abi3-12", "aba2-1", 2, 6, 3, 7),
    "DB_given_A0": ("dog1-4", "aba2-1", 0, 2, 1, 3),
    "DB_given_A1": ("dog1-4", "aba2-1", 4, 6, 5, 7),
}
MODELS = ("additive", "masking by first", "masking by second", "suppression")
EFFECT = 1.0
ALPHA = 0.05


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
    genes_all = df["gene_id"].astype(str).to_numpy()
    cols = [c for c in df.columns if c != "gene_id"]
    design = pd.read_csv(DESIGN)
    idx = [cols.index(s) for s in design["sample_id"]]
    y_all = df.drop(columns="gene_id").to_numpy(dtype=np.float64)[:, idx]
    lrt = pd.read_csv(LRT)
    pos = pd.Series(np.arange(len(genes_all)), index=genes_all)
    keep = pos[lrt["gene_id"].astype(str)].to_numpy()
    y = y_all[keep].T
    cells = (design["A"] * 4 + design["D"] * 2 + design["B"]).astype(int).to_numpy()
    return lrt, y, cells


def fit_cells(y, cells, sf, alpha, cell_ids, n_iter=60, tol=1e-10):
    """Cell-means NB fit with fixed dispersion; returns log means and variances.

    With one parameter per genotype cell the likelihood separates by cell, so
    each cell mean is found by Fisher scoring on its own log scale.  This gives
    the same estimates as the factorial parametrisation of the main model.
    """
    logm = np.empty((y.shape[1], len(cell_ids)))
    var = np.empty_like(logm)
    a = alpha[None, :]
    for j, g in enumerate(cell_ids):
        rows = cells == g
        yy = y[rows]
        s = sf[rows][:, None]
        eta = np.log(np.maximum((yy / s).mean(axis=0), 1e-10))
        for _ in range(n_iter):
            m = s * np.exp(eta)
            den = 1.0 + a * m
            score = ((yy - m) / den).sum(axis=0)
            info = (m / den).sum(axis=0)
            new = np.clip(eta + score / np.maximum(info, 1e-300), -30.0, 30.0)
            done = np.max(np.abs(new - eta)) < tol
            eta = new
            if done:
                break
        m = s * np.exp(eta)
        info = (m / (1.0 + a * m)).sum(axis=0)
        logm[:, j] = eta
        var[:, j] = 1.0 / np.maximum(info, 1e-300)
    return logm, var


def pair_effects(logm, var, ref, gp, gq, gpq):
    """log2 changes and their standard errors for one pair."""
    L = logm / LN2
    V = var / LN2 ** 2
    out = {}
    out["p"] = L[:, gp] - L[:, ref]
    out["q"] = L[:, gq] - L[:, ref]
    out["pq"] = L[:, gpq] - L[:, ref]
    out["delta"] = out["pq"] - out["p"] - out["q"]
    out["pq_minus_p"] = L[:, gpq] - L[:, gp]
    out["pq_minus_q"] = L[:, gpq] - L[:, gq]
    out["p_se"] = np.sqrt(V[:, gp] + V[:, ref])
    out["q_se"] = np.sqrt(V[:, gq] + V[:, ref])
    out["pq_se"] = np.sqrt(V[:, gpq] + V[:, ref])
    out["delta_se"] = np.sqrt(V[:, gpq] + V[:, gp] + V[:, gq] + V[:, ref])
    out["pq_minus_p_se"] = np.sqrt(V[:, gpq] + V[:, gp])
    out["pq_minus_q_se"] = np.sqrt(V[:, gpq] + V[:, gq])
    # x = p + q shares the reference cell twice
    out["x"] = out["p"] + out["q"]
    out["x_se"] = np.sqrt(V[:, gp] + V[:, gq] + 4.0 * V[:, ref])
    for k in ("p", "q", "pq", "delta", "pq_minus_p", "pq_minus_q"):
        z = out[k] / out[f"{k}_se"]
        out[f"{k}_pval"] = 2.0 * stats.norm.sf(np.abs(z))
        out[f"{k}_qval"] = bh_qvalues(out[f"{k}_pval"])
    return out


def shared_responders(e):
    """Genes changed by both single mutants, with estimable effects."""
    return ((e["p_qval"] < ALPHA) & (np.abs(e["p"]) >= EFFECT) &
            (e["q_qval"] < ALPHA) & (np.abs(e["q"]) >= EFFECT) &
            (e["p_se"] <= 1.0) & (e["q_se"] <= 1.0) & (e["pq_se"] <= 1.0))


def odr_slope(x, y, sx, sy):
    data = odr.RealData(x, y, sx=np.maximum(sx, 1e-3), sy=np.maximum(sy, 1e-3))
    model = odr.Model(lambda beta, t: beta[0] * t)
    fit = odr.ODR(data, model, beta0=[0.0]).run()
    return float(fit.beta[0])


def epistasis_coefficient(e, mask, rng):
    x, y = e["x"][mask], e["delta"][mask]
    sx, sy = e["x_se"][mask], e["delta_se"][mask]
    s = odr_slope(x, y, sx, sy)
    boot = np.empty(N_BOOT)
    n = mask.sum()
    for b in range(N_BOOT):
        i = rng.integers(0, n, n)
        boot[b] = odr_slope(x[i], y[i], sx[i], sy[i])
    ols = float(np.sum(x * y) / np.sum(x * x))
    return s, np.percentile(boot, [2.5, 97.5]), ols


def classify(e, stable):
    def eff(k):
        return (e[f"{k}_qval"] < ALPHA) & (np.abs(e[k]) >= EFFECT)

    def differs(k):
        return (e[f"{k}_qval"] < ALPHA) & (np.abs(e[k]) >= EFFECT)

    ep, eq, epq = eff("p"), eff("q"), eff("pq")
    same_p, same_q = ~differs("pq_minus_p"), ~differs("pq_minus_q")
    sp, sq, spq = np.sign(e["p"]), np.sign(e["q"]), np.sign(e["pq"])
    inverted = epq & (ep | eq) & (~ep | (sp != spq)) & (~eq | (sq != spq))
    beyond = (np.sign(e["delta"]) == spq) & (np.abs(e["pq"]) > np.abs(e["x"]))
    cls = np.full(len(stable), "", dtype=object)
    rules = [
        ("synthetic", ~ep & ~eq & epq),
        ("suppression", (ep | eq) & ~epq),
        ("inversion", inverted),
        ("pathway-like", epq & same_p & same_q),
        ("masking by first", epq & same_p & ~same_q),
        ("masking by second", epq & ~same_p & same_q),
        ("synergistic", epq & beyond),
    ]
    todo = stable.copy()
    for name, rule in rules:
        hit = todo & rule
        cls[hit] = name
        todo &= ~hit
    cls[todo] = "quantitative"
    cls[~stable] = ""
    return cls


def simulate(cells_sub, sf_sub, mu_true, alpha, rng):
    """Draw NB counts (gamma-Poisson) for the 12 samples of one pair."""
    mean = sf_sub[:, None] * mu_true[cells_sub]
    shape = np.broadcast_to(1.0 / alpha[None, :], mean.shape)
    lam = rng.gamma(shape, mean / shape)
    return rng.poisson(lam).astype(np.float64)


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    lrt, y, cells = load()
    genes = lrt["gene_id"].astype(str).to_numpy()
    alpha = lrt["dispersion"].to_numpy(dtype=np.float64)
    sf = nb.poscounts_size_factors(y)

    logm, var = fit_cells(y, cells, sf, alpha, list(range(8)))
    ct = pd.read_csv(CONTRASTS)
    assert (ct["gene_id"].astype(str).to_numpy() == genes).all()
    # genes with counts in every cell have finite MLEs; others are separation-like
    finite = np.all(np.vstack([(y[cells == g] > 0).any(axis=0) for g in range(8)]), axis=0)
    check = {}
    for name, (_, _, ref, gp, gq, gpq) in PAIRS.items():
        e = pair_effects(logm, var, ref, gp, gq, gpq)
        diff = np.abs(e["delta"] - ct[f"{name}_log2"].to_numpy())
        check[name] = {"max_abs_diff_finite": float(np.max(diff[finite])),
                       "frac_within_1e-3_all": float(np.mean(diff < 1e-3))}
    print("reproduction of main contrasts:", json.dumps(check), flush=True)

    stable_rule = {}
    for name in PAIRS:
        stable_rule[name] = ((ct[f"{name}_q"] < 0.05) & (ct[f"{name}_log2"].abs() >= 1.0) &
                             (ct[f"{name}_se"] <= 1.0) & (lrt["qvalue"] < 0.05)).to_numpy()

    long_rows, coef_rows, class_rows = [], [], []
    for name, (first, second, ref, gp, gq, gpq) in PAIRS.items():
        e = pair_effects(logm, var, ref, gp, gq, gpq)
        mask = shared_responders(e)
        s, ci, ols = epistasis_coefficient(e, mask, rng)
        cls = classify(e, stable_rule[name])

        # simulations under explicit genetic models
        sel = np.isin(cells, [ref, gp, gq, gpq])
        cells_sub = cells[sel]
        sf_sub = sf[sel]
        local = {ref: 0, gp: 1, gq: 2, gpq: 3}
        cells_loc = np.array([local[c] for c in cells_sub])
        # Additivity is undefined when the reference cell has only zero counts.
        estimable = (y[cells == ref] > 0).any(axis=0)
        mu = np.exp(logm[estimable][:, [ref, gp, gq]])
        sim_alpha = alpha[estimable]
        sims = {}
        for model in MODELS:
            if model == "additive":
                mu_pq = mu[:, 1] * mu[:, 2] / mu[:, 0]
            elif model == "masking by first":
                mu_pq = mu[:, 1]
            elif model == "masking by second":
                mu_pq = mu[:, 2]
            else:
                mu_pq = mu[:, 0]
            mu_true = np.column_stack([mu, mu_pq]).T  # no truncation of finite MLEs
            vals, ns = [], []
            for _ in range(N_SIM):
                ysim = simulate(cells_loc, sf_sub, mu_true, sim_alpha, rng)
                lm, vv = fit_cells(ysim, cells_loc, sf_sub, sim_alpha, [0, 1, 2, 3])
                es = pair_effects(lm, vv, 0, 1, 2, 3)
                ms = shared_responders(es)
                vals.append(odr_slope(es["x"][ms], es["delta"][ms],
                                      es["x_se"][ms], es["delta_se"][ms]))
                ns.append(int(ms.sum()))
            sims[model] = (float(np.mean(vals)), float(np.std(vals, ddof=1)), float(np.mean(ns)))

        # where does the observed slope sit between additive and the masking models
        dist = {m: abs(s - sims[m][0]) / max(sims[m][1], 1e-6) for m in MODELS}
        coef_rows.append({
            "pair": name, "first": first, "second": second,
            "n_shared_responders": int(mask.sum()),
            "s_odr": s, "s_ci_low": float(ci[0]), "s_ci_high": float(ci[1]), "s_ols": ols,
            **{f"sim_{m.replace(' ', '_')}_mean": sims[m][0] for m in MODELS},
            **{f"sim_{m.replace(' ', '_')}_sd": sims[m][1] for m in MODELS},
            **{f"sim_{m.replace(' ', '_')}_n": sims[m][2] for m in MODELS},
            "nearest_benchmark_not_acceptance": min(dist, key=dist.get),
            "n_benchmark_genes": int(estimable.sum()),
            "n_stable_interaction": int(stable_rule[name].sum()),
        })

        counts = pd.Series(cls[stable_rule[name]]).value_counts()
        for c, n in counts.items():
            sub = stable_rule[name] & (cls == c)
            class_rows.append({"pair": name, "class": c, "n": int(n),
                               "fraction": float(n / stable_rule[name].sum()),
                               "n_delta_positive": int((e["delta"][sub] > 0).sum()),
                               "n_delta_negative": int((e["delta"][sub] < 0).sum()),
                               "median_abs_delta": float(np.median(np.abs(e["delta"][sub])))})

        tab = pd.DataFrame({"gene_id": genes, "pair": name})
        for k in ("p", "q", "pq", "delta", "pq_minus_p", "pq_minus_q", "x"):
            tab[k] = e[k]
            tab[f"{k}_se"] = e[f"{k}_se"]
        for k in ("p", "q", "pq", "delta", "pq_minus_p", "pq_minus_q"):
            tab[f"{k}_qval"] = e[f"{k}_qval"]
        tab["shared_responder"] = mask
        tab["stable_interaction"] = stable_rule[name]
        tab["interaction_type"] = cls
        long_rows.append(tab)
        print(f"{name}: shared responders {mask.sum():,}, s = {s:.3f} "
              f"[{ci[0]:.3f}, {ci[1]:.3f}]; sims " +
              ", ".join(f"{m} {sims[m][0]:.3f}±{sims[m][1]:.3f}" for m in MODELS), flush=True)

    gene_level = pd.concat(long_rows, ignore_index=True)
    gene_level.to_csv(OUT / "epistasis_gene_level.csv.gz", index=False)
    coefs = pd.DataFrame(coef_rows)
    coefs.to_csv(OUT / "epistasis_coefficients.csv", index=False)
    classes = pd.DataFrame(class_rows)
    classes.to_csv(OUT / "interaction_type_counts.csv", index=False)

    summary = {
        "reproduces_main_contrasts_max_abs_diff": check,
        "shared_responder_rule": "q<0.05 and |log2|>=1 for both single mutants; SE<=1 for p, q, pq",
        "slope": "orthogonal distance regression through the origin, weighted by SEs",
        "bootstrap": N_BOOT,
        "simulations_per_model": N_SIM,
        "seed": SEED,
        "coefficients": coefs.to_dict(orient="records"),
        "seconds": time.time() - t0,
    }
    (OUT / "epistasis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(classes.pivot_table(index="class", columns="pair", values="n", fill_value=0).to_string())


if __name__ == "__main__":
    main()
