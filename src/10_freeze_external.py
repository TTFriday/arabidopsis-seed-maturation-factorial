"""Freeze the signatures and decision rules for the external validations.

This script runs before any external expression matrix is opened.  It writes
the gene lists, the exact analysis rules and their hashes to
results/external/frozen_external.json.  Re-running after the file exists only
verifies the hash and refuses to redefine anything.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
COUNTS = ROOT / "data" / "raw" / "GSE270988_raw_counts.csv"
SAMPLES = ROOT / "data" / "processed" / "samples.csv"
OUT = ROOT / "results" / "external"
OUT.mkdir(parents=True, exist_ok=True)
PATH = OUT / "frozen_external.json"

STAGES = [12, 14, 16, 18, 20, 22, 24]
MARKERS = {
    "AT4G25140": "OLEOSIN1",
    "AT1G03880": "CRUCIFERIN2",
    "AT3G17520": "LEA",
    "AT5G45830": "DOG1",
    "AT3G24650": "ABI3",
    "AT3G26790": "FUS3",
}


def sha(genes):
    return hashlib.sha256("\n".join(sorted(map(str, genes))).encode()).hexdigest()


def spearman_with_day(mat, day):
    ranks = stats.rankdata(mat, axis=0)
    dr = stats.rankdata(day)
    gr = ranks - ranks.mean(axis=0)
    dd = dr - dr.mean()
    num = (gr * dd[:, None]).sum(axis=0)
    den = np.sqrt((gr ** 2).sum(axis=0) * (dd ** 2).sum())
    return num / np.maximum(den, 1e-12)


def derive_maturity(mat, genes, day, keep):
    """Derive up/down lists from a subset of samples (used for LOSO too).

    The endpoint contrast uses the first and last day present in ``day`` so
    that leave-one-stage-out runs still have anchors when 12 or 24 DAP is
    held out.
    """
    sub = mat[keep][:, :]
    rho = spearman_with_day(sub.T, day)
    logc = np.log2(sub / sub.sum(axis=0) * 1e6 + 1)
    d0, d1 = day.min(), day.max()
    first = logc[:, day == d0].mean(axis=1)
    last = logc[:, day == d1].mean(axis=1)
    delta = last - first
    up = (rho >= 0.8) & (delta >= 2)
    down = (rho <= -0.8) & (delta <= -2)
    score = np.abs(rho) * np.abs(delta)
    kept_genes = genes[keep]
    up_idx = np.where(up)[0][np.argsort(-score[up])[:500]]
    down_idx = np.where(down)[0][np.argsort(-score[down])[:500]]
    up_genes = [kept_genes[i] for i in up_idx]
    down_genes = [kept_genes[i] for i in down_idx]
    return up_genes, down_genes, rho, delta


def maturity_score(mat, genes, up, down):
    logc = np.log2(mat / mat.sum(axis=0) * 1e6 + 1)
    idx = {g: i for i, g in enumerate(genes)}
    u = [idx[g] for g in up if g in idx]
    d = [idx[g] for g in down if g in idx]
    z = (logc - logc.mean(axis=1, keepdims=True)) / np.maximum(
        logc.std(axis=1, keepdims=True), 1e-9)
    return z[u].mean(axis=0) - z[d].mean(axis=0), len(u), len(d)


def main() -> None:
    df = pd.read_csv(COUNTS)
    genes = df["gene_id"].astype(str).to_numpy()
    cols = [c for c in df.columns if c != "gene_id"]
    samples = pd.read_csv(SAMPLES)
    lrt = pd.read_csv(ROOT / "results" / "tables" / "lrt_all_genes.csv.gz").set_index("gene_id")
    contr = pd.read_csv(ROOT / "results" / "tables" / "contrasts_all_genes.csv.gz").set_index("gene_id")

    # ---- S_D (dog1-4 main effect) -----------------------------------------
    set_d = lrt[(lrt["qvalue"] < 0.05) & (lrt["coef_D"].abs() >= 1)]
    s_d = sorted(set_d.index)
    robust = set_d[(contr.loc[set_d.index, "DB_given_A0_log2"].abs() < 1)]
    s_dr = sorted(robust.index)

    # ---- maturity signature from Col-0 round 2, 12-24 DAP ----------------
    col = samples[(samples["genotype"] == "Col-0") & (samples["round"] == 2) &
                  (samples["stage"].isin([f"{d}d" for d in STAGES]))].copy()
    col["day"] = col["stage"].str.replace("d", "").astype(int)
    col = col.sort_values(["day", "rep"])
    assert len(col) == 20, len(col)
    idx = [cols.index(s) for s in col["sample_id"]]
    mat = df.iloc[:, 1:].to_numpy(dtype=np.float64)[:, idx]
    cpm = mat / mat.sum(axis=0) * 1e6
    keep = (cpm >= 1).sum(axis=1) >= 3
    day_real = np.array([12, 12, 12, 14, 14, 14, 16, 16, 16, 18, 18, 18,
                         20, 20, 20, 22, 22, 22, 24, 24], float)
    assert (col["day"].to_numpy() == day_real).all(), col["day"].tolist()
    up, down, rho, delta = derive_maturity(mat, genes, day_real, keep)
    print(f"S_D {len(s_d)}, S_D_robust {len(s_dr)}, maturity up {len(up)}, down {len(down)}")

    # leave-one-stage-out check of the score construction
    loso = {}
    for held in STAGES:
        tr = day_real != held
        keep_tr = (cpm[:, tr] >= 1).sum(axis=1) >= 3
        up_i, down_i, _, _ = derive_maturity(mat[:, tr], genes, day_real[tr], keep_tr)
        score, nu, nd = maturity_score(mat, genes, up_i, down_i)
        held_score = float(score[day_real == held].mean())
        stage_means = {int(d): float(score[day_real == d].mean()) for d in STAGES}
        lower = stage_means.get(held - 2)
        upper = stage_means.get(held + 2)
        checks = []
        if lower is not None:
            checks.append(held_score > lower)
        if upper is not None:
            checks.append(held_score < upper)
        loso[str(held)] = {"held_out_score": held_score,
                           "lower_stage": lower if lower is None else round(lower, 3),
                           "upper_stage": upper if upper is None else round(upper, 3),
                           "ordinal_pass": bool(all(checks)),
                           "n_up": nu, "n_down": nd}
    ok = sum(v["ordinal_pass"] for v in loso.values())

    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "status": "frozen before opening any external expression matrix",
        "S_D": {"rule": "LRT q<0.05 and |coef_D|>=1 (DOG1 main effect at A=0, B=0)",
                "n": len(s_d), "sha256": sha(s_d), "genes": s_d},
        "S_D_robust": {"rule": "S_D plus |DB_given_A0|<1 (dog1 effect similar in both ABA backgrounds)",
                       "n": len(s_dr), "sha256": sha(s_dr), "genes": s_dr},
        "S_maturity": {
            "rule": ("Col-0 round 2, 12-24 DAP (n=20). CPM>=1 in >=3 samples; Spearman rho "
                     "with day >=0.8 (up) or <=-0.8 (down) and |log2 CPM(24d)-log2 CPM(12d)| "
                     ">=2; top 500 per direction by |rho|*|delta|."),
            "up": {"n": len(up), "sha256": sha(up), "genes": up},
            "down": {"n": len(down), "sha256": sha(down), "genes": down},
            "loso": loso, "loso_ordinal_pass": int(ok),
        },
        "markers": MARKERS,
        "V1_rules": {
            "data": "GSE251921 processed counts; Col-0 GSM7989996-7989999 vs dog1-4 GSM7990000-7990003",
            "filter": "CPM>=1 in >=3 of 8 samples",
            "model": "NB GLM, two group means, poscounts size factors, CR dispersion + EB shrinkage",
            "effect": "log2(dog1-4 / Col-0) per gene",
            "primary": ("sign concordance and Spearman/Slope of V1 effect vs main coef_D on S_D; "
                        "sample-level signature score (mean z of S_D genes, sign from main coef_D) "
                        "compared between Col-0 and dog1-4"),
            "gate": "if <50% of S_D genes map and pass the filter, report insufficient coverage",
            "mapping": "uppercase, strip version suffix, TAIR AGI IDs only",
        },
        "V2_rules": {
            "data": "GSE5634 series matrix (MAS5 values as deposited); GSM131694-131708, stages 6-10, 3 arrays each",
            "platform": "GPL198 ATH1; probe->gene by GPL198 annotation; multiple probes: keep highest mean expression",
            "primary": ("maturity score = mean z(up) - mean z(down) per array; Spearman with ordinal "
                        "stage 6-10 and stage means reported; MAS5 values used as deposited, no second log"),
            "secondary": "marker genes OLEOSIN1/CRUCIFERIN2/LEA/DOG1/ABI3/FUS3 across stage",
            "gate": "if <50% of signature genes map to GPL198, report insufficient coverage",
        },
    }
    if PATH.exists():
        prev = json.loads(PATH.read_text(encoding="utf-8"))
        same = (prev["S_D"]["sha256"] == payload["S_D"]["sha256"] and
                prev["S_D_robust"]["sha256"] == payload["S_D_robust"]["sha256"] and
                prev["S_maturity"]["up"]["sha256"] == payload["S_maturity"]["up"]["sha256"] and
                prev["S_maturity"]["down"]["sha256"] == payload["S_maturity"]["down"]["sha256"])
        if not same:
            raise SystemExit("frozen file exists with different signatures; refusing to redefine")
        print(f"verified existing frozen signatures: {PATH}")
        return

    PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"frozen: {PATH}")
    print(json.dumps(loso, indent=1))


if __name__ == "__main__":
    main()
