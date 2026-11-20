"""Promoter cis-element content of the interaction gene classes.

Promoters are the 1,000 bp upstream of the annotated gene start (strand
aware) of nuclear genes in the Ensembl Plants TAIR10 annotation (release 60);
500 bp is a secondary window.  Elements are counted on both strands:

  RY      CATGCA        bound by B3-domain factors (ABI3, FUS3, LEC2)
  ABRE    ACGTGKC       ABA-responsive element, bound by ABI5/ABF bZIPs
  G-box   CACGTG        bZIP/bHLH element overlapping some ABREs
  DRE     RCCGAC        dehydration-responsive element (control element)

For each gene set the fraction of promoters with at least one element is
compared with all tested genes by Fisher's exact test and by a
Mantel-Haenszel test stratified by expression decile and promoter GC tertile,
so that differences in expression level or base composition do not drive the
result.  Genes reduced in abi3-12 serve as a positive control for RY, and
genes reduced in aba2-1 as a positive control for ABRE.
"""

import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
GFF = ROOT / "data" / "raw" / "external" / "TAIR10" / "Arabidopsis_thaliana.TAIR10.60.gff3.gz"
FNA = ROOT / "data" / "raw" / "external" / "TAIR10" / "Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.gz"
LRT = ROOT / "results" / "tables" / "lrt_all_genes.csv.gz"
SETS = ROOT / "results" / "function" / "gene_sets.json"
OUT = ROOT / "results" / "promoters"
OUT.mkdir(parents=True, exist_ok=True)

NUCLEAR = {"1", "2", "3", "4", "5"}
MOTIFS = {
    "RY": "CATGCA",
    "ABRE": "ACGTG[GT]C",
    "G-box": "CACGTG",
    "DRE": "[AG]CCGAC",
}
COMP = str.maketrans("ACGTN", "TGCAN")


def revcomp(s):
    return s.translate(COMP)[::-1]


def iupac_rc(pattern):
    """Reverse complement of a pattern written with [..] classes."""
    tokens = re.findall(r"\[[^\]]+\]|.", pattern)
    out = []
    for t in reversed(tokens):
        if t.startswith("["):
            out.append("[" + "".join(sorted(c.translate(COMP) for c in t[1:-1])) + "]")
        else:
            out.append(t.translate(COMP))
    return "".join(out)


def read_genome(path):
    seqs, name, buf = {}, None, []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if name in NUCLEAR:
                    seqs[name] = "".join(buf).upper()
                name, buf = line[1:].split()[0], []
            elif name in NUCLEAR:
                buf.append(line.strip())
    if name in NUCLEAR:
        seqs[name] = "".join(buf).upper()
    return seqs


def read_genes(path):
    rows = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "gene" or f[0] not in NUCLEAR:
                continue
            m = re.search(r"ID=gene:(AT[1-5]G\d{5})", f[8])
            if not m:
                continue
            rows.append((m.group(1), f[0], int(f[3]), int(f[4]), f[6]))
    genes = pd.DataFrame(rows, columns=["gene_id", "seqid", "start", "end", "strand"])
    return genes.drop_duplicates("gene_id")


def promoter(seqs, seqid, start, end, strand, width):
    chrom = seqs[seqid]
    if strand == "+":
        s = chrom[max(start - 1 - width, 0): start - 1]
    else:
        s = revcomp(chrom[end: end + width])
    return s


def count_motifs(seq):
    out = {}
    for name, pat in MOTIFS.items():
        fwd = len(re.findall(f"(?={pat})", seq))
        rc = iupac_rc(pat)
        rev = 0 if rc == pat else len(re.findall(f"(?={rc})", seq))
        out[name] = fwd + rev
    return out


def mantel_haenszel(in_set, has, strata):
    """MH common odds ratio and continuity-corrected CMH test."""
    num = den = a_sum = e_sum = v_sum = 0.0
    for s in np.unique(strata):
        m = strata == s
        a = np.sum(in_set[m] & has[m])
        b = np.sum(in_set[m] & ~has[m])
        c = np.sum(~in_set[m] & has[m])
        d = np.sum(~in_set[m] & ~has[m])
        n = a + b + c + d
        if n < 2 or (a + b) == 0 or (c + d) == 0:
            continue
        num += a * d / n
        den += b * c / n
        r1, c1 = a + b, a + c
        a_sum += a
        e_sum += r1 * c1 / n
        v_sum += r1 * (n - r1) * c1 * (n - c1) / (n ** 2 * (n - 1))
    odds = num / den if den > 0 else np.inf
    chi2 = max(abs(a_sum - e_sum) - 0.5, 0.0) ** 2 / v_sum if v_sum > 0 else np.nan
    p = stats.chi2.sf(chi2, 1) if np.isfinite(chi2) else np.nan
    return odds, p


def bh_qvalues(p):
    p = np.asarray(p, dtype=np.float64)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def main() -> None:
    lrt = pd.read_csv(LRT)[["gene_id", "base_mean"]]
    lrt["gene_id"] = lrt["gene_id"].astype(str).str.upper()
    seqs = read_genome(FNA)
    genes = read_genes(GFF)
    genes = genes[genes["gene_id"].isin(set(lrt["gene_id"]))]

    rows = []
    for g in genes.itertuples():
        row = {"gene_id": g.gene_id}
        for width in (1000, 500):
            s = promoter(seqs, g.seqid, g.start, g.end, g.strand, width)
            c = count_motifs(s)
            for k, v in c.items():
                row[f"{k}_{width}"] = v
            row[f"gc_{width}"] = (s.count("G") + s.count("C")) / max(len(s), 1)
            row[f"len_{width}"] = len(s)
        rows.append(row)
    prom = pd.DataFrame(rows).merge(lrt, on="gene_id")
    prom["RY+ABRE_1000"] = ((prom["RY_1000"] > 0) & (prom["ABRE_1000"] > 0)).astype(int)
    prom["RY+ABRE_500"] = ((prom["RY_500"] > 0) & (prom["ABRE_500"] > 0)).astype(int)
    prom.to_csv(OUT / "promoter_element_counts.csv.gz", index=False)

    expr_dec = pd.qcut(np.log10(prom["base_mean"] + 1), 10, labels=False).to_numpy()
    strata = {width: expr_dec * 3 + pd.qcut(prom[f"gc_{width}"], 3, labels=False).to_numpy()
              for width in (1000, 500)}

    sets = json.loads(SETS.read_text(encoding="utf-8"))
    universe = prom["gene_id"].to_numpy()
    res = []
    for name, members in sets.items():
        in_set = np.isin(universe, [m.upper() for m in members])
        if in_set.sum() < 25:
            continue
        for width in (1000, 500):
            for motif in list(MOTIFS) + ["RY+ABRE"]:
                col = f"{motif}_{width}"
                has = prom[col].to_numpy() > 0
                a = int(np.sum(in_set & has))
                n = int(in_set.sum())
                bg = float(has.mean())
                table = [[a, n - a], [int(has.sum()) - a, len(has) - int(has.sum()) - (n - a)]]
                odds, p = stats.fisher_exact(table)
                mh_or, mh_p = mantel_haenszel(in_set, has, strata[width])
                res.append({"set": name, "window": width, "element": motif, "n_set": n,
                            "n_with": a, "fraction_set": a / n, "fraction_background": bg,
                            "odds_ratio": odds, "fisher_p": p,
                            "mh_odds_ratio": mh_or, "mh_p": mh_p})
    res = pd.DataFrame(res)
    for width in (1000, 500):
        m = res["window"] == width
        res.loc[m, "mh_q"] = bh_qvalues(res.loc[m, "mh_p"].fillna(1.0).to_numpy())
    res.to_csv(OUT / "promoter_element_enrichment.csv", index=False)

    summary = {
        "genes_with_promoter": int(len(prom)),
        "tested_genes": int(len(lrt)),
        "background_fraction_1000": {k: float((prom[f"{k}_1000"] > 0).mean())
                                     for k in list(MOTIFS) + ["RY+ABRE"]},
        "strata": "expression decile x promoter GC tertile (30 strata)",
    }
    (OUT / "promoter_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    pd.set_option("display.width", 220)
    show = res[(res["window"] == 1000)].sort_values("mh_p").head(40)
    print(show[["set", "element", "n_set", "fraction_set", "fraction_background",
                "mh_odds_ratio", "mh_q"]].to_string(index=False, float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main()
