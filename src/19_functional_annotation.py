"""Functional annotation of the interaction gene classes.

Three layers, all against the 18,014 tested genes as background:

1. GO biological-process enrichment.  Annotations: GO Consortium GAF for
   Arabidopsis (TAIR source); ontology: go-basic.obo.  Annotations are
   propagated to all is_a / part_of ancestors; NOT-qualified and ND-evidence
   annotations are dropped; terms with 10-500 annotated background genes are
   tested by one-sided Fisher's exact test with BH correction within each
   gene set.
2. Overlap with ABI3-regulated genes (direct and indirect targets reported by
   Tian et al. 2020, as compiled in Table S8 of the source study).
3. A literature panel of seed maturation, dormancy, degreening and ABA genes.
   Every panel gene that passed the expression filter is reported.

Gene sets are interaction types from 18_epistasis.py split by the direction of
the double-mutant change, background-switch sets (an interaction present in
one background only), and single-mutant response sets used as controls.
"""

import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
GAF = ROOT / "data" / "raw" / "external" / "GO" / "tair.gaf.gz"
OBO = ROOT / "data" / "raw" / "external" / "GO" / "go-basic.obo"
GENE_INFO = ROOT / "data" / "raw" / "external" / "GPL198" / "Arabidopsis_thaliana.gene_info.gz"
XLSX = ROOT / "data" / "raw" / "supplemental_tables.xlsx"
EPI = ROOT / "results" / "epistasis" / "epistasis_gene_level.csv.gz"
LRT = ROOT / "results" / "tables" / "lrt_all_genes.csv.gz"
OUT = ROOT / "results" / "function"
OUT.mkdir(parents=True, exist_ok=True)

MIN_TERM, MAX_TERM, MIN_SET = 10, 500, 25

PANEL = [
    ("ABI3", "AT3G24650", "B3 transcription factor, maturation master regulator"),
    ("FUS3", "AT3G26790", "B3 transcription factor"),
    ("LEC1", "AT1G21970", "NF-YB maturation regulator"),
    ("LEC2", "AT1G28300", "B3 transcription factor"),
    ("WRI1", "AT3G54320", "oil biosynthesis regulator"),
    ("ABI4", "AT2G40220", "AP2 transcription factor, ABA response"),
    ("ABI5", "AT2G36270", "bZIP transcription factor, ABA response"),
    ("DOG1", "AT5G45830", "dormancy regulator"),
    ("MFT", "AT1G18100", "dormancy and ABA/GA crosstalk"),
    ("AHG1", "AT5G51760", "PP2C, DOG1 interactor"),
    ("AHG3", "AT3G11410", "PP2C (PP2CA), DOG1 interactor"),
    ("ABI1", "AT4G26080", "PP2C, ABA co-receptor"),
    ("ABI2", "AT5G57050", "PP2C, ABA co-receptor"),
    ("HAI1", "AT5G59220", "PP2C"),
    ("SRK2D", "AT3G50500", "SnRK2.2"),
    ("SRK2I", "AT5G66880", "SnRK2.3"),
    ("OST1", "AT4G33950", "SnRK2.6"),
    ("ABA2", "AT1G52340", "ABA biosynthesis"),
    ("NCED6", "AT3G24220", "ABA biosynthesis"),
    ("NCED9", "AT1G78390", "ABA biosynthesis"),
    ("CYP707A1", "AT4G19230", "ABA catabolism"),
    ("CYP707A2", "AT2G29090", "ABA catabolism"),
    ("GA3OX1", "AT1G15550", "GA biosynthesis"),
    ("RGL2", "AT3G03450", "DELLA, germination repressor"),
    ("EM1", "AT3G51810", "LEA, ABI3/ABI5 target"),
    ("EM6", "AT2G40170", "LEA, ABI3/ABI5 target"),
    ("RAB18", "AT5G66400", "dehydrin"),
    ("KIN2", "AT5G15970", "stress/ABA-responsive"),
    ("RD29B", "AT5G52300", "ABA-responsive"),
    ("PER1", "AT1G48130", "1-Cys peroxiredoxin, seed longevity"),
    ("HSFA9", "AT5G54070", "seed heat-shock factor, ABI3 target"),
    ("HSP17.4", "AT3G46230", "small heat-shock protein"),
    ("CRA1", "AT5G44120", "12S cruciferin"),
    ("CRU3", "AT4G28520", "12S cruciferin"),
    ("SESA1", "AT4G27140", "2S albumin"),
    ("OLE1", "AT4G25140", "oleosin"),
    ("SGR1", "AT4G22920", "chlorophyll degradation (stay-green)"),
    ("SGR2", "AT4G11910", "chlorophyll degradation"),
    ("NYC1", "AT4G13250", "chlorophyll b reductase"),
    ("PPH", "AT5G13800", "pheophytinase"),
    ("CAB1", "AT1G29930", "light-harvesting complex protein"),
]


def bh_qvalues(p):
    p = np.asarray(p, dtype=np.float64)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def read_obo(path):
    terms, cur = {}, None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line == "[Term]":
                cur = {"parents": set(), "obsolete": False}
            elif line.startswith("[") and line.endswith("]"):
                cur = None
            elif cur is not None:
                if line.startswith("id: "):
                    cur["id"] = line[4:]
                    terms[cur["id"]] = cur
                elif line.startswith("name: "):
                    cur["name"] = line[6:]
                elif line.startswith("namespace: "):
                    cur["namespace"] = line[11:]
                elif line.startswith("is_a: "):
                    cur["parents"].add(line[6:].split(" ")[0])
                elif line.startswith("relationship: part_of "):
                    cur["parents"].add(line.split(" ")[2])
                elif line.startswith("is_obsolete: true"):
                    cur["obsolete"] = True
    return terms


def ancestors(terms):
    memo = {}

    def up(t):
        if t in memo:
            return memo[t]
        out = {t}
        for p in terms.get(t, {}).get("parents", ()):
            out |= up(p)
        memo[t] = out
        return out

    for t in terms:
        up(t)
    return memo


def read_gaf(path, terms, anc, background):
    gene_terms = defaultdict(set)
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("!"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 15 or f[8] != "P" or "NOT" in f[3] or f[6] == "ND":
                continue
            gene = f[1].upper()
            if gene not in background or f[4] not in terms:
                continue
            for t in anc.get(f[4], {f[4]}):
                if terms.get(t, {}).get("namespace") == "biological_process":
                    gene_terms[gene].add(t)
    return gene_terms


def go_enrichment(set_genes, gene_terms, term_genes, n_bg, terms):
    annotated = [g for g in set_genes if g in gene_terms]
    n_set = len(annotated)
    rows = []
    if n_set == 0:
        return pd.DataFrame()
    counts = defaultdict(int)
    for g in annotated:
        for t in gene_terms[g]:
            counts[t] += 1
    for t, k in counts.items():
        K = len(term_genes[t])
        if K < MIN_TERM or K > MAX_TERM or k < 2:
            continue
        p = stats.hypergeom.sf(k - 1, n_bg, K, n_set)
        rows.append({"term": t, "name": terms[t].get("name", ""), "k": k, "n_set": n_set,
                     "K": K, "N": n_bg, "fold": (k / n_set) / (K / n_bg), "pvalue": p})
    out = pd.DataFrame(rows)
    if len(out):
        # every testable term contributes to the correction, including k < 2
        n_tests = sum(1 for t, g in term_genes.items() if MIN_TERM <= len(g) <= MAX_TERM)
        q = bh_qvalues(np.r_[out["pvalue"].to_numpy(), np.ones(max(n_tests - len(out), 0))])
        out["qvalue"] = q[: len(out)]
        out = out.sort_values("pvalue")
    return out


def fisher_overlap(set_genes, target, background):
    s = set(set_genes)
    a = len(s & target)
    b = len(s) - a
    c = len(target & background) - a
    d = len(background) - a - b - c
    odds, p = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
    return a, len(s), odds, p


def gene_sets(epi):
    sets = {}
    wide = {}
    for pair, sub in epi.groupby("pair"):
        sub = sub.set_index("gene_id")
        wide[pair] = sub
        st = sub[sub["stable_interaction"]]
        for cls, grp in st.groupby("interaction_type"):
            for sign, lab in ((1, "up"), (-1, "down")):
                g = grp.index[np.sign(grp["pq"]) == sign]
                if len(g) >= MIN_SET:
                    sets[f"{pair} | {cls} | double {lab}"] = list(g)
        for sign, lab in ((1, "positive"), (-1, "negative")):
            g = st.index[np.sign(st["delta"]) == sign]
            if len(g) >= MIN_SET:
                sets[f"{pair} | all stable | delta {lab}"] = list(g)

    def switch(on, off, label):
        a = wide[on]
        b = wide[off]
        on_g = set(a.index[a["stable_interaction"]])
        off_any = set(b.index[(b["delta_qval"] < 0.05) & (b["delta"].abs() >= 1.0)])
        only = sorted(on_g - off_any)
        both = sorted(on_g & set(b.index[b["stable_interaction"]]))
        sets[f"{label} only"] = only
        return only, both

    ad_b0_only, ad_both = switch("AD_given_B0", "AD_given_B1", "ABI3xDOG1 with ABA2")
    ad_b1_only, _ = switch("AD_given_B1", "AD_given_B0", "ABI3xDOG1 in aba2-1")
    db_a1_only, db_both = switch("DB_given_A1", "DB_given_A0", "DOG1xABA2 in abi3-12")
    sets["ABI3xDOG1 in both ABA2 backgrounds"] = ad_both
    sets["DOG1xABA2 in both ABI3 backgrounds"] = db_both

    ref = wide["AD_given_B0"]
    for col, lab in (("p", "abi3-12"), ("q", "dog1-4")):
        for sign, d in ((1, "up"), (-1, "down")):
            m = (ref[f"{col}_qval"] < 0.05) & (np.sign(ref[col]) == sign) & (ref[col].abs() >= 1)
            sets[f"single {lab} {d}"] = list(ref.index[m])
    ref = wide["AB_given_D0"]
    for sign, d in ((1, "up"), (-1, "down")):
        m = (ref["q_qval"] < 0.05) & (np.sign(ref["q"]) == sign) & (ref["q"].abs() >= 1)
        sets[f"single aba2-1 {d}"] = list(ref.index[m])
    return {k: v for k, v in sets.items() if len(v) >= MIN_SET}, wide


def main() -> None:
    lrt = pd.read_csv(LRT)
    background = set(lrt["gene_id"].astype(str).str.upper())
    epi = pd.read_csv(EPI)
    epi["interaction_type"] = epi["interaction_type"].fillna("")

    terms = read_obo(OBO)
    terms = {k: v for k, v in terms.items() if not v.get("obsolete")}
    anc = ancestors(terms)
    gene_terms = read_gaf(GAF, terms, anc, background)
    term_genes = defaultdict(set)
    for g, ts in gene_terms.items():
        for t in ts:
            term_genes[t].add(g)
    n_bg = len(gene_terms)
    print(f"BP-annotated background genes: {n_bg:,} of {len(background):,}")

    sets, wide = gene_sets(epi)
    pd.DataFrame([{"set": k, "n": len(v)} for k, v in sets.items()]).to_csv(
        OUT / "gene_set_sizes.csv", index=False)
    (OUT / "gene_sets.json").write_text(json.dumps(sets, indent=1), encoding="utf-8")

    go_rows = []
    for name, genes in sets.items():
        res = go_enrichment(genes, gene_terms, term_genes, n_bg, terms)
        if len(res):
            res.insert(0, "set", name)
            go_rows.append(res)
    go = pd.concat(go_rows, ignore_index=True)
    go.to_csv(OUT / "go_bp_enrichment_all.csv.gz", index=False)
    top = (go[go["qvalue"] < 0.05].sort_values(["set", "pvalue"]).groupby("set").head(8))
    top.to_csv(OUT / "go_bp_enrichment_top.csv", index=False)

    s8 = pd.read_excel(XLSX, sheet_name="Table S8", header=2)
    abi3_targets = set(s8.iloc[:, 0].dropna().astype(str).str.strip().str.upper())
    abi3_targets = {g for g in abi3_targets if g.startswith("AT")}
    ov_rows = []
    for name, genes in sets.items():
        a, n, odds, p = fisher_overlap(genes, abi3_targets, background)
        ov_rows.append({"set": name, "n_set": n, "n_abi3_targets": a,
                        "fraction": a / n if n else np.nan, "odds_ratio": odds, "pvalue": p})
    ov = pd.DataFrame(ov_rows)
    ov["qvalue"] = bh_qvalues(ov["pvalue"].to_numpy())
    ov.to_csv(OUT / "abi3_target_overlap.csv", index=False)

    info = pd.read_csv(GENE_INFO, sep="\t", usecols=["Symbol", "LocusTag", "Synonyms"])
    info["LocusTag"] = info["LocusTag"].astype(str).str.upper()
    sym = dict(zip(info["LocusTag"], info["Symbol"]))
    panel_rows = []
    for name, agi, role in PANEL:
        row = {"gene": name, "gene_id": agi, "ncbi_symbol": sym.get(agi, ""), "role": role,
               "tested": agi in background}
        for pair, sub in wide.items():
            if agi in sub.index:
                r = sub.loc[agi]
                row[f"{pair}_single_first"] = r["p"]
                row[f"{pair}_single_second"] = r["q"]
                row[f"{pair}_double"] = r["pq"]
                row[f"{pair}_delta"] = r["delta"]
                row[f"{pair}_delta_q"] = r["delta_qval"]
                row[f"{pair}_type"] = r["interaction_type"]
        panel_rows.append(row)
    panel = pd.DataFrame(panel_rows)
    panel.to_csv(OUT / "literature_panel.csv", index=False)

    summary = {
        "background_tested": len(background),
        "background_bp_annotated": n_bg,
        "go_release": open(OBO, encoding="utf-8").read(200).split("data-version: ")[1].split("\n")[0],
        "n_sets": len(sets),
        "abi3_targets_total": len(abi3_targets),
        "abi3_targets_in_background": len(abi3_targets & background),
        "panel_tested": int(panel["tested"].sum()),
        "panel_total": len(panel),
    }
    (OUT / "function_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    pd.set_option("display.width", 220)
    print(ov.sort_values("pvalue").to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(top.groupby("set").head(4)[["set", "name", "k", "n_set", "fold", "qvalue"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main()
