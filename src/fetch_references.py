"""Download the public reference files used by the annotation scripts.

Files are fetched with HTTP range requests in small pieces so that a reset
connection only costs one piece; every piece is resumed until complete and
the assembled file must match the server's Content-Length.  SHA-256 sums of
the finished files are written to data/raw/external/reference_files.json.

  GO Consortium annotation (TAIR source) and go-basic ontology
  Ensembl Plants release 60 TAIR10 genome sequence and gene annotation
"""

import hashlib
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "raw" / "external"
ENSEMBL = "https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-60/"
FILES = [
    ("https://current.geneontology.org/annotations/tair.gaf.gz", EXT / "GO" / "tair.gaf.gz"),
    ("https://current.geneontology.org/ontology/go-basic.obo", EXT / "GO" / "go-basic.obo"),
    (ENSEMBL + "gff3/arabidopsis_thaliana/Arabidopsis_thaliana.TAIR10.60.gff3.gz",
     EXT / "TAIR10" / "Arabidopsis_thaliana.TAIR10.60.gff3.gz"),
    (ENSEMBL + "fasta/arabidopsis_thaliana/dna/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.gz",
     EXT / "TAIR10" / "Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.gz"),
]
PIECE = 1024 * 1024
WORKERS = 10
HEADERS = {"User-Agent": "Mozilla/5.0 (reference-download)"}
EXPECTED_SHA256 = {
    "tair.gaf.gz": "582ea7a8cbbae9427b6f68909681adae82d5cfb028b3189d00986ad918c23b45",
    "go-basic.obo": "b08d45b268b8c24ccb2513dbbbc7d4df9f6521c099b413f79eb31e06e0fa3bcc",
    "Arabidopsis_thaliana.TAIR10.60.gff3.gz": "aa845e700335e44cfbccfaa233fe1d648bc0b9efa1f8cbfc8c5c5b5550a9eba7",
    "Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.gz": "80a13166003333ba4982b57abdb3d2b62037bc831f4da29ea378c1b776fff8e2",
}


def content_length(url):
    """Total size from the Content-Range of a one-byte ranged GET."""
    req = urllib.request.Request(url, headers={**HEADERS, "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        crange = r.headers.get("Content-Range", "")
        if "/" in crange:
            return int(crange.rsplit("/", 1)[1])
        return int(r.headers["Content-Length"])


def fetch_piece(url, path, start, end):
    want = end - start + 1
    for _ in range(60):
        have = path.stat().st_size if path.exists() else 0
        if have >= want:
            return
        req = urllib.request.Request(url, headers={**HEADERS,
                                                   "Range": f"bytes={start + have}-{end}"})
        try:
            with urllib.request.urlopen(req, timeout=90) as r, open(path, "ab") as fh:
                while True:
                    block = r.read(65536)
                    if not block:
                        break
                    fh.write(block)
        except Exception:
            time.sleep(2)
    raise RuntimeError(f"piece {path.name} incomplete")


def fetch(url, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    size = content_length(url)
    if out.exists() and out.stat().st_size == size:
        return size
    tmp = out.parent / (out.name + ".pieces")
    tmp.mkdir(exist_ok=True)
    jobs = [(url, tmp / f"{i:05d}", s, min(s + PIECE, size) - 1)
            for i, s in enumerate(range(0, size, PIECE))]
    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(lambda j: fetch_piece(*j), jobs))
    with open(out, "wb") as fh:
        for _, p, _, _ in jobs:
            fh.write(p.read_bytes())
    if out.stat().st_size != size:
        raise RuntimeError(f"{out.name}: size mismatch")
    for _, p, _, _ in jobs:
        p.unlink()
    tmp.rmdir()
    return size


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    record = {}
    manifest = EXT / "reference_files.json"
    existing = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    for url, out in FILES:
        t0 = time.time()
        if out.exists() and sha256(out) == EXPECTED_SHA256[out.name]:
            size = out.stat().st_size
        else:
            size = fetch(url, out)
        digest = sha256(out)
        if digest != EXPECTED_SHA256[out.name]:
            raise ValueError(f"{out.name}: upstream release changed; restore the study-pinned "
                             "snapshot rather than silently updating the analysis")
        key = str(out.relative_to(ROOT)).replace(os.sep, "/")
        record[key] = existing.get(key, {
            "url": url, "bytes": size, "sha256": digest,
            "retrieved": time.strftime("%Y-%m-%d"),
        })
        if record[key]["sha256"] != digest:
            raise ValueError("recorded study checksum differs from verified input")
        print(f"{out.name}: {size:,} bytes, study checksum verified", flush=True)
    manifest.write_text(json.dumps(record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
