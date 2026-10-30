#!/usr/bin/env python
"""Download AlphaFold DB models (needs internet). Resumable: existing files are skipped.
   python tools/fetch_alphafold.py --csv clinical_substitutions.csv --out af_models
Proteins > 2700 aa are split into fragments in AlphaFold DB and are skipped (features -> NaN)."""
import argparse
import json
from pathlib import Path

from _common import accessions, get, pmap


def fetch(acc, out):
    if list(out.glob(f"AF-{acc}-F1-model*.pdb")):
        return "cached"
    try:
        meta = json.loads(get(f"https://alphafold.ebi.ac.uk/api/prediction/{acc}"))
        url = meta[0]["pdbUrl"]
        (out / Path(url).name).write_bytes(get(url))
        return "ok"
    except Exception as e:
        return f"fail:{type(e).__name__}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--out", default="af_models")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    ids = accessions(a.csv)
    res = pmap(lambda x: fetch(x, out), ids, a.workers)
    from collections import Counter
    print(Counter(r.split(":")[0] for r in res), "(failures = no AF model / network error; features become NaN)")
