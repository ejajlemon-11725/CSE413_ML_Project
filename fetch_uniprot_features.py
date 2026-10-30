#!/usr/bin/env python
"""Download UniProt feature annotations (needs internet) -> annotations.csv (+ domains.csv for track plots).
   python tools/fetch_uniprot_features.py --csv clinical_substitutions.csv --out annotations.csv"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mutpred.annotations import GROUPS, parse_uniprot_json
from _common import accessions, get, pmap


def fetch(acc):
    try:
        obj = json.loads(get(f"https://rest.uniprot.org/uniprotkb/{acc}.json"))
        return parse_uniprot_json(acc, obj)
    except Exception:
        return []


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--out", default="annotations.csv")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    rows = [r for rs in pmap(fetch, accessions(a.csv), a.workers) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    dom = df[df["type"].isin(GROUPS["domain"] | {"Region"})].rename(columns={"description": "name"})
    dom[["protein_id", "start", "end", "name"]].to_csv(Path(a.out).with_name("domains.csv"), index=False)
    print(f"{df.protein_id.nunique()} proteins, {len(df)} rows -> {a.out} (+ domains.csv)")
