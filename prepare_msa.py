#!/usr/bin/env python
"""Prepare MSA generation. Writes msa_input.fasta (one entry per unique protein) and prints the search command.
MSAs are expensive and need a sequence database (UniRef30 / ColabFold DB) - run them on a server/cluster:

   python tools/prepare_msa.py --csv clinical_substitutions.csv --out msa_work
   colabfold_search msa_work/msa_input.fasta <path/to/colabfold_db> msa_work/raw      # ColabFold (MMseqs2)
   python tools/prepare_msa.py --rename msa_work                                       # -> msa_work/msa/<protein_key>.a3m

Then:  python run_pipeline.py ... --msa-dir msa_work/msa
(Check your colabfold_search version's output naming: files are matched by the order in index.tsv, or by header id.)"""
import argparse
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--out", default="msa_work"); ap.add_argument("--rename")
    a = ap.parse_args()
    if a.rename:
        w = Path(a.rename); (w / "msa").mkdir(exist_ok=True)
        idx = pd.read_csv(w / "index.tsv", sep="\t")
        n = 0
        for i, key in enumerate(idx["key"]):
            for cand in (w / "raw" / f"{key}.a3m", w / "raw" / f"{i}.a3m", w / "raw" / f"{i:06d}.a3m"):
                if cand.exists():
                    shutil.copy(cand, w / "msa" / f"{key}.a3m"); n += 1; break
        print(f"{n}/{len(idx)} a3m files collected into {w/'msa'}")
        sys.exit()
    df = pd.read_csv(a.csv).drop_duplicates("protein_id")
    out = Path(a.out); out.mkdir(exist_ok=True)
    keys = [re.sub(r"[^\w.\-]", "_", str(p)) for p in df.protein_id]
    with open(out / "msa_input.fasta", "w") as f:
        for k, s in zip(keys, df.protein_sequence.str.upper()):
            f.write(f">{k}\n{s}\n")
    pd.DataFrame({"key": keys}).to_csv(out / "index.tsv", sep="\t", index=False)
    print(f"wrote {len(keys)} sequences to {out/'msa_input.fasta'}")
