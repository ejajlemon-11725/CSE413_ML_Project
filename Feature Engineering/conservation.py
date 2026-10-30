"""MSA conservation features from A3M files (Henikoff sequence weights, pseudocounts, PSSM log-odds)."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import AA, AA_INDEX

_BG = np.array([0.0825, 0.0553, 0.0406, 0.0545, 0.0137, 0.0393, 0.0675, 0.0707, 0.0227, 0.0596,
                0.0966, 0.0584, 0.0242, 0.0386, 0.0470, 0.0656, 0.0534, 0.0108, 0.0292, 0.0687])
_BG = _BG / _BG.sum()
CONS_COLS = ["cons_available", "cons_entropy", "cons_entropy_win", "cons_gap", "cons_pssm_wt", "cons_pssm_mut",
             "cons_pssm_delta", "cons_llr", "cons_log_neff"]


def read_a3m(path, max_seqs=3000):
    seqs, cur = [], []
    for ln in open(path):
        ln = ln.rstrip()
        if ln.startswith(">"):
            if cur:
                seqs.append("".join(cur))
            cur = []
        elif ln and not ln.startswith("#"):
            cur.append(re.sub(r"[a-z]", "", ln))          # lowercase = insertions relative to the query
    if cur:
        seqs.append("".join(cur))
    return seqs[:max_seqs]


def msa_profile(seqs, pseudo=5.0):
    L = len(seqs[0])
    seqs = [s for s in seqs if len(s) == L]
    M = np.full((len(seqs), L), 20, dtype=np.int8)         # 20 = gap / unknown
    for i, s in enumerate(seqs):
        M[i] = [AA_INDEX.get(c, 20) for c in s]
    # Henikoff position-based weights
    w = np.zeros(len(seqs))
    for j in range(L):
        vals, inv, cnt = np.unique(M[:, j], return_inverse=True, return_counts=True)
        w += 1.0 / (len(vals) * cnt[inv])
    w /= w.sum()
    counts = np.zeros((L, 21))
    for a in range(21):
        counts[:, a] = ((M == a) * w[:, None]).sum(0)
    gap = counts[:, 20]
    neff_pos = counts[:, :20].sum(1) * len(seqs)               # effective #seqs informative at the position
    f = (counts[:, :20] * len(seqs) + pseudo * _BG) / (neff_pos[:, None] + pseudo)
    ent = -(f * np.log2(f)).sum(1)
    neff = float(np.exp(-(w * np.log(w + 1e-12)).sum()))
    return dict(freq=f, entropy=ent, gap=gap, neff=neff)


class MSAStore:
    """Expects <msa_dir>/<protein_key or protein_id>.a3m (query = first sequence)."""

    def __init__(self, msa_dir):
        self.dir = Path(msa_dir)
        self.cache = {}

    def _profile(self, pk, pid, seq):
        if pk in self.cache:
            return self.cache[pk]
        prof = None
        for name in (pk, pid):
            p = self.dir / (re.sub(r"[^\w.\-]", "_", str(name)) + ".a3m")
            if p.exists():
                seqs = read_a3m(p)
                if seqs and re.sub(r"-", "", seqs[0]).upper() == seq and len(seqs) > 1:
                    prof = msa_profile(seqs)
                break
        self.cache[pk] = prof
        return prof

    def features(self, df, log=print):
        out = np.full((len(df), len(CONS_COLS)), np.nan)
        out[:, 0] = 0.0
        hit = 0
        for i, (pk, pid, seq, pos, wt, mt) in enumerate(zip(df["protein_key"], df["protein_id"], df["protein_sequence"],
                                                           df["pos"], df["wt_aa"], df["mut_aa"])):
            p = self._profile(pk, pid, seq)
            if p is None:
                continue
            j, f = pos - 1, p["freq"]
            iw, im = AA_INDEX[wt], AA_INDEX[mt]
            win = p["entropy"][max(0, j - 3): j + 4].mean()
            out[i] = [1.0, p["entropy"][j], win, p["gap"][j], np.log2(f[j, iw] / _BG[iw]), np.log2(f[j, im] / _BG[im]),
                      np.log2(f[j, im] / _BG[im]) - np.log2(f[j, iw] / _BG[iw]), np.log(f[j, im] / f[j, iw]),
                      np.log(p["neff"] + 1)]
            hit += 1
        log(f"[conservation] MSA features for {hit}/{len(df)} variants")
        return pd.DataFrame(out, columns=CONS_COLS, index=df.index)
