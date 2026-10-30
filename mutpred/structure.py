"""AlphaFold-structure features (dependency-free: numpy + scipy only).

Per residue: pLDDT (B-factor column), burial proxies (CA/CB contact numbers, half-sphere exposure),
backbone dihedrals and a coarse Ramachandran secondary-structure call. WT structure only.
"""
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .constants import THREE_TO_ONE

_T2O = {k.upper(): v for k, v in THREE_TO_ONE.items()}
AF_COLS = ["af_available", "af_plddt", "af_plddt_win", "af_cn10", "af_cn10_pct", "af_cb_cn8", "af_hse_up", "af_hse_down",
           "af_hse_ratio", "af_phi_sin", "af_phi_cos", "af_psi_sin", "af_psi_cos", "af_helix_win", "af_strand_win"]


def parse_pdb(path):
    """First chain only. Returns dict(seq, plddt, N, CA, C, CB) with (L,3) float arrays (NaN if absent)."""
    res, order, chain0 = {}, [], None
    with open(path) as fh:
        for ln in fh:
            if not ln.startswith("ATOM"):
                continue
            ch = ln[21]
            chain0 = chain0 or ch
            if ch != chain0:
                break
            key = (int(ln[22:26]), ln[26])
            if key not in res:
                res[key] = dict(name=ln[17:20].strip().upper(), b=float(ln[60:66]))
                order.append(key)
            res[key][ln[12:16].strip()] = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
    L = len(order)
    out = {k: np.full((L, 3), np.nan) for k in ("N", "CA", "C", "CB")}
    seq, plddt = [], np.zeros(L)
    for i, k in enumerate(order):
        r = res[k]
        seq.append(_T2O.get(r["name"], "X"))
        plddt[i] = r["b"]
        for a in out:
            if a in r:
                out[a][i] = r[a]
    # virtual CB for glycine (ideal geometry)
    for i in np.where(np.isnan(out["CB"][:, 0]))[0]:
        n, ca, c = out["N"][i], out["CA"][i], out["C"][i]
        if not (np.isnan(n).any() or np.isnan(ca).any() or np.isnan(c).any()):
            b, cc = ca - n, c - ca
            a = np.cross(b, cc)
            out["CB"][i] = -0.58273431 * a + 0.56802827 * b - 0.54067466 * cc + ca
    return dict(seq="".join(seq), plddt=plddt, **out)


def dihedral(p0, p1, p2, p3):
    """Signed torsion angle in degrees (vectorised over rows)."""
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - (b0 * b1n).sum(-1, keepdims=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdims=True) * b1n
    x = (v * w).sum(-1)
    y = (np.cross(b1n, v) * w).sum(-1)
    return np.degrees(np.arctan2(y, x))


def residue_features(st):
    L = len(st["seq"])
    CA, CB, N, C = st["CA"], st["CB"], st["N"], st["C"]
    ok = ~np.isnan(CA).any(1)
    tree = cKDTree(np.nan_to_num(CA))
    cn10 = np.array([len(x) - 1 for x in tree.query_ball_point(np.nan_to_num(CA), 10.0)], float)
    cbt = cKDTree(np.nan_to_num(np.where(np.isnan(CB), CA, CB)))
    cb8 = np.array([len(x) - 1 for x in cbt.query_ball_point(np.nan_to_num(np.where(np.isnan(CB), CA, CB)), 8.0)], float)
    up, down = np.zeros(L), np.zeros(L)
    for i, nb in enumerate(tree.query_ball_point(np.nan_to_num(CA), 13.0)):
        nb = [j for j in nb if j != i]
        if not nb or np.isnan(CB[i]).any():
            continue
        u = CB[i] - CA[i]
        d = ((CA[nb] - CA[i]) @ u) > 0
        up[i], down[i] = d.sum(), (~d).sum()
    phi, psi = np.full(L, np.nan), np.full(L, np.nan)
    if L > 2:
        phi[1:] = dihedral(C[:-1], N[1:], CA[1:], C[1:])
        psi[:-1] = dihedral(N[:-1], CA[:-1], C[:-1], N[1:])
    helix = ((phi > -100) & (phi < -30) & (psi > -80) & (psi < -5)).astype(float)
    strand = ((phi < -40) & ((psi > 90) | (psi < -150))).astype(float)
    helix[np.isnan(phi) | np.isnan(psi)] = np.nan
    strand[np.isnan(phi) | np.isnan(psi)] = np.nan

    def win(a, w):
        return pd.Series(a).rolling(2 * w + 1, center=True, min_periods=1).mean().values

    f = pd.DataFrame(dict(
        af_plddt=st["plddt"], af_plddt_win=win(st["plddt"], 5), af_cn10=cn10,
        af_cn10_pct=pd.Series(cn10).rank(pct=True).values, af_cb_cn8=cb8, af_hse_up=up, af_hse_down=down,
        af_hse_ratio=up / (up + down + 1.0),
        af_phi_sin=np.sin(np.radians(phi)), af_phi_cos=np.cos(np.radians(phi)),
        af_psi_sin=np.sin(np.radians(psi)), af_psi_cos=np.cos(np.radians(psi)),
        af_helix_win=win(helix, 3), af_strand_win=win(strand, 3)))
    f.loc[~ok] = np.nan
    return f


class AFStore:
    """Looks up AF-<accession>-F1-model*.pdb files by protein_id (UniProt accession)."""

    def __init__(self, af_dir):
        self.files = {}
        for p in glob.glob(str(Path(af_dir) / "AF-*-F1-model*.pdb")):
            self.files[Path(p).name.split("-")[1]] = p
        self.cache = {}

    def _table(self, pid, seq):
        acc = re.split(r"[-_.]", pid)[0] if pid not in self.files else pid
        if acc not in self.files:
            return None
        if acc not in self.cache:
            st = parse_pdb(self.files[acc])
            self.cache[acc] = (st["seq"], residue_features(st))
        s, tab = self.cache[acc]
        return tab if s == seq else None            # sequence mismatch (isoform / version) -> unavailable

    def features(self, df, log=print):
        out = np.full((len(df), len(AF_COLS)), np.nan)
        out[:, 0] = 0.0
        hit = 0
        for i, (pid, seq, pos) in enumerate(zip(df["protein_id"], df["protein_sequence"], df["pos"])):
            t = self._table(pid, seq)
            if t is not None:
                out[i, 1:] = t.iloc[pos - 1].values
                out[i, 0] = 1.0
                hit += 1
        log(f"[structure] AlphaFold features for {hit}/{len(df)} variants "
            f"({len(self.files)} structure files indexed)")
        return pd.DataFrame(out, columns=AF_COLS, index=df.index)
