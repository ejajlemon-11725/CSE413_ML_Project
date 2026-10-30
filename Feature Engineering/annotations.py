"""UniProt feature annotations (domains, functional sites, PTMs, topology) -> per-variant features.
Variant-type features ('Natural variant', 'Mutagenesis') are deliberately NOT used (label circularity)."""
import pandas as pd
import numpy as np

GROUPS = {
    "domain": {"Domain", "Repeat", "Zinc finger", "DNA binding", "Coiled coil"},
    "region": {"Motif", "Region", "Compositional bias"},
    "func": {"Active site", "Binding site", "Metal binding", "Site"},
    "disulfide": {"Disulfide bond", "Cross-link"},
    "ptm": {"Modified residue", "Glycosylation", "Lipidation"},
    "tm": {"Transmembrane", "Intramembrane"},
    "signal": {"Signal", "Transit peptide", "Propeptide"},
}
ANN_COLS = ["ann_available", "ann_in_domain", "ann_in_region", "ann_in_func_site", "ann_dist_func_site",
            "ann_in_disulfide", "ann_in_ptm", "ann_in_tm", "ann_in_signal", "ann_n_overlap"]


def parse_uniprot_json(acc, obj):
    """Rows (protein_id,type,start,end,description) from a UniProt REST entry (always emits a FETCHED marker)."""
    rows = [dict(protein_id=acc, type="FETCHED", start=0, end=0, description="")]
    for f in obj.get("features", []):
        loc = f.get("location", {})
        try:
            s, e = int(loc["start"]["value"]), int(loc["end"]["value"])
        except (KeyError, TypeError, ValueError):
            continue
        rows.append(dict(protein_id=acc, type=f.get("type", ""), start=s, end=e, description=f.get("description", "")))
    return rows


class AnnotationStore:
    def __init__(self, csv_path):
        a = pd.read_csv(csv_path)
        self.fetched = set(a.loc[a["type"] == "FETCHED", "protein_id"])
        self.by = {k: v for k, v in a[a["type"] != "FETCHED"].groupby("protein_id")}

    def features(self, df, log=print):
        out = np.full((len(df), len(ANN_COLS)), np.nan)
        out[:, 0] = 0.0
        hit = 0
        for i, (pid, pos) in enumerate(zip(df["protein_id"], df["pos"])):
            if pid not in self.fetched and pid not in self.by:
                continue
            hit += 1
            t = self.by.get(pid)
            row = dict(ann_available=1.0, ann_in_domain=0.0, ann_in_region=0.0, ann_in_func_site=0.0,
                       ann_dist_func_site=50.0, ann_in_disulfide=0.0, ann_in_ptm=0.0, ann_in_tm=0.0,
                       ann_in_signal=0.0, ann_n_overlap=0.0)
            if t is not None:
                ov = t[(t.start <= pos) & (t.end >= pos)]
                row["ann_n_overlap"] = float(len(ov))
                for g, key in (("domain", "ann_in_domain"), ("region", "ann_in_region"), ("func", "ann_in_func_site"),
                               ("disulfide", "ann_in_disulfide"), ("ptm", "ann_in_ptm"), ("tm", "ann_in_tm"),
                               ("signal", "ann_in_signal")):
                    row[key] = float(ov["type"].isin(GROUPS[g]).any())
                fs = t[t["type"].isin(GROUPS["func"] | GROUPS["disulfide"])]
                if len(fs):
                    d = np.minimum(np.abs(fs.start.values - pos), np.abs(fs.end.values - pos))
                    d[(fs.start.values <= pos) & (fs.end.values >= pos)] = 0
                    row["ann_dist_func_site"] = float(min(d.min(), 50))
            out[i] = [row[c] for c in ANN_COLS]
        log(f"[annotation] UniProt features for {hit}/{len(df)} variants")
        return pd.DataFrame(out, columns=ANN_COLS, index=df.index)
