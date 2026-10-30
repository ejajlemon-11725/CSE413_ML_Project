"""Optional feature blocks built from external files (structure / conservation / annotation)."""


def build_extra_blocks(df, cfg, log=print):
    blocks = {}
    if getattr(cfg, "af_dir", None):
        from .structure import AFStore
        blocks["struct"] = AFStore(cfg.af_dir).features(df, log)
    if getattr(cfg, "msa_dir", None):
        from .conservation import MSAStore
        blocks["cons"] = MSAStore(cfg.msa_dir).features(df, log)
    if getattr(cfg, "annot_csv", None):
        from .annotations import AnnotationStore
        blocks["annot"] = AnnotationStore(cfg.annot_csv).features(df, log)
    return blocks
