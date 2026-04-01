from __future__ import annotations



import copy
import re
import shutil
from pathlib import Path

def root_workspace(cfg_root: Dict[str, Any]) -> Path:
    return Path(cfg_root["project"]["workspace_root"]).resolve()

def root_artifact_root(cfg_root: Dict[str, Any]) -> Path:
    path = root_workspace(cfg_root) / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path

def root_results_json_path(cfg_root: Dict[str, Any]) -> Path:
    return root_artifact_root(cfg_root) / "seq_only_results_ver15.json"

def shared_download_root(cfg_root: Dict[str, Any]) -> Path:
    path = Path(cfg_root["project"]["download_root"]).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path

def _domain_display_name(domain: str) -> str:
    domain = str(domain).lower().strip()
    mapping = {
        "cb": "creative birds",
        "cc": "creative creatures",
    }
    if domain not in mapping:
        raise ValueError(f"Unknown CreativeSketch domain: {domain}")
    return mapping[domain]

def _scaled_cap(base_cap: int, fraction: float) -> int:
    base_cap = int(base_cap)
    fraction = float(fraction)
    return max(1, int(round(base_cap * max(0.0, min(1.0, fraction)))))

def build_stage_cfg(cfg_root: Dict[str, Any], stage_name: str, stage_kind: str, domains: Sequence[str] | None = None) -> Dict[str, Any]:
    out = copy.deepcopy(cfg_root)
    defaults = dict(out["dataset_defaults"])
    base_dir = root_workspace(cfg_root) / stage_name

    if stage_kind == "quickdraw":
        frac = float(out["target"]["dataset_fraction"])
        rep_cap = _scaled_cap(int(out["target"]["representation_max_drawings_per_class"]), frac)
        gen_cap = _scaled_cap(int(out["target"]["generator_max_drawings_per_class"]), frac)
        out["dataset"] = {
            "base_dir": str(base_dir),
            "dataset_kind": "quickdraw",
            "variant": str(out["target"]["variant"]),
            "raw_url_template": str(out["target"]["raw_url_template"]),
            "simplified_url_template": str(out["target"]["simplified_url_template"]),
            "classes": list(out["target"]["classes"]),
            "representation_max_drawings_per_class": int(rep_cap),
            "generator_max_drawings_per_class": int(gen_cap),
            "train_ratio": float(out["target"]["train_ratio"]),
            "val_ratio": float(out["target"]["val_ratio"]),
            "test_ratio": float(out["target"]["test_ratio"]),
            "filter_recognized": bool(out["target"]["filter_recognized"]),
            "quickdraw_canonical_stroke_order": str(out["dataset_defaults"].get("quickdraw_canonical_stroke_order", "none")),
            **defaults,
        }
    elif stage_kind == "creativesketch":
        domains = [str(x).lower().strip() for x in (domains or [])]
        if not domains:
            raise ValueError(f"{stage_name}: domains must be non-empty for CreativeSketch stage")
        use_domain_as_class = bool(out["source"]["use_domain_as_class_label"])
        classes = [_domain_display_name(d) for d in domains] if use_domain_as_class else ["source"]
        frac = 1.0
        if "shape_ae" in stage_name:
            frac = float(out["source"]["shape_ae_fraction"])
        elif "tokenizer" in stage_name or "location" in stage_name:
            frac = float(out["source"]["tokenizer_fraction"])
        out["dataset"] = {
            "base_dir": str(base_dir),
            "dataset_kind": "creativesketch",
            "variant": "simplified",
            "classes": list(classes),
            "source_domains": list(domains),
            "source_fraction": float(frac),
            "use_domain_as_class_label": bool(use_domain_as_class),
            "representation_max_drawings_per_class": 10**9,
            "generator_max_drawings_per_class": 10**9,
            "train_ratio": float(out["source"]["train_ratio"]),
            "val_ratio": float(out["source"]["val_ratio"]),
            "test_ratio": float(out["source"]["test_ratio"]),
            "filter_recognized": False,
            **defaults,
        }
    else:
        raise ValueError(f"Unknown stage_kind: {stage_kind}")

    out["project"]["stage_name"] = str(stage_name)
    out["project"]["config_signature"] = str(cfg_root["project"]["config_signature"])
    return out

def describe_stage_cfg(cfg_stage: Dict[str, Any]) -> None:
    print(
        f"[stage] {cfg_stage['project'].get('stage_name', '?')} | "
        f"kind={cfg_stage['dataset']['dataset_kind']} | "
        f"classes={cfg_stage['dataset']['classes']} | "
        f"max_strokes={cfg_stage['dataset']['max_strokes']}"
    )

def copy_embedding_stats_between_cfgs(src_cfg: Dict[str, Any], dst_cfg: Dict[str, Any]) -> Path:
    ensure_workspace(dst_cfg)
    src = embedding_stats_path(src_cfg)
    dst = embedding_stats_path(dst_cfg)
    if not src.exists():
        raise FileNotFoundError(f"Source embedding stats not found: {src}")
    shutil.copy2(src, dst)
    _STATS_CACHE.pop(str(dst), None)
    return dst

