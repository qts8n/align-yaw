from dataclasses import MISSING, dataclass, fields
from typing import Any, Dict

import yaml


@dataclass
class AlignConfig:
    pano_ref: str
    pano_late: str
    out: str = "aligned.png"
    maxw: int = 2048
    ransac_iters: int = 2500
    thresh_deg: float = 1.5
    seed: int = 0
    prefer_sift: bool = False
    ratio: float = 0.75
    bottom_mask_frac: float = 0.18
    yolo_model: str = "yolo26s-seg.pt"
    yolo_conf: float = 0.25
    yolo_iou: float = 0.5
    yolo_seam_shifts: int = 3
    mask_people: bool = False
    mask_dilate_px: int = 18
    save_mask: str = ""

    @classmethod
    def from_namespace(cls, ns) -> "AlignConfig":
        """Build AlignConfig from argparse.Namespace-like object, merging optional YAML config."""
        yaml_config = cls._load_yaml(getattr(ns, "config_yaml", None))
        defaults = cls._default_values()
        overrides = cls._cli_overrides(ns)

        known_fields = {f.name for f in fields(cls)}
        unknown_yaml = [k for k in yaml_config if k not in known_fields]
        if unknown_yaml:
            extras = ", ".join(sorted(unknown_yaml))
            raise ValueError(f"Unknown config keys in YAML: {extras}")

        cfg: Dict[str, Any] = {**defaults, **yaml_config, **overrides}

        missing = [k for k in ("pano_ref", "pano_late") if not cfg.get(k)]
        if missing:
            missing_s = ", ".join(missing)
            raise ValueError(f"Missing required config values: {missing_s}")

        return cls(**cfg)

    @classmethod
    def _default_values(cls) -> Dict[str, Any]:
        defaults: Dict[str, Any] = {}
        for f in fields(cls):
            if f.default is not MISSING:
                defaults[f.name] = f.default
            elif f.default_factory is not MISSING:  # type: ignore[attr-defined]
                defaults[f.name] = f.default_factory()
        return defaults

    @staticmethod
    def _load_yaml(path: str | None) -> Dict[str, Any]:
        if not path:
            return {}
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError("YAML config must contain a mapping at the top level")
        return data

    @staticmethod
    def _cli_overrides(ns) -> Dict[str, Any]:
        overrides: Dict[str, Any] = {}
        for key, val in vars(ns).items():
            if key == "config_yaml":
                continue
            if val is None:
                continue
            overrides[key] = val
        return overrides
