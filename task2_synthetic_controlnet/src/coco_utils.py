from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_COCO_CACHE_PATH = Path(
    "/root/.cache/kagglehub/datasets/abdelrahmanelgharibx/coco2017-subset/versions/1"
)
DATASET_SLUG = "abdelrahmanelgharibx/coco2017-subset"


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9а-яё]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "class"


def resolve_data_root(data_root: str | None = None, allow_download: bool = True) -> Path:
    """Resolve COCO-subset root.

    Priority:
    1. explicit CLI argument;
    2. COCO_DATA_ROOT environment variable;
    3. known VM KaggleHub cache path;
    4. kagglehub.dataset_download(...).
    """
    candidates: List[Path] = []

    if data_root:
        candidates.append(Path(data_root).expanduser())

    env_root = os.environ.get("COCO_DATA_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())

    candidates.append(DEFAULT_COCO_CACHE_PATH)

    for candidate in candidates:
        if is_valid_coco_root(candidate):
            return candidate

    if allow_download:
        try:
            import kagglehub
        except ImportError as exc:
            raise RuntimeError(
                "Dataset was not found locally and kagglehub is not installed. "
                "Install requirements or pass --data-root explicitly."
            ) from exc

        path = Path(kagglehub.dataset_download(DATASET_SLUG))
        if is_valid_coco_root(path):
            return path

    checked = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not find COCO-subset dataset. Checked paths:\n"
        f"{checked}\n"
        "Expected files: train2017.json, val2017.json and coco/*.jpg"
    )


def is_valid_coco_root(path: Path) -> bool:
    return (
        path.exists()
        and (path / "train2017.json").exists()
        and (path / "val2017.json").exists()
        and (path / "coco").exists()
    )


def get_coco_paths(data_root: Path) -> Tuple[Path, Path, Path]:
    image_dir = data_root / "coco"
    train_json = data_root / "train2017.json"
    val_json = data_root / "val2017.json"
    return image_dir, train_json, val_json


def existing_image_names(image_dir: Path) -> set[str]:
    return {p.name for p in image_dir.glob("*.jpg")}


def sanitize_bbox_xywh(bbox: Iterable[float], width: int, height: int) -> list[float] | None:
    x, y, w, h = map(float, bbox)
    x1 = max(0.0, min(x, float(width - 1)))
    y1 = max(0.0, min(y, float(height - 1)))
    x2 = max(0.0, min(x + w, float(width)))
    y2 = max(0.0, min(y + h, float(height)))
    new_w = max(0.0, x2 - x1)
    new_h = max(0.0, y2 - y1)
    if new_w <= 1.0 or new_h <= 1.0:
        return None
    return [x1, y1, new_w, new_h]


def crop_box_with_context(
    bbox_xywh: Iterable[float],
    width: int,
    height: int,
    padding_ratio: float = 0.15,
) -> tuple[int, int, int, int] | None:
    clean = sanitize_bbox_xywh(bbox_xywh, width, height)
    if clean is None:
        return None

    x, y, w, h = clean
    pad_x = w * padding_ratio
    pad_y = h * padding_ratio
    x1 = int(max(0, x - pad_x))
    y1 = int(max(0, y - pad_y))
    x2 = int(min(width, x + w + pad_x))
    y2 = int(min(height, y + h + pad_y))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def categories_by_id(coco: dict) -> Dict[int, str]:
    return {int(c["id"]): str(c["name"]) for c in coco["categories"]}


def images_by_id(coco: dict) -> Dict[int, dict]:
    return {int(img["id"]): img for img in coco["images"]}
