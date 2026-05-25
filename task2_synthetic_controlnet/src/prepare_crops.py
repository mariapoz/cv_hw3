from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from src.coco_utils import (
    categories_by_id,
    crop_box_with_context,
    existing_image_names,
    get_coco_paths,
    images_by_id,
    load_json,
    resolve_data_root,
    sanitize_bbox_xywh,
    slugify,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare object crops from COCO-subset for classification.")
    parser.add_argument("--data-root", type=str, default=None, help="COCO-subset root. If omitted, KaggleHub cache is used.")
    parser.add_argument("--output-dir", type=str, default="data/crops", help="Output directory for real object crops.")
    parser.add_argument("--metadata-dir", type=str, default="data/metadata", help="Directory for metadata CSV/JSON files.")
    parser.add_argument("--num-classes", type=int, default=5, help="Number of rare classes to select.")
    parser.add_argument("--min-train-objects", type=int, default=30, help="Minimum train objects per selected class.")
    parser.add_argument("--max-train-objects", type=int, default=500, help="Maximum train objects per selected class.")
    parser.add_argument("--min-val-objects", type=int, default=5, help="Minimum val objects per selected class.")
    parser.add_argument("--max-train-per-class", type=int, default=250, help="Limit real train crops per class.")
    parser.add_argument("--max-val-per-class", type=int, default=80, help="Limit real val crops per class.")
    parser.add_argument("--min-bbox-size", type=int, default=20, help="Ignore boxes with width/height smaller than this.")
    parser.add_argument("--padding-ratio", type=float, default=0.15, help="Context padding around bounding boxes.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Optional comma-separated class names or COCO category ids to use instead of automatic rare class selection.",
    )
    return parser.parse_args()


def collect_valid_annotations(coco: dict, image_dir: Path, min_bbox_size: int) -> list[dict]:
    imgs = images_by_id(coco)
    existing = existing_image_names(image_dir)
    valid = []

    for ann in coco["annotations"]:
        if int(ann.get("iscrowd", 0)) == 1:
            continue

        image_id = int(ann["image_id"])
        if image_id not in imgs:
            continue

        img_info = imgs[image_id]
        if img_info["file_name"] not in existing:
            continue

        width, height = int(img_info["width"]), int(img_info["height"])
        bbox = sanitize_bbox_xywh(ann["bbox"], width, height)
        if bbox is None:
            continue

        _, _, w, h = bbox
        if w < min_bbox_size or h < min_bbox_size:
            continue

        valid.append(ann)

    return valid


def select_classes(
    train_coco: dict,
    val_coco: dict,
    train_valid: list[dict],
    val_valid: list[dict],
    num_classes: int,
    min_train_objects: int,
    max_train_objects: int,
    min_val_objects: int,
    explicit_classes: str | None,
) -> pd.DataFrame:
    cat_by_id = categories_by_id(train_coco)
    name_to_id = {name.lower(): cat_id for cat_id, name in cat_by_id.items()}

    train_counts = Counter(int(a["category_id"]) for a in train_valid)
    val_counts = Counter(int(a["category_id"]) for a in val_valid)

    if explicit_classes:
        selected_ids = []
        for raw in explicit_classes.split(","):
            item = raw.strip()
            if not item:
                continue
            if item.isdigit():
                cat_id = int(item)
            else:
                cat_id = name_to_id.get(item.lower())
                if cat_id is None:
                    raise ValueError(f"Unknown class name: {item}")
            selected_ids.append(cat_id)
    else:
        candidates = []
        for cat_id, count in train_counts.items():
            val_count = val_counts.get(cat_id, 0)
            if min_train_objects <= count <= max_train_objects and val_count >= min_val_objects:
                candidates.append((cat_id, count, val_count))

        candidates = sorted(candidates, key=lambda x: (x[1], x[0]))
        selected_ids = [cat_id for cat_id, _, _ in candidates[:num_classes]]

    if len(selected_ids) == 0:
        raise RuntimeError("No classes were selected. Try lowering --min-train-objects or --min-val-objects.")

    rows = []
    for idx, cat_id in enumerate(selected_ids):
        rows.append(
            {
                "label_id": idx,
                "category_id": int(cat_id),
                "class_name": cat_by_id[int(cat_id)],
                "class_slug": slugify(cat_by_id[int(cat_id)]),
                "train_objects": int(train_counts.get(int(cat_id), 0)),
                "val_objects": int(val_counts.get(int(cat_id), 0)),
            }
        )
    return pd.DataFrame(rows)


def save_crops_for_split(
    split: str,
    coco: dict,
    valid_annotations: list[dict],
    selected_df: pd.DataFrame,
    image_dir: Path,
    output_dir: Path,
    max_per_class: int,
    padding_ratio: float,
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(seed)
    selected_ids = set(int(x) for x in selected_df["category_id"].tolist())
    cat_to_slug = {int(row.category_id): row.class_slug for row in selected_df.itertuples()}
    cat_to_name = {int(row.category_id): row.class_name for row in selected_df.itertuples()}
    cat_to_label = {int(row.category_id): int(row.label_id) for row in selected_df.itertuples()}

    imgs = images_by_id(coco)
    anns_by_class: Dict[int, List[dict]] = defaultdict(list)
    for ann in valid_annotations:
        cat_id = int(ann["category_id"])
        if cat_id in selected_ids:
            anns_by_class[cat_id].append(ann)

    rows = []
    for cat_id in selected_ids:
        anns = anns_by_class[cat_id]
        rng.shuffle(anns)
        anns = anns[:max_per_class]

        class_slug = cat_to_slug[cat_id]
        class_dir = output_dir / split / class_slug
        class_dir.mkdir(parents=True, exist_ok=True)

        for idx, ann in enumerate(tqdm(anns, desc=f"crop {split}/{class_slug}", leave=False)):
            image_id = int(ann["image_id"])
            img_info = imgs[image_id]
            image_path = image_dir / img_info["file_name"]

            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as exc:
                print(f"Skip unreadable image {image_path}: {exc}")
                continue

            width, height = image.size
            box = crop_box_with_context(ann["bbox"], width, height, padding_ratio=padding_ratio)
            if box is None:
                continue

            crop = image.crop(box)
            if crop.width < 8 or crop.height < 8:
                continue

            out_name = f"{split}_{class_slug}_{idx:05d}_img{image_id}_ann{int(ann.get('id', idx))}.jpg"
            out_path = class_dir / out_name
            crop.save(out_path, quality=95)

            rows.append(
                {
                    "split": split,
                    "source": "real",
                    "label_id": cat_to_label[cat_id],
                    "category_id": cat_id,
                    "class_name": cat_to_name[cat_id],
                    "class_slug": class_slug,
                    "image_id": image_id,
                    "annotation_id": int(ann.get("id", -1)),
                    "source_image": str(image_path),
                    "crop_path": str(out_path),
                    "bbox_xywh": json.dumps(list(map(float, ann["bbox"]))),
                    "crop_xyxy": json.dumps(list(map(int, box))),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    image_dir, train_json, val_json = get_coco_paths(data_root)

    output_dir = Path(args.output_dir)
    metadata_dir = Path(args.metadata_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    train_coco = load_json(train_json)
    val_coco = load_json(val_json)

    print(f"DATA_ROOT: {data_root}")
    print(f"IMAGE_DIR: {image_dir}")
    print("Collecting valid annotations...")

    train_valid = collect_valid_annotations(train_coco, image_dir, args.min_bbox_size)
    val_valid = collect_valid_annotations(val_coco, image_dir, args.min_bbox_size)
    print(f"Valid train annotations: {len(train_valid)}")
    print(f"Valid val annotations: {len(val_valid)}")

    selected_df = select_classes(
        train_coco=train_coco,
        val_coco=val_coco,
        train_valid=train_valid,
        val_valid=val_valid,
        num_classes=args.num_classes,
        min_train_objects=args.min_train_objects,
        max_train_objects=args.max_train_objects,
        min_val_objects=args.min_val_objects,
        explicit_classes=args.classes,
    )
    print("Selected rare classes:")
    print(selected_df)

    selected_df.to_csv(metadata_dir / "selected_classes.csv", index=False)
    selected_df.to_json(metadata_dir / "selected_classes.json", orient="records", force_ascii=False, indent=2)

    train_meta = save_crops_for_split(
        split="train",
        coco=train_coco,
        valid_annotations=train_valid,
        selected_df=selected_df,
        image_dir=image_dir,
        output_dir=output_dir,
        max_per_class=args.max_train_per_class,
        padding_ratio=args.padding_ratio,
        seed=args.seed,
    )
    val_meta = save_crops_for_split(
        split="val",
        coco=val_coco,
        valid_annotations=val_valid,
        selected_df=selected_df,
        image_dir=image_dir,
        output_dir=output_dir,
        max_per_class=args.max_val_per_class,
        padding_ratio=args.padding_ratio,
        seed=args.seed + 1,
    )

    metadata = pd.concat([train_meta, val_meta], ignore_index=True)
    metadata.to_csv(metadata_dir / "real_crops_metadata.csv", index=False)

    print("Saved crops:")
    print(metadata.groupby(["split", "class_slug"]).size().reset_index(name="count"))
    print(f"Metadata saved to: {metadata_dir}")


if __name__ == "__main__":
    main()
