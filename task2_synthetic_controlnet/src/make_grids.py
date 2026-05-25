from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create grids of real and synthetic samples.")
    parser.add_argument("--real-dir", type=str, default="data/crops/train")
    parser.add_argument("--synthetic-dir", type=str, default="data/synthetic_controlnet/train")
    parser.add_argument("--output-dir", type=str, default="artifacts/synthetic_examples")
    parser.add_argument("--samples-per-class", type=int, default=4)
    parser.add_argument("--thumb-size", type=int, default=160)
    return parser.parse_args()


def list_images(root: Path):
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS])


def create_grid(real_dir: Path, synthetic_dir: Path, out_path: Path, samples_per_class: int, thumb: int):
    classes = sorted([p.name for p in real_dir.iterdir() if p.is_dir()])
    rows = []
    for cls in classes:
        real_imgs = list_images(real_dir / cls)[:samples_per_class]
        syn_imgs = list_images(synthetic_dir / cls)[:samples_per_class]
        rows.append((cls, real_imgs, syn_imgs))

    cols = 1 + samples_per_class * 2
    cell_w = thumb
    cell_h = thumb
    label_w = 180
    width = label_w + (cols - 1) * cell_w
    height = max(1, len(rows)) * cell_h
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    for r, (cls, real_imgs, syn_imgs) in enumerate(rows):
        y = r * cell_h
        draw.text((8, y + 8), cls, fill="black")
        x = label_w
        for p in real_imgs:
            img = Image.open(p).convert("RGB").resize((thumb, thumb))
            canvas.paste(img, (x, y))
            x += cell_w
        for p in syn_imgs:
            img = Image.open(p).convert("RGB").resize((thumb, thumb))
            canvas.paste(img, (x, y))
            x += cell_w

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=95)


def main():
    args = parse_args()
    create_grid(
        real_dir=Path(args.real_dir),
        synthetic_dir=Path(args.synthetic_dir),
        out_path=Path(args.output_dir) / "real_vs_synthetic_grid.jpg",
        samples_per_class=args.samples_per_class,
        thumb=args.thumb_size,
    )


if __name__ == "__main__":
    main()
