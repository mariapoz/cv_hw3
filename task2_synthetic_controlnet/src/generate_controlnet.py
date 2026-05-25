from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic rare-class images using Stable Diffusion + ControlNet Canny.")
    parser.add_argument("--real-train-dir", type=str, default="data/crops/train")
    parser.add_argument("--metadata-dir", type=str, default="data/metadata")
    parser.add_argument("--output-dir", type=str, default="data/synthetic_controlnet/train")
    parser.add_argument("--examples-dir", type=str, default="artifacts/synthetic_examples")
    parser.add_argument("--base-model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--controlnet-model", type=str, default="lllyasviel/sd-controlnet-canny")
    parser.add_argument("--images-per-class", type=int, default=40)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=1.0)
    parser.add_argument("--canny-low", type=int, default=100)
    parser.add_argument("--canny-high", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prompt-prefix", type=str, default="a realistic high quality photo of")
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="low quality, blurry, distorted, watermark, text, cropped, duplicate, deformed",
    )
    return parser.parse_args()


def load_selected_classes(metadata_dir: Path) -> pd.DataFrame:
    path = metadata_dir / "selected_classes.csv"
    if not path.exists():
        raise FileNotFoundError(f"Selected classes metadata not found: {path}. Run prepare_crops.py first.")
    return pd.read_csv(path)


def make_canny_condition(image: Image.Image, size: int, low: int, high: int) -> Image.Image:
    image = image.convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.array(image)
    edges = cv2.Canny(arr, low, high)
    edges = edges[:, :, None]
    edges = np.concatenate([edges, edges, edges], axis=2)
    return Image.fromarray(edges)


def list_images(class_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted([p for p in class_dir.rglob("*") if p.suffix.lower() in exts])


def save_contact_sheet(image_paths: list[Path], output_path: Path, thumb_size: int = 160) -> None:
    if not image_paths:
        return
    images = [Image.open(p).convert("RGB").resize((thumb_size, thumb_size)) for p in image_paths]
    cols = min(4, len(images))
    rows = int(np.ceil(len(images) / cols))
    sheet = Image.new("RGB", (cols * thumb_size, rows * thumb_size), "white")
    for i, img in enumerate(images):
        x = (i % cols) * thumb_size
        y = (i // cols) * thumb_size
        sheet.paste(img, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=95)


def main() -> None:
    args = parse_args()
    real_train_dir = Path(args.real_train_dir)
    metadata_dir = Path(args.metadata_dir)
    output_dir = Path(args.output_dir)
    examples_dir = Path(args.examples_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)

    selected_df = load_selected_classes(metadata_dir)

    print("Loading Stable Diffusion + ControlNet pipeline...")
    try:
        from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
    except ImportError as exc:
        raise RuntimeError("diffusers is not installed. Run: pip install -r requirements.txt") from exc

    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    controlnet = ControlNetModel.from_pretrained(args.controlnet_model, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.base_model,
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=dtype,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    if args.device.startswith("cuda"):
        pipe.to(args.device)
        pipe.enable_attention_slicing()
    else:
        pipe.to("cpu")

    rng = random.Random(args.seed)
    all_rows = []
    example_paths = []

    for row in selected_df.itertuples(index=False):
        class_slug = str(row.class_slug)
        class_name = str(row.class_name)
        class_real_dir = real_train_dir / class_slug
        source_images = list_images(class_real_dir)
        if not source_images:
            print(f"No real crops found for class {class_slug}; skipping.")
            continue

        class_out_dir = output_dir / class_slug
        class_out_dir.mkdir(parents=True, exist_ok=True)
        prompt = f"{args.prompt_prefix} {class_name}, centered object, natural lighting, detailed"

        for i in tqdm(range(args.images_per_class), desc=f"generate/{class_slug}"):
            source_path = rng.choice(source_images)
            source_img = Image.open(source_path).convert("RGB")
            condition = make_canny_condition(source_img, args.image_size, args.canny_low, args.canny_high)

            generator = torch.Generator(device=args.device if args.device.startswith("cuda") else "cpu").manual_seed(
                args.seed + int(row.label_id) * 10000 + i
            )

            result = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                image=condition,
                width=args.image_size,
                height=args.image_size,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                controlnet_conditioning_scale=args.controlnet_conditioning_scale,
                generator=generator,
            )
            image = result.images[0]

            out_path = class_out_dir / f"synthetic_{class_slug}_{i:05d}.png"
            image.save(out_path)
            if len(example_paths) < 16:
                example_paths.append(out_path)

            all_rows.append(
                {
                    "source": "synthetic_controlnet_canny",
                    "label_id": int(row.label_id),
                    "category_id": int(row.category_id),
                    "class_name": class_name,
                    "class_slug": class_slug,
                    "prompt": prompt,
                    "source_crop": str(source_path),
                    "synthetic_path": str(out_path),
                    "base_model": args.base_model,
                    "controlnet_model": args.controlnet_model,
                    "seed": args.seed + int(row.label_id) * 10000 + i,
                }
            )

    synthetic_meta = pd.DataFrame(all_rows)
    synthetic_meta.to_csv(metadata_dir / "synthetic_metadata.csv", index=False)
    save_contact_sheet(example_paths, examples_dir / "synthetic_examples_grid.jpg")

    print(f"Synthetic images saved to: {output_dir}")
    print(f"Synthetic metadata saved to: {metadata_dir / 'synthetic_metadata.csv'}")
    print(f"Example grid saved to: {examples_dir / 'synthetic_examples_grid.jpg'}")


if __name__ == "__main__":
    main()
