from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from tqdm.auto import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


class FixedClassImageFolder(Dataset):
    """A lightweight ImageFolder with a fixed class_to_idx mapping.

    This avoids issues when a synthetic directory contains only a subset of classes.
    """

    def __init__(self, root: str | Path, class_to_idx: Dict[str, int], transform=None):
        self.root = Path(root)
        self.class_to_idx = dict(class_to_idx)
        self.classes = [c for c, _ in sorted(self.class_to_idx.items(), key=lambda x: x[1])]
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []

        if self.root.exists():
            for class_name, label in self.class_to_idx.items():
                class_dir = self.root / class_name
                if not class_dir.exists():
                    continue
                for path in sorted(class_dir.rglob("*")):
                    if path.suffix.lower() in IMG_EXTS:
                        self.samples.append((path, label))

        if not self.samples:
            raise RuntimeError(f"No images found in {self.root}")

        self.targets = [label for _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet18 classifier with/without synthetic data.")
    parser.add_argument("--real-train-dir", type=str, default="data/crops/train")
    parser.add_argument("--real-val-dir", type=str, default="data/crops/val")
    parser.add_argument("--synthetic-train-dir", type=str, default="data/synthetic_controlnet/train")
    parser.add_argument("--metadata-dir", type=str, default="data/metadata")
    parser.add_argument("--output-dir", type=str, default="artifacts/metrics")
    parser.add_argument("--plots-dir", type=str, default="artifacts/plots")
    parser.add_argument("--tensorboard-dir", type=str, default="artifacts/tensorboard")
    parser.add_argument("--experiment", choices=["baseline", "synthetic", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-model", action="store_true", help="Save best model weights. Disabled by default to keep repo small.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms(image_size: int):
    train_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tfms, val_tfms


def load_class_mapping(metadata_dir: Path, real_train_dir: Path) -> tuple[list[str], dict[str, int], dict[int, str]]:
    selected_path = metadata_dir / "selected_classes.csv"
    if selected_path.exists():
        selected = pd.read_csv(selected_path).sort_values("label_id")
        classes = selected["class_slug"].astype(str).tolist()
        id_to_name = {int(row.label_id): str(row.class_name) for row in selected.itertuples()}
    else:
        classes = sorted([p.name for p in real_train_dir.iterdir() if p.is_dir()])
        id_to_name = {idx: name for idx, name in enumerate(classes)}

    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    return classes, class_to_idx, id_to_name


def build_model(num_classes: int, pretrained: bool) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def compute_class_weights(targets: list[int], num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(np.array(targets), minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict, list[int], list[int]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    total_loss = 0.0
    n_batches = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            total_loss += float(loss.item())
            n_batches += 1

    metrics = {
        "val_loss": total_loss / max(1, n_batches),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    return metrics, y_true, y_pred


def plot_confusion(cm: np.ndarray, labels: list[str], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), max(5, len(labels) * 1.0)))
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run_experiment(
    name: str,
    train_dataset: Dataset,
    val_dataset: Dataset,
    class_names: list[str],
    id_to_name: dict[int, str],
    args: argparse.Namespace,
    output_dir: Path,
    plots_dir: Path,
    tensorboard_dir: Path,
) -> dict:
    device = torch.device(args.device)
    model = build_model(num_classes=len(class_names), pretrained=args.pretrained).to(device)

    targets = []
    if isinstance(train_dataset, ConcatDataset):
        for ds in train_dataset.datasets:
            targets.extend(getattr(ds, "targets", []))
    else:
        targets = getattr(train_dataset, "targets", [])
    class_weights = compute_class_weights(targets, len(class_names), device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    writer = SummaryWriter(log_dir=str(tensorboard_dir / name))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    best_macro_f1 = -1.0
    best_metrics = None
    history = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"{name} epoch {epoch}/{args.epochs}")

        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = running_loss / max(1, n_batches)
        val_metrics, y_true, y_pred = evaluate(model, val_loader, device)
        row = {"experiment": name, "epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(row)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_metrics["val_loss"], epoch)
        writer.add_scalar("metrics/accuracy", val_metrics["accuracy"], epoch)
        writer.add_scalar("metrics/macro_f1", val_metrics["macro_f1"], epoch)

        print(row)
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_metrics = row.copy()
            if args.save_model:
                model_path = output_dir / f"{name}_best_model.pt"
                torch.save(model.state_dict(), model_path)

    writer.flush()
    writer.close()

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / f"{name}_history.csv", index=False)

    final_metrics, y_true, y_pred = evaluate(model, val_loader, device)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=[id_to_name.get(i, class_names[i]) for i in range(len(class_names))],
        zero_division=0,
        output_dict=True,
    )
    pd.DataFrame(report).transpose().to_csv(output_dir / f"{name}_classification_report.csv")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(output_dir / f"{name}_confusion_matrix.csv")
    plot_confusion(cm, class_names, plots_dir / f"{name}_confusion_matrix.png", f"Confusion matrix: {name}")

    summary = {
        "experiment": name,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "epochs": args.epochs,
        "best_macro_f1": best_macro_f1,
        "final_accuracy": final_metrics["accuracy"],
        "final_macro_f1": final_metrics["macro_f1"],
        "final_weighted_f1": final_metrics["weighted_f1"],
        "runtime_sec": time.time() - start_time,
    }
    if best_metrics is not None:
        summary.update({f"best_{k}": v for k, v in best_metrics.items() if k not in {"experiment"}})

    with open(output_dir / f"{name}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    real_train_dir = Path(args.real_train_dir)
    real_val_dir = Path(args.real_val_dir)
    synthetic_train_dir = Path(args.synthetic_train_dir)
    metadata_dir = Path(args.metadata_dir)
    output_dir = Path(args.output_dir)
    plots_dir = Path(args.plots_dir)
    tensorboard_dir = Path(args.tensorboard_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    class_names, class_to_idx, id_to_name = load_class_mapping(metadata_dir, real_train_dir)
    print("Classes:", class_names)
    print("Device:", args.device)

    train_tfms, val_tfms = build_transforms(args.image_size)

    real_train = FixedClassImageFolder(real_train_dir, class_to_idx, transform=train_tfms)
    val_dataset = FixedClassImageFolder(real_val_dir, class_to_idx, transform=val_tfms)

    experiments = []
    if args.experiment in {"baseline", "both"}:
        experiments.append(("baseline_real_only", real_train))

    if args.experiment in {"synthetic", "both"}:
        synthetic_train = FixedClassImageFolder(synthetic_train_dir, class_to_idx, transform=train_tfms)
        combined = ConcatDataset([real_train, synthetic_train])
        experiments.append(("real_plus_synthetic", combined))

    summaries = []
    for name, train_dataset in experiments:
        summary = run_experiment(
            name=name,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            class_names=class_names,
            id_to_name=id_to_name,
            args=args,
            output_dir=output_dir,
            plots_dir=plots_dir,
            tensorboard_dir=tensorboard_dir,
        )
        summaries.append(summary)

    ablation = pd.DataFrame(summaries)
    ablation.to_csv(output_dir / "ablation_results.csv", index=False)
    print("Ablation results:")
    print(ablation)
    print(f"Saved metrics to: {output_dir}")


if __name__ == "__main__":
    main()
