"""Training script for CASIA-B silhouette gait recognition."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data import CasiaBSilhouetteDataset, build_subject_label_map, discover_subject_ids, split_subjects
from src.model import GaitRecognitionModel


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_datasets(cfg: Dict[str, Any]) -> Tuple[CasiaBSilhouetteDataset, CasiaBSilhouetteDataset]:
    data_cfg = cfg.get("data", {})
    root = Path(data_cfg.get("root", "data/GaitDatasetB-silh")).expanduser()
    all_subjects = discover_subject_ids(root)
    if not all_subjects:
        raise RuntimeError(f"No subject folders discovered under {root}.")

    if data_cfg.get("train_subjects") and data_cfg.get("val_subjects"):
        train_subjects = [f"{int(s):03d}" for s in data_cfg["train_subjects"]]
        val_subjects = [f"{int(s):03d}" for s in data_cfg["val_subjects"]]
    else:
        train_ratio = float(data_cfg.get("train_ratio", 0.8))
        seed = int(data_cfg.get("split_seed", 42))
        train_subjects, val_subjects = split_subjects(all_subjects, train_ratio=train_ratio, seed=seed)

    label_mapping = build_subject_label_map(all_subjects)
    shared_kwargs = {
        "root": root,
        "label_mapping": label_mapping,
        "frames_per_clip": int(data_cfg.get("frames_per_clip", 30)),
        "min_frames": int(data_cfg.get("min_frames", 8)),
        "sampling_strategy": data_cfg.get("sampling_strategy", "uniform"),
    }

    train_dataset = CasiaBSilhouetteDataset(subjects=train_subjects, **shared_kwargs)
    val_dataset = CasiaBSilhouetteDataset(subjects=val_subjects, **shared_kwargs)
    return train_dataset, val_dataset


def build_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, int]:
    train_ds, val_ds = prepare_datasets(cfg)
    data_cfg = cfg.get("data", {})
    batch_size = int(data_cfg.get("batch_size", 8))
    num_workers = int(data_cfg.get("num_workers", 4))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, train_ds.num_classes


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter | None = None,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for i, batch in enumerate(loader):
        clips = batch["clip"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(clips)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = clips.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()

        if writer is not None:
            global_step = (epoch - 1) * len(loader) + i
            writer.add_scalar("Train/BatchLoss", loss.item(), global_step)

    return total_loss / max(1, total_samples), total_correct / max(1, total_samples)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.inference_mode():
        for batch in loader:
            clips = batch["clip"].to(device)
            labels = batch["label"].to(device)
            logits, _ = model(clips)
            loss = criterion(logits, labels)
            batch_size = clips.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
    return total_loss / max(1, total_samples), total_correct / max(1, total_samples)


def save_checkpoint(state: Dict[str, Any], output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_dir / filename)


def main(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    cfg = load_config(config_path)
    train_loader, val_loader, num_classes = build_dataloaders(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(cfg.get("experiment", {}).get("seed", 42)))

    model_cfg = cfg.get("model", {})
    model = GaitRecognitionModel(
        num_classes=num_classes,
        in_channels=int(model_cfg.get("in_channels", 1)),
        frame_feature_dims=tuple(model_cfg.get("frame_feature_dims", [32, 64, 128])),
        pyramid_bins=tuple(model_cfg.get("pyramid_bins", [1, 2, 4])),
        dropout=float(model_cfg.get("dropout", 0.3)),
    ).to(device)

    optim_cfg = cfg.get("optim", {})
    criterion = nn.CrossEntropyLoss(label_smoothing=float(optim_cfg.get("label_smoothing", 0.0)))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(optim_cfg.get("lr", 3e-4)),
        weight_decay=float(optim_cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(args.epochs or optim_cfg.get("epochs", 50))

    output_dir = Path(cfg.get("experiment", {}).get("output_dir", "runs/exp"))
    history = []
    best_acc = 0.0

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, writer)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Log epoch metrics
        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Train/Accuracy", train_acc, epoch)
        writer.add_scalar("Val/Loss", val_loss, epoch)
        writer.add_scalar("Val/Accuracy", val_acc, epoch)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}",
            flush=True,
        )
        save_checkpoint(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": cfg,
                "metrics": history[-1],
            },
            output_dir,
            "latest.pt",
        )
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": cfg,
                    "metrics": history[-1],
                },
                output_dir,
                "best.pt",
            )

    with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GaitSet-inspired model on CASIA-B silhouettes.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to YAML config file.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs.")
    cli_args = parser.parse_args()
    main(cli_args)
