"""Evaluation script for CASIA-B gait recognition model."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report
from torch.utils.data import DataLoader

from src.model import GaitRecognitionModel
from src.train import build_dataloaders, load_config, set_seed


def evaluate_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    model.eval()
    all_preds = []
    all_labels = []

    print("Running evaluation...")
    with torch.inference_mode():
        for batch in loader:
            clips = batch["clip"].to(device)
            labels = batch["label"].to(device)
            
            logits, _ = model(clips)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    # Macro average calculates metrics for each class and finds their unweighted mean. 
    # This does not take label imbalance into account.
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, zero_division=0))

    return {
        "Accuracy": acc,
        "Recall (Macro)": recall,
        "F1 Score (Macro)": f1
    }


def main(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Load Config
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set seed for reproducibility (to get same data split)
    set_seed(int(cfg.get("experiment", {}).get("seed", 42)))

    # Build Dataloaders (We only need val_loader)
    # Note: This assumes the config defines the split same as training
    _, val_loader, num_classes = build_dataloaders(cfg)
    
    print(f"Validation set size: {len(val_loader.dataset)}")
    print(f"Number of classes: {num_classes}")

    # Initialize Model
    model_cfg = cfg.get("model", {})
    model = GaitRecognitionModel(
        num_classes=num_classes,
        in_channels=int(model_cfg.get("in_channels", 1)),
        frame_feature_dims=tuple(model_cfg.get("frame_feature_dims", [32, 64, 128])),
        pyramid_bins=tuple(model_cfg.get("pyramid_bins", [1, 2, 4])),
        dropout=float(model_cfg.get("dropout", 0.3)),
    )

    # Load Checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    # Evaluate
    metrics = evaluate_metrics(model, val_loader, device)
    
    print("-" * 30)
    print(f"Evaluation Results:")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"Recall:   {metrics['Recall (Macro)']:.4f}")
    print(f"F1 Score: {metrics['F1 Score (Macro)']:.4f}")
    print("-" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Gait Recognition Model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()
    main(args)
