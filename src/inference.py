"""Inference script for CASIA-B silhouette gait recognition."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from PIL import Image
from torch import nn
from torchvision import transforms

from src.model import GaitRecognitionModel


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def preprocess_sequence(image_dir: Path, transform: Any, min_frames: int = 8) -> torch.Tensor | None:
    """Load and preprocess a sequence of silhouette images from a directory."""
    if not image_dir.is_dir():
        print(f"Error: {image_dir} is not a directory.")
        return None

    # Find all images
    image_paths = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))
    if len(image_paths) < min_frames:
        print(f"Warning: Not enough frames in {image_dir} (found {len(image_paths)}, required {min_frames}).")
        return None

    frames = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("L")  # Convert to grayscale (silhouette)
            frames.append(transform(img))
        except Exception as e:
            print(f"Error loading {p}: {e}")

    if not frames:
        return None

    # Stack frames: (T, C, H, W)
    clip = torch.stack(frames)
    # Add batch dimension: (1, T, C, H, W)
    return clip.unsqueeze(0)


def main(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    input_dir = Path(args.input_dir)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Load Config
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Model
    # We need to know num_classes to initialize the model, even if we only care about embeddings or if we want to predict class.
    # If the checkpoint contains the model state, we can load it.
    # However, num_classes is usually data-dependent.
    # We can try to infer it from the checkpoint if it stores config, or pass it as an arg.
    # For now, let's assume the checkpoint has the config or we use the config file.
    # But the config file usually doesn't hardcode num_classes (it's derived from dataset).
    # Let's check if the checkpoint has 'config' key as saved in train.py.
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to get num_classes from checkpoint config or model state
    if "config" in checkpoint and "model" in checkpoint["config"]:
        # This might not have num_classes if it was dynamic.
        # But the model state dict has 'classifier.weight' which shape is (num_classes, embed_dim).
        classifier_weight = checkpoint["model_state"]["classifier.weight"]
        num_classes = classifier_weight.shape[0]
    else:
        # Fallback: try to guess from state dict
        classifier_weight = checkpoint["model_state"]["classifier.weight"]
        num_classes = classifier_weight.shape[0]

    print(f"Detected {num_classes} classes from checkpoint.")

    model_cfg = cfg.get("model", {})
    model = GaitRecognitionModel(
        num_classes=num_classes,
        in_channels=int(model_cfg.get("in_channels", 1)),
        frame_feature_dims=tuple(model_cfg.get("frame_feature_dims", [32, 64, 128])),
        pyramid_bins=tuple(model_cfg.get("pyramid_bins", [1, 2, 4])),
        dropout=float(model_cfg.get("dropout", 0.3)),
    )
    
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    # Define Transform (Same as training)
    transform = transforms.Compose([
        transforms.Resize((64, 44)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Run Inference
    print(f"Processing sequence in {input_dir}...")
    clip = preprocess_sequence(input_dir, transform)
    
    if clip is not None:
        clip = clip.to(device)
        with torch.inference_mode():
            logits, embedding = model(clip)
            pred_idx = logits.argmax(dim=1).item()
            # If we had a label mapping, we could map back to subject ID.
            # For now, we print the class index.
            print(f"Predicted Class Index: {pred_idx}")
            # If label mapping is saved in checkpoint, use it
            if "label_mapping" in checkpoint:
                 # Invert mapping
                 inv_map = {v: k for k, v in checkpoint["label_mapping"].items()}
                 print(f"Predicted Subject ID: {inv_map.get(pred_idx, 'Unknown')}")
            elif "config" in checkpoint and "data" in checkpoint["config"]:
                 # Try to reconstruct if possible, but usually hard without dataset scan.
                 pass
    else:
        print("Failed to process input.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for Gait Recognition Model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to folder containing silhouette sequence")
    args = parser.parse_args()
    main(args)
