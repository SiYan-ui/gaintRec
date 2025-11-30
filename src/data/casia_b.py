"""CASIA-B silhouette dataset utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


ConditionFilter = Callable[[str, str, str], bool]   # 用于做类型检查


@dataclass(frozen=True)
class SequenceIndex:
    """Metadata describing one CASIA-B walking sequence."""

    subject_id: str
    condition: str
    angle: str
    frame_paths: Tuple[Path, ...]
    label: int


class CasiaBSilhouetteDataset(Dataset):
    """PyTorch dataset for CASIA-B silhouette sequences."""

    def __init__(
        self,
        root: Path | str,
        subjects: Sequence[str],
        label_mapping: Optional[Dict[str, int]] = None,
        condition_filter: Optional[ConditionFilter] = None,
        min_frames: int = 8,
        frames_per_clip: int = 30,
        sampling_strategy: str = "uniform",
        frame_transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        clip_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.root = Path(root)
        self.subjects = [f"{int(s):03d}" for s in subjects]
        self.condition_filter = condition_filter
        self.min_frames = min_frames
        self.frames_per_clip = frames_per_clip
        self.sampling_strategy = sampling_strategy
        self.frame_transform = frame_transform or transforms.Compose(
            [transforms.Resize((64, 44)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        self.clip_transform = clip_transform

        if not self.root.exists():
            raise FileNotFoundError(f"CASIA-B root not found: {self.root}")

        self.label_mapping = label_mapping or build_subject_label_map(self.subjects)
        self.sequences: List[SequenceIndex] = self._build_index()

        if not self.sequences:
            raise RuntimeError("No valid CASIA-B sequences were indexed. Check dataset path and filters.")

    def _build_index(self) -> List[SequenceIndex]:
        sequences: List[SequenceIndex] = []
        for subject_id in self.subjects:
            subject_dir = self.root / subject_id
            if not subject_dir.is_dir():
                continue
            for condition_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
                condition = condition_dir.name
                for angle_dir in sorted(p for p in condition_dir.iterdir() if p.is_dir()):
                    angle = angle_dir.name
                    if self.condition_filter and not self.condition_filter(subject_id, condition, angle):
                        continue
                    frame_paths = tuple(sorted(angle_dir.glob("*.png")))
                    if len(frame_paths) < self.min_frames:
                        continue
                    label = self.label_mapping.get(subject_id)
                    if label is None:
                        continue
                    sequences.append(
                        SequenceIndex(
                            subject_id=subject_id,
                            condition=condition,
                            angle=angle,
                            frame_paths=frame_paths,
                            label=label,
                        )
                    )
        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str | int]:
        meta = self.sequences[index]
        frame_paths = self._select_frames(meta.frame_paths)
        frames: List[torch.Tensor] = []
        for path in frame_paths:
            with Image.open(path) as img:
                frame = img.convert("L")    # 转为8位灰度图
                tensor_frame = self.frame_transform(frame)
                frames.append(tensor_frame)
        clip = torch.stack(frames, dim=0)
        if self.clip_transform:
            clip = self.clip_transform(clip)
        return {
            "clip": clip,
            "label": meta.label,
            "subject_id": meta.subject_id,
            "condition": meta.condition,
            "angle": meta.angle,
        }

    def _select_frames(self, frame_paths: Sequence[Path]) -> List[Path]:
        total = len(frame_paths)
        fps = self.frames_per_clip
        if total == 0:
            raise RuntimeError("Empty frame list encountered in dataset index.")
        if total >= fps:
            if self.sampling_strategy == "random":
                indices = sorted(random.sample(range(total), fps))
            else:
                step = total / fps
                indices = [int(i * step) for i in range(fps)]
        else:
            indices = list(range(total))
            while len(indices) < fps:
                indices.extend(indices[: max(1, fps - len(indices))])
            indices = indices[:fps]
        return [frame_paths[i] for i in indices]

    @property
    def num_classes(self) -> int:
        return len(set(self.label_mapping.values()))


def discover_subject_ids(root: Path | str) -> List[str]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    subject_ids: List[str] = []
    for candidate in root_path.iterdir():
        if candidate.is_dir() and candidate.name.isdigit():
            subject_ids.append(candidate.name)
    return sorted(subject_ids)


def split_subjects(subject_ids: Sequence[str], train_ratio: float = 0.8, seed: int = 42) -> Tuple[List[str], List[str]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    ids = [f"{int(s):03d}" for s in subject_ids]
    rng = random.Random(seed)
    rng.shuffle(ids)
    pivot = int(len(ids) * train_ratio)
    train_ids = sorted(ids[:pivot])
    val_ids = sorted(ids[pivot:])
    return train_ids, val_ids


def build_subject_label_map(subject_ids: Iterable[str]) -> Dict[str, int]:
    return {f"{int(s):03d}": idx for idx, s in enumerate(sorted({f"{int(s):03d}" for s in subject_ids}))}
