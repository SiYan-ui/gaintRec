"""Data loading utilities for CASIA-B silhouettes."""

from .casia_b import (
    CasiaBSilhouetteDataset,
    discover_subject_ids,
    split_subjects,
    build_subject_label_map,
)

__all__ = [
    "CasiaBSilhouetteDataset",
    "discover_subject_ids",
    "split_subjects",
    "build_subject_label_map",
]
