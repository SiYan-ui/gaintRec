"""Model exports for gait recognition."""

from .gaitset import GaitSetBackbone, SetPooling, GaitRecognitionModel

__all__ = [
	"GaitSetBackbone",
	"SetPooling",
	"GaitRecognitionModel",
]
