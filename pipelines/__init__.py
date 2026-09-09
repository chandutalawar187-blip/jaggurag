"""Isolated document-understanding pipeline."""
from .pipeline import DocumentPipeline
from .face_recognition import (
    DetectedFace,
    EmployeeFaceRegistry,
    FaceRecognitionError,
    FaceRecognitionProvider,
    FaceRecognitionUnavailable,
    Match,
)

__all__ = [
    "DocumentPipeline",
    "DetectedFace",
    "EmployeeFaceRegistry",
    "FaceRecognitionError",
    "FaceRecognitionProvider",
    "FaceRecognitionUnavailable",
    "Match",
]
