"""Opt-in employee face enrollment and recognition.

The registry stores face embeddings, not source photographs. The actual face
detector is injected so deployments can choose a model and tests can remain
deterministic without downloading model data.
"""

from __future__ import annotations

import io
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence


class FaceProvider(Protocol):
    """Detect faces and return one embedding and location per detected face."""

    def extract(self, image: bytes) -> list["DetectedFace"]:
        ...


@dataclass(frozen=True)
class DetectedFace:
    embedding: Sequence[float]
    location: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class Match:
    employee_id: str
    name: str
    distance: float
    confidence: float
    location: tuple[int, int, int, int] | None = None


class FaceRecognitionError(ValueError):
    """Raised for invalid enrollment or recognition input."""


class FaceRecognitionUnavailable(RuntimeError):
    """Raised when the optional default face model is not installed."""


class FaceRecognitionProvider:
    """Default provider backed by the optional ``face_recognition`` package."""

    def __init__(self, model: str = "hog") -> None:
        self.model = model

    def extract(self, image: bytes) -> list[DetectedFace]:
        try:
            import face_recognition
        except ImportError as exc:
            raise FaceRecognitionUnavailable(
                "Install the optional face-recognition dependency to use "
                "the default provider, or inject a FaceProvider."
            ) from exc

        try:
            loaded = face_recognition.load_image_file(io.BytesIO(image))
            locations = face_recognition.face_locations(loaded, model=self.model)
            encodings = face_recognition.face_encodings(loaded, locations)
        except Exception as exc:
            raise FaceRecognitionError("Could not decode or analyze the image") from exc
        return [
            DetectedFace(tuple(float(value) for value in encoding), tuple(location))
            for encoding, location in zip(encodings, locations)
        ]


def _normalise(values: Iterable[float]) -> list[float]:
    vector = [float(value) for value in values]
    length = math.sqrt(sum(value * value for value in vector))
    if not vector or length == 0:
        raise FaceRecognitionError("Face embeddings must be non-empty and non-zero")
    return [value / length for value in vector]


def _average(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        raise FaceRecognitionError("At least one face embedding is required")
    size = len(vectors[0])
    if any(len(vector) != size for vector in vectors):
        raise FaceRecognitionError("All face embeddings must have the same size")
    averaged = [
        sum(vector[index] for vector in vectors) / len(vectors)
        for index in range(size)
    ]
    return _normalise(averaged)


def _cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise FaceRecognitionError("Stored and query embeddings have different sizes")
    return 1.0 - sum(a * b for a, b in zip(left, right))


class EmployeeFaceRegistry:
    """SQLite-backed enrollment and nearest-neighbor face matching."""

    def __init__(
        self,
        database: str | Path,
        provider: FaceProvider | None = None,
        threshold: float = 0.6,
    ) -> None:
        if not 0 < threshold < 2:
            raise ValueError("threshold must be between 0 and 2")
        self.database = str(database)
        self.provider = provider or FaceRecognitionProvider()
        self.threshold = threshold
        parent = Path(self.database).parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)
        self._initialise()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialise(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS employee_faces (
                    employee_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    enrolled_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    @staticmethod
    def _read_image(image: bytes | str | Path) -> bytes:
        if isinstance(image, bytes):
            return image
        try:
            return Path(image).read_bytes()
        except OSError as exc:
            raise FaceRecognitionError(f"Could not read image: {image}") from exc

    def enroll(
        self,
        employee_id: str,
        name: str,
        images: Sequence[bytes | str | Path],
        *,
        consent: bool = False,
    ) -> None:
        employee_id, name = employee_id.strip(), name.strip()
        if not employee_id or not name:
            raise FaceRecognitionError("employee_id and name are required")
        if not consent:
            raise FaceRecognitionError(
                "Explicit employee consent is required before enrollment"
            )
        if not images:
            raise FaceRecognitionError("At least one enrollment image is required")

        embeddings: list[Sequence[float]] = []
        for image in images:
            faces = self.provider.extract(self._read_image(image))
            if len(faces) != 1:
                raise FaceRecognitionError(
                    "Each enrollment image must contain exactly one detectable face"
                )
            embeddings.append(_normalise(faces[0].embedding))
        embedding = _average(embeddings)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO employee_faces (employee_id, name, embedding)
                VALUES (?, ?, ?)
                ON CONFLICT(employee_id) DO UPDATE SET
                    name = excluded.name,
                    embedding = excluded.embedding,
                    enrolled_at = CURRENT_TIMESTAMP
                """,
                (employee_id, name, json.dumps(embedding)),
            )

    def recognize(
        self, image: bytes | str | Path, *, threshold: float | None = None
    ) -> list[Match]:
        limit = self.threshold if threshold is None else threshold
        if not 0 < limit < 2:
            raise ValueError("threshold must be between 0 and 2")
        faces = self.provider.extract(self._read_image(image))
        if not faces:
            return []
        with self._connect() as connection:
            enrolled = [
                (row["employee_id"], row["name"], _normalise(json.loads(row["embedding"])))
                for row in connection.execute(
                    "SELECT employee_id, name, embedding FROM employee_faces"
                )
            ]

        matches: list[Match] = []
        for face in faces:
            query = _normalise(face.embedding)
            candidates = [
                (employee_id, name, _cosine_distance(query, embedding))
                for employee_id, name, embedding in enrolled
            ]
            if not candidates:
                continue
            employee_id, name, distance = min(candidates, key=lambda item: item[2])
            if distance <= limit:
                matches.append(
                    Match(
                        employee_id=employee_id,
                        name=name,
                        distance=distance,
                        confidence=max(0.0, min(1.0, 1.0 - distance / limit)),
                        location=face.location,
                    )
                )
        return matches

    def remove(self, employee_id: str) -> bool:
        with self._connect() as connection:
            result = connection.execute(
                "DELETE FROM employee_faces WHERE employee_id = ?", (employee_id,)
            )
            return result.rowcount > 0

    def employees(self) -> list[dict[str, str]]:
        with self._connect() as connection:
            return [
                {"employee_id": row["employee_id"], "name": row["name"]}
                for row in connection.execute(
                    "SELECT employee_id, name FROM employee_faces ORDER BY employee_id"
                )
            ]
