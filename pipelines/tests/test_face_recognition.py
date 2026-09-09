import pytest

from pipelines.face_recognition import (
    DetectedFace,
    EmployeeFaceRegistry,
    FaceRecognitionError,
)


class FakeProvider:
    def __init__(self, faces_by_image):
        self.faces_by_image = faces_by_image

    def extract(self, image):
        return self.faces_by_image[image]


def face(*values):
    return [DetectedFace(values)]


def test_enrolls_and_recognizes_without_storing_images(tmp_path):
    provider = FakeProvider(
        {
            b"alice": face(1.0, 0.0),
            b"query": face(0.99, 0.01),
            b"unknown": face(0.0, 1.0),
        }
    )
    registry = EmployeeFaceRegistry(tmp_path / "faces.sqlite3", provider, threshold=0.2)

    registry.enroll("e-1", "Alice", [b"alice"], consent=True)

    matches = registry.recognize(b"query")
    assert matches[0].employee_id == "e-1"
    assert matches[0].name == "Alice"
    assert registry.recognize(b"unknown") == []
    assert registry.employees() == [{"employee_id": "e-1", "name": "Alice"}]


def test_enrollment_requires_consent_and_one_face(tmp_path):
    provider = FakeProvider({b"two": [DetectedFace((1, 0)), DetectedFace((0, 1))]})
    registry = EmployeeFaceRegistry(tmp_path / "faces.sqlite3", provider)

    with pytest.raises(FaceRecognitionError, match="consent"):
        registry.enroll("e-1", "Alice", [b"two"])
    with pytest.raises(FaceRecognitionError, match="exactly one"):
        registry.enroll("e-1", "Alice", [b"two"], consent=True)
