from pathlib import Path

from PIL import Image

from perception.detector_wrapper import ObjectDetector
from perception.pipeline import build_belief_state


class FakeValue:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeBox:
    def __init__(self, cls_idx, conf, xyxy):
        self.cls = FakeValue(cls_idx)
        self.conf = FakeValue(conf)
        self.xyxy = [xyxy]


class FakeResult:
    def __init__(self):
        self.names = {0: "building", 1: "car"}
        self.boxes = [
            FakeBox(0, 0.91, [10.0, 20.0, 110.0, 140.0]),
            FakeBox(1, 0.65, [120.0, 50.0, 180.0, 100.0]),
        ]


class FakeYOLOModel:
    names = {0: "building", 1: "car"}

    def predict(self, source, conf, imgsz, verbose):
        return [FakeResult()]


class StubDetector:
    def __init__(self, detections):
        self._detections = detections

    def detect(self, image):
        return self._detections


def test_object_detector_normalizes_results_without_ultralytics() -> None:
    detector = ObjectDetector(model=FakeYOLOModel())

    detections = detector.detect(Image.new("RGB", (200, 200)))

    assert detections == [
        {
            "label": "building",
            "bbox_xyxy": [10.0, 20.0, 110.0, 140.0],
            "confidence": 0.91,
        },
        {
            "label": "car",
            "bbox_xyxy": [120.0, 50.0, 180.0, 100.0],
            "confidence": 0.65,
        },
    ]


def test_build_belief_state_returns_belief_state_from_image_path(tmp_path) -> None:
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (300, 240), color="white").save(image_path)

    belief_state = build_belief_state(
        image=image_path,
        detector=StubDetector(
            [
                {
                    "label": "building",
                    "bbox_xyxy": [100.0, 50.0, 220.0, 180.0],
                    "confidence": 0.92,
                }
            ]
        ),
    )

    assert belief_state.image_width == 300
    assert belief_state.image_height == 240
    assert belief_state.source_path == str(Path(image_path))
    assert belief_state.get_objects_by_label("building")[0]["area"] == 15600.0


def test_build_belief_state_attaches_drone_state(tmp_path) -> None:
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (300, 240), color="white").save(image_path)

    belief_state = build_belief_state(
        image=image_path,
        detector=StubDetector([]),
        drone_state={"altitude_m": 12.5, "heading_deg": 45.0},
    )

    assert belief_state.drone_state == {"altitude_m": 12.5, "heading_deg": 45.0}
    assert belief_state.to_dict()["drone_state"]["altitude_m"] == 12.5
