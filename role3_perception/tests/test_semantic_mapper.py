from perception.semantic_mapper import build_scene_dict


def test_build_scene_dict_computes_geometry_and_regions() -> None:
    detections = [
        {
            "label": "building",
            "bbox_xyxy": [100.0, 50.0, 220.0, 180.0],
            "confidence": 0.92,
        }
    ]

    scene = build_scene_dict(detections, image_width=300, image_height=300, source_path="sample.jpg")
    obj = scene["objects"][0]

    assert obj["label"] == "building"
    assert obj["center"] == [160.0, 115.0]
    assert obj["width"] == 120.0
    assert obj["height"] == 130.0
    assert obj["area"] == 15600.0
    assert obj["region_horizontal"] == "center"
    assert obj["region_vertical"] == "middle"
    assert scene["image"]["source_path"] == "sample.jpg"


def test_build_scene_dict_handles_empty_detections() -> None:
    scene = build_scene_dict([], image_width=640, image_height=480)

    assert scene["objects"] == []
    assert scene["image"]["width"] == 640
    assert scene["image"]["height"] == 480
