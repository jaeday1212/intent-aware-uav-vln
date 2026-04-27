from perception.belief_state import BeliefState


def test_belief_state_helpers_and_serialization() -> None:
    scene_dict = {
        "image": {"width": 640, "height": 480, "source_path": "frame.jpg"},
        "objects": [
            {"label": "car", "bbox_xyxy": [0.0, 0.0, 10.0, 10.0], "confidence": 0.8},
            {"label": "tree", "bbox_xyxy": [5.0, 5.0, 20.0, 20.0], "confidence": 0.7},
            {"label": "car", "bbox_xyxy": [10.0, 10.0, 30.0, 30.0], "confidence": 0.9},
        ],
    }

    belief_state = BeliefState.from_scene_dict(scene_dict)

    assert len(belief_state.get_objects()) == 3
    assert len(belief_state.get_objects_by_label("car")) == 2
    assert belief_state.to_dict()["image"]["source_path"] == "frame.jpg"
    assert "objects=3" in belief_state.summary()
    assert "car: 2" in belief_state.summary()


def test_belief_state_summary_for_empty_objects() -> None:
    belief_state = BeliefState(image_width=320, image_height=240)

    assert belief_state.summary() == "BeliefState(image=320x240, objects=0)"
