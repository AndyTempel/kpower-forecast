import pandas as pd

from kpower_forecast.ml.storage import MLModelManifest, MLModelStorage


def test_ml_storage_round_trips_manifest(tmp_path) -> None:
    storage = MLModelStorage(storage_path=str(tmp_path), model_id="roundtrip")
    manifest = MLModelManifest(
        model_id="roundtrip",
        backend_type="nixtla_hybrid",
        target_type="solar",
        interval_levels=[50, 80, 90],
        feature_columns=["temperature_2m"],
        conformal_quantiles={"90": 0.2},
    )

    storage.save_manifest(manifest)
    loaded = storage.load_manifest()

    assert loaded == manifest


def test_ml_storage_round_trips_training_frame(tmp_path) -> None:
    storage = MLModelStorage(storage_path=str(tmp_path), model_id="roundtrip")
    frame = pd.DataFrame({"ds": pd.to_datetime(["2026-08-12T10:30:00Z"]), "y": [0.25]})

    storage.save_training_frame(frame)

    pd.testing.assert_frame_equal(storage.load_training_frame(), frame)
