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
