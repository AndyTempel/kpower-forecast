import time
import pandas as pd
from prophet import Prophet
from kpower_forecast.storage import ModelStorage
import shutil
from pathlib import Path
import logging

# Configure logging to avoid cluttering output
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("kpower_forecast").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

def benchmark():
    print("Setting up benchmark...")

    # Setup temporary storage
    storage_path = Path("./temp_benchmark_data")
    if storage_path.exists():
        shutil.rmtree(storage_path)
    storage_path.mkdir(parents=True)

    storage = ModelStorage(storage_path=str(storage_path))
    model_id = "benchmark_model"

    # Train a simple model
    print("Training a simple Prophet model to save...")
    df = pd.DataFrame({
        "ds": pd.date_range("2023-01-01", periods=100, freq="D"),
        "y": range(100)
    })
    m = Prophet()
    m.fit(df)

    # Save the model
    storage.save_model(m, model_id)
    print("Model saved.")

    iterations = 100

    # Measure Baseline: load_model
    print(f"\nBenchmarking 'load_model' (baseline) over {iterations} iterations...")
    start_time = time.time()
    for _ in range(iterations):
        model = storage.load_model(model_id)
        if not model:
            raise Exception("Model should exist")
    end_time = time.time()
    baseline_duration = end_time - start_time
    avg_baseline = baseline_duration / iterations
    print(f"Total time: {baseline_duration:.4f}s")
    print(f"Average time per call: {avg_baseline:.4f}s")

    # Measure Optimization: exists (simulated or actual if implemented)
    print(f"\nBenchmarking 'exists' (optimization) over {iterations} iterations...")

    # We will simulate what the exists method would do if it's not implemented yet
    # checking file existence directly using the same logic as ModelStorage
    model_path = storage._get_model_path(model_id)

    start_time = time.time()
    for _ in range(iterations):
        if hasattr(storage, 'exists'):
            exists = storage.exists(model_id)
        else:
            # Simulation of what exists() will do
            exists = model_path.exists()

        if not exists:
             raise Exception("Model file should exist")

    end_time = time.time()
    opt_duration = end_time - start_time
    avg_opt = opt_duration / iterations
    print(f"Total time: {opt_duration:.4f}s")
    print(f"Average time per call: {avg_opt:.4f}s")

    # Results
    if avg_opt > 0:
        speedup = avg_baseline / avg_opt
        print(f"\nSpeedup: {speedup:.2f}x")
    else:
        print("\nSpeedup: Infinite (optimization took ~0s)")

    # Cleanup
    shutil.rmtree(storage_path)

if __name__ == "__main__":
    benchmark()
