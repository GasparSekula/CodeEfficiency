import os
import time
from absl import app, flags

import numpy as np
import pandas as pd

from matrix_multiplier import MatrixMultiplier


_SEED = 42

_METHOD = flags.DEFINE_enum(
    "method",
    "numpy",
    ["naive", "numpy", "numba", "gpu"],
    "Method used to calculate matrix multiplication. Possible values: naive, numpy, numba, gpu",
)

_SIZES = flags.DEFINE_list("sizes", [10], "List of matrices' sizes.")

_N_RUNS = flags.DEFINE_integer(
    "n_runs", 10, "Number of executions in experiment."
)

_ITERS = flags.DEFINE_integer(
    "iters", 100, "Number of iterations for each multiplication (for averaging)."
)

_SAVE_DIR = flags.DEFINE_string(
    "save_dir",
    os.getenv("SAVE_DIR", "default_path_for_docs"),
    "Path where results will be stored.",
)

def main(argv):
    np.random.seed(_SEED)
    results = []
    
    print(f"Warming up for method: {_METHOD.value}...")
    dummy_size = 100
    A_dummy = np.random.rand(dummy_size, dummy_size).astype(np.float32)
    B_dummy = np.random.rand(dummy_size, dummy_size).astype(np.float32)
    warmup_multiplier = MatrixMultiplier(A_dummy, B_dummy)
    
    if _METHOD.value == "numpy":
        warmup_multiplier.multiply_numpy()
    elif _METHOD.value == "numba":
        warmup_multiplier.multiply_numba()
    # elif _METHOD.value == "gpu":
    #     warmup_multiplier.multiply_gpu()
    
    for size_str in _SIZES.value:
        size = int(size_str)
        print(f"Running size: {size}x{size}...")
        
        for i in range(_N_RUNS.value):
            
            A = np.random.rand(size, size).astype(np.float32)
            B = np.random.rand(size, size).astype(np.float32)
            multiplier = MatrixMultiplier(A, B)
            
            if _METHOD.value == "naive":
                if size > 1000: continue
                
                start_time = time.perf_counter()
                for _ in range(_ITERS.value):
                    C = multiplier.multiply_naive()
                end_time = time.perf_counter()
                avg_duration = (end_time - start_time) / _ITERS.value
                
            elif _METHOD.value == "numpy":
                start_time = time.perf_counter()
                for _ in range(_ITERS.value):
                    C = multiplier.multiply_numpy()
                end_time = time.perf_counter()
                avg_duration = (end_time - start_time) / _ITERS.value
                
            elif _METHOD.value == "numba":
                start_time = time.perf_counter()
                for _ in range(_ITERS.value):
                    C = multiplier.multiply_numba()
                end_time = time.perf_counter()
                avg_duration = (end_time - start_time) / _ITERS.value
                
            elif _METHOD.value == "gpu":
                start_time = time.perf_counter()
                for _ in range(_ITERS.value):
                    C = multiplier.multiply_gpu()
                end_time = time.perf_counter()
                avg_duration = (end_time - start_time) / _ITERS.value
            
            
            
            results.append({
                "size": size,
                "method": _METHOD.value,
                "run": i,
                "time_seconds": avg_duration,
                "gflops": (2 * size**3) / (avg_duration * 1e9)
            })
            
            del A, B, multiplier
    
    df = pd.DataFrame(results)
    os.makedirs(_SAVE_DIR.value, exist_ok=True)
    output_path = os.path.join(_SAVE_DIR.value, f"{_METHOD.value}_results.csv")
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    app.run(main)
