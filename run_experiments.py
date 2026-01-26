from optimize import optimize
import torch
from tabulate import tabulate
import numpy as np

def run_experiments():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_runs_per_env = 1
    env_idxs = list(range(10))
    all_triangles = [i for i in range(3, 7)]

    results = []
    for num_triangles in all_triangles:
        for env_idx in env_idxs:
            print("-" * 100)
            print(f"Running experiments for {num_triangles} triangles and environment {env_idx}")
            time_to_solutions = []
            for run_idx in range(num_runs_per_env):
                time_to_solution = optimize(
                    num_triangles, env_idx, num_particles=512, visualize=True, device=device
                )
                time_to_solutions.append(time_to_solution)

            results.append(
                {
                    "num_triangles": num_triangles,
                    "env_idx": env_idx,
                    "time_to_solutions": time_to_solutions,
                }
            )

    # Collect results into a table
    table_data = []
    for result in results:
        ts = [t for t in result["time_to_solutions"] if t!=float('inf')]
        table_data.append([
            result["num_triangles"],
            result["env_idx"],
            np.mean(ts) if ts else float('inf'),
            np.std(ts) if ts else "N/A"
        ])
    headers = ["No. triangles", "Env index", "Mean Time", "Std Time"]
    print(tabulate(table_data, headers, tablefmt='grid'))


if __name__ == "__main__":
    torch.manual_seed(13)
    torch.cuda.manual_seed(13)
    run_experiments()
