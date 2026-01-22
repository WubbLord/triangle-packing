import json
import os.path
import time
from datetime import datetime

import rerun as rr
import torch

from typing import Optional

assets_dir = os.path.join(os.path.dirname(__file__), "assets")


def load_env(num_triangles: int, env_idx: int) -> dict:
    """
    Load the environment. You don't need to modify this function.
    """
    tri_env_path = os.path.join(assets_dir, f"{num_triangles}_triangles.jsonl")
    if not os.path.exists(tri_env_path):
        raise FileNotFoundError(f"File {tri_env_path} does not exist")

    with open(tri_env_path, "r") as f:
        for i, line in enumerate(f):
            if i == env_idx:
                break
        if i != env_idx:
            raise IndexError(f"Trial index {env_idx} out of range for {tri_env_path}")

    # Check there are only num_triangles + 1 objects in the environment
    env = json.loads(line)
    assert (
        len(env) == num_triangles + 1
    ), f"Expected {num_triangles + 1} objects, got {len(env)}"
    # print(f"Loaded num_triangles={num_triangles} and idx={env_idx} from {tri_env_path}")
    # print("env=", env)
    return env


def visualize_env(env: dict):
    """
    Visualize environment in rerun. You don't need to modify this function.
    """
    for label, obj in env.items():
        shape = obj["shape"]

        if shape == "box":
            extents = obj["extents"]
            half_sizes = [extents[0] / 2, extents[1] / 2, extents[2] / 2]
            centroid = obj["centroid"]

            rr.log(
                f"world/{label}",
                rr.Boxes3D(
                    half_sizes=half_sizes,
                    centers=centroid,
                ),
            )
        elif shape == "arbitrary_triangle":
            vertices = obj["vertices"]
            faces = obj["faces"]
            rgb = obj["color"]

            vertices_3d = [v + [0.0] for v in vertices]
            rr.log(
                f"world/{label}",
                rr.Mesh3D(
                    vertex_positions=vertices_3d,
                    triangle_indices=faces,
                    vertex_colors=[rgb],
                ),
            )
        else:
            raise ValueError(f"Unknown shape {shape}")


def get_goal_aabb(env: dict) -> torch.Tensor:
    """Get goal axis-aligned bounding box (AABB) from environment. You don't need to modify this function."""
    # We only care about x and y so take the first two elements
    goal_extents = torch.tensor(env["goal"]["extents"][:2])
    goal_centroid = torch.tensor(env["goal"]["centroid"][:2])

    # goal_aabb is a 2x2 tensor, where the first row is the min corner and the second row is the max corner
    goal_aabb = torch.stack(
        [
            goal_centroid - goal_extents / 2,
            goal_centroid + goal_extents / 2,
        ]
    )
    return goal_aabb


def optimize(
    num_triangles: int,
    env_idx: int,
    num_particles: int,
    visualize: bool = True,
    device: Optional[str] = "cuda",
) -> float:
    """
    Solve the triangle packing problem. Returns the time required to find a satisfying particle.
    """

    env = load_env(num_triangles, env_idx)

    if visualize:
        recording_id = datetime.now().isoformat().split(".")[0]
        rr.init("triangle_world", recording_id=recording_id, spawn=True)
        visualize_env(env)

    goal_aabb = get_goal_aabb(env).to(device)
    # print(f"Goal AABB: {goal_aabb}")

    triangles = {
        label: torch.tensor(obj["vertices"], device=device)
        for label, obj in env.items()
        if label != "goal"
    }

    # Randomly sample xy positions and rotations for each triangle
    # These form the particles to be optimized
    particles = {}
    for triangle in triangles:
        xy = torch.rand(num_particles, 2, device=device)
        xy = xy * (goal_aabb[1] - goal_aabb[0]) + goal_aabb[0]
        rot = torch.rand(num_particles, 1, device=device) * 2 * torch.pi
        xy_rot = torch.cat((xy, rot), dim=1)
        particles[triangle] = xy_rot
        # print(f"Triangle {triangle} particles shape: {xy_rot.shape}")

    # Your code starts here. Feel free to write any additional methods you need.
    # You should track the time required to find a satisfying particle along with other metrics you think are relevant
    start_time = time.perf_counter()

    for _ in range(1):
        found_solution = False
        if found_solution:
            torch.cuda.synchronize()
            time_to_solution = time.perf_counter() - start_time
            return time_to_solution

        # Incomplete implementation for visualizing a particle. Might be helpful for debugging.
        if visualize:
            best_idx = 0
            for triangle, vertices in triangles.items():
                xy_rot = particles[triangle][best_idx].detach().clone()
                # Transform the triangle vertices by the rotation and xy translation
                xy = xy_rot[:2]
                rot = xy_rot[2]
                xy = torch.tensor([0, 0])
                # rot = torch.tensor(torch.pi/2)
                rot_matrix = torch.tensor([[rot.cos(), rot.sin()], [-rot.sin(), rot.cos()]])
                triangle_centroid = vertices.sum(dim=0).div(vertices.shape[0])
                centered_vertices = vertices - triangle_centroid
                print(vertices)
                vertices = centered_vertices @ rot_matrix.T + triangle_centroid + xy
                print(vertices)
                print(vertices.sum(dim=0).div(vertices.shape[0]))

                vertices_3d = torch.cat(
                    (vertices, torch.zeros_like(vertices[:, :1])), dim=1
                )
                vertices_3d += 0.25  # offset for sake of this demo
                rr.log(
                    f"world/{triangle}",
                    rr.Mesh3D(
                        vertex_positions=vertices_3d.cpu(), triangle_indices=[[0, 1, 2]]
                    ),
                )

    # No solution found
    return float("inf")


if __name__ == "__main__":
    # Use device="cpu" if you don't have a GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(13)
    duration = optimize(num_triangles=3, env_idx=0, num_particles=512, visualize=True, device=device)
    # print("Time to solution:", duration)
