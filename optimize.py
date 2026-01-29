import json
import os.path
import time
from datetime import datetime

import rerun as rr
import torch
import torch.nn as nn

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



def transform_vertices(vertices, params):
    """
    Transform vertices for all particles.
    Args:
        vertices: (T, 3, 2) - original triangle vertices
        params: (P, T, 3) - [x, y, rotation] for each particle
    Returns:
        (P, T, 3, 2) - transformed vertices for each particle
    """
    xy = params[..., :2]  # (P, T, 2)
    rot = params[..., 2]   # (P, T)
    cos = rot.cos()
    sin = rot.sin()
    rot_matrices = torch.stack([
        torch.stack([cos, -sin], dim=-1),  # (P, T, 2)
        torch.stack([sin, cos], dim=-1),   # (P, T, 2)
    ], dim=-2)  # (P, T, 2, 2)
    centroid = vertices.mean(dim=-2)  # (T, 2)
    centered = vertices - centroid.unsqueeze(dim=-2)  # (T, 3, 2)
    rotated = centered @ rot_matrices.transpose(-1, -2)  # (P, T, 3, 2)
    transformed = rotated + xy.unsqueeze(-2)  # (P, T, 3, 2)
    
    return transformed

def compute_bbox_loss(transformed_vertices, goal_aabb, threshold=1e-3):
    """
    compute relu loss for vertices outside bbox
    transformed_vertices: (P, T, 3, 2) tensor
    goal_aabb: (2, 2) - [min_corner, max_corner]
    """
    min_corner = goal_aabb[0]  # (2,)
    max_corner = goal_aabb[1]  # (2,)
    
    loss = (torch.relu(min_corner - transformed_vertices + threshold) + torch.relu(transformed_vertices - max_corner + threshold)).sum(dim=(3, 2, 1))
    return loss

def compute_SAT_loss(transformed_vertices, threshold=1e-3):
    '''
    compute relu loss on max gap between every pair of triangles over all projection axes; candidate axes are normal to the 6 edges in the pair
    transformed_vertices: (P, T, 3, 2) tensor
    '''
    T = transformed_vertices.shape[-3]
    
    X = transformed_vertices.unsqueeze(dim=-3).unsqueeze(dim=-5) # (P, 1, T, 1, 3, 2)

    edge_dirs = torch.stack([
        transformed_vertices[..., 1, :] - transformed_vertices[..., 0, :],
        transformed_vertices[..., 2, :] - transformed_vertices[..., 1, :],
        transformed_vertices[..., 0, :] - transformed_vertices[..., 2, :]
    ], dim = -2) # (P, T, 3, 2)

    axes = torch.stack([-edge_dirs[..., 1], edge_dirs[..., 0]], dim=-1) # (P, T, 3, 2)
    axes = axes.unsqueeze(dim=-2).unsqueeze(dim=-4) # (P, T, 1, 3, 1, 2)
    proj = torch.linalg.vecdot(X, axes, dim=-1) # (P, T, T, 3, 3)

    self_proj = proj.diagonal(dim1=-4, dim2=-3).movedim(-1, -3) # (P, T, 3, 3)
    self_proj = torch.stack([self_proj] * T, dim=-3) # (P, T, T, 3, 3)
    proj_pairs = torch.stack([proj, self_proj], dim=-2) # (P, T, T, 3, 2, 3)
    first = proj_pairs[..., 0, :]
    second = proj_pairs[..., 1, :]

    gap = torch.maximum(second.amin(dim=-1) - first.amax(dim=-1), first.amin(dim=-1) - second.amax(dim=-1)) # (P, T, T, 3)
    max_gap = gap.amax(dim=-1) # (P, T, T)
    diag_mask = torch.eye(T, device=transformed_vertices.device).unsqueeze(dim=0)
    max_gap = max_gap * (1 - diag_mask)

    loss = torch.relu(threshold - max_gap)
    # loss = torch.relu(-max_gap)
    total_loss = torch.sum(loss, dim=(-1, -2)) - threshold * T
    return total_loss

def optimize(
    num_triangles: int,
    env_idx: int,
    num_particles: int,
    visualize: bool,
    device: str
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

    # make all vertices in CCW in triangles
    for key in triangles:
        verts = triangles[key]  # (3, 2)
        x = verts[:, 0] # (3,)
        y = verts[:, 1] # (3,)
        area_twice = (
            (x[0] * y[1] - x[1] * y[0]) +
            (x[1] * y[2] - x[2] * y[1]) +
            (x[2] * y[0] - x[0] * y[2])
        )
        if area_twice < 0:
            triangles[key] = verts[[0, 2, 1]]

    # Your code starts here. Feel free to write any additional methods you need.
    # You should track the time required to find a satisfying particle along with other metrics you think are relevant
    start_time = time.perf_counter()
    found_solution = False
    
    vertices = torch.stack([triangle_vertices for triangle_vertices in triangles.values()]) # (T, 3, 2)
    params = torch.stack([particles[triangle] for triangle in triangles], dim=1).requires_grad_(True) # (P, T, 3)
    optimizer = torch.optim.Adam([params], lr=0.01)

    epochs = 1000
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        transformed_vertices = transform_vertices(vertices, params) # (P, T, 3, 2)
        
        threshold = 1e-3
        bbox_loss = compute_bbox_loss(transformed_vertices, goal_aabb, threshold=threshold)  # (P,)
        SAT_loss = compute_SAT_loss(transformed_vertices, threshold=threshold)
        particle_losses = bbox_loss + SAT_loss # (P,)
        total_loss = particle_losses.sum()
        
        if particle_losses.min().item() < 1e-8:
            found_solution = True
            if device == "cuda":
                torch.cuda.synchronize()
            time_to_solution = time.perf_counter() - start_time
            best_idx = particle_losses.argmin().item()
            print(f"Solution found at epoch {epoch}, particle {best_idx}")
            print(f"Time to solution: {time_to_solution:.4f}s")
            # print("total loss:", particle_losses[best_idx].item())
            break

        total_loss.backward()
        optimizer.step()

    best_idx = particle_losses.argmin().item()
    if not found_solution:
        print(f"No solution found. Best particle {best_idx} has loss {particle_losses[best_idx].item():.6f}")
        time_to_solution = time.perf_counter() - start_time
        print(f"Time spent: {time_to_solution:.4f}s")
        # print("bbox loss:", bbox_loss[best_idx].item())
        # print("SAT loss:", SAT_loss[best_idx].item())
    
    # Visualize best attempt
    if visualize:
        for i, (triangle, vertices) in enumerate(triangles.items()):
            transformed_vertices = transform_vertices(vertices, params[best_idx, i, :]).detach().clone()
            # print(transformed_vertices)
            vertices_3d = torch.cat(
                (transformed_vertices, torch.zeros_like(transformed_vertices[:, :1])), dim=1
            )
            # vertices_3d += 0.25  # offset for sake of this demo
            rr.log(
                f"world/{triangle}",
                rr.Mesh3D(
                    vertex_positions=vertices_3d.cpu(), triangle_indices=[[0, 1, 2]]
                ),
            )

    if found_solution: return time_to_solution
    return float("inf")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_particles = 512
    torch.manual_seed(13)
    torch.cuda.manual_seed(13)
    duration = optimize(num_triangles=6, env_idx=7, num_particles=num_particles, visualize=True, device=device)
