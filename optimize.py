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
    centered = vertices - centroid.unsqueeze(dim=-2)   # (T, 3, 2)
    rotated = centered @ rot_matrices.transpose(-1, -2)  # (P, T, 3, 2)
    transformed = rotated + xy.unsqueeze(-2)  # (P, T, 3, 2)
    
    return transformed

def compute_bbox_loss(transformed_vertices, goal_aabb):
    """
    Compute ReLU loss for vertices outside bounding box.
    Args:
        transformed_vertices: (P, T, 3, 2) tensor
        goal_aabb: (2, 2) - [min_corner, max_corner]
    Returns:
        (P,) - loss for each particle
    """
    min_corner = goal_aabb[0]  # (2,)
    max_corner = goal_aabb[1]  # (2,)
    
    # total_loss = torch.zeros(num_particles, device=device)
    # for verts in transformed_vertices:
    #     loss = (torch.relu(min_corner - verts) + torch.relu(verts - max_corner)).sum(dim=(1, 2))
    #     total_loss += loss
    loss = (torch.relu(min_corner - transformed_vertices) + torch.relu(transformed_vertices - max_corner)).sum(dim=(3, 2, 1)) # MAYBE optimize
    return loss

def compute_edge_intersection_loss(transformed_vertices, threshold=1e-3):
    """
    Compute ReLU signed distance loss for edge intersections between triangles.
    Args:
        transformed_vertices: (P, T, 3, 2) tensor
        threshold: threshold for considering distance as zero
    Returns:
        (P,) - loss for each particle
    """
    T = transformed_vertices.shape[-3]
    
    v0, v1, v2 = transformed_vertices[..., 0, :], transformed_vertices[..., 1, :], transformed_vertices[..., 2, :]  # each (P, T, 2)
    edges = torch.stack([
        torch.stack([v0, v1], dim=-2),  # edge 0: v0 -> v1
        torch.stack([v1, v2], dim=-2),  # edge 1: v1 -> v2
        torch.stack([v2, v0], dim=-2),  # edge 2: v2 -> v0
    ], dim=-3)  # (P, T, 3, 2, 2)
    
    # reshape for broadcasting
    edges_i = edges.unsqueeze(2).unsqueeze(4)  # (P, T, 1, 3, 1, 2, 2)
    edges_j = edges.unsqueeze(1).unsqueeze(3)  # (P, 1, T, 1, 3, 2, 2)
    # endpoints A, B from edges_i; C, D from edges_j
    A = edges_i[..., 0, :]  # (P, T, 1, 3, 1, 2)
    B = edges_i[..., 1, :]  # (P, T, 1, 3, 1, 2)
    C = edges_j[..., 0, :]  # (P, 1, T, 1, 3, 2)
    D = edges_j[..., 1, :]  # (P, 1, T, 1, 3, 2)
    
    # Compute direction vectors and AC
    # These broadcast to (P, T, T, 3, 3, 2)
    AB = B - A
    CD = D - C
    AC = C - A
    
    denom = AB[..., 0] * CD[..., 1] - AB[..., 1] * CD[..., 0]  # (P, T, T, 3, 3)
    eps = 1e-8
    parallel_mask = denom.abs() < eps
    denom_safe = torch.where(parallel_mask, torch.ones_like(denom) * 1e-8, denom)
    
    # parametric intersection parameters
    t = (AC[..., 0] * CD[..., 1] - AC[..., 1] * CD[..., 0]) / denom_safe  # cross(AC, CD) / cross(AB, CD) => (P, T, T, 3, 3)
    s = (AC[..., 0] * AB[..., 1] - AC[..., 1] * AB[..., 0]) / denom_safe  # cross(AC, AB) / cross(AB, CD) => (P, T, T, 3, 3)
    
    # Distance from [0, 1] range: 0 when inside, positive when outside
    dist_t = torch.relu(-t) + torch.relu(t - 1)
    dist_s = torch.relu(-s) + torch.relu(s - 1)
    
    # Total separation distance (0 means edges intersect)
    dist = dist_t + dist_s  # (P, T, T, 3, 3)
    
    # For parallel edges, set large distance (no intersection)
    dist = torch.where(parallel_mask, torch.ones_like(dist) * 10.0, dist)
    
    # ReLU signed distance: penalize when dist < threshold
    loss = torch.relu(threshold - dist)  # (P, T, T, 3, 3)
    
    # Create upper triangular mask to only count each triangle pair once (i < j)
    # This avoids double-counting and self-intersection
    upper_tri_mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
    # Expand mask to match loss shape: (1, T, T, 1, 1)
    upper_tri_mask = upper_tri_mask.view(1, T, T, 1, 1)
    
    # Apply mask and sum over all dimensions except particles
    masked_loss = loss * upper_tri_mask
    total_loss = masked_loss.sum(dim=(1, 2, 3, 4))  # (P,)
    
    return total_loss


def optimize(
    num_triangles: int,
    env_idx: int,
    num_particles: int,
    visualize: bool = True,
    device: str = "cpu",
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

    # Your code starts here. Feel free to write any additional methods you need.
    # You should track the time required to find a satisfying particle along with other metrics you think are relevant
    start_time = time.perf_counter()
    found_solution = False
    
    # vertices = torch.stack([torch.tensor(triangles[label]) for label in triangles.keys()]) # (T, 3, 2)\
    vertices = torch.stack([triangle_vertices for triangle_vertices in triangles.values()]) # (T, 3, 2)
    # triangle_labels = list(triangles.keys())
    params = torch.stack([particles[triangle] for triangle in triangles], dim=1).requires_grad_(True) # (P, T, 3)
    # active_mask = torch.ones(num_particles, dtype=torch.bool, device=device)
    optimizer = torch.optim.Adam([params], lr=0.01)

    epochs = 2000
    for epoch in range(epochs):
        optimizer.zero_grad()
        # transformed_vertices = [transform_vertices_batch(triangles[label], params[:, i, :]) for i, label in enumerate(triangle_labels)]
        # transformed_vertices = torch.stack([transform_vertices_batch(triangles[label], params[:, i, :]) for i, label in enumerate(triangle_labels)])
        transformed_vertices = transform_vertices(vertices, params) # (P, T, 3, 2)
        
        bbox_loss = compute_bbox_loss(transformed_vertices, goal_aabb)  # (P,)
        intersection_loss = compute_edge_intersection_loss(transformed_vertices, threshold=1e-3)  # (P,)
        particle_losses = bbox_loss + intersection_loss  # (P,)
        # masked_losses = particle_losses * active_mask.float()
        # total_loss = masked_losses.sum()
        total_loss = particle_losses.sum()
        
        # Check for solution before backward pass
        if particle_losses.min().item() < 1e-8:
        # if particle_losses[best_idx] < 1e-8:
            found_solution = True
            if device == "cuda":
                torch.cuda.synchronize()
            time_to_solution = time.perf_counter() - start_time
            best_idx = particle_losses.argmin().item()
            print(f"Solution found at epoch {epoch}, particle {best_idx}")
            print(f"Time to solution: {time_to_solution:.4f}s")
            print("bbox loss:", bbox_loss[best_idx])
            print("intersection loss:", intersection_loss[best_idx])

            # debug bbox
            min_corner = goal_aabb[0]  # (2,)
            max_corner = goal_aabb[1]  # (2,)
            print(min_corner, max_corner)
            # transformed_vertices is (P, T, 3, 2), so iterate over T dimension
            T = transformed_vertices.shape[1]
            for t in range(T):
                print(f"Triangle {t}:", transformed_vertices[best_idx, t])

            break
        
        total_loss.backward()
        
        # # Zero out gradients for frozen particles
        # if not active_mask.all():
        #     if params.grad is not None:
        #         params.grad[~active_mask] = 0
        
        optimizer.step()
        
        # # Freeze particles with zero loss after this epoch
        # newly_frozen = particle_losses < 1e-8
        # if newly_frozen.any():
        #     active_mask = active_mask & ~newly_frozen
            # num_frozen = (~active_mask).sum().item()
            # print(f"Epoch {epoch}: Froze {newly_frozen.sum().item()} particles (total frozen: {num_frozen})")
        
        # if epoch % 5 == 0 or epoch == epochs - 1:
        #     print(f"Epoch {epoch}: Loss = {total_loss.item():.6f}, "
        #           f"Active particles: {active_mask.sum().item()}, "
        #           f"Min particle loss: {particle_losses.min().item():.6f}")

    best_idx = particle_losses.argmin().item()
    if not found_solution:
        print(f"No solution found. Best particle {best_idx} has loss {particle_losses[best_idx].item():.6f}")
        print(f"Time spent: {time.perf_counter() - start_time:.4f}s")
    
    # Visualize best attempt
    if visualize:
        for i, (triangle, vertices) in enumerate(triangles.items()):
            # xy_rot = particles[triangle][best_idx].detach().clone()
            # Transform the triangle vertices by the rotation and xy translation
            # xy = xy_rot[:2]
            # rot = xy_rot[2]
            # rot_matrix = torch.tensor([[rot.cos(), -rot.sin()], [rot.sin(), rot.cos()]])
            # triangle_centroid = vertices.sum(dim=0).div(vertices.shape[0])
            # centered_vertices = vertices - triangle_centroid
            # vertices = centered_vertices @ rot_matrix.T + xy
            transformed_vertices = transform_vertices(vertices, params[best_idx, i, :]).detach().clone()
            print("visualized vertices=", vertices)

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
    # Use device="cpu" if you don't have a GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_particles = 1024
    torch.manual_seed(13)
    duration = optimize(num_triangles=3, env_idx=1, num_particles=num_particles, visualize=True, device=device)
    # print("Time to solution:", duration)
