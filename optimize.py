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
    centered = vertices - centroid.unsqueeze(dim=-2)   # (T, 3, 2)
    rotated = centered @ rot_matrices.transpose(-1, -2)  # (P, T, 3, 2)
    transformed = rotated + xy.unsqueeze(-2)  # (P, T, 3, 2)
    
    return transformed

def compute_bbox_loss(transformed_vertices, goal_aabb):
    """
    Compute ReLU loss for vertices outside goal box
    transformed_vertices: (P, T, 3, 2) tensor
    goal_aabb: (2, 2) - [min_corner, max_corner]
    """
    min_corner = goal_aabb[0]  # (2,)
    max_corner = goal_aabb[1]  # (2,)
    
    loss = (torch.relu(min_corner - transformed_vertices) + torch.relu(transformed_vertices - max_corner)).sum(dim=(3, 2, 1)) # MAYBE optimize
    return loss

def compute_edge_intersection_loss(transformed_vertices, threshold=1e-3):
    """
    Compute ReLU distance loss for edge intersections between triangles
    transformed_vertices: (P, T, 3, 2) tensor
    threshold: threshold for considering distance as zero
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
    
    AB = B - A # (P, T, T, 3, 3, 2)
    CD = D - C # (P, T, T, 3, 3, 2)
    AC = C - A # (P, T, T, 3, 3, 2)
    
    # solve for A + t * AB = C + s * CD
    denom = AB[..., 0] * CD[..., 1] - AB[..., 1] * CD[..., 0]  # (P, T, T, 3, 3)
    eps = 1e-8
    parallel_mask = denom.abs() < eps
    denom_safe = torch.where(parallel_mask, torch.ones_like(denom) * 1e-8, denom)
    t = (AC[..., 0] * CD[..., 1] - AC[..., 1] * CD[..., 0]) / denom_safe  # (AC x CD) / (AB x CD) => (P, T, T, 3, 3)
    s = (AC[..., 0] * AB[..., 1] - AC[..., 1] * AB[..., 0]) / denom_safe  # (AC x AB) / (AB x CD) => (P, T, T, 3, 3)

    dist_t = torch.relu(-t) + torch.relu(t - 1)                                                                                                           
    dist_s = torch.relu(-s) + torch.relu(s - 1)                                                                                                           
    # dist_t = (torch.relu(t + threshold)) * (t < 0.5) + torch.relu(1 + threshold - t) * (t >= 0.5)                                                         
    # dist_s = (torch.relu(s + threshold)) * (s < 0.5) + torch.relu(1 + threshold - s) * (s >= 0.5)
    dist = dist_t + dist_s # (P, T, T, 3, 3)

    # softplus = nn.Softplus(beta=10e7, threshold=2e-7)
    # blend_t = torch.sigmoid(20 * (t - 0.5))
    # blend_s = torch.sigmoid(20 * (s - 0.5))
    # dist_t = (1 - blend_t) * softplus(t + threshold) + blend_t * softplus(1 + threshold - t)
    # dist_s = (1 - blend_s) * softplus(s + threshold) + blend_s * softplus(1 + threshold - s)
    # loss = dist_t + dist_s # (P, T, T, 3, 3)
        
    # # set large distance for parallel edges
    dist = torch.where(parallel_mask, torch.ones_like(dist) * 5.0, dist)
    loss = torch.relu(threshold - dist)  # (P, T, T, 3, 3)
    # loss = torch.relu(dist - threshold)  # (P, T, T, 3, 3)
    
    
    # remove double counting and comparisons within same triangle
    upper_triangle_mask = torch.triu(torch.ones(T, T, device=transformed_vertices.device, dtype=torch.bool), diagonal=1)
    upper_triangle_mask = upper_triangle_mask.view(1, T, T, 1, 1)
    masked_loss = loss * upper_triangle_mask # (P, T, T, 3, 3)
    total_loss = masked_loss.sum(dim=(-1, -2, -3, -4))  # (P,)
    
    return total_loss


def compute_vertex_inside_loss(transformed_vertices):
    """
    Check if each vertex is inside another triangle, and if yes, adds perpendicular distance to closest edge.
    transformed_vertices: (P, T, 3, 2) tensor
    """
    T = transformed_vertices.shape[-3]
    
    X = transformed_vertices.unsqueeze(2)  # (P, T, 1, 3, 2)
    
    tri = transformed_vertices.unsqueeze(1)  # (P, 1, T, 3, 2)
    A = tri[..., 0, :].unsqueeze(-2)  # (P, 1, T, 1, 2)
    B = tri[..., 1, :].unsqueeze(-2)
    C = tri[..., 2, :].unsqueeze(-2)
    
    XA = A - X  # (P, T, T, 3, 2)
    XB = B - X
    XC = C - X
    
    # point X is inside tri if all cross products >= 0
    cross_ab = XA[..., 0] * XB[..., 1] - XA[..., 1] * XB[..., 0]  # (P, T, T, 3)
    cross_bc = XB[..., 0] * XC[..., 1] - XB[..., 1] * XC[..., 0]
    cross_ca = XC[..., 0] * XA[..., 1] - XC[..., 1] * XA[..., 0]
    
    inside_mask = (cross_ab >= 0) & (cross_bc >= 0) & (cross_ca >= 0)  # (P, T, T, 3)
    
    # same_tri_mask = torch.eye(T, device=transformed_vertices.device, dtype=torch.bool)
    # same_tri_mask = same_tri_mask.view(1, T, T, 1)  # (1, T_i, T_j, 1)
    # inside_mask = inside_mask & ~same_tri_mask
    
    # dist from point X to edge PQ = |PQ × PX| / |PQ|
    AB = B - A # (P, 1, T, 1, 2)
    BC = C - B
    CA = A - C
    AX = X - A # (P, T, T, 3, 2)
    BX = X - B
    CX = X - C
    cross_AB_AX = AB[..., 0] * AX[..., 1] - AB[..., 1] * AX[..., 0] # (P, T, T, 3)
    cross_BC_BX = BC[..., 0] * BX[..., 1] - BC[..., 1] * BX[..., 0]
    cross_CA_CX = CA[..., 0] * CX[..., 1] - CA[..., 1] * CX[..., 0]
    
    eps = 1e-8
    dist_AB = cross_AB_AX.abs() / (AB.norm(dim=-1) + eps) # (P, T, T, 3)
    dist_BC = cross_BC_BX.abs() / (BC.norm(dim=-1) + eps)
    dist_CA = cross_CA_CX.abs() / (CA.norm(dim=-1) + eps)
    min_dist = torch.minimum(torch.minimum(dist_AB, dist_BC), dist_CA) # (P, T, T, 3)
    
    loss = min_dist * inside_mask.float() # (P, T, T, 3)
    total_loss = loss.sum(dim=(-1, -2, -3)) # (P,)
    
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

    epochs = 2000
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        transformed_vertices = transform_vertices(vertices, params) # (P, T, 3, 2)
        
        bbox_loss = compute_bbox_loss(transformed_vertices, goal_aabb)  # (P,)
        intersection_loss = compute_edge_intersection_loss(transformed_vertices, threshold=1e-3)  # (P,)
        inside_loss = compute_vertex_inside_loss(transformed_vertices)  # (P,)
        particle_losses = bbox_loss + inside_loss + intersection_loss # (P,)
        total_loss = particle_losses.sum()
        
        if epoch == 0:
            print("initial bbox loss:", bbox_loss[0].item())
            print("initial intersection loss:", intersection_loss[0].item())
            print("initial inside loss:", inside_loss[0].item())
            print("initial total loss:", particle_losses[0].item())

        # check for sol
        if particle_losses.min().item() < 1e-8:
            found_solution = True
            if device == "cuda":
                torch.cuda.synchronize()
            time_to_solution = time.perf_counter() - start_time
            best_idx = particle_losses.argmin().item()
            print(f"Solution found at epoch {epoch}, particle {best_idx}")
            print(f"Time to solution: {time_to_solution:.4f}s")
            print("total loss:", particle_losses[best_idx].item())
            break
        
        total_loss.backward()
        optimizer.step()

    best_idx = particle_losses.argmin().item()
    if not found_solution:
        print(f"No solution found. Best particle {best_idx} has loss {particle_losses[best_idx].item():.6f}")
        print(f"Time spent: {time.perf_counter() - start_time:.4f}s")
    
        print("bbox loss:", bbox_loss[best_idx].item())
        print("intersection loss:", intersection_loss[best_idx].item())
        print("inside loss:", inside_loss[best_idx].item())
    
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
    duration = optimize(num_triangles=3, env_idx=2, num_particles=num_particles, visualize=True, device=device)
