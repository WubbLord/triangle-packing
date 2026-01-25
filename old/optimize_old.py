import torch

def cross_2d(v1, v2):
    """2D cross product: v1.x * v2.y - v1.y * v2.x"""
    return v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]

def transform_vertices_batch(vertices, xy_rot):
    """
    Transform vertices for all particles.
    Args:
        vertices: (3, 2) - original triangle vertices
        xy_rot: (P, 3) - [x, y, rotation] for each particle
    Returns:
        (P, 3, 2) - transformed vertices for each particle
    """
    xy = xy_rot[:, :2]  # (P, 2)
    rot = xy_rot[:, 2]   # (P,)
    cos = rot.cos()
    sin = rot.sin()
    rot_matrices = torch.stack([
        torch.stack([cos, -sin], dim=1),
        torch.stack([sin, cos], dim=1)
    ], dim=1)  # (P, 2, 2)
    
    centroid = vertices.mean(dim=0)  # (2,)
    centered = vertices - centroid   # (3, 2)
    rotated = centered @ rot_matrices.transpose(1, 2)  # (3, 2) x (P, 2, 2) => (P, 3, 2)
    # rotated = rotated.transpose(0, 1)  # (3, P, 2) -> (P, 3, 2)
    transformed = rotated + xy.unsqueeze(1)  # (P, 3, 2)
    
    return transformed

def get_edges(verts):
    """
    Get edges from vertices of a particular tile.
    Args:
        verts: (P, 3, 2)
    Returns:
        (P, 3, 2, 2) - [particle, edge_idx, endpoint_idx (0=start, 1=end), coord]
    """
    v0, v1, v2 = verts[:, 0], verts[:, 1], verts[:, 2]  # each (P, 2)
    edges = torch.stack([
        torch.stack([v0, v1], dim=1),  # (P, 2, 2)
        torch.stack([v1, v2], dim=1),
        torch.stack([v2, v0], dim=1),
    ], dim=1)  # (P, 3, 2, 2)
    return edges

def compute_edge_intersection_loss(transformed_vertices, threshold=1e-3):
    """
    Compute ReLU signed distance loss for edge intersections between triangles.
    Args:
        transformed_vertices: list of (P, 3, 2) tensors
        threshold: threshold for considering distance as zero
    Returns:
        (P,) - loss for each particle
    """
    T = len(transformed_vertices)
    total_loss = torch.zeros(num_particles, device=device)
    
    # Precompute edges for all triangles
    all_edges = [get_edges(verts) for verts in transformed_vertices]
    
    # For each pair of triangles
    for i in range(T):
        for j in range(i + 1, T):
            edges_i = all_edges[i]  # (P, 3, 2, 2)
            edges_j = all_edges[j]  # (P, 3, 2, 2)
            
            # Reshape for broadcasting: check all 9 edge pairs
            # (P, 3, 1, 2, 2) vs (P, 1, 3, 2, 2) -> (P, 3, 3, ...)
            ei = edges_i.unsqueeze(2)  # (P, 3, 1, 2, 2)
            ej = edges_j.unsqueeze(1)  # (P, 1, 3, 2, 2)
            
            # Extract endpoints: A-B is edge from ei, C-D is edge from ej
            A = ei[..., 0, :]  # (P, 3, 1, 2)
            B = ei[..., 1, :]  # (P, 3, 1, 2)
            C = ej[..., 0, :]  # (P, 1, 3, 2)
            D = ej[..., 1, :]  # (P, 1, 3, 2)
            
            # Parametric form: intersection at A + t*(B-A) = C + s*(D-C)
            # t = cross(AC, CD) / cross(AB, CD)
            # s = cross(AC, AB) / cross(AB, CD)
            AB = B - A  # (P, 3, 1, 2)
            CD = D - C  # (P, 1, 3, 2)
            AC = C - A  # (P, 3, 3, 2)
            
            denom = cross_2d(AB, CD)  # (P, 3, 3)
            
            # Handle near-parallel edges
            eps = 1e-8
            parallel_mask = denom.abs() < eps
            denom_safe = torch.where(parallel_mask, torch.ones_like(denom), denom)
            
            t = cross_2d(AC, CD) / denom_safe  # (P, 3, 3)
            s = cross_2d(AC, AB) / denom_safe  # (P, 3, 3)
            
            # Distance from [0,1] range: 0 when inside, positive when outside
            dist_t = torch.relu(-t) + torch.relu(t - 1)  # (P, 3, 3)
            dist_s = torch.relu(-s) + torch.relu(s - 1)  # (P, 3, 3)
            
            # Total separation distance (0 when edges intersect)
            dist = dist_t + dist_s  # (P, 3, 3)
            
            # For parallel edges, set large distance (no intersection)
            dist = torch.where(parallel_mask, torch.ones_like(dist) * 10.0, dist)
            
            # ReLU signed distance: penalize when dist < threshold
            # signed_dist = dist - threshold (negative means intersection)
            # loss = relu(-signed_dist) = relu(threshold - dist)
            loss = torch.relu(threshold - dist)  # (P, 3, 3)
            
            # Sum over all edge pairs
            total_loss = total_loss + loss.sum(dim=(1, 2))  # (P,)
    
    return total_loss