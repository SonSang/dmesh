import torch as th
import numpy as np
import trimesh
import time
import os
from pytorch3d.ops import knn_points

def import_mesh(fname, device, scale=0.8):
    if fname.endswith(".obj"):
        target_mesh = trimesh.load_mesh(fname, 'obj')
    elif fname.endswith(".stl"):
        target_mesh = trimesh.load_mesh(fname, 'stl')    
    elif fname.endswith(".ply"):
        target_mesh = trimesh.load_mesh(fname, 'ply')
    else:
        raise ValueError(f"unknown mesh file type: {fname}")

    target_vertices, target_faces = target_mesh.vertices,target_mesh.faces
    target_vertices, target_faces = \
        th.tensor(target_vertices, dtype=th.float32, device=device), \
        th.tensor(target_faces, dtype=th.long, device=device)
    
    # normalize to fit mesh into a sphere of radius [scale];
    if scale > 0:
        target_vertices = target_vertices - target_vertices.mean(dim=0, keepdim=True)
        max_norm = th.max(th.norm(target_vertices, dim=-1)) + 1e-6
        target_vertices = (target_vertices / max_norm) * scale

    return target_vertices, target_faces

def extract_faces(tet_points_id: th.Tensor):
    '''
    @ tet_points_id: [# tet, 4], indices of points in each tet
    '''
    faces = th.cat([
        tet_points_id[:, [0, 1, 2]],
        tet_points_id[:, [0, 1, 3]],
        tet_points_id[:, [0, 2, 3]],
        tet_points_id[:, [1, 2, 3]]
    ], dim=0)
    faces = th.sort(faces, dim=-1)[0]
    faces = th.unique(faces, dim=0)
    return faces

def time_str():
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

def setup_logdir(path: str):
    # n_path = os.path.join(path, time_str())
    if os.path.exists(path) == False:
        os.makedirs(path)
    else:
        raise ValueError(f"Directory already exists: {path}")
    return path

def sample_bary(num_faces: int, device):    
    face_bary = th.zeros((num_faces, 3), dtype=th.float32, device=device)

    face_coords = th.rand((num_faces, 2), dtype=th.float32, device=device)
    face_coords_valid = (face_coords[:, 0] + face_coords[:, 1]) < 1.0
    face_coords[~face_coords_valid] = 1.0 - face_coords[~face_coords_valid]
    assert th.all(face_coords[:, 0] + face_coords[:, 1] <= 1.0), "face coords should be valid."

    face_bary[:, 0] = 1.0 - face_coords[:, 0] - face_coords[:, 1]
    face_bary[:, 1] = face_coords[:, 0]
    face_bary[:, 2] = face_coords[:, 1]

    return face_bary

def sample_points_on_mesh(positions: th.Tensor,
                            faces: th.Tensor,
                            num_points: int,
                            device):

    '''
    Sample points on faces.
    The points are distributed based on the size of triangular faces.

    @ return
    @ sample_positions: [num_points, 3]
    '''

    with th.no_grad():
        # compute face areas;
        face_areas = triangle_area(positions, faces)
        assert th.all(face_areas >= 0.0), "face area should be non-negative."
        assert th.sum(face_areas) > 0, "sum of face area should be positive."

        # sample faces;
        with th.no_grad():
            face_idx = th.multinomial(
                face_areas, 
                num_points, 
                replacement=True
            )

        face_bary = sample_bary(num_points, device)

    # compute sample positions;
    sample_positions = th.sum(
        positions[faces[face_idx]] * face_bary.unsqueeze(-1),
        dim=-2
    )

    # compute sample normals;
    sample_normals = th.nn.functional.normalize(
        th.cross(
            positions[faces[face_idx][:, 1]] - positions[faces[face_idx][:, 0]],
            positions[faces[face_idx][:, 2]] - positions[faces[face_idx][:, 0]],
            dim=-1
        ), dim=-1
    )

    return sample_positions, sample_normals

def grid_points(resolution: int, lower_bound: float, upper_bound: float, device):
    N1 = resolution
    N2 = N1 * N1
    N3 = N2 * N1
    GRID_SIZE = (upper_bound - lower_bound) / (N1)

    points0 = th.arange(0, N3, 1, dtype=th.int64, device=device)
    points_z = points0 // (N2)
    points_y = (points0 % (N2)) // (N1)
    points_x = (points0 % (N2)) % (N1)

    points = th.stack([points_x, points_y, points_z], dim=-1) / (N1 - 1)
    points = (lower_bound + GRID_SIZE * 0.5) + (upper_bound - lower_bound - GRID_SIZE) * points
    grid_lb = points - (GRID_SIZE * 0.5)
    grid_ub = points + (GRID_SIZE * 0.5)

    return points, grid_lb, grid_ub

def triangle_aspect_ratio(points: th.Tensor, faces: th.Tensor):
    '''
    Compute aspect ratio of triangles, which is in range [1, inf).
    '''

    face_vertex_0 = points[faces[:, 0]]
    face_vertex_1 = points[faces[:, 1]]
    face_vertex_2 = points[faces[:, 2]]

    face_edge_dir_0 = face_vertex_1 - face_vertex_0
    face_edge_dir_1 = face_vertex_2 - face_vertex_0
    face_edge_dir_2 = face_vertex_2 - face_vertex_1

    face_edge_len_0 = th.norm(face_edge_dir_0, dim=-1)
    face_edge_len_1 = th.norm(face_edge_dir_1, dim=-1)
    face_edge_len_2 = th.norm(face_edge_dir_2, dim=-1)

    face_area_0 = th.norm(th.cross(face_edge_dir_0, face_edge_dir_1, dim=-1), dim=-1)
    face_area_1 = th.norm(th.cross(face_edge_dir_1, face_edge_dir_2, dim=-1), dim=-1)
    face_area_2 = th.norm(th.cross(face_edge_dir_2, face_edge_dir_0, dim=-1), dim=-1)

    face_height_0 = face_area_0 / (face_edge_len_0 + 1e-6)
    face_height_1 = face_area_1 / (face_edge_len_1 + 1e-6)
    face_height_2 = face_area_2 / (face_edge_len_2 + 1e-6)

    max_face_edge_len = th.max(th.stack([face_edge_len_0, face_edge_len_1, face_edge_len_2], dim=-1), dim=-1)[0]
    min_face_height = th.min(th.stack([face_height_0, face_height_1, face_height_2], dim=-1), dim=-1)[0]
    min_face_height = min_face_height * (2.0 / np.sqrt(3.0))

    ar = max_face_edge_len / (min_face_height + 1e-6)

    # there is numerical instability when [min_face_height] is too small;
    # assert th.all(ar >= 0.99), "Aspect ratio should be >= 1.0"
    
    return ar

def triangle_area(points: th.Tensor, faces: th.Tensor):
    face_areas = th.norm(th.cross(
        points[faces[:, 1]] - points[faces[:, 0]],
        points[faces[:, 2]] - points[faces[:, 0]],
        dim=-1
    ), dim=-1) * 0.5

    return face_areas

'''
Distance functions
'''
def point_lineseg_distance(points: th.Tensor, lines: th.Tensor):
    '''
    @points: [..., 3]
    @lines: [..., 2, 3]
    '''

    assert points.ndim == lines.ndim - 1, "Invalid input dimensions"
    assert points.shape[-1] == 3, "Invalid input dimensions"
    assert lines.shape[-1] == 3, "Invalid input dimensions"
    assert lines.shape[-2] == 2, "Invalid input dimensions"

    p0 = lines[..., 0, :]
    p1 = lines[..., 1, :]
    v = p1 - p0

    # project points onto the line
    t = th.sum(v * (points - p0), dim=-1) / (th.sum(v * v, dim=-1) + 1e-6)  # [...]
    t = th.clamp(t, 0, 1)

    foot = p0 + t.unsqueeze(-1) * v  # [..., 3]
    dist = th.norm(points - foot, dim=-1)  # [...]

    return dist

def point_tri_distance(points: th.Tensor, tris: th.Tensor):
    '''
    @ points: [M, N, 3]
    @ tris: [M, N, 3, 3]
    '''

    # Helper function to compute the area of a triangle given its vertices
    def triangle_area(v1, v2, v3):
        return 0.5 * th.norm(th.cross(v2 - v1, v3 - v1), dim=-1)

    # Expand dimensions for broadcasting
    points_expanded = points               # Shape: [M, N, 3]
    triangles_expanded = tris              # Shape: [M, N, 3, 3]

    # Compute vectors for the edges of the triangles
    edge1 = triangles_expanded[:, :, 1] - triangles_expanded[:, :, 0]  # Shape: [M, N, 3]
    edge2 = triangles_expanded[:, :, 2] - triangles_expanded[:, :, 0]  # Shape: [M, N, 3]

    # Compute the normal vector for each triangle
    normals = th.cross(edge1, edge2, dim=-1)  # Shape: [M, N, 3]

    # Normalize the normal vectors
    normals = th.nn.functional.normalize(normals, dim=-1)

    # Compute vectors from points to a vertex of each triangle
    point_to_vertex = points_expanded - triangles_expanded[:, :, 0]     # Shape: [M, N, 3]

    # Compute perpendicular distance from points to triangle planes
    dot_product = th.sum(point_to_vertex * normals, dim=-1)        # Shape: [M, N]
    foot_points = points_expanded - dot_product.unsqueeze(-1) * normals     # Shape: [M, N, 3]
    
    # Compute barycentric coordinates to check if the perpendicular point is inside the triangle
    area_full = triangle_area(tris[:, :, 0], tris[:, :, 1], tris[:, :, 2])  # Shape: [M, N]
    area1 = triangle_area(foot_points, triangles_expanded[:, :, 1], triangles_expanded[:, :, 2])  # Shape: [M, N]
    area2 = triangle_area(foot_points, triangles_expanded[:, :, 2], triangles_expanded[:, :, 0])  # Shape: [M, N]
    area3 = triangle_area(foot_points, triangles_expanded[:, :, 0], triangles_expanded[:, :, 1])  # Shape: [M, N]

    # Check if point is inside the triangle using barycentric coordinates
    is_inside = (area1 + area2 + area3 <= area_full + 1e-5)

    # Compute distance from point to each edge of the triangle if outside
    edge_dist1 = point_lineseg_distance(points, tris[:, :, [0, 1]])  # Shape: [M, N]
    edge_dist2 = point_lineseg_distance(points, tris[:, :, [0, 2]])  # Shape: [M, N]
    edge_dist3 = point_lineseg_distance(points, tris[:, :, [1, 2]])  # Shape: [M, N]
    
    # Choose the minimum distance from the point to an edge of the triangle
    min_edge_dist = th.min(th.min(edge_dist1, edge_dist2), edge_dist3)

    # Final distance is either perpendicular or edge distance based on whether the point is inside the triangle
    perpendicular_distance = th.abs(dot_product)
    final_distance = th.where(is_inside, perpendicular_distance, min_edge_dist)

    return final_distance

'''
KNN
'''
def run_knn(src: th.Tensor, tgt: th.Tensor, k: int):

    '''
    @ src: [M, 3]
    @ tgt: [N, 3]
    '''

    assert src.ndim == 2, "Invalid input dimensions"
    assert tgt.ndim == 2, "Invalid input dimensions"
    assert src.shape[-1] == 3, "Invalid input dimensions"
    assert tgt.shape[-1] == 3, "Invalid input dimensions"

    t_src = src.unsqueeze(0)
    t_tgt = tgt.unsqueeze(0)
    knn_result = knn_points(t_src, t_tgt, K=k)

    knn_idx = knn_result.idx.squeeze(0)
    knn_dist = th.sqrt(knn_result.dists.squeeze(0))

    return knn_idx, knn_dist

def nd_chamfer_dist(x: th.Tensor, y: th.Tensor):
    '''
    @ x: [M, 3]
    @ y: [N, 3]
    '''
    with th.no_grad():

        assert x.ndim == 2, "Invalid input dimensions"
        assert y.ndim == 2, "Invalid input dimensions"
        assert x.shape[-1] == 3, "Invalid input dimensions"
        assert y.shape[-1] == 3, "Invalid input dimensions"

        knn_idx_x2y, knn_dist_x2y = run_knn(x, y, k=1)
        knn_idx_y2x, knn_dist_y2x = run_knn(y, x, k=1)

        return th.mean(knn_dist_x2y), th.mean(knn_dist_y2x)
    
'''
Differentiable max & min
'''
def dmax(val: th.Tensor, k: float = 100):
    '''
    @ val: [# elem, N]
    '''
    with th.no_grad():
        e_val_denom = val * k
        e_val_denom_max = th.max(e_val_denom, dim=-1, keepdim=True)[0]
        e_val_denom = e_val_denom - e_val_denom_max

        e_val_denom = th.exp(e_val_denom)
        e_val_nom = th.sum(e_val_denom, dim=-1, keepdim=True)
        e_val = e_val_denom / e_val_nom

    return th.sum(e_val * val, dim=-1)

def dmin(val: th.Tensor, k: float = 1000):
    return -dmax(-val, k)