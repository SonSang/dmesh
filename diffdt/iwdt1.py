import torch as th
from diffdt.pd import PDStruct
from diffdt.sso import SSO, th_compute_hplane
from . import _C

'''
Differentiable dual line computation for each face.
'''
def compute_face_dual_line(pd: PDStruct,
                        faces: th.Tensor,
                        device,
                        eps=1e-6):

    num_faces = len(faces)
    
    # compute dual half planes;
    faces_points_id0 = faces[:, 0].to(dtype=th.long)
    faces_points_id1 = faces[:, 1].to(dtype=th.long)
    faces_points_id2 = faces[:, 2].to(dtype=th.long)

    faces_points_pos0 = pd.points.positions[faces_points_id0]
    faces_points_pos1 = pd.points.positions[faces_points_id1]
    faces_points_pos2 = pd.points.positions[faces_points_id2]

    faces_points_w0 = pd.points.weights[faces_points_id0]
    faces_points_w1 = pd.points.weights[faces_points_id1]
    faces_points_w2 = pd.points.weights[faces_points_id2]

    faces_points_pos_merge0 = th.cat([faces_points_pos0, faces_points_pos0], dim=0)
    faces_points_pos_merge1 = th.cat([faces_points_pos1, faces_points_pos2], dim=0)
    faces_points_w_merge0 = th.cat([faces_points_w0, faces_points_w0], dim=0)
    faces_points_w_merge1 = th.cat([faces_points_w1, faces_points_w2], dim=0)

    faces_hp_normal_merge, faces_hp_offset_merge = \
        th_compute_hplane(
            faces_points_pos_merge0,
            faces_points_w_merge0[:, None],
            faces_points_pos_merge1,
            faces_points_w_merge1[:, None]
        )
    faces_hp_normal0 = faces_hp_normal_merge[:num_faces]        # half plane bw 0 and 1
    faces_hp_normal1 = faces_hp_normal_merge[num_faces:]        # half plane bw 0 and 2
    faces_hp_offset0 = faces_hp_offset_merge[:num_faces]
    faces_hp_offset1 = faces_hp_offset_merge[num_faces:]

    # compute dual lines;
    dl_directions = th.cross(faces_hp_normal0, faces_hp_normal1, dim=-1)
    dl_directions = th.nn.functional.normalize(dl_directions, dim=-1)

    dot = th.sum(faces_hp_normal0 * faces_hp_normal1, dim=-1, keepdim=True)
    dot_sq = dot * dot
    dot_sq = th.clamp(dot_sq, max=1.0 - eps)
    c1 = (faces_hp_offset0 - (faces_hp_offset1 * dot)) / (1.0 - dot_sq)
    c2 = (faces_hp_offset1 - (faces_hp_offset0 * dot)) / (1.0 - dot_sq)
    dl_origins = (c1 * faces_hp_normal0) + (c2 * faces_hp_normal1)

    # compute axis on the plane that is perpendicular to the dual line;
    dummy0 = th.zeros_like(dl_directions)
    dummy1 = th.zeros_like(dl_directions)
    dummy0[:, 0] = 1.0
    dummy1[:, 1] = 1.0
    
    dl_axis_0 = th.cross(dl_directions, dummy0, dim=-1)
    dl_axis_0_norm = th.norm(dl_axis_0, p=2, dim=-1)
    dl_axis_0[dl_axis_0_norm < eps] = th.cross(dl_directions, dummy1, dim=-1)[dl_axis_0_norm < eps]
    dl_axis_0 = th.nn.functional.normalize(dl_axis_0, dim=-1)
    dl_axis_1 = th.cross(dl_directions, dl_axis_0, dim=-1)

    return dl_origins, dl_directions, dl_axis_0, dl_axis_1

'''
Differentiable line-line seg distance computation.
'''
def lineseg_line_distance(
    # lineseg;
    lineseg_beg: th.Tensor,
    lineseg_end: th.Tensor,
    lineseg_inf: th.Tensor,

    # line;
    line_origin: th.Tensor,
    line_direction: th.Tensor,
    line_axis_0: th.Tensor,
    line_axis_1: th.Tensor,

    eps=1e-6
):
    '''
    @ lineseg_beg: [..., 3]
    @ lineseg_end: [..., 3]
    @ lineseg_inf: [..., 1]

    @ line_origin: [..., 3]
    @ line_direction: [..., 3]
    @ line_axis_0: [..., 3]
    @ line_axis_1: [..., 3]

    @ return: [..., 1]
    '''
    # compute distance between lineseg and line;
    diff_0 = lineseg_beg - line_origin
    diff_1 = lineseg_end - line_origin

    lineseg0_x = th.sum(diff_0 * line_axis_0, dim=-1, keepdim=True)
    lineseg0_y = th.sum(diff_0 * line_axis_1, dim=-1, keepdim=True)
    lineseg1_x = th.sum(diff_1 * line_axis_0, dim=-1, keepdim=True)
    lineseg1_y = th.sum(diff_1 * line_axis_1, dim=-1, keepdim=True)

    lineseg0_xy = th.cat([lineseg0_x, lineseg0_y], dim=-1)
    lineseg1_xy = th.cat([lineseg1_x, lineseg1_y], dim=-1)

    lineseg_dir_xy = lineseg1_xy - lineseg0_xy

    nom = th.sum(lineseg0_xy * lineseg_dir_xy, dim=-1, keepdim=True)
    denom = th.sum(lineseg_dir_xy * lineseg_dir_xy, dim=-1, keepdim=True)
    denom = th.clamp(denom, min=eps)

    t = -(nom / denom)
    t[~lineseg_inf] = th.clamp(t[~lineseg_inf], 0., 1.)
    t[lineseg_inf] = th.clamp(t[lineseg_inf], 0., float("inf"))

    lineseg_foot_xy = lineseg0_xy + t * lineseg_dir_xy

    diff_norm = th.norm(lineseg_foot_xy, p=2, dim=-1, keepdim=True)

    return diff_norm

'''
Differentiable delta computation for each face.
'''
def compute_face_delta(pd: PDStruct, 
                        faces_dl_origin: th.Tensor,
                        faces_dl_direction: th.Tensor,
                        faces_dl_axis_0: th.Tensor,
                        faces_dl_axis_1: th.Tensor,
                        per_face_dl_nearest_pd_edge_idx: th.Tensor,
                        device):

    num_faces = len(faces_dl_origin)

    '''
    Compute delta for each pd edge.
    '''
    
    pf_pd_edges_vertex_id = pd.pd_edges.vertex_id[per_face_dl_nearest_pd_edge_idx.to(dtype=th.long)]  # [# faces, 3, 2]
    pf_pd_edges_vertex_id_0 = pf_pd_edges_vertex_id[:, :, 0].to(dtype=th.long)                        # [# faces, 3]
    pf_pd_edges_vertex_id_1 = pf_pd_edges_vertex_id[:, :, 1].to(dtype=th.long)                        # [# faces, 3]

    pf_pd_edges_origin = pd.pd_edges.origin[per_face_dl_nearest_pd_edge_idx.to(dtype=th.long)]  # [# faces, 3, 3]
    pf_pd_edges_direction = pd.pd_edges.direction[per_face_dl_nearest_pd_edge_idx.to(dtype=th.long)]  # [# faces, 3, 3]

    # finite;
    pf_pd_edges_vertex_pos_0 = pd.pd_vertices.positions[pf_pd_edges_vertex_id_0]    # [# faces, 3, 3]
    pf_pd_edges_vertex_pos_1 = pd.pd_vertices.positions[pf_pd_edges_vertex_id_1]    # [# faces, 3, 3]

    # infinite;
    inf_edge = (pf_pd_edges_vertex_id_1 == -1)  # [# faces, 3]
    pf_pd_edges_vertex_pos_0[inf_edge] = pf_pd_edges_origin[inf_edge]
    pf_pd_edges_vertex_pos_1[inf_edge] = pf_pd_edges_origin[inf_edge] + pf_pd_edges_direction[inf_edge]

    face_delta = lineseg_line_distance(
        pf_pd_edges_vertex_pos_0,
        pf_pd_edges_vertex_pos_1,
        inf_edge[..., None],

        faces_dl_origin.unsqueeze(1),
        faces_dl_direction.unsqueeze(1),
        faces_dl_axis_0.unsqueeze(1),
        faces_dl_axis_1.unsqueeze(1)
    )   # [# faces, 3, 1]

    face_delta = th.squeeze(face_delta, dim=-1)                                     # [# faces, 3]
    face_delta = th.mean(face_delta, dim=-1)                                        # [# faces]
    face_delta = face_delta * -1.
    
    return face_delta

'''
CUDA
'''
class IWDT1ComputationLayer(th.autograd.Function):

    @staticmethod
    def forward(ctx, 
                pd: PDStruct,
                faces: th.Tensor,
                faces_dl_origin: th.Tensor,
                faces_dl_direction: th.Tensor,
                faces_dl_axis_0: th.Tensor,
                faces_dl_axis_1: th.Tensor,
                device):

        pd_rp_id = pd.rp_point_id 
        pd_per_point_rp_id = pd.per_point_rp_id

        pd_vertex_positions = pd.pd_vertices.positions
        pd_vertex_points_id = pd.pd_vertices.points_id

        pd_edge_vertex_id = pd.pd_edges.vertex_id
        pd_edge_points_id = pd.pd_edges.points_id
        pd_edge_origin = pd.pd_edges.origin
        pd_edge_direction = pd.pd_edges.direction

        pd_per_rp_edge_index_array = pd.per_rp_edge_info.index_array
        pd_per_rp_edge_beg_array = pd.per_rp_edge_info.beg_array
        pd_per_rp_edge_end_array = pd.per_rp_edge_info.end_array
        pd_per_rp_edge_dist_array = pd.per_rp_edge_info.dist_array

        ctx.save_for_backward(pd_rp_id, pd_per_point_rp_id,
                            pd_vertex_positions, pd_vertex_points_id,
                            pd_edge_vertex_id, pd_edge_points_id,
                            pd_edge_origin, pd_edge_direction,
                            pd_per_rp_edge_index_array,
                            pd_per_rp_edge_beg_array,
                            pd_per_rp_edge_end_array,
                            pd_per_rp_edge_dist_array,
                            
                            faces,
                            faces_dl_origin,
                            faces_dl_direction,
                            faces_dl_axis_0,
                            faces_dl_axis_1)
        return _C.compute_iwdt1(
            pd_rp_id, pd_per_point_rp_id,
            pd_vertex_positions, pd_vertex_points_id,
            pd_edge_vertex_id, pd_edge_points_id,
            pd_edge_origin, pd_edge_direction,
            pd_per_rp_edge_index_array,
            pd_per_rp_edge_beg_array,
            pd_per_rp_edge_end_array,
            pd_per_rp_edge_dist_array,
            
            faces,
            faces_dl_origin,
            faces_dl_direction,
            faces_dl_axis_0,
            faces_dl_axis_1)
    
    @staticmethod
    def backward(ctx, grad_output):

        raise NotImplementedError()

'''
Torch: Slow, but easy to debug
'''
def th_iwdt1(pd: PDStruct,
            faces: th.Tensor,
            faces_dl_origin: th.Tensor,
            faces_dl_direction: th.Tensor,
            faces_dl_axis_0: th.Tensor,
            faces_dl_axis_1: th.Tensor,
            device):
    
    '''
    For each dual line of faces, find the nearest pd edge of each power cell.
    '''
    pd_edges_vertex_id = pd.pd_edges.vertex_id.to(dtype=th.long)

    per_face_dl_nearest_pd_edge_idx = []

    num_faces = len(faces)
    for i in range(num_faces):
        
        curr_face_dl_origin = faces_dl_origin[i]
        curr_face_dl_direction = faces_dl_direction[i]
        curr_face_dl_axis_0 = faces_dl_axis_0[i]
        curr_face_dl_axis_1 = faces_dl_axis_1[i]

        # for each point in this face, get pd edges of each power cell
        # and find min distance between pd edge and dual line to get delta;
        
        nearest_pd_edge_idx = []
        for j in range(3):
            curr_point_id = faces[i, j]
            curr_point_rp_id = pd.per_point_rp_id[curr_point_id]
            assert curr_point_rp_id != -1, "Point does not have a power cell!"

            # get pd edges of this power cell;
            curr_point_pd_edge_beg = pd.per_rp_edge_info.beg_array[curr_point_rp_id]
            curr_point_pd_edge_end = pd.per_rp_edge_info.end_array[curr_point_rp_id]
            curr_point_pd_edge_idx = pd.per_rp_edge_info.index_array[curr_point_pd_edge_beg:curr_point_pd_edge_end]

            min_dist = float("inf")
            min_dist_hp_id = -1
            for k in range(len(curr_point_pd_edge_idx)):
                cmp_pd_edge_id = curr_point_pd_edge_idx[k]

                cmp_pd_edge_vertex_id_0 = pd_edges_vertex_id[cmp_pd_edge_id, 0]
                cmp_pd_edge_vertex_id_1 = pd_edges_vertex_id[cmp_pd_edge_id, 1]

                cmp_pd_edge_vertex_pos_0 = pd.pd_vertices.positions[cmp_pd_edge_vertex_id_0]
                cmp_pd_edge_vertex_pos_1 = pd.pd_vertices.positions[cmp_pd_edge_vertex_id_1]

                if pd_edges_vertex_id[cmp_pd_edge_id, 1] == -1:
                    cmp_pd_edge_origin = pd.pd_edges.origin[cmp_pd_edge_id]
                    cmp_pd_edge_direction = pd.pd_edges.direction[cmp_pd_edge_id]

                    cmp_pd_edge_vertex_pos_0 = cmp_pd_edge_origin
                    cmp_pd_edge_vertex_pos_1 = cmp_pd_edge_origin + cmp_pd_edge_direction

                # compute distance between dual line and pd edge;
                dist = lineseg_line_distance(
                    cmp_pd_edge_vertex_pos_0[None, :],
                    cmp_pd_edge_vertex_pos_1[None, :],
                    (pd_edges_vertex_id[cmp_pd_edge_id, 1] == -1)[None,][None, :],

                    curr_face_dl_origin[None, :],
                    curr_face_dl_direction[None, :],
                    curr_face_dl_axis_0[None, :],
                    curr_face_dl_axis_1[None, :]
                )
                if dist < min_dist or min_dist_hp_id == -1:
                    min_dist = dist
                    min_dist_hp_id = cmp_pd_edge_id

            nearest_pd_edge_idx.append(min_dist_hp_id)
        
        nearest_pd_edge_idx = th.stack(nearest_pd_edge_idx, dim=0)
        per_face_dl_nearest_pd_edge_idx.append(nearest_pd_edge_idx)

    per_face_dl_nearest_pd_edge_idx = th.stack(per_face_dl_nearest_pd_edge_idx, dim=0)
    return per_face_dl_nearest_pd_edge_idx
        
class IWDT1:

    def __init__(self):
        '''
        Values needed for computing inclusion score for non-existing faces,
        which are the duals of the lines in R^3.
        '''
        self.faces: th.Tensor = None                            # [# faces, 3]
        self.per_face_delta: th.Tensor = None                   # [# faces]
        self.per_face_dl_origin: th.Tensor = None               # [# faces, 3], dual line
        self.per_face_dl_direction: th.Tensor = None            # [# faces, 3], dual line
        self.per_face_dl_axis_0: th.Tensor = None               # [# faces, 3], dual line
        self.per_face_dl_axis_1: th.Tensor = None               # [# faces, 3], dual line
        self.per_face_dl_nearest_pd_edge_idx: th.Tensor = None  # [# faces, 3]

        self.faces_wo_pc: th.Tensor = None                      # [# faces_wo_pc, 3]

    @staticmethod
    def forward(pd: PDStruct,
                faces: th.Tensor,
                device,
                mode: str='cuda'):
        '''
        Compute differentiable inclusion score for given faces.

        @ mode: ['torch', 'cuda']
        '''
        result = IWDT1()

        with th.no_grad():
            # extract faces with points of no power cell;
            faces_w_pc, faces_wo_pc = IWDT1.classify_faces_w_power_cell(pd, faces)

            # extract non-existing faces only;
            # [faces_wo_pc] does not exist already;
            faces_w_pc = IWDT1.extract_non_existing_faces(pd, faces_w_pc)

        # compute dual lines;
        per_face_dl_origin, per_face_dl_direction, \
            per_face_dl_axis_0, per_face_dl_axis_1 = \
            compute_face_dual_line(pd, faces_w_pc, device)

        with th.no_grad():

            if mode == 'torch':

                per_face_dl_nearest_pd_edge_idx = \
                    th_iwdt1(pd, 
                            faces_w_pc, 
                            per_face_dl_origin,
                            per_face_dl_direction,
                            per_face_dl_axis_0,
                            per_face_dl_axis_1,
                            device)
            
            elif mode == 'cuda':

                per_face_dl_nearest_pd_edge_idx = \
                    IWDT1ComputationLayer.apply(
                        pd, 
                        faces_w_pc,
                        per_face_dl_origin,
                        per_face_dl_direction,
                        per_face_dl_axis_0,
                        per_face_dl_axis_1,
                        device)
            
            else:

                raise ValueError()

        face_delta = \
            compute_face_delta(
                pd, 
                per_face_dl_origin,
                per_face_dl_direction,
                per_face_dl_axis_0,
                per_face_dl_axis_1,
                per_face_dl_nearest_pd_edge_idx,
                device
            )

        result.faces = faces_w_pc
        result.per_face_delta = face_delta
        result.per_face_dl_origin = per_face_dl_origin
        result.per_face_dl_direction = per_face_dl_direction
        result.per_face_dl_axis_0 = per_face_dl_axis_0
        result.per_face_dl_axis_1 = per_face_dl_axis_1
        result.per_face_dl_nearest_pd_edge_idx = per_face_dl_nearest_pd_edge_idx

        result.faces_wo_pc = faces_wo_pc

        return result

    @staticmethod
    def extract_non_existing_faces(pd: PDStruct, faces: th.Tensor):

        with th.no_grad():
            t_faces = th.sort(faces, dim=-1)[0]
            t_faces = th.unique(t_faces, dim=0)

            existing_faces = pd.pd_edges.points_id
            existing_faces = th.sort(existing_faces, dim=-1)[0]

            # find common faces;
            merge_faces = th.cat([t_faces, existing_faces], dim=0)
            u_merge_faces, u_merge_faces_cnt = th.unique(merge_faces, return_counts=True, dim=0)
            common_faces = u_merge_faces[u_merge_faces_cnt > 1]

            # subtract common faces from t_faces;
            merge_faces = th.cat([t_faces, common_faces], dim=0)
            u_merge_faces, u_merge_faces_cnt = th.unique(merge_faces, return_counts=True, dim=0)
            non_existing_faces = u_merge_faces[u_merge_faces_cnt == 1]

            return non_existing_faces

    @staticmethod
    def classify_faces_w_power_cell(pd: PDStruct, faces: th.Tensor):

        with th.no_grad():
            faces_points_rp_id = pd.per_point_rp_id[faces.to(dtype=th.long)]
            target_faces = th.all(faces_points_rp_id != -1, dim=-1)

            return faces[target_faces], faces[~target_faces]