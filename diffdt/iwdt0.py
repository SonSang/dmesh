import torch as th
from diffdt.pd import PDStruct
from diffdt.sso import SSO
from . import _C

'''
Differentiable delta computation for each pd edge.
'''
def compute_pd_edge_delta(pd: PDStruct, 
                        sso: SSO, 
                        per_pd_edge_nearest_hp_idx: th.Tensor, 
                        infinite_edge_delta: float,
                        device):

    num_pd_edges = len(pd.pd_edges.points_id)

    '''
    Compute sample point on each pd edge.
    '''
    pd_edges_sample_points = th.zeros((num_pd_edges, 3), dtype=th.float32, device=device)

    pd_edges_vertex_id_0 = pd.pd_edges.vertex_id[:, 0]
    pd_edges_vertex_id_1 = pd.pd_edges.vertex_id[:, 1]
    finite_pd_edges = (pd_edges_vertex_id_1 != -1)

    pd_edges_vertex_pos_0 = pd.pd_vertices.positions[pd_edges_vertex_id_0.to(dtype=th.long)]

    # finite edges;
    pd_edges_vertex_pos_1 = pd.pd_vertices.positions[pd_edges_vertex_id_1.to(dtype=th.long)]

    # infinite edges;
    pd_edges_vertex_pos_1[~finite_pd_edges] = \
        pd.pd_edges.origin[~finite_pd_edges] + \
        infinite_edge_delta * pd.pd_edges.direction[~finite_pd_edges]

    pd_edges_sample_points = \
        0.5 * (pd_edges_vertex_pos_0 + pd_edges_vertex_pos_1)

    '''
    Compute delta for each pd edge.
    '''
    pd_edges_nearest_hp_normal = sso.per_rp_hp_normal_array[per_pd_edge_nearest_hp_idx.to(dtype=th.long)] # [# edges, 3, 3]
    pd_edges_nearest_hp_offset = sso.per_rp_hp_offset_array[per_pd_edge_nearest_hp_idx.to(dtype=th.long)] # [# edges, 3]
    pd_edges_deltas = th.abs(
        th.sum(pd_edges_nearest_hp_normal * pd_edges_sample_points.unsqueeze(1), dim=-1) - \
        pd_edges_nearest_hp_offset) # [# edges, 3]
    pd_edges_delta = th.mean(pd_edges_deltas, dim=-1) # [# edges]

    return pd_edges_delta, pd_edges_sample_points

'''
CUDA
'''
class IWDT0ComputationLayer(th.autograd.Function):

    @staticmethod
    def forward(ctx, pd: PDStruct, sso: SSO, infinite_edge_delta: float):

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

        sso_per_rp_hp_pid_array = sso.per_rp_hp_pid_array
        sso_per_rp_hp_normal_array = sso.per_rp_hp_normal_array
        sso_per_rp_hp_offset_array = sso.per_rp_hp_offset_array
        sso_per_rp_beg_array = sso.per_rp_beg_array
        sso_per_rp_end_array = sso.per_rp_end_array

        ctx.save_for_backward(pd_rp_id, pd_per_point_rp_id,
                            pd_vertex_positions, pd_vertex_points_id,
                            pd_edge_vertex_id, pd_edge_points_id,
                            pd_edge_origin, pd_edge_direction,
                            pd_per_rp_edge_index_array,
                            pd_per_rp_edge_beg_array,
                            pd_per_rp_edge_end_array,
                            pd_per_rp_edge_dist_array,
                            sso_per_rp_hp_pid_array,
                            sso_per_rp_hp_normal_array,
                            sso_per_rp_hp_offset_array,
                            sso_per_rp_beg_array,
                            sso_per_rp_end_array)
        return _C.compute_iwdt0(
            pd_rp_id, pd_per_point_rp_id,
            pd_vertex_positions, pd_vertex_points_id,
            pd_edge_vertex_id, pd_edge_points_id,
            pd_edge_origin, pd_edge_direction,
            pd_per_rp_edge_index_array,
            pd_per_rp_edge_beg_array,
            pd_per_rp_edge_end_array,
            pd_per_rp_edge_dist_array,
            sso_per_rp_hp_pid_array,
            sso_per_rp_hp_normal_array,
            sso_per_rp_hp_offset_array,
            sso_per_rp_beg_array,
            sso_per_rp_end_array,
            infinite_edge_delta)
    
    @staticmethod
    def backward(ctx, grad_output):

        raise NotImplementedError()

'''
Torch: Slow, but easy to debug
'''
def th_iwdt0(pd: PDStruct,
            sso: SSO,
            infinite_edge_delta: float,
            device):
    
    '''
    Compute scores for existing faces.
    '''
    pd_edges_points_id = pd.pd_edges.points_id
    pd_edges_vertex_id = pd.pd_edges.vertex_id

    per_pd_edge_nearest_hp_idx = []

    for i in range(len(pd_edges_points_id)):
        
        curr_pd_edge_points_id = pd_edges_points_id[i]
        curr_pd_edge_vertex_id = pd_edges_vertex_id[i]

        # two pd vertices that comprise this edge;
        curr_pd_edge_vertex_id_0 = curr_pd_edge_vertex_id[0]
        curr_pd_edge_vertex_id_1 = curr_pd_edge_vertex_id[1]

        curr_pd_edge_vertex_pos_0 = pd.pd_vertices.positions[curr_pd_edge_vertex_id_0]
        curr_pd_edge_vertex_pos_1 = pd.pd_vertices.positions[curr_pd_edge_vertex_id_1]

        if curr_pd_edge_vertex_id_1 == -1:
            # infinite edge;
            curr_pd_edge_direction = pd.pd_edges.direction[i]
            curr_pd_edge_origin = pd.pd_edges.origin[i]
            curr_pd_edge_vertex_pos_1 = curr_pd_edge_origin + \
                infinite_edge_delta * curr_pd_edge_direction

        # take sample point on this edge;
        curr_pd_edge_sample_point = 0.5 * (curr_pd_edge_vertex_pos_0 + curr_pd_edge_vertex_pos_1)
        
        # for each point in this face, get half planes nearyby each power cell
        # and project the above point onto the half planes to find delta;
        delta = 0.0
        nearest_hp_idx = []
        for j in range(3):
            curr_pd_edge_point_id = curr_pd_edge_points_id[j]
            curr_pd_edge_point_rp_id = pd.per_point_rp_id[curr_pd_edge_point_id]
            assert curr_pd_edge_point_rp_id != -1, "Point does not have a power cell!"

            # nearby half planes;
            curr_pd_edge_point_hp_beg = sso.per_rp_beg_array[curr_pd_edge_point_rp_id]
            curr_pd_edge_point_hp_end = sso.per_rp_end_array[curr_pd_edge_point_rp_id]

            # for each nearby half plane, project point onto half plane;
            min_dist = float("inf")
            min_dist_hp_id = -1
            for k in range(curr_pd_edge_point_hp_beg, curr_pd_edge_point_hp_end):
                curr_pd_edge_point_hp_pid = sso.per_rp_hp_pid_array[k]

                # skip if this half plane includes the current pd edge;
                icnt = 0
                for m in range(3):
                    for n in range(2):
                        if curr_pd_edge_points_id[m] == curr_pd_edge_point_hp_pid[n]:
                            icnt += 1
                if icnt >= 2:
                    continue

                curr_pd_edge_point_hp_normal = sso.per_rp_hp_normal_array[k]
                curr_pd_edge_point_hp_offset = sso.per_rp_hp_offset_array[k]

                # project point onto half plane;
                dist = th.abs(th.sum(curr_pd_edge_point_hp_normal * curr_pd_edge_sample_point) - \
                            curr_pd_edge_point_hp_offset)
                
                if dist < min_dist:
                    min_dist = dist
                    min_dist_hp_id = k

            delta = delta + min_dist
            nearest_hp_idx.append(min_dist_hp_id)
        
        delta = delta / 3.0
        per_pd_edge_nearest_hp_idx.append(nearest_hp_idx)

    per_pd_edge_nearest_hp_idx = th.tensor(per_pd_edge_nearest_hp_idx, dtype=th.int32, device=device)
    return per_pd_edge_nearest_hp_idx
        
class IWDT0:

    def __init__(self):
        '''
        Values needed for computing inclusion score for existing faces,
        which are the duals of the edges of the current Power Diagram.
        '''
        self.pd_edge_delta: th.Tensor = None                    # [# edges]
        self.per_pd_edge_sample_point: th.Tensor = None         # [# edges, 3]
        self.per_pd_edge_nearest_hp_idx: th.Tensor = None       # [# edges, 3]

    @staticmethod
    def forward(pd: PDStruct,
                sso: SSO,
                device,
                infinite_edge_delta: float=1e3,
                mode: str='cuda'):
        '''
        Compute differentiable inclusion score for given faces.

        @ mode: ['torch', 'cuda']
        '''
        result = IWDT0()

        with th.no_grad():

            if mode == 'torch':

                per_pd_edge_nearest_hp_idx = \
                    th_iwdt0(pd, sso, infinite_edge_delta, device)
            
            elif mode == 'cuda':

                per_pd_edge_nearest_hp_idx = \
                    IWDT0ComputationLayer.apply(pd, sso, infinite_edge_delta)
            
            else:

                raise ValueError()

        pd_edge_delta, per_pd_edge_sample_point = \
            compute_pd_edge_delta(
                pd, 
                sso, 
                per_pd_edge_nearest_hp_idx, 
                infinite_edge_delta, 
                device
            )
        
        result.pd_edge_delta = pd_edge_delta
        result.per_pd_edge_sample_point = per_pd_edge_sample_point
        result.per_pd_edge_nearest_hp_idx = per_pd_edge_nearest_hp_idx

        return result

    def compute_rp_cell_delta(self, pd: PDStruct, thresh: float, device):
        '''
        Cell existence value, which is sum of delta values for all edges
        of a power cell that is related to a point.
        '''

        # compute cell existence value;
        pd_edge_delta = th.clamp(self.pd_edge_delta, max=thresh)

        per_rp_edge_delta = pd_edge_delta[pd.per_rp_edge_info.index_array.to(dtype=th.long)]
        per_rp_edge_delta_cumsum = th.cumsum(per_rp_edge_delta, dim=0)
        tmp_a = per_rp_edge_delta_cumsum[pd.per_rp_edge_info.end_array.to(dtype=th.long) - 1]
        tmp_b = th.cat([th.zeros(1, dtype=th.float32, device=device), tmp_a])
        rp_cell_delta = tmp_b[1:] - tmp_b[:-1]

        return rp_cell_delta