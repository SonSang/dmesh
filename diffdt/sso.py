import torch as th
from diffdt.pd import PDStruct
from . import _C

'''
Differentiable half plane computation.
@TODO: CUDA impl? (not necessary)
'''
def th_compute_hplane(this_pos: th.Tensor,
                    this_w: th.Tensor,
                    other_pos: th.Tensor,
                    other_w: th.Tensor,
                    eps: float=1e-6):
    '''
    @this_pos: [..., 3]
    @this_w: [..., 1]
    @other_pos: [..., 3]
    @other_w: [..., 1]

    @return: [..., 3], [..., 1]
    '''
    # normal;
    diff = other_pos - this_pos
    dist_sq = th.sum(diff * diff, dim=-1, keepdim=True)
    dist = th.sqrt(dist_sq)
    hp_normal = diff / th.clamp(dist, min=eps)

    # offset;
    delta = 0.5 * (1.0 - ((other_w - this_w) / th.clamp(dist_sq, eps)))
    mid_point = (delta * other_pos) + ((1 - delta) * this_pos)
    hp_offset = th.sum(mid_point * hp_normal, dim=-1, keepdim=True)

    return hp_normal, hp_offset

'''
Torch: Slow, but easy to debug
'''
def th_sso(pd: PDStruct,
            thresh: float,
            device):
    
    num_rp = len(pd.rp_point_id)

    hp_pid = []

    # iterate through each real point;
    for i in range(num_rp):
        rp_id = pd.rp_point_id[i]
        
        this_rp_pos = pd.points.positions[rp_id]
        this_rp_w = pd.points.weights[rp_id]

        # find out pd vertices related to this real point;
        rp_pdv_beg = pd.per_rp_vertex_info.beg_array[i]
        rp_pdv_end = pd.per_rp_vertex_info.end_array[i]
        rp_pdv_idx = pd.per_rp_vertex_info.index_array[rp_pdv_beg:rp_pdv_end]

        # iterate through other real points;
        for j in range(num_rp):

            if i == j:
                continue

            other_rp_id = pd.rp_point_id[j]
            
            other_rp_pos = pd.points.positions[other_rp_id]
            other_rp_w = pd.points.weights[other_rp_id]

            hp_normal, hp_offset = th_compute_hplane(
                this_rp_pos[None, :], 
                this_rp_w[None][None, :], 
                other_rp_pos[None, :], 
                other_rp_w[None][None, :]
            )
            hp_normal = hp_normal[0]
            hp_offset = hp_offset[0]
            
            # iterate through each pd vertex of the real point;
            # find min distance between half plane and power cell of [rp_id];
            min_dist_hp_pc = 1e10
            for k in range(len(rp_pdv_idx)):
                curr_pdv_id = rp_pdv_idx[k]
                curr_pdv_position = pd.pd_vertices.positions[curr_pdv_id]
                curr_pdv_points_id = pd.pd_vertices.points_id[curr_pdv_id]

                # if [other_rp_id] is already in [cc_tri_idx], skip;
                if th.any(curr_pdv_points_id == other_rp_id):
                    min_dist_hp_pc = 0.
                    break
                
                # compute distance from half plane to this cc;
                dist = th.abs(th.sum(hp_normal * curr_pdv_position) - hp_offset)
                if dist < min_dist_hp_pc:
                    min_dist_hp_pc = dist

            if min_dist_hp_pc < thresh:
                hp_pid.append([rp_id.cpu().item(), other_rp_id.cpu().item()])
                
    hp_pid = th.tensor(hp_pid, dtype=th.int32, device=device)
    return hp_pid

class SSO:

    def __init__(self):

        # half plane info;
        self.per_rp_hp_pid_array: th.Tensor = None              # id of points that define half planes;      
        self.per_rp_hp_normal_array: th.Tensor = None
        self.per_rp_hp_offset_array: th.Tensor = None
        self.per_rp_beg_array: th.Tensor = None
        self.per_rp_end_array: th.Tensor = None
        self.thresh: float = None
        
    def get_ith_rp_info(self, i: int):
        hp_pid = self.per_rp_hp_pid_array[self.per_rp_beg_array[i]:self.per_rp_end_array[i]]
        hp_normal = self.per_rp_hp_normal_array[self.per_rp_beg_array[i]:self.per_rp_end_array[i]]
        hp_offset = self.per_rp_hp_offset_array[self.per_rp_beg_array[i]:self.per_rp_end_array[i]]
        return hp_pid, hp_normal, hp_offset
    
    @staticmethod
    def forward(pd: PDStruct, thresh: float, cuda_max_num_point_per_batch: int, mode: str='cuda'):

        result = SSO()
        with th.no_grad():
            device = pd.points.positions.device
            result.thresh = thresh

            if mode == 'torch':
                hp_pid = th_sso(pd, thresh, device)
            else:
                hp_pid = _C.compute_sso(
                    pd.points.positions,
                    pd.points.weights,

                    pd.rp_point_id,
                    pd.per_rp_vertex_info.beg_array,
                    pd.per_rp_vertex_info.end_array,
                    pd.per_rp_vertex_info.index_array,
                    pd.per_rp_vertex_info.dist_array,

                    pd.pd_vertices.points_id,
                    pd.pd_vertices.positions,
                    
                    thresh,
                    cuda_max_num_point_per_batch
                )
                
            hp_pid = th.unique(hp_pid, dim=0)
            u_hplane_x, u_hplane_x_cnt = th.unique_consecutive(hp_pid[:, 0], return_counts=True)
            u_hplane_x_cnt_cumsum = th.cumsum(u_hplane_x_cnt, dim=0).to(dtype=th.int32)
            assert th.all(pd.rp_point_id == u_hplane_x), "SSO failed"

        result.per_rp_hp_pid_array = hp_pid

        # compute half plane info;
        long_hp_pid = hp_pid.to(dtype=th.long)
        # long_hp_pid = long_hp_pid[0]    # debug
        # long_hp_pid = long_hp_pid[:0]   # debug
        if long_hp_pid.ndim == 1:
            long_hp_pid = long_hp_pid[None, :]
        this_positions = pd.points.positions[long_hp_pid[:, 0]]
        this_weights = pd.points.weights[long_hp_pid[:, 0]][:, None]
        other_positions = pd.points.positions[long_hp_pid[:, 1]]
        other_weights = pd.points.weights[long_hp_pid[:, 1]][:, None]
        result.per_rp_hp_normal_array, result.per_rp_hp_offset_array = \
            th_compute_hplane(
                this_positions,
                this_weights,
                other_positions,
                other_weights
            )
        result.per_rp_hp_offset_array = result.per_rp_hp_offset_array[:, 0]
        
        result.per_rp_beg_array = th.cat([th.zeros((1,), dtype=th.int32, device=device), u_hplane_x_cnt_cumsum[:-1]], dim=0)
        result.per_rp_end_array = u_hplane_x_cnt_cumsum

        return result

class PseudoSSO:

    def __init__(self):
        '''
        Rather than computing exact SSO, this class uses history to estimate SSO.
        For each point, it accumulates the history of hplanes that have been
        related to the point.
        '''

        # [N, 2]
        # First col: rp_id
        # Second col: the other rp_id that defines the half plane
        self.per_rp_hp_pid: th.Tensor = None

    def update(self, per_rp_hp_pid: th.Tensor):

        if self.per_rp_hp_pid is None:
            self.per_rp_hp_pid = per_rp_hp_pid
        else:
            self.per_rp_hp_pid = th.cat([self.per_rp_hp_pid, per_rp_hp_pid], dim=0)
        self.per_rp_hp_pid = th.unique(self.per_rp_hp_pid, dim=0)

    def update_with_pd(self, pd: PDStruct):

        # for each real point, find out real points that define tets;
        pairs = []
        for comb in [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]:
            pairs.append(pd.tets_point_id[:, comb])
        pairs = th.cat(pairs, dim=0)
        pairs = th.cat([pairs, pairs.flip(1)], dim=0)
        pairs = th.unique(pairs, dim=0)
        pairs = pairs.to(dtype=th.int32)

        self.update(pairs)

    def get_sso(self, pd: PDStruct):

        '''
        For given real points in [pd], find corresponding entries 
        in [per_rp_hp_pid] and generate SSO according to it.
        '''

        # not necessary, but for safety;
        self.update_with_pd(pd)
        
        result = SSO()
        with th.no_grad():
            device = self.per_rp_hp_pid.device
            result.thresh = 1e10        # no threshold in pseudo SSO

            # in [self.per_rp_hp_pid], pick entries where 
            # the first value is in [rp_pid];
            rp_pid = pd.rp_point_id
            mask = th.isin(self.per_rp_hp_pid[:, 0], rp_pid)
            
            hp_pid = self.per_rp_hp_pid[mask]    
            hp_pid = th.unique(hp_pid, dim=0)

            u_hplane_x, u_hplane_x_cnt = th.unique_consecutive(hp_pid[:, 0], return_counts=True)
            u_hplane_x_cnt_cumsum = th.cumsum(u_hplane_x_cnt, dim=0).to(dtype=th.int32)
            assert th.all(pd.rp_point_id == u_hplane_x), "SSO failed"

        result.per_rp_hp_pid_array = hp_pid

        # compute half plane info;
        long_hp_pid = hp_pid.to(dtype=th.long)
        # long_hp_pid = long_hp_pid[0]    # debug
        # long_hp_pid = long_hp_pid[:0]   # debug
        if long_hp_pid.ndim == 1:
            long_hp_pid = long_hp_pid[None, :]
        this_positions = pd.points.positions[long_hp_pid[:, 0]]
        this_weights = pd.points.weights[long_hp_pid[:, 0]][:, None]
        other_positions = pd.points.positions[long_hp_pid[:, 1]]
        other_weights = pd.points.weights[long_hp_pid[:, 1]][:, None]
        result.per_rp_hp_normal_array, result.per_rp_hp_offset_array = \
            th_compute_hplane(
                this_positions,
                this_weights,
                other_positions,
                other_weights
            )
        result.per_rp_hp_offset_array = result.per_rp_hp_offset_array[:, 0]
        
        result.per_rp_beg_array = th.cat([th.zeros((1,), dtype=th.int32, device=device), u_hplane_x_cnt_cumsum[:-1]], dim=0)
        result.per_rp_end_array = u_hplane_x_cnt_cumsum

        return result