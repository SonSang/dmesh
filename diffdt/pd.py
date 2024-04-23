'''
Script to build Power diagram from WDT.
'''

import torch as th
from diffdt.wp import WPoints
from diffdt.torch_func import th_cc
from . import _C

class PDVertexComputationLayer(th.autograd.Function):
    '''
    Compute positions of power diagram vertices.
    '''
    @staticmethod
    def forward(ctx, positions, weights, vertex_point_id):
        dtype = positions.dtype
        d_positions = positions.to(dtype=th.float64)
        d_weights = weights.to(dtype=th.float64)

        d_cc, d_cc_unnorm = _C.compute_circumcenters(d_positions, d_weights, vertex_point_id)
        ctx.save_for_backward(d_positions, d_weights, vertex_point_id, d_cc_unnorm)       
        return d_cc.to(dtype=dtype)
    
    @staticmethod
    def backward(ctx, grad_output):
        dtype = grad_output.dtype

        d_positions, d_weights, vertex_point_id, d_cc_unnorm = ctx.saved_tensors
        d_grad_output = grad_output.to(dtype=th.float64)
        
        # cc_unnorm_w = cc_unnorm[:, -1]
        # print("cc_unnorm_w: ", cc_unnorm_w.mean().cpu().item()) 
        
        d_grad_positions, d_grad_weights = _C.compute_circumcenters_backward(
            d_positions, 
            d_weights, 
            vertex_point_id, 
            d_cc_unnorm, 
            d_grad_output)
        
        return d_grad_positions.to(dtype=dtype), d_grad_weights.to(dtype=dtype), None

class PDEdgeComputationLayer:
    '''
    Compute direction of power diagram edges.
    '''
    @staticmethod
    def apply(positions: th.Tensor, edge_point_id: th.Tensor):
        # @TODO: CUDA impl? But not seems to be necessary.
        normal_0 = positions[edge_point_id[:, 1]] - positions[edge_point_id[:, 0]]
        normal_1 = positions[edge_point_id[:, 2]] - positions[edge_point_id[:, 0]]
        direction = th.cross(normal_0, normal_1, dim=-1)
        direction = th.nn.functional.normalize(direction, dim=-1)
        return direction

class PDVertex:
    def __init__(self):
        # [# vertex, 3], positions of vertices;
        self.positions: th.Tensor = None
        # [# vertex, 4], indices of points that form the vertex;
        self.points_id: th.Tensor = None

class PDEdge:
    def __init__(self):
        # [# edge, 2], indices of vertices that form the edge;
        self.vertex_id: th.Tensor = None
        # [# edge, 3], indices of points that form the edge;
        self.points_id: th.Tensor = None
        # [# edge, 3], origin of edge lines;
        # these are required because there are half-infinite edges in PD;
        self.origin: th.Tensor = None
        # [# edge, 3], direction of edge lines;
        self.direction: th.Tensor = None

class PDPerRPInfo:
    '''
    For ease of parallel computation, we store indices of PD vertex or edge
    for each real point in a flattened array, [index_array]. Then, we specify
    the range of indices for each real point in [beg_array] and [end_array].
    Finally, we store distances between each real point and its corresponding
    PD vertex or edge in [dist_array].
    '''
    def __init__(self):
        self.index_array: th.Tensor = None
        # [# real point], begin index in [index_array] for each real point;
        self.beg_array: th.Tensor = None
        # [# real point], end index in [index_array] for each real point;
        self.end_array: th.Tensor = None
        # [# real point], distances between each real point and its corresponding
        # PD vertex or edge;
        self.dist_array: th.Tensor = None

class PDStruct:
    def __init__(self):
        
        # Weighted points;
        self.points: WPoints = None 
        
        # [# tet, 4], indices of points for each tetrahedron;
        self.tets_point_id: th.Tensor = None
        
        # [# real point], indices of real points,
        # which have cells in the power diagram;
        self.rp_point_id: th.Tensor = None

        # for each point, index of it in [rp_point_id]
        # if the point is rp; -1 otherwise;
        self.per_point_rp_id: th.Tensor = None
        
        '''
        Power diagram vertices.

        Number of vertices equals to number of tetrahedra,
        because they are dual to each other.
        '''

        self.pd_vertices: PDVertex = None
        self.per_rp_vertex_info: PDPerRPInfo = None
        
        '''
        Power diagram edges.

        Number of edges equals to number of faces,
        because they are dual to each other.
        '''

        self.pd_edges: PDEdge = None
        self.per_rp_edge_info: PDPerRPInfo = None

        # float, computation time;
        self.time_sec: float = -1.

    @staticmethod
    def forward(points: WPoints, tets_point_id: th.Tensor):
        
        device = points.positions.device

        result = PDStruct()
        result.points = points
        result.tets_point_id = tets_point_id

        '''
        Get real points.
        '''
        with th.no_grad():
            result.rp_point_id = th.unique(tets_point_id.reshape(-1)).to(dtype=th.long)
            result.per_point_rp_id = th.zeros((points.positions.shape[0],), dtype=th.long, device=device) - 1
            result.per_point_rp_id[result.rp_point_id] = th.arange(0, result.rp_point_id.shape[0], device=device, dtype=th.long)
            assert th.all(result.rp_point_id[result.per_point_rp_id[result.rp_point_id]] == result.rp_point_id), "Real point id does not match."

            result.rp_point_id = result.rp_point_id.to(dtype=th.int32)
            result.per_point_rp_id = result.per_point_rp_id.to(dtype=th.int32)

        '''
        Construct vertices and per-point vertex info.
        '''
        result.pd_vertices = PDStruct.construct_pd_vertices(points, 
                                                            tets_point_id, 
                                                            device)
        with th.no_grad():
            result.per_rp_vertex_info = PDStruct.construct_per_rp_vertex_info(
                                                                points, 
                                                                result.rp_point_id,
                                                                tets_point_id, 
                                                                result.pd_vertices.positions,
                                                                device)
            
        '''
        Construct edges and per-point edge info.
        '''
        result.pd_edges = PDStruct.construct_pd_edges(points,
                                                    tets_point_id,
                                                    result.pd_vertices.positions,
                                                    result.pd_vertices.points_id,
                                                    device)

        with th.no_grad():
            result.per_rp_edge_info = PDStruct.construct_per_rp_edge_info(
                                                                points,
                                                                result.rp_point_id,
                                                                result.pd_edges.points_id,
                                                                result.pd_edges.vertex_id,
                                                                result.pd_vertices.positions,
                                                                device)

        return result

    @staticmethod
    def construct_pd_vertices(points: WPoints,
                            tets_point_id: th.Tensor,
                            device):
        '''
        Construct PD vertices.
        '''
        pdv = PDVertex()
        pdv.positions = PDVertexComputationLayer.apply(points.positions, points.weights, tets_point_id)
        # pdv.positions = th_cc(points.positions, points.weights, tets_point_id.to(dtype=th.long))
        pdv.points_id = tets_point_id
        return pdv

    @staticmethod
    def construct_pd_edges(points: WPoints,
                        tets_point_id: th.Tensor,
                        pd_vertex_positions: th.Tensor,
                        pd_vertex_points_id: th.Tensor,
                        device):
        
        # get edges;
        tet_ordinal = th.arange(0, tets_point_id.shape[0], device=device)[:, None] # [# tet, 1]
        edges_list = []
        for comb in [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]:
            # [3: edge id, 1: tet id]
            edges_list.append(th.cat([tets_point_id[:, comb], tet_ordinal], dim=1))
        edges_list = th.cat(edges_list, dim=0)                      # [# tet * 4, 4]
        edges_list[:, :3] = th.sort(edges_list[:, :3], dim=1)[0]    # [# tet * 4, 4]
        edges_list = th.unique(edges_list, dim=0)                   # [# edge, 4]

        u_edges, u_edges_cnt = th.unique_consecutive(edges_list[:, :3], dim=0, return_counts=True) # [# edge, 3], [# edge]
        edges_with_finite_length = th.where(u_edges_cnt == 2)[0]
        edges_with_infinite_length = th.where(u_edges_cnt == 1)[0]
        u_edges_cnt_cumsum = th.cumsum(u_edges_cnt, dim=0) # [# edge]
        u_edges_beg = th.cat([th.zeros((1,), dtype=th.int32, device=device), u_edges_cnt_cumsum[:-1]], dim=0) # [# edge]
        u_edges_end = u_edges_cnt_cumsum # [# edge]

        result = PDEdge()
        result.points_id = u_edges
        result.vertex_id = th.zeros((u_edges.shape[0], 2), dtype=th.long, device=device) - 1
        result.vertex_id[:, 0] = edges_list[u_edges_beg, -1]
        # finite edges have two vertices;
        # infinite edges have only one vertex, and the other one is -1;
        result.vertex_id[edges_with_finite_length, 1] = \
            edges_list[u_edges_end[edges_with_finite_length] - 1, -1]

        '''
        Compute origin and direction for each edge
        '''
        edge_pd_vertices_0 = pd_vertex_positions[result.vertex_id[:, 0]]
        edge_pd_vertices_1 = pd_vertex_positions[result.vertex_id[:, 1]]
        result.origin = edge_pd_vertices_0
        result.direction = edge_pd_vertices_1 - edge_pd_vertices_0
        
        # deal with edges of infinite length;
        result.direction[edges_with_infinite_length] = \
            PDEdgeComputationLayer.apply(
                points.positions, 
                result.points_id[edges_with_infinite_length])

        '''
        Reorient directions of infinite edges
        '''
        infinite_edge_test_point = (result.origin + result.direction * 1e3)[edges_with_infinite_length]

        infinite_edge_anchor_point_id = result.points_id[edges_with_infinite_length, 0]
        infinite_edge_residual_point_id = th.zeros_like(infinite_edge_anchor_point_id) - 1
        infinite_edge_vertex_id = result.vertex_id[edges_with_infinite_length, 0]

        # among four points that comprise the one vertex of infinite edge,
        # find the one that does not belong to the infinite edge;
        for i in range(4):
            pid = pd_vertex_points_id[infinite_edge_vertex_id, i]
            not_residual = th.zeros_like(pid, dtype=th.bool)
            for j in range(3):
                not_residual = not_residual | (pid == result.points_id[edges_with_infinite_length, j])
            is_residual = ~not_residual
            infinite_edge_residual_point_id[is_residual] = pid[is_residual].to(dtype=th.long)
        assert th.all(infinite_edge_residual_point_id > -1), "Some infinite edge residual point id is -1."

        # find if test point falls into H_{anchor > residual};
        anchor_positions = points.positions[infinite_edge_anchor_point_id]
        anchor_weights = points.weights[infinite_edge_anchor_point_id]
        residual_positions = points.positions[infinite_edge_residual_point_id]
        residual_weights = points.weights[infinite_edge_residual_point_id]

        test_anchor_positions_diff = infinite_edge_test_point - anchor_positions
        test_point_anchor_val = th.sum(test_anchor_positions_diff * test_anchor_positions_diff, dim=-1) - anchor_weights
        test_residual_positions_diff = infinite_edge_test_point - residual_positions
        test_point_residual_val = th.sum(test_residual_positions_diff * test_residual_positions_diff, dim=-1) - residual_weights

        infinite_edge_correct_direction = (test_point_anchor_val < test_point_residual_val)
        infinite_edge_with_wrong_direction = edges_with_infinite_length[~infinite_edge_correct_direction]
        result.direction[infinite_edge_with_wrong_direction] = \
            -result.direction[infinite_edge_with_wrong_direction]
        
        # type fix;
        result.points_id = result.points_id.to(dtype=th.int32)
        result.vertex_id = result.vertex_id.to(dtype=th.int32)

        return result

    @staticmethod
    def construct_per_rp_vertex_info(points: WPoints,
                                rp_point_id: th.Tensor,
                                tets_point_id: th.Tensor,
                                pd_vertex_positions: th.Tensor,
                                device):

        per_point_tets = []
        for comb in [[0, 1, 2, 3], [1, 0, 2, 3], [2, 0, 1, 3], [3, 0, 1, 2]]:
            per_point_tets.append(tets_point_id[:, comb])
        per_point_tets = th.cat(per_point_tets, dim=0)    # [# tet * 4, 4]

        # compute distances between points and cc;
        per_point_tets_cc = th.cat([pd_vertex_positions for _ in range(4)], dim=0)      # [# tri * 4, 3]
        per_point_position = points.positions[per_point_tets[:, 0].to(th.long)] # [# tri * 4, 3]
        per_point_dist = th.norm(per_point_position - per_point_tets_cc, dim=-1, keepdim=True) # [# tri * 4, 1]
        
        num_tets = tets_point_id.shape[0]
        tet_id = th.arange(0, num_tets, device=device)[:, None] # [# tet, 1]
        per_point_tet_id = th.cat([tet_id for _ in range(4)], dim=0) # [# tet * 4, 1]

        # sort by distance from each point to cc in descending order;
        # [0: point id, 1: distance, 2: cc id]
        sort_target = th.cat([per_point_tets[:, 0:1], -per_point_dist, per_point_tet_id], dim=-1) # [# tri * 4, 3]
        sort_target = th.unique(sort_target, dim=0) # [# tri * 4, 3]

        # get points that have cc (remove points that do not have cc);
        real_point_id, real_points_cnt = th.unique_consecutive(
            sort_target[:, 0].to(dtype=th.int32), 
            dim=0, 
            return_counts=True)
        assert th.all(real_point_id == rp_point_id), "Real point id does not match."
        real_points_cnt_cumsum = th.cumsum(real_points_cnt, dim=0)
        
        # store results;
        per_rp_vertex_info = PDPerRPInfo()
        per_rp_vertex_info.index_array = sort_target[:, -1].to(dtype=th.int32)
        per_rp_vertex_info.beg_array = th.cat([
            th.zeros((1,), dtype=th.int32, device=device), 
            real_points_cnt_cumsum[:-1]], 
            dim=0).to(dtype=th.int32)
        per_rp_vertex_info.end_array = real_points_cnt_cumsum.to(dtype=th.int32)
        per_rp_vertex_info.dist_array = -sort_target[:, 1]

        return per_rp_vertex_info

    @staticmethod
    def construct_per_rp_edge_info(points: WPoints,
                                rp_point_id: th.Tensor,
                                edge_point_id: th.Tensor,
                                edge_vertex_id: th.Tensor,
                                pd_vertex_positions: th.Tensor,
                                device):
        
        per_point_edges = []
        for comb in [[0, 1, 2], [1, 0, 2], [2, 0, 1]]:
            per_point_edges.append(edge_point_id[:, comb])
        per_point_edges = th.cat(per_point_edges, dim=0)    # [# edge * 3, 4]

        # compute distances between points and edges;
        per_point_edges_vertex_0 = pd_vertex_positions[edge_vertex_id[:, 0].to(th.long)]
        per_point_edges_vertex_1 = pd_vertex_positions[edge_vertex_id[:, 1].to(th.long)]
        per_point_edges_vertex_0 = th.cat([per_point_edges_vertex_0 for _ in range(3)], dim=0) # [# edge * 3, 3]
        per_point_edges_vertex_1 = th.cat([per_point_edges_vertex_1 for _ in range(3)], dim=0) # [# edge * 3, 3]
        is_infinite_edge = (edge_vertex_id[:, 1] == -1)
        is_infinite_edge = th.cat([is_infinite_edge for _ in range(3)], dim=0) # [# edge * 3]

        per_point_position = points.positions[per_point_edges[:, 0].to(th.long)] # [# edge * 3, 3]
        per_point_edges_dist_0 = th.norm(per_point_position - per_point_edges_vertex_0, dim=-1, keepdim=True) # [# edge * 3, 1]
        per_point_edges_dist_1 = th.norm(per_point_position - per_point_edges_vertex_1, dim=-1, keepdim=True) # [# edge * 3, 1]
        per_point_edges_dist_1[is_infinite_edge] = float('inf')
        per_point_edges_dist = th.max(per_point_edges_dist_0, per_point_edges_dist_1)
        
        num_edges = edge_point_id.shape[0]
        edge_id = th.arange(0, num_edges, device=device)[:, None] # [# edge, 1]
        per_point_edge_id = th.cat([edge_id for _ in range(3)], dim=0) # [# edge * 3, 1]

        # sort by distance from each point to edge in descending order;
        # [0: point id, 1: distance, 2: edge id]
        sort_target = th.cat([per_point_edges[:, 0:1], -per_point_edges_dist, per_point_edge_id], dim=-1) # [# edge * 3, 3]
        sort_target = th.unique(sort_target, dim=0) # [# edge * 3, 3]

        # get points that have cell (remove points that do not have cell);
        real_point_id, real_points_cnt = th.unique_consecutive(
            sort_target[:, 0].to(dtype=th.int32), 
            dim=0, 
            return_counts=True)
        assert th.all(real_point_id == rp_point_id), "Real point id does not match."
        real_points_cnt_cumsum = th.cumsum(real_points_cnt, dim=0)
        
        # store results;
        per_rp_edge_info = PDPerRPInfo()
        per_rp_edge_info.index_array = sort_target[:, -1].to(dtype=th.int32)
        per_rp_edge_info.beg_array = th.cat([
            th.zeros((1,), dtype=th.int32, device=device), 
            real_points_cnt_cumsum[:-1]], 
            dim=0).to(dtype=th.int32)
        per_rp_edge_info.end_array = real_points_cnt_cumsum.to(dtype=th.int32)
        per_rp_edge_info.dist_array = -sort_target[:, 1]
        
        return per_rp_edge_info