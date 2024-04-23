import torch as th
from diffdt.wp import WPoints
from . import _C

class CGALWDTStruct:
    def __init__(self):
        # WPoints
        self.points: WPoints = None
        
        # [# tet, 4], indices of points for each tetrahedron
        self.tets_point_id: th.Tensor = None

        # [# tet, # dim], circumcenters of tetrahedra
        self.tets_cc: th.Tensor = None

        # float, computation time
        self.time_sec: float = -1.

    @staticmethod
    def forward(points: WPoints, 
                weighted: bool, 
                parallelize: bool, 
                p_lock_grid_size: int, 
                compute_cc: bool):

        with th.no_grad():
            t_positions, t_weights = points.positions, points.weights
            if t_positions.device != th.device('cpu'):
                t_positions = points.positions.cpu()
            if t_weights.device != th.device('cpu'):
                t_weights = points.weights.cpu()
            result = _C.delaunay_triangulation(t_positions, 
                                            t_weights, 
                                            weighted, 
                                            parallelize, 
                                            p_lock_grid_size, 
                                            compute_cc)

            rstruct = CGALWDTStruct()
            rstruct.points = points
            rstruct.tets_point_id = result[0].to(points.positions.device)
            rstruct.tets_cc = result[1].to(points.positions.device)
            rstruct.time_sec = result[2]
            
        return rstruct