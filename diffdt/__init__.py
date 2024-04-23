from typing import NamedTuple
import torch as th
from diffdt.torch_func import *
from diffdt.wp import WPoints
from diffdt.pd import PDVertexComputationLayer, PDStruct
from diffdt.cgalwdt import CGALWDTStruct
from diffdt.iwdt0 import IWDT0
from diffdt.iwdt1 import IWDT1
from diffdt.sso import SSO
from . import _C

class DiffDTSettings(NamedTuple):
    debug_level : int

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, th.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def run_diff_dt(
    positions: th.Tensor,
    weights: th.Tensor,
    context: dict,
    settings: DiffDTSettings
):
    raise NotImplementedError()

class DiffDT(th.nn.Module):

    def __init__(self, settings: DiffDTSettings):
        super().__init__()
        self.settings = settings

    '''
    Main routine
    '''
    def forward(self, 
                positions: th.Tensor,
                weights: th.Tensor,
                context: dict):
        
        '''
        @ positions: [# vertex, # dim], positions of points
        @ weights: [# vertex, # dim], weights of points
        @ context: dict, context for diff dt (mainly data structure for speed up)
        '''
        
        settings = self.settings

        weights = weights.unsqueeze(-1) if weights.ndim == 1 else weights

        # Invoke C++/CUDA volume rendering routine
        return run_diff_dt(
            positions.to(dtype=th.float32),
            weights.to(dtype=th.float32),
            context,
            settings
        )

    '''
    Subroutines
    '''
    @staticmethod
    def compute_kmeans_cluster(positions: th.Tensor, 
                            num_clusters: int,
                            random_seed: int):
        
        centroids, assignments = \
            _C.compute_kmeans_cluster(
                positions.to(dtype=th.float32),
                num_clusters,
                random_seed
            )

        return centroids, assignments

    @staticmethod
    def CGAL_WDT(points: WPoints,
            weighted: bool,
            parallelize: bool,
            p_lock_grid_size: int = 50,
            compute_cc: bool = False):
        '''
        Run CGAL's Weighted Delaunay Triangulation.

        @ points: WPoints, weighted points fed into WDT
        @ weighted: bool, whether to use weighted DT
        @ parallelize: bool, whether to use parallelized DT
        @ p_lock_grid_size: int, grid size for parallelized DT
        @ compute_cc: bool, whether to compute circumcenters

        Return:
        @ CGALWDTStruct
        '''

        return CGALWDTStruct.forward(
            points,
            weighted,
            parallelize,
            p_lock_grid_size,
            compute_cc
        )

    @staticmethod
    def compute_circumcenters(
        positions: th.Tensor,
        weights: th.Tensor,
        tri_idx: th.Tensor,
        mode: str='cuda'):
        '''
        Compute tri-wise circumcenters.

        @ mode: ['torch', 'cuda']
        '''
        if mode == 'torch':

            return th_cc(positions, weights, tri_idx.to(dtype=th.long))
        
        elif mode == 'cuda':

            return PDVertexComputationLayer.apply(positions, weights, tri_idx)
        
        else:

            raise ValueError()
    
    @staticmethod
    def sso(
        pd: PDStruct,
        dist_thresh: float,
        cuda_max_num_point_per_batch: int=10_000,
        mode: str='cuda'):

        '''
        Compute SSO.

        @ mode: ['torch', 'cuda']
        '''

        return SSO.forward(pd, dist_thresh, cuda_max_num_point_per_batch, mode)  

    @staticmethod
    def sst(
        sso: SSO,
        pd: PDStruct,
        thresh: float,
        cuda_max_num_face_per_batch: int=100_000_000,
        mode: str='cuda'):
        '''
        Compute SST.

        @ mode: ['torch', 'cuda']
        '''

        return SST.forward(sso, pd, thresh, cuda_max_num_face_per_batch, mode)
    
    @staticmethod
    def iwdt0(
        pd: PDStruct,
        sso: SSO,
        device,
        mode: str='cuda'):
        '''
        Compute differentiable inclusion score for faces already exist in the current WDT.
        
        @ mode: ['torch', 'cuda']
        '''
        return IWDT0.forward(pd, sso, device, mode=mode)

    @staticmethod
    def iwdt1(
        pd: PDStruct,
        faces: th.Tensor,
        device,
        mode: str='cuda'):
        '''
        Compute differentiable inclusion score for faces not exist in the current WDT.
        
        @ mode: ['torch', 'cuda']
        '''
        return IWDT1.forward(pd, faces, device, mode=mode)