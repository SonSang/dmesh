import torch as th
import trimesh
import igl

from diffdt import DiffDT
from diffdt.wp import WPoints

from exp.utils.utils import extract_faces
from exp.utils.tetra import TetraSet

'''
Format to store our shape representation.
'''
class DMesh:

    def __init__(self, 
                point_positions: th.Tensor, 
                point_weights: th.Tensor, 
                point_real: th.Tensor):

        assert len(point_positions) == len(point_weights), \
            f"len(point_positions) != len(point_weights): {len(point_positions)} != {len(point_weights)}"
        assert len(point_positions) == len(point_real), \
            f"len(point_positions) != len(point_real): {len(point_positions)} != {len(point_real)}"
        
        self.point_positions = point_positions
        self.point_weights = point_weights
        self.point_real = point_real

    def boolean_point_real(self):

        return self.point_real > 0.5

    def save(self, path: str):
        
        pos = self.point_positions
        w = self.point_weights
        r = self.point_real

        assert pos.shape[-1] == 3, f"pos.shape[-1] != 3: {pos.shape[-1]}"
        if self.point_weights.ndim == 1:
            w = self.point_weights.unsqueeze(-1)
        if self.point_real.ndim == 1:
            r = self.point_real.unsqueeze(-1)

        merge = th.cat([pos, w, r], dim=-1)

        th.save(merge, path)

    @staticmethod
    def load(path: str, dtype, device):

        merge = th.load(path)
        pos = merge[:, :3].to(dtype=dtype, device=device)
        w = merge[:, 3].to(dtype=dtype, device=device)
        r = merge[:, 4].to(dtype=dtype, device=device)
        
        return DMesh(pos, w, r)

    @staticmethod
    def eval_tet_faces(point_positions: th.Tensor, 
                    point_reals: th.Tensor, 
                    tets: th.Tensor,
                    remove_invisible: bool):

        if point_reals.dtype != th.bool:
            b_point_real = point_reals > 0.5
        else:
            b_point_real = point_reals

        faces = extract_faces(tets)

        faces_reality_0 = th.all(b_point_real[faces.to(dtype=th.long)], dim=-1)

        if remove_invisible:
            # remove invisible faces;
            tetset = TetraSet(point_positions, tets.to(dtype=th.long))
            tetset_faces = tetset.faces
            tetset_faces_real = b_point_real[tetset_faces.to(dtype=th.long)]
            tetset_faces_real = th.all(tetset_faces_real, dim=-1)
            
            tetset_faces_apex = tetset.face_apex
            tetset_faces_apex_0_real = b_point_real[tetset_faces_apex[:, 0].to(dtype=th.long)]
            tetset_faces_apex_1_real = b_point_real[tetset_faces_apex[:, 1].to(dtype=th.long)]
            tetset_faces_apex_1_real[tetset_faces_apex[:, 1] == -1] = False
            tetset_faces_apex_real = th.all(th.stack([tetset_faces_apex_0_real, tetset_faces_apex_1_real], dim=-1), dim=-1)
            
            faces_reality_1 = th.logical_and(faces_reality_0, ~tetset_faces_apex_real)
        else:
            faces_reality_1 = faces_reality_0

        real_faces = faces[faces_reality_1]
        imag_faces = faces[~faces_reality_1]

        return faces, real_faces, imag_faces
    
    def get_faces(self, remove_invisible: bool):
        '''
        Convert to mesh.
        '''
        points = WPoints(
            self.point_positions, 
            self.point_weights
        )
        wdt_result = DiffDT.CGAL_WDT(
            points,
            True,
            False
        )

        return DMesh.eval_tet_faces(
            self.point_positions, 
            self.point_real, 
            wdt_result.tets_point_id,
            remove_invisible
        )
    
    def get_mesh(self, remove_invisible: bool):

        return self.get_real_mesh(remove_invisible)

    def get_real_mesh(self, remove_invisible: bool):

        _, real_faces, _ = self.get_faces(remove_invisible)

        # reorient using BFS
        faces = real_faces.detach().cpu().numpy()
        faces, _ = igl.bfs_orient(faces)

        mesh = trimesh.base.Trimesh(
            vertices=self.point_positions.detach().cpu().numpy(),
            faces=faces
        )
        return mesh
    
    def get_imag_mesh(self, remove_invisible: bool):

        _, _, imag_faces = self.get_faces(remove_invisible)
        mesh = trimesh.base.Trimesh(
            vertices=self.point_positions.detach().cpu().numpy(),
            faces=imag_faces.detach().cpu().numpy()
        )
        return mesh