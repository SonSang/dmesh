'''
Script to represent domain tetrahedralization.
'''

import torch as th

def add_ordinal_axis(mat: th.Tensor):
    '''
    Add an ordinal axis to 2-D tensor's last axis.

    @ mat: (num_row, num_col)
    @ return: (num_row, num_col + 1)
    '''

    num_row = mat.shape[0]
    new_col = th.arange(num_row, device=mat.device).reshape((-1, 1))
    return th.cat([mat, new_col], dim=1)

class TetraSet:

    def __init__(self, 
                verts: th.Tensor, 
                tets: th.Tensor):
        '''
        @ verts: (# vert, 3)
        @ tets: (# tet, 4)
        '''
        self.device = verts.device
        self.verts = verts
        self.tets = tets
        self.tet_faces = None       # (# tet, 4), membership of each tet to four faces

        assert th.min(self.tets) >= 0 and th.max(self.tets) < self.num_verts, \
            'Invalid tetrahedron indices.'

        # faces;
        self.faces = None           # (# face, 3)
        self.face_tet = None        # (# face, 2), membership of each face to two tets
        self.face_apex = None       # (# face, 2), membership of each face to two apexes
        self.face_normals = None    # (# face, 3), normal of each face, oriented toward first tet of [face_tet];
        self.raw_face_normals = None    # (# face, 3), normal of each face, oriented according to ordering of [faces];
        self.face_aligned_with_tet0 = None  # (# face), whether each face is aligned with tet0;
                                            # true if raw face normal is oriented toward tet0;
        self.init_faces()

        # edges;
        self.edges = None           # (# edge, 2)
        self.edge_faces = None      # (# edge * # face_per_edge, 2), membership of each edge [edge id, face id]
        self.face_edges = None      # (# face, 3), membership of each face to three edges 
        self.init_edges()

    @property
    def num_verts(self):
        return self.verts.shape[0]

    @property
    def num_faces(self):
        return self.faces.shape[0]

    @property
    def num_tets(self):
        return self.tets.shape[0]
    
    def init_faces(self):
        '''
        Initialize face information.
        '''
        # faces;
        faces = []
        face_tet = []
        face_apex = []
        for comb in [[0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 3, 1], [1, 2, 3, 0]]:
            faces.append(self.tets[:, comb[:3]])
            face_tet.append(th.arange(self.num_tets, device=self.device))
            face_apex.append(self.tets[:, comb[3]])
        faces = th.cat(faces, dim=0)        # (# dup face, 3), duplicates are possible
        face_tet = th.cat(face_tet, dim=0)  # (# dup face)
        face_apex = th.cat(face_apex, dim=0)  # (# dup face)

        # sort;
        tmp = th.cat([faces, face_tet.unsqueeze(-1)], dim=1)      # (# dup face, 4)
        tmp2 = th.cat([faces, face_apex.unsqueeze(-1)], dim=1)    # (# dup face, 4)
        tmp[:, :3] = th.sort(tmp[:, :3], dim=1)[0]
        tmp2[:, :3] = th.sort(tmp2[:, :3], dim=1)[0]
        tmp = th.unique(tmp, dim=0)
        tmp2 = th.unique(tmp2, dim=0)

        # remove duplicate faces;
        u_faces, u_faces_cnt = th.unique(tmp[:, :3], dim=0, return_counts=True)
        assert th.all(u_faces_cnt <= 2), "A face can be shared by at most 2 tets."
        u_faces_first_id = th.cumsum(u_faces_cnt, dim=0)[:-1]
        u_faces_first_id = th.cat([th.zeros(1, dtype=u_faces_first_id.dtype, device=self.device), 
                                    u_faces_first_id], 
                                    dim=0)
        
        self.faces = u_faces
        self.face_tet = th.zeros((len(u_faces), 2), dtype=u_faces.dtype, device=self.device) - 1
        self.face_apex = th.zeros((len(u_faces), 2), dtype=u_faces.dtype, device=self.device) - 1
        for i in range(2):
            valid_i = i < u_faces_cnt
            self.face_tet[valid_i, i] = \
                tmp[u_faces_first_id[valid_i] + i, -1]
            self.face_apex[valid_i, i] = \
                tmp2[u_faces_first_id[valid_i] + i, -1]

        # set tet faces;
        face_tet_ordinal = add_ordinal_axis(self.face_tet)
        tet_faces = [face_tet_ordinal[:, [0, 2]], face_tet_ordinal[:, [1, 2]]]
        tet_faces = th.cat(tet_faces, dim=0)
        tet_faces = th.unique(tet_faces, dim=0)
        tet_faces = tet_faces[tet_faces[:, 0] >= 0]
        self.tet_faces = tet_faces.reshape((self.num_tets, 8))
        self.tet_faces = self.tet_faces[:, [1, 3, 5, 7]]

        # face normals;
        self.raw_face_normals, self.face_normals = \
            self.face_normals_toward_tet0()

        self.face_aligned_with_tet0 = \
            th.sum(self.raw_face_normals * self.face_normals, dim=1) > 0

    def init_edges(self):
        '''
        Initialize edge information.
        '''
        faces_with_id = add_ordinal_axis(self.faces)
        edges_with_face_id = []
        for comb in [[0, 1, 3], [0, 2, 3], [1, 2, 3]]:
            edges_with_face_id.append(faces_with_id[:, comb])
        edges_with_face_id = th.cat(edges_with_face_id, dim=0)      # (# face * 3, 3) [edge (2), face id(1)]
        edges_with_face_id[:, :2] = th.sort(edges_with_face_id[:, :2], dim=1)[0]
        edges_with_face_id = th.unique(edges_with_face_id, dim=0)   # [edge (2), face id(1)]

        # [edges_cnt] means how many faces share each edge;
        edges, edges_cnt = th.unique(edges_with_face_id[:, :2], dim=0, return_counts=True)  # (# edge, 2)
        edges_end = th.cumsum(edges_cnt, dim=0)
        edges_beg = th.cat([th.zeros(1, dtype=edges_end.dtype, device=self.device), 
                            edges_end[:-1]], dim=0)
        edge_id_axis = th.zeros_like(edges_with_face_id[:, 0])
        edge_id = th.arange(len(edges), device=self.device, dtype=edge_id_axis.dtype)
        max_edges_cnt = th.max(edges_cnt)
        for i in range(max_edges_cnt):
            j = edges_end - i - 1
            valid = (j >= edges_beg)
            edge_id_axis[j[valid]] = edge_id[valid]
        
        edges_with_face_id_and_edge_id = \
            th.cat([edges_with_face_id, edge_id_axis.unsqueeze(-1)], dim=1)  # (# face * 3, 4) [edge (2), face id(1), edge id(1)]
        face_id_and_edge_id = \
            edges_with_face_id_and_edge_id[:, [2, 3]]  # (# face * 3, 2) [face id(1), edge id(1)]
        face_id_and_edge_id = th.unique(face_id_and_edge_id, dim=0)  # (# face * 3, 2) [face id(1), edge id(1)]
        assert len(face_id_and_edge_id) == 3 * self.num_faces, "Each face should have 3 edges."
        
        edge_id_and_face_id = \
            edges_with_face_id_and_edge_id[:, [3, 2]]  # (# face * 3, 2) [edge id(1), face id(1)]
        edge_id_and_face_id = th.unique(edge_id_and_face_id, dim=0)  # (# face * 3, 2) [edge id(1), face id(1)]
        
        self.edges = edges
        self.edge_faces = edge_id_and_face_id
        self.face_edges = face_id_and_edge_id[:, 1].reshape((self.num_faces, 3))

    def tet_center(self, tet_id: th.Tensor):
        '''
        Compute center of each tetrahedron.

        @ tet_id: (# tet)
        '''
        return th.mean(self.verts[self.tets[tet_id]], dim=1)
    

    def face_center(self, face_id: th.Tensor):
        '''
        Compute center of each face.

        @ face_id: (# face)
        '''
        return th.mean(self.verts[self.faces[face_id]], dim=1)

    def vertex_normals(self, 
                        face_normals: th.Tensor, 
                        face_taus: th.Tensor):
        '''
        Compute vertex normals based on faces adjacent to them.

        @ face_normals: (# face, 3)
        @ face_taus: (# face), tau value of each face.
        '''

        weighted_face_normals = face_normals * face_taus[:, None]
        
        # each vertex can be the first, second, or third vertex of a triangle;
        # vertex_normals[0, 0] is the sum of face normals that has the 0-th vertex as the first vertex;
        vertex_normals = th.zeros(
            (self.num_verts, 3, 3), 
            dtype=th.float32, 
            device=self.device)
        
        vertex_normals.scatter_add_(
            dim=0, 
            index=self.faces[:,:,None].expand(self.num_faces, 3, 3), 
            src=weighted_face_normals[:,None,:].expand(self.num_faces, 3, 3))
        
        vertex_normals = vertex_normals.sum(dim=1)  #V,3
        
        # weighted average;
        vertex_face_weights = th.zeros(
            (self.num_verts, 3), 
            dtype=th.float32, 
            device=self.device)
        vertex_face_weights.scatter_add_(
            dim=0,
            index=self.faces,
            src=face_taus[:,None].expand(self.num_faces, 3))
        with th.no_grad():
            vertex_face_weights = vertex_face_weights.sum(dim=1, keepdim=True) # V,1
            vertex_face_weights = th.clamp(vertex_face_weights, min=1e-6)

        vertex_normals = vertex_normals / vertex_face_weights
        return th.nn.functional.normalize(vertex_normals, eps=1e-6, dim=1)

    def get_face_taus(self, vert_taus: th.Tensor) -> th.Tensor:
        '''
        Tau of a face is the minimum tau of its three vertices.

        @ vert_taus: (# vert)
        '''
        face_taus = vert_taus[self.faces]           # (# face, 3)
        face_taus = th.min(face_taus, dim=1)[0]     # (# face)

        return face_taus

    def get_oriented_faces(self, tet_occ: th.Tensor):

        # raw face normals;
        oriented_faces = self.faces.clone()
        v0 = self.verts[oriented_faces[:, 0]]
        v1 = self.verts[oriented_faces[:, 1]]
        v2 = self.verts[oriented_faces[:, 2]]
        raw_normals = th.cross(v1 - v0, v2 - v0)
        raw_normals = th.nn.functional.normalize(raw_normals, eps=1e-6, dim=1)

        '''
        Fix orientation using tet occupancy.
        '''
        # get occupancy for two tets that share each face;
        face_tet0 = self.face_tet[:, 0]     # (# face)
        face_tet1 = self.face_tet[:, 1]     # (# face)
        def sel_tet_occ(tet_occ: th.Tensor, tet_id: th.Tensor):
            res = th.zeros_like(tet_id, dtype=tet_occ.dtype, device=self.device)
            valid = tet_id >= 0
            res[valid] = tet_occ[tet_id[valid]]
            return res
        
        face_tet0_occ = sel_tet_occ(tet_occ, face_tet0)
        face_tet1_occ = sel_tet_occ(tet_occ, face_tet1)

        # for each face, find out which tet is located outside;
        is_tet0_outside = face_tet0_occ < face_tet1_occ
        is_tet1_outside = face_tet0_occ >= face_tet1_occ

        face_center = self.face_center(th.arange(len(self.faces), device=self.device))
        face_tet0_center = self.tet_center(face_tet0)
        face_tet0_center_dir = face_tet0_center - face_center
        is_raw_normal_aligned_with_tet0 = \
            th.sum(face_tet0_center_dir * raw_normals, dim=1) > 0

        reorder_mask0 = th.logical_and(is_raw_normal_aligned_with_tet0,
                                        is_tet1_outside)
        reorder_mask1 = th.logical_and(~is_raw_normal_aligned_with_tet0,
                                        is_tet0_outside)
        reorder_mask = th.logical_or(reorder_mask0, reorder_mask1)
        oriented_faces[reorder_mask] = oriented_faces[reorder_mask][:,[0,2,1]]

        return oriented_faces

    def face_normals_toward_tet0(self, face_id: th.Tensor=None):
        '''
        Compute face normals that are oriented toward tet0.
        That is, normal that assuming tet0 is outside and tet1 is inside.
        '''

        if face_id == None:
            face_id = th.arange(self.num_faces, device=self.device)

        curr_faces = self.faces[face_id]
        curr_face_tet0 = self.face_tet[face_id, 0]

        # raw face normals;
        v0 = self.verts[curr_faces[:, 0]]
        v1 = self.verts[curr_faces[:, 1]]
        v2 = self.verts[curr_faces[:, 2]]
        raw_normals = th.cross(v1 - v0, v2 - v0, dim=-1)
        raw_normals = th.nn.functional.normalize(raw_normals, eps=1e-6, dim=1)

        # find out if [raw_normals] are oriented toward tet0;
        face_center = self.face_center(face_id)
        face_tet0_center = self.tet_center(curr_face_tet0)
        face_tet0_center_dir = face_tet0_center - face_center
        is_raw_normal_oriented_to_tet0 = \
            th.sum(face_tet0_center_dir * raw_normals, dim=1) > 0
        
        is_raw_normal_oriented_to_tet0 = \
            th.where(is_raw_normal_oriented_to_tet0,
                    th.ones_like(is_raw_normal_oriented_to_tet0, dtype=th.float32, device=self.device),
                    -1.0 * th.ones_like(is_raw_normal_oriented_to_tet0, dtype=th.float32, device=self.device))
        
        normals = raw_normals * is_raw_normal_oriented_to_tet0.unsqueeze(1).float()

        return raw_normals, normals

    def face_tau_normal_from_tet_tau(self, 
                                    tet_taus: th.Tensor,
                                    face_tau_thresh: float,
                                    differentiable: bool,):
        '''
        Compute face tau and oriented normal from tet tau.
        '''

        '''
        Face taus
        '''
        face_tet0 = self.face_tet[:, 0]     # (# face)
        face_tet1 = self.face_tet[:, 1]     # (# face)
        def sel_tet_tau(tet_taus: th.Tensor, tet_id: th.Tensor):
            res = th.zeros_like(tet_id, dtype=tet_taus.dtype, device=self.device)
            valid = tet_id >= 0
            res[valid] = tet_taus[tet_id[valid]]
            return res
        
        face_tet0_tau = sel_tet_tau(tet_taus, face_tet0)    # (# face)
        face_tet1_tau = sel_tet_tau(tet_taus, face_tet1)    # (# face)

        # if difference between tet taus are large enough,
        # there is a face;
        # if differentiable:
        #     face_taus = th.sigmoid((th.abs(face_tet0_tau - face_tet1_tau) - 0.5) * 16.0)
        # else:
        #     face_taus = (th.abs(face_tet0_tau - face_tet1_tau) > 0.5).float()

        # choose the larger tau;
        face_taus = th.max(face_tet0_tau, face_tet1_tau)

        '''
        Face normals
        '''
        if differentiable:
            EXP_C = 4.0
            exp_face_tet0_tau = th.exp(face_tet0_tau * EXP_C)
            exp_face_tet1_tau = th.exp(face_tet1_tau * EXP_C)
            denom = exp_face_tet0_tau + exp_face_tet1_tau

            # relative occupancy;
            # tetrahedron with larger occupancy has 1, and
            # the one with smaller occupancy has 0;
            rel_face_tet0_tau = exp_face_tet0_tau / denom
            rel_face_tet1_tau = exp_face_tet1_tau / denom
        else:
            rel_face_tet0_tau = face_tet0_tau > face_tet1_tau
            rel_face_tet1_tau = face_tet0_tau <= face_tet1_tau

        # interpolate;
        face_normals_tet0 = self.face_normals
        face_normals_tet1 = self.face_normals * -1.0

        # normal should be oriented toward tet0 if tet1's occupancy is larger than tet0's;
        face_normals = face_normals_tet0 * rel_face_tet1_tau.unsqueeze(1).float() + \
                    face_normals_tet1 * rel_face_tet0_tau.unsqueeze(1).float()

        # only select faces with larger taus than threshold;
        if not differentiable:
            face_tau_thresh = 0.5

        face_indices = th.where(face_taus > face_tau_thresh)[0]
        faces = self.faces[face_indices]
        face_taus = face_taus[face_indices]
        face_normals = face_normals[face_indices]

        # reorder faces so that it aligns with [face_normals];
        with th.no_grad():
            order_aligned_to_curr_normal = th.sum(face_normals * self.raw_face_normals[face_indices], dim=1) > 0
            faces[~order_aligned_to_curr_normal] = faces[~order_aligned_to_curr_normal][:,[0,2,1]]

        return face_indices, faces, face_taus, face_normals

    def find_face_idx(self, query_faces: th.Tensor):
        '''
        @ query_faces: [# query face, 3]
        '''
        assert query_faces.shape[1] == 3, "Each query face should have 3 vertices."

        # sort queries;
        query = th.sort(query_faces, dim=1)[0]
        query = th.unique(query, dim=0)
        query = th.cat([query, th.zeros((len(query), 1), dtype=th.int64, device=self.device)], dim=1)  # [# query face, 3 + 1]
        query[:, -1] = -1

        face_and_idx = add_ordinal_axis(self.faces)     # [# face, 3 + 1]

        tmp = th.cat([face_and_idx, query], dim=0)

        # if a face in [query] was in [self.faces], there should be duplicates;
        tmp = th.unique(tmp, dim=0)

        u_faces, u_faces_cnt = th.unique(tmp[:, :3], dim=0, return_counts=True)
        tmp_end = th.cumsum(u_faces_cnt, dim=0)
        tmp_end = tmp_end[u_faces_cnt == 2]

        query_face_and_idx = tmp[tmp_end - 1]

        return query_face_and_idx[:, :3], query_face_and_idx[:, 3]

    def face_tau_normal_from_tet_probs(self, 
                                    tet_probs: th.Tensor,
                                    prob_thresh: float = 0.5,
                                    remove_airpockets: bool = False,):
        '''
        Compute face tau and oriented normal from prob to pick tets.
        '''

        '''
        @TODO: Remove airpockets, which mean outside tetra surrounded by inside tetra.
        '''

        '''
        Choose faces.
        '''

        face_tet0 = self.face_tet[:, 0]     # (# face)
        face_tet1 = self.face_tet[:, 1]     # (# face)
        
        def sel_tet_prob(tet_probs: th.Tensor, tet_id: th.Tensor):
            res = th.zeros_like(tet_id, dtype=tet_probs.dtype, device=self.device)
            valid = tet_id >= 0
            res[valid] = tet_probs[tet_id[valid]]
            return res
        
        face_tet0_prob = sel_tet_prob(tet_probs, face_tet0)    # (# face)
        face_tet1_prob = sel_tet_prob(tet_probs, face_tet1)    # (# face)

        # choose faces that only one of the tets has probability over threshold;
        # if both of them have probability over threshold, then the face is not selected,
        # because it is not a boundary face;
        face_tet0_prob_over_thresh = face_tet0_prob > prob_thresh
        face_tet1_prob_over_thresh = face_tet1_prob > prob_thresh

        # face that tet0 is inside, and tet1 is outside;
        face_tet0_prob_over_thresh_exc = th.logical_and(
            face_tet0_prob_over_thresh, 
            ~face_tet1_prob_over_thresh)
        
        # face that tet1 is inside, and tet0 is outside;
        face_tet1_prob_over_thresh_exc = th.logical_and(
            face_tet1_prob_over_thresh, 
            ~face_tet0_prob_over_thresh)

        p_face_idx_0 = th.where(face_tet0_prob_over_thresh_exc)[0]
        p_face_idx_1 = th.where(face_tet1_prob_over_thresh_exc)[0]

        p_faces_0 = self.faces[p_face_idx_0]
        p_faces_1 = self.faces[p_face_idx_1]

        # normals;
        p_face_normals_0 = -self.face_normals[p_face_idx_0]  # normals toward tet1;
        p_face_normals_1 = self.face_normals[p_face_idx_1]  # normals toward tet0;

        # reorder faces so that it aligns with [face_normals];
        with th.no_grad():
            raw_normals_0 = self.raw_face_normals[p_face_idx_0]
            raw_normals_1 = self.raw_face_normals[p_face_idx_1]

            order_aligned_to_curr_normal_0 = th.sum(p_face_normals_0 * raw_normals_0, dim=1) > 0
            order_aligned_to_curr_normal_1 = th.sum(p_face_normals_1 * raw_normals_1, dim=1) > 0

            p_faces_0[~order_aligned_to_curr_normal_0] = p_faces_0[~order_aligned_to_curr_normal_0][:,[0,2,1]]
            p_faces_1[~order_aligned_to_curr_normal_1] = p_faces_1[~order_aligned_to_curr_normal_1][:,[0,2,1]]

        p_faces_normals_0 = th.cat([p_faces_0, p_face_normals_0], dim=-1)
        p_faces_normals_1 = th.cat([p_faces_1, p_face_normals_1], dim=-1)

        p_faces_normals = th.cat([p_faces_normals_0, p_faces_normals_1], dim=0)
        p_faces_normals = th.unique(p_faces_normals, dim=0)

        p_faces = p_faces_normals[:, :3].to(th.int64)
        p_face_normals = p_faces_normals[:, 3:].to(th.float32)

        return p_faces, p_face_normals

    def compute_face_taus_from_tet_taus(self, 
                                        tetra_taus: th.Tensor, 
                                        exp_k: float=16):

        # for each face, compute difference between the two tets
        # that share it to find tau for the face;
        
        face_taus = th.zeros((len(self.faces),), dtype=th.float32, device='cuda')      # [# face,]
        face_tet0 = self.face_tet[:, 0]                                                # [# face,]
        face_tet1 = self.face_tet[:, 1]                                                # [# face,]
        face_tet0_tau = th.where(face_tet0 != -1, tetra_taus[face_tet0], th.zeros_like(face_taus))  # [# face,]
        face_tet1_tau = th.where(face_tet1 != -1, tetra_taus[face_tet1], th.zeros_like(face_taus))  # [# face,]
        face_taus = th.abs(face_tet0_tau - face_tet1_tau)                                   # [# face,]

        # for each tet, pick the face with the highest tau
        # and adjust face taus according to it;
        tetra_face_taus = face_taus[self.tet_faces]                                    # [# tetra, 3]
        with th.no_grad():
            tetra_face_taus_max = th.max(tetra_face_taus, dim=1, keepdim=True)[0]           # [# tetra, 1]
        if exp_k > 0:
            tetra_face_taus_tmp = (tetra_face_taus - tetra_face_taus_max) * exp_k                  # [# tetra, 3]
            tetra_face_taus_nom = th.exp(tetra_face_taus_tmp)                                   # [# tetra, 3]
            tetra_face_taus_denom = th.sum(tetra_face_taus_nom, dim=1, keepdim=True)            # [# tetra, 1]
            tetra_face_taus = (tetra_face_taus_nom / tetra_face_taus_denom) * tetra_face_taus   # [# tetra, 3]
        else:
            tetra_face_taus[tetra_face_taus < tetra_face_taus_max] = 0.0
        
        # for each face, choose the smaller one;
        assert th.all(tetra_face_taus <= 1.0), ""
        tetra_face_id_taus = th.stack([self.tet_faces, tetra_face_taus], dim=2)        # [# tetra, 4, 2]
        tetra_face_id_taus = tetra_face_id_taus.reshape((-1, 2))                            # [# tetra * 4, 2]
        
        tetra_face_id_sort = th.sort(tetra_face_id_taus[:, 0].to(dtype=th.long), dim=0)[1]  # [# tetra * 4,]
        tetra_face_id_taus = tetra_face_id_taus[tetra_face_id_sort]                         # [# tetra * 4, 2]
        
        _, tetra_face_id_cnt = th.unique(tetra_face_id_taus[:, 0].to(dtype=th.long), return_counts=True)                    # [# face,]
        assert th.max(tetra_face_id_cnt) <= 2, ""
        tetra_face_id_end = th.cumsum(tetra_face_id_cnt, dim=0)                        # [# face - 1,]
        
        n_face_taus0 = tetra_face_id_taus[tetra_face_id_end - 1][:, 1]
        n_face_taus1 = th.ones_like(n_face_taus0)
        n_face_taus1[tetra_face_id_cnt == 2] = \
            tetra_face_id_taus[tetra_face_id_end[tetra_face_id_cnt == 2] - 2][:, 1]
        n_face_taus = th.min(th.stack([n_face_taus0, n_face_taus1], dim=1), dim=1)[0]       # [# face,]
        
        assert len(n_face_taus) == len(self.faces), ""

        return n_face_taus
    
    def boundary_faces(self):
        '''
        Return faces that are shared by only one tet, which means they are 
        on the boundary of the convex hull of the tetrahedral grid.
        '''
        return th.where(self.face_tet[:, 1] == -1)[0]