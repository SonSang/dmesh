import torch as th
import pytorch3d.ops.knn as knn
from diffdt.pd import PDStruct

def rp_knn(
    pd: PDStruct,
    k: int,
    p_r: th.Tensor=None,
    min_p_r: float=0.0,
):
    '''
    Find K nearest real points for each valid real point in PD.
    Valid real point is a real point with real value larger than min_p_r.

    Note that the returned indices are indices in real points in PD,
    not indices in all points in PD.
    '''
    if p_r is None:
        valid_rp = th.arange(pd.rp_point_id.shape[0])
        valid_rp_point_id = pd.rp_point_id
    else:
        valid_rp = th.nonzero(p_r[pd.rp_point_id.to(dtype=th.long)] > min_p_r).squeeze(dim=-1)
        valid_rp_point_id = pd.rp_point_id[valid_rp]

    rp_positions = pd.points.positions[valid_rp_point_id.to(th.long)].unsqueeze(0)

    # find K nearest real points for each valid real point;
    # (num_rp, k)
    knn_result = knn.knn_points(rp_positions, rp_positions, K=k+1)
    knn_idx = knn_result.idx[0, :, 1:]

    src_rp_point_id = valid_rp_point_id
    knn_rp_point_id = valid_rp_point_id[knn_idx]

    return src_rp_point_id, knn_rp_point_id

def pd_knn_faces(
    pd: PDStruct,
    k: int,
    p_r: th.Tensor=None,
    min_p_r: float=0.0,
):
    '''
    Construct faces for each rp by combining K nearest real points.

    Among real points, only use points with real value larger than min_p_r.
    '''
    if p_r is not None:
        rp_validity = p_r[pd.rp_point_id.to(dtype=th.long)] > min_p_r
        num_rp = rp_validity.count_nonzero()
    else:
        num_rp = pd.rp_point_id.shape[0]

    if num_rp < 3:
        return th.zeros((0, 3), dtype=th.int64, device=pd.rp_point_id.device)

    if k > num_rp - 1:
        k = num_rp - 1
    elif k < 2:
        k = 2

    combs = th.combinations(th.arange(k), 2)        # (n = k*(k-1)/2, 2)
    n_combs = combs.shape[0]
    combs = combs.unsqueeze(0).expand(num_rp, -1, -1)    # (num_rp, n, 2)
    
    # point_index_0 = [# valid rp]
    # knn_idx = [# valid rp, k]
    point_index_0, knn_idx = rp_knn(pd, k, p_r, min_p_r)
    knn_idx = knn_idx.unsqueeze(1).expand(-1, n_combs, -1)    # (num_rp, n, k)
    
    combs = combs.to(device=knn_idx.device)
    face_idx = th.gather(knn_idx, 2, combs)         # (num_rp, n, 2)

    point_index_0 = point_index_0.unsqueeze(1).expand(-1, n_combs)    # (num_rp, n)
    face_idx = th.cat([point_index_0.unsqueeze(-1), face_idx], dim=-1)    # (num_rp, n, 3)

    face_idx = face_idx.reshape(-1, 3)              # (num_rp*n, 3)
    face_idx = th.sort(face_idx, dim=-1)[0]         # (num_rp*n, 3)
    face_idx = th.unique(face_idx, dim=0)           # (num_faces, 3)

    assert th.any(face_idx[:, 0] == face_idx[:, 1]) == False, 'Invalid face'
    assert th.any(face_idx[:, 1] == face_idx[:, 2]) == False, 'Invalid face'
    assert th.any(face_idx[:, 2] == face_idx[:, 0]) == False, 'Invalid face'

    return face_idx