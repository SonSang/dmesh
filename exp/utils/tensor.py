import torch as th

def tensor_intersect(a: th.Tensor, b: th.Tensor):
    '''
    Get intersection of two tensors of shape [N, D] and [M, D].
    Assume [a] and [b] does not have duplicates in its own.
    '''
    assert a.shape[1] == b.shape[1], "Tensor dimension mismatch"
    merge = th.cat([a, b], dim=0)
    u_merge, u_merge_cnt = th.unique(merge, return_counts=True, dim=0)
    return u_merge[u_merge_cnt > 1]

def tensor_intersect_idx(a: th.Tensor, b: th.Tensor):
    '''
    For tensors a and b, return boolean tensors of the same size of a and b,
    where True indicates the corresponding element in a and b are shared.

    Assume [a] and [b] does not have duplicates in its own.
    '''
    assert a.shape[1] == b.shape[1], "Tensor dimension mismatch"
    a_i = th.zeros((a.shape[0],), dtype=th.bool, device=a.device)
    b_i = th.zeros((b.shape[0],), dtype=th.bool, device=b.device)

    a_z = th.zeros((a.shape[0], 1), dtype=th.int32, device=a.device)
    b_z = th.ones((b.shape[0], 1), dtype=th.int32, device=b.device)
    
    a_ordinal = th.arange(a.shape[0], device=a.device).view(-1, 1)
    b_ordinal = th.arange(b.shape[0], device=b.device).view(-1, 1)
    
    a_0 = th.cat([a, a_z, a_ordinal], dim=1)
    b_0 = th.cat([b, b_z, b_ordinal], dim=1)
    
    merge = th.cat([a_0, b_0], dim=0)
    u_merge = th.unique(merge, dim=0)
    
    _, u_merge2_cnt = th.unique(u_merge[:, :-2], return_counts=True, dim=0)
    u_merge2_cnt_cumsum = th.cumsum(u_merge2_cnt, dim=0)
    u_merge2_beg = th.cat([th.zeros((1,), dtype=th.long, device=a.device), u_merge2_cnt_cumsum[:-1]], dim=0)
    u_merge2_end = u_merge2_cnt_cumsum

    u_merge2_beg = u_merge2_beg[u_merge2_cnt == 2]
    u_merge2_end = u_merge2_end[u_merge2_cnt == 2]

    a_i[u_merge[u_merge2_beg, -1]] = True
    b_i[u_merge[u_merge2_beg + 1, -1]] = True

    return a_i, b_i

def tensor_subtract_1(a: th.Tensor, b: th.Tensor):
    '''
    Subtract common elements in a & b from a.
    Assume [a] and [b] does not have duplicates in its own.
    '''
    assert a.shape[1] == b.shape[1], "Tensor dimension mismatch"
    intersect = tensor_intersect(a, b)
    
    merge_a = th.cat([a, intersect], dim=0)
    u_merge_a, u_merge_cnt_a = th.unique(merge_a, return_counts=True, dim=0)

    return u_merge_a[u_merge_cnt_a == 1]

def tensor_subtract_2(a: th.Tensor, b: th.Tensor):
    '''
    Subtract common elements in a & b from a, b.
    Assume [a] and [b] does not have duplicates in its own.
    '''
    assert a.shape[1] == b.shape[1], "Tensor dimension mismatch"
    intersect = tensor_intersect(a, b)
    
    merge_a = th.cat([a, intersect], dim=0)
    merge_b = th.cat([b, intersect], dim=0)
    
    u_merge_a, u_merge_cnt_a = th.unique(merge_a, return_counts=True, dim=0)
    u_merge_b, u_merge_cnt_b = th.unique(merge_b, return_counts=True, dim=0)

    return u_merge_a[u_merge_cnt_a == 1], u_merge_b[u_merge_cnt_b == 1]
