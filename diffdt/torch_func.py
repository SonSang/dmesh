import torch as th
from diffdt.wp import WPoints
# from diffdt.pd import PDStruct

'''
Circumcenter
'''
def th_cc(positions: th.Tensor,
        weights: th.Tensor,
        tri_idx: th.Tensor):
    
    dimension = positions.shape[1]

    '''
    1. Gather point coordinates for each simplex.
    '''
    num_simplex = tri_idx.shape[0]

    # [# simplex, # dim + 1, # dim]
    simplex_points = positions[tri_idx]

    # [# simplex, # dim + 1]
    simplex_weights = weights[tri_idx]

    '''
    2. Change points in [# dim] dimension to hyperplanes in [# dim + 1] dimension
    '''
    # [# simplex, # dim + 1, # dim + 2]
    hyperplanes0 = th.ones_like(simplex_points[:, :, [0]]) * -1.
    hyperplanes1 = simplex_weights.unsqueeze(-1) - \
        th.sum(simplex_points * simplex_points, dim=-1, keepdim=True)
    hyperplanes = th.cat([simplex_points * 2., hyperplanes0, hyperplanes1], dim=-1)

    '''
    3. Find intersection of hyperplanes above to get circumcenter.
    '''
    # @TODO: Speedup by staying on original dimension...
    mats = []
    for dim in range(dimension + 2):
        cols = list(range(dimension + 2))
        cols = cols[:dim] + cols[(dim + 1):]

        # [# simplex, # dim + 1, # dim + 1]
        mat = hyperplanes[:, :, cols]
        mats.append(mat)

    # [# simplex * (# dim + 2), # dim + 1, # dim + 1]
    detmat = th.cat(mats, dim=0)

    # [# simplex * (# dim + 2)]
    det = th.det(detmat)

    # [# simplex, # dim + 2]
    hyperplane_intersections0 = det.reshape((dimension + 2, num_simplex))
    hyperplane_intersections0 = th.transpose(hyperplane_intersections0.clone(), 0, 1)
    sign = 1.
    for dim in range(dimension + 2):
        hyperplane_intersections0[:, dim] = hyperplane_intersections0[:, dim] * sign
        sign *= -1.
        
    # [# simplex, # dim + 2]
    eps = 1e-6
    last_dim = hyperplane_intersections0[:, [-1]]
    last_dim = th.sign(last_dim) * th.clamp(th.abs(last_dim), min=eps)
    last_dim = th.where(last_dim == 0., th.ones_like(last_dim) * eps, last_dim)
    hyperplane_intersections = hyperplane_intersections0[:, :] / \
                                    last_dim

    '''
    Projection
    '''
    # [# tri, # dim]

    circumcenters = hyperplane_intersections[:, :-2]
    if th.any(th.isnan(circumcenters)) or th.any(th.isinf(circumcenters)):
        raise ValueError()

    return circumcenters