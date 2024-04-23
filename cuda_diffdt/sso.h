#ifndef CUDA_DIFFDT_SELECT_SIMPLEX_ONE_H_INCLUDED
#define CUDA_DIFFDT_SELECT_SIMPLEX_ONE_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <torch/extension.h>

#include "rp.h"

namespace DIFFDT
{
    /*
    ================================================================
    Select one-simplexes that are likely included in WDT.

    A one-simplex is a line segment that connects two points, of which
    dual is a plane in 3D. The indices of point pairs are returned in
    simplexes, which would be allocated dynamically by this function.

    Input:

    Output:
    ================================================================
    */
    torch::Tensor sso(
        int num_points,
        int dim,
        const float* positions,
        const float* weights,
        const float dist_thresh,
        // info about real points;
        const RPI* rp_info,
        const int* cc_pt_id,
        const float* cc_pos,
        const int max_num_point_per_batch=10000
    );
}

#endif