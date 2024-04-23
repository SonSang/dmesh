#ifndef CUDA_DIFFDT_CIRCUMCENTER_H_INCLUDED
#define CUDA_DIFFDT_CIRCUMCENTER_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <torch/extension.h>

namespace DIFFDT
{
    /*
    ================================================================
    Compute weighted circumcenter of tetrahedra.

    Input:
        @ positions: (N, 3) tensor, the position of the points.
        @ weights: (N,) tensor, the weight of the points.
        @ tri_idx: (M, 4) tensor, the index of the points in the tetrahedra.

    Output:
        @ tri_cc: (M, 3) tensor, the circumcenter of the tetrahedra.
        @ tri_cc_unnorm: (M, 3) tensor, intermediate data for backward.
    ================================================================
    */
    void cc(
        // point info;
        torch::PackedTensorAccessor64<double, 2> positions,
        torch::PackedTensorAccessor64<double, 1> weights,
        
        // tri info;
        torch::PackedTensorAccessor64<int, 2> tri_idx,
        torch::PackedTensorAccessor64<double, 2> tri_cc,
        torch::PackedTensorAccessor64<double, 2> tri_cc_unnorm
    );

    void cc_backward(
        // point info;
        torch::PackedTensorAccessor64<double, 2> positions,
        torch::PackedTensorAccessor64<double, 1> weights,

        // tri info;
        torch::PackedTensorAccessor64<int, 2> tri_idx,
        torch::PackedTensorAccessor64<double, 2> tri_cc_unnorm,
        torch::PackedTensorAccessor64<double, 2> grad_tri_cc,
        
        // point grad;
        torch::PackedTensorAccessor64<double, 2> grad_positions,
        torch::PackedTensorAccessor64<double, 1> grad_weights
    );
}


#endif