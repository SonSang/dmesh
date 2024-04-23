#ifndef CUDA_DIFFDT_IWDT0_H_INCLUDED
#define CUDA_DIFFDT_IWDT0_H_INCLUDED

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
    Compute delta values for each existing face in the current WDT.

    The delta value is computed as the average minimum distance from a sample point
    on the dual line segment of each face to the nearby faces of each power cell.

    Input:

    Output:
    ================================================================
    */
    void iwdt0(
        // PDStruct;
        const torch::Tensor& pd_rp_id,
        const torch::Tensor& pd_per_point_rp_id,

        // pd vertex;
        const torch::Tensor& pd_vertex_positions,
        const torch::Tensor& pd_vertex_points_id,
        
        // pd edge;
        const torch::Tensor& pd_edge_vertex_id,
        const torch::Tensor& pd_edge_points_id,
        const torch::Tensor& pd_edge_origin,
        const torch::Tensor& pd_edge_direction,
        const torch::Tensor& pd_per_rp_edge_index_array,
        const torch::Tensor& pd_per_rp_edge_beg_array_beg,
        const torch::Tensor& pd_per_rp_edge_end_array_end,
        const torch::Tensor& pd_per_rp_edge_dist_array,

        // sso;
        const torch::Tensor& sso_per_rp_hp_pid_array,
        const torch::Tensor& sso_per_rp_hp_normals_array,
        const torch::Tensor& sso_per_rp_hp_offsets_array,
        const torch::Tensor& sso_per_rp_beg_array,
        const torch::Tensor& sso_per_rp_end_array,

        torch::Tensor& per_pd_edge_nearest_hp_idx,

        float infinite_edge_delta
    );
}

#endif