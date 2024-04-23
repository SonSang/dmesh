#ifndef CUDA_DIFFDT_IWDT1_H_INCLUDED
#define CUDA_DIFFDT_IWDT1_H_INCLUDED

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
    Compute delta values for the given non-existing faces.

    The delta value is computed as the average minimum distance from 
    the dual line of each face to the pd edges of each power cell.

    Input:

    Output:
    ================================================================
    */
    void iwdt1(
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

        // faces;
        const torch::Tensor& faces_points_id,
        const torch::Tensor& faces_dl_origin,
        const torch::Tensor& faces_dl_direction,
        const torch::Tensor& faces_dl_axis_0,
        const torch::Tensor& faces_dl_axis_1,

        // output;
        torch::Tensor& per_face_dl_nearest_pd_edge_idx
    );
}

#endif