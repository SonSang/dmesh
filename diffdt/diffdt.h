#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

/*
Intermediate functions used for differentiable Delaunay triangulation
*/

// Non differentiable Delaunay triangulation
std::tuple<torch::Tensor, torch::Tensor, float>
ComputeDT(
    const torch::Tensor& positions,
    const torch::Tensor& weights,
    const bool weighted,
    const bool parallelize,
    const int p_lock_grid_size,      // number of voxel grids lock per axis to use for parallelization
    const bool compute_cc
);

// Circumcenter functions
std::tuple<torch::Tensor, torch::Tensor>
ComputeCircumcentersCUDA(
    const torch::Tensor& positions,
    const torch::Tensor& weights,
    const torch::Tensor& tri_idx);

std::tuple<torch::Tensor, torch::Tensor>
ComputeCircumcentersBackwardCUDA(
    const torch::Tensor& positions,
    const torch::Tensor& weights,
    const torch::Tensor& tri_idx,
    const torch::Tensor& cc_unnorm,
    const torch::Tensor& dL_dCC);

// SSO function
torch::Tensor
ComputeSSOCUDA(
    const torch::Tensor& positions,
    const torch::Tensor& weights,
    // ppsc info;
    const torch::Tensor& rp_id,
    const torch::Tensor& rp_cc_beg,
    const torch::Tensor& rp_cc_end,
    const torch::Tensor& rp_cc_id,
    const torch::Tensor& rp_cc_d,
    // circumcenters info;
    const torch::Tensor& tri_idx,
    const torch::Tensor& circumcenters,
    // dist thresh;
    const float dist_thresh,
    const int max_num_point_per_batch=10000);

// IWDT function
torch::Tensor
ComputeIWDT0CUDA(
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
    const torch::Tensor& pd_per_rp_edge_beg_array,
    const torch::Tensor& pd_per_rp_edge_end_array,
    const torch::Tensor& pd_per_rp_edge_dist_array,

    // sso;
    const torch::Tensor& sso_per_rp_hp_pid_array,
    const torch::Tensor& sso_per_rp_hp_normals_array,
    const torch::Tensor& sso_per_rp_hp_offsets_array,
    const torch::Tensor& sso_per_rp_beg_array,
    const torch::Tensor& sso_per_rp_end_array,

    float infinite_edge_delta
);

torch::Tensor
ComputeIWDT1CUDA(
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
    const torch::Tensor& pd_per_rp_edge_beg_array,
    const torch::Tensor& pd_per_rp_edge_end_array,
    const torch::Tensor& pd_per_rp_edge_dist_array,

    const torch::Tensor& faces_points_id,
    const torch::Tensor& faces_dl_origin,
    const torch::Tensor& faces_dl_direction,
    const torch::Tensor& faces_dl_axis_0,
    const torch::Tensor& faces_dl_axis_1
);