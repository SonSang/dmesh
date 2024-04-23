#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <fstream>
#include <string>
#include <functional>
#include <chrono>

#include "ops.h"
#include "cc.h"
#include "sso.h"
#include "iwdt0.h"
#include "iwdt1.h"

std::tuple<torch::Tensor, torch::Tensor, float>
ComputeDT(
    const torch::Tensor& positions,
    const torch::Tensor& weights,
    const bool weighted,
    const bool parallelize,
    const int p_lock_grid_size,
    const bool compute_cc) {
    
    int num_points = positions.size(0);
    int dimension = positions.size(1);

    const float* positions_ptr = positions.data_ptr<float>();
    const float* weights_ptr = weights.data_ptr<float>();

    auto dt_result = CGALDDT::WDT(
        num_points, dimension,
        positions_ptr,
        weights_ptr,
        true,
        parallelize,
        p_lock_grid_size,
        compute_cc
    );
    
    auto int_options = positions.options().dtype(torch::kInt32);
    auto float_options = positions.options().dtype(torch::kFloat32);
    torch::Tensor tet_tensor = torch::zeros({dt_result.num_tri, 4}, int_options);
    std::memcpy(tet_tensor.contiguous().data_ptr<int>(), 
                dt_result.tri_verts_idx, 
                dt_result.num_tri * 4 * sizeof(int));
    
    torch::Tensor cc_tensor = torch::zeros({0,}, float_options);
    if (compute_cc) {
        cc_tensor = torch::zeros({dt_result.num_tri, dimension}, float_options);
        std::memcpy(cc_tensor.contiguous().data_ptr<float>(), 
                    dt_result.tri_cc, 
                    dt_result.num_tri * dimension * sizeof(float));
    }
    
    return std::make_tuple(tet_tensor, cc_tensor, dt_result.time_sec);
}

torch::Tensor
ComputeSSOCUDA(
    // points info;
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
    
    const float dist_thresh,
    const int max_num_point_per_batch)
{
    int num_points = positions.size(0);
    int dimension = positions.size(1);
    int num_rp = rp_id.size(0);

    assert(dimension == 3);

    DIFFDT::RPI rp_info;
    rp_info.num_rp = num_rp;
    rp_info.rp_id = rp_id.contiguous().data_ptr<int>();
    rp_info.rp_cc_beg = rp_cc_beg.contiguous().data_ptr<int>();
    rp_info.rp_cc_end = rp_cc_end.contiguous().data_ptr<int>();
    rp_info.rp_cc_id = rp_cc_id.contiguous().data_ptr<int>();
    rp_info.rp_cc_d = rp_cc_d.contiguous().data_ptr<float>();

    auto sso_result = DIFFDT::sso(
        num_points,
        dimension,
        positions.data_ptr<float>(),
        weights.data_ptr<float>(),
        dist_thresh,

        &rp_info,

        tri_idx.contiguous().data_ptr<int>(),
        circumcenters.contiguous().data_ptr<float>(),
        max_num_point_per_batch
    );

    return sso_result;
}


std::tuple<torch::Tensor, torch::Tensor>
ComputeCircumcentersCUDA(
    const torch::Tensor& positions,
    const torch::Tensor& weights,
    const torch::Tensor& tri_idx
) 
{
    int num_tri = tri_idx.size(0);
    int dimension = positions.size(1);
    assert(dimension == 3);

    auto double_options = positions.options().dtype(torch::kFloat64);
    torch::Tensor cc_tensor = torch::zeros({num_tri, dimension}, double_options);
    torch::Tensor cc_unnorm_tensor = torch::zeros({num_tri, dimension + 2}, double_options);

    DIFFDT::cc(
        positions.packed_accessor64<double, 2>(),
        weights.packed_accessor64<double, 1>(),

        tri_idx.packed_accessor64<int, 2>(),
        cc_tensor.packed_accessor64<double, 2>(),
        cc_unnorm_tensor.packed_accessor64<double, 2>()
    );

    return std::make_tuple(cc_tensor, cc_unnorm_tensor);
}

std::tuple<torch::Tensor, torch::Tensor>
ComputeCircumcentersBackwardCUDA(
    const torch::Tensor& positions,
    const torch::Tensor& weights,
    const torch::Tensor& tri_idx,
    const torch::Tensor& cc_unnorm,
    const torch::Tensor& dL_dCC) 
{
    int num_points = positions.size(0);
    int dimension = positions.size(1);
    int num_tri = tri_idx.size(0);

    assert(dimension == 3);

    auto double_options = positions.options().dtype(torch::kFloat64);
    torch::Tensor dL_dPositions = torch::zeros({num_points, dimension}, double_options);
    torch::Tensor dL_dWeights = torch::zeros({num_points}, double_options);
    
    DIFFDT::cc_backward(
        positions.packed_accessor64<double, 2>(),
        weights.packed_accessor64<double, 1>(),

        tri_idx.packed_accessor64<int, 2>(),
        cc_unnorm.packed_accessor64<double, 2>(),
        dL_dCC.packed_accessor64<double, 2>(),

        dL_dPositions.packed_accessor64<double, 2>(),
        dL_dWeights.packed_accessor64<double, 1>()
    );

    return std::make_tuple(dL_dPositions, dL_dWeights);
}

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
) {
    int num_pd_edges = pd_edge_vertex_id.size(0);
    auto int_options = pd_rp_id.options().dtype(torch::kInt32);
    
    torch::Tensor per_pd_edge_nearest_hp_idx = torch::zeros({num_pd_edges, 3}, int_options);
    
    DIFFDT::iwdt0(
        pd_rp_id,
        pd_per_point_rp_id,

        pd_vertex_positions,
        pd_vertex_points_id,

        pd_edge_vertex_id,
        pd_edge_points_id,
        pd_edge_origin,
        pd_edge_direction,
        pd_per_rp_edge_index_array,
        pd_per_rp_edge_beg_array,
        pd_per_rp_edge_end_array,
        pd_per_rp_edge_dist_array,

        sso_per_rp_hp_pid_array,
        sso_per_rp_hp_normals_array,
        sso_per_rp_hp_offsets_array,
        sso_per_rp_beg_array,
        sso_per_rp_end_array,

        per_pd_edge_nearest_hp_idx,

        infinite_edge_delta
    );

    return per_pd_edge_nearest_hp_idx;
}

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
) {
    int num_faces = faces_points_id.size(0);
    auto int_options = pd_rp_id.options().dtype(torch::kInt32);
    
    torch::Tensor per_face_dl_nearest_pd_edge_idx = torch::zeros({num_faces, 3}, int_options);
    
    DIFFDT::iwdt1(
        pd_rp_id,
        pd_per_point_rp_id,

        pd_vertex_positions,
        pd_vertex_points_id,

        pd_edge_vertex_id,
        pd_edge_points_id,
        pd_edge_origin,
        pd_edge_direction,
        pd_per_rp_edge_index_array,
        pd_per_rp_edge_beg_array,
        pd_per_rp_edge_end_array,
        pd_per_rp_edge_dist_array,

        faces_points_id,
        faces_dl_origin,
        faces_dl_direction,
        faces_dl_axis_0,
        faces_dl_axis_1,

        per_face_dl_nearest_pd_edge_idx
    );

    return per_face_dl_nearest_pd_edge_idx;
}