#include "iwdt0.h"
#include "cuda_math.h"
#include "common.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
namespace cg = cooperative_groups;

// ========================================================================
// Kernel function for IWDT0
// ========================================================================
__global__ void iwdt0CUDA(
    // PDStruct;
    torch::PackedTensorAccessor64<int, 1> pd_rp_id,
    torch::PackedTensorAccessor64<int, 1> pd_per_point_rp_id,

    // pd vertex;
    torch::PackedTensorAccessor64<float, 2> pd_vertex_positions,
    torch::PackedTensorAccessor64<int, 2> pd_vertex_points_id,

    // pd edge;
    torch::PackedTensorAccessor64<int, 2> pd_edge_vertex_id,
    torch::PackedTensorAccessor64<int, 2> pd_edge_points_id,
    torch::PackedTensorAccessor64<float, 2> pd_edge_origin,
    torch::PackedTensorAccessor64<float, 2> pd_edge_direction,
    torch::PackedTensorAccessor64<int, 1> pd_per_rp_edge_index_array,
    torch::PackedTensorAccessor64<int, 1> pd_per_rp_edge_beg_array,
    torch::PackedTensorAccessor64<int, 1> pd_per_rp_edge_end_array,
    torch::PackedTensorAccessor64<float, 1> pd_per_rp_edge_dist_array,

    // sso;
    torch::PackedTensorAccessor64<int, 2> sso_per_rp_hp_pid_array,
    torch::PackedTensorAccessor64<float, 2> sso_per_rp_hp_normals_array,
    torch::PackedTensorAccessor64<float, 1> sso_per_rp_hp_offsets_array,
    torch::PackedTensorAccessor64<int, 1> sso_per_rp_beg_array,
    torch::PackedTensorAccessor64<int, 1> sso_per_rp_end_array,

    torch::PackedTensorAccessor64<int, 2> per_pd_edge_nearest_hp_idx,

    float infinite_edge_delta
) {
    int num_pd_edges = pd_edge_vertex_id.size(0);
    auto idx = cg::this_grid().thread_rank();
	if (idx >= num_pd_edges)
		return;

    // get information about current pd edge;
    int curr_pd_edge_points_id[3] = {
        pd_edge_points_id[idx][0],
        pd_edge_points_id[idx][1],
        pd_edge_points_id[idx][2]
    };
    int curr_pd_edge_vertex_id[2] = {
        pd_edge_vertex_id[idx][0],
        pd_edge_vertex_id[idx][1]
    };
    
    // two end points of current pd edge;
    float3 curr_pd_edge_vertex_pos[2];
    curr_pd_edge_vertex_pos[0] = make_float3(
        pd_vertex_positions[curr_pd_edge_vertex_id[0]][0],
        pd_vertex_positions[curr_pd_edge_vertex_id[0]][1],
        pd_vertex_positions[curr_pd_edge_vertex_id[0]][2]
    );

    if (curr_pd_edge_vertex_id[1] != -1) {
        // finite edge;
        curr_pd_edge_vertex_pos[1] = make_float3(
            pd_vertex_positions[curr_pd_edge_vertex_id[1]][0],
            pd_vertex_positions[curr_pd_edge_vertex_id[1]][1],
            pd_vertex_positions[curr_pd_edge_vertex_id[1]][2]
        );
    }
    else {
        // infinite edge;
        float3 curr_pd_edge_origin = make_float3(
            pd_edge_origin[idx][0],
            pd_edge_origin[idx][1],
            pd_edge_origin[idx][2]
        );
        float3 curr_pd_edge_direction = make_float3(
            pd_edge_direction[idx][0],
            pd_edge_direction[idx][1],
            pd_edge_direction[idx][2]
        );

        curr_pd_edge_vertex_pos[1] = 
            curr_pd_edge_origin + 
            curr_pd_edge_direction * infinite_edge_delta;
    }

    // get sample point of current pd edge;
    float3 curr_pd_edge_sample_point = (curr_pd_edge_vertex_pos[0] + curr_pd_edge_vertex_pos[1]) * 0.5f;
    
    // for each point in this face, get half planes nearyby each power cell
    // and project the above point onto the half planes to find delta;
    for (int j = 0; j < 3; j++) {
        per_pd_edge_nearest_hp_idx[idx][j] = -1;

        int curr_pd_edge_point_id = curr_pd_edge_points_id[j];
        int curr_pd_edge_point_rp_id = pd_per_point_rp_id[curr_pd_edge_point_id];
        if (curr_pd_edge_point_rp_id == -1) {
            // point does not have a power cell;
            printf("Error (IWDT0 CUDA): Point does not have a power cell!\n");
            continue;
        }

        int curr_pd_edge_point_hp_beg = sso_per_rp_beg_array[curr_pd_edge_point_rp_id];
        int curr_pd_edge_point_hp_end = sso_per_rp_end_array[curr_pd_edge_point_rp_id];

        // for each nearby half plane, project point onto half plane;
        float min_dist = 1e10;
        int min_dist_hp_id = -1;

        for (int k = curr_pd_edge_point_hp_beg; k < curr_pd_edge_point_hp_end; k++) {
            int curr_pd_edge_point_hp_pid[2] = {
                sso_per_rp_hp_pid_array[k][0],
                sso_per_rp_hp_pid_array[k][1]
            };

            // skip if this half plane includes the current pd edge;
            int icnt = 0;
            for (int m = 0; m < 3; m++) {
                for (int n = 0; n < 2; n++) {
                    if (curr_pd_edge_points_id[m] == curr_pd_edge_point_hp_pid[n]) {
                        icnt += 1;
                    }
                }
            }
            if (icnt >= 2)
                continue;

            float3 curr_pd_edge_point_hp_normal = make_float3(
                sso_per_rp_hp_normals_array[k][0],
                sso_per_rp_hp_normals_array[k][1],
                sso_per_rp_hp_normals_array[k][2]
            );

            float curr_pd_edge_point_hp_offset = sso_per_rp_hp_offsets_array[k];

            // project point onto half plane;
            float dist = fabs(dot(curr_pd_edge_point_hp_normal, curr_pd_edge_sample_point) - \
                            curr_pd_edge_point_hp_offset);
            
            if (dist < min_dist || min_dist_hp_id == -1) {
                min_dist = dist;
                min_dist_hp_id = k;
            }
        }
        per_pd_edge_nearest_hp_idx[idx][j] = min_dist_hp_id;
    }
}


void DIFFDT::iwdt0(
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

    torch::Tensor& per_pd_edge_nearest_hp_idx,

    float infinite_edge_delta
) {
    int num_pd_edges = pd_edge_vertex_id.size(0);

    // launch kernel;
    iwdt0CUDA<<<(num_pd_edges + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        // PDStruct;
        pd_rp_id.packed_accessor64<int, 1>(),
        pd_per_point_rp_id.packed_accessor64<int, 1>(),

        // pd vertex;
        pd_vertex_positions.packed_accessor64<float, 2>(),
        pd_vertex_points_id.packed_accessor64<int, 2>(),

        // pd edge;
        pd_edge_vertex_id.packed_accessor64<int, 2>(),
        pd_edge_points_id.packed_accessor64<int, 2>(),
        pd_edge_origin.packed_accessor64<float, 2>(),
        pd_edge_direction.packed_accessor64<float, 2>(),
        pd_per_rp_edge_index_array.packed_accessor64<int, 1>(),
        pd_per_rp_edge_beg_array.packed_accessor64<int, 1>(),
        pd_per_rp_edge_end_array.packed_accessor64<int, 1>(),
        pd_per_rp_edge_dist_array.packed_accessor64<float, 1>(),

        // sso;
        sso_per_rp_hp_pid_array.packed_accessor64<int, 2>(),
        sso_per_rp_hp_normals_array.packed_accessor64<float, 2>(),
        sso_per_rp_hp_offsets_array.packed_accessor64<float, 1>(),
        sso_per_rp_beg_array.packed_accessor64<int, 1>(),
        sso_per_rp_end_array.packed_accessor64<int, 1>(),

        per_pd_edge_nearest_hp_idx.packed_accessor64<int, 2>(),

        infinite_edge_delta
    );
}