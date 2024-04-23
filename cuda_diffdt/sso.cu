#include "sso.h"
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

__forceinline__ __device__ void compute_simplex_dual(
    const float3& this_position,
    const float this_weight,
    const float3& other_position,
    const float other_weight,
    float3& dual_normal,
    float& dual_offset,
    float& dual_plane_dist
) {
    // dual normal;
    float3 diff = other_position - this_position;
    float dist_sq = dot(diff, diff);
    float dist = sqrtf(dist_sq);
    dual_normal = diff / max(dist, EPS);

    // dual offset;
    float delta = 0.5 * (1.0 - ((other_weight - this_weight) / max(dist_sq, EPS)));
    float3 mid_point = (delta * other_position) + ((1 - delta) * this_position);
    dual_offset = dot(mid_point, dual_normal);
    dual_plane_dist = delta * dist;
}

// ========================================================================
// Kernel function for retrieving possible one-simplices
// ========================================================================
__global__ void ssoKernelCUDA(
    const int grid_offset_x,
    const int grid_offset_y,

	// real point info
	const int num_rp,           // number of real points
    const int* rp_id,           // point indices of real points
    const int* rp_cc_beg,       // begin indices in [rp_cc_pid] for each real point
    const int* rp_cc_end,       // end indices in [rp_cc_pid] for each real point
    const int* rp_cc_id,        // circumcenter ids of real points
    const float* rp_cc_d,       // distances of [cc]s to real points, sorted in descending order

    // point info
	const int dim,
    const float* positions,
    const float* weights,

    // cc info
    const float* cc_pos,
    const int* cc_pt_id,
    
    const float dist_thresh,

    const int validity_mat_size_x,
    const int validity_mat_size_y,
    torch::PackedTensorAccessor32<bool, 2> validity_mat
) {
    auto block = cg::this_thread_block();
    auto local_x = block.group_index().x * BLOCK_X + block.thread_index().x;
    auto local_y = block.group_index().y * BLOCK_Y + block.thread_index().y;
    auto global_x = grid_offset_x + local_x;
    auto global_y = grid_offset_y + local_y;

    if (local_x >= validity_mat_size_x || local_y >= validity_mat_size_y)
        return;

    // initialize to false;
    validity_mat[local_x][local_y] = false;

    // if the two rp's were the same, just break;
    if (global_x == global_y)
        return;

    if (global_x >= num_rp || global_y >= num_rp)
        return;

    // this rp info
    int this_rp_id = rp_id[global_x];
    float3 this_rp_position = make_float3(
        positions[this_rp_id * dim], 
        positions[this_rp_id * dim + 1], 
        positions[this_rp_id * dim + 2]);
    float this_rp_weight = weights[this_rp_id];

    int this_rp_cc_beg = rp_cc_beg[global_x];
    int this_rp_cc_end = rp_cc_end[global_x];
    int this_rp_num_cc = this_rp_cc_end - this_rp_cc_beg;

    // other rp info
    int other_rp_id = rp_id[global_y];
    float3 other_rp_position = make_float3(
        positions[other_rp_id * dim], 
        positions[other_rp_id * dim + 1], 
        positions[other_rp_id * dim + 2]);
    float other_rp_weight = weights[other_rp_id];
    
    // find half plane
    float3 dual_plane_normal;
    float dual_plane_offset;
    float this_rp_dual_plane_dist;  // signed distance;
    compute_simplex_dual(
        this_rp_position, this_rp_weight,
        other_rp_position, other_rp_weight,
        dual_plane_normal, dual_plane_offset, this_rp_dual_plane_dist);

    // iterate through cc's and see the distance between 
    // power cell and the half plane...    
    for (int i = 0; i < this_rp_num_cc; i++) {
        float curr_ccd = rp_cc_d[this_rp_cc_beg + i];        // radius of bounding ball
                                                            // centered at [this_rp],
                                                            // encompassing every [cc]s
                                                            // under consideration
        int curr_cc_id = rp_cc_id[this_rp_cc_beg + i];
        float3 curr_cc_pos = make_float3(
            cc_pos[curr_cc_id * dim + 0],
            cc_pos[curr_cc_id * dim + 1],
            cc_pos[curr_cc_id * dim + 2]);

        // if [other_rp] alreay comprises the power cell,
        // it is definitely possible;
        for (int j = 0; j < 4; j++) {
            if (other_rp_id == cc_pt_id[4 * curr_cc_id + j]) {
                validity_mat[local_x][local_y] = true;
                return;
            }
        }
            
        // if the dual plane intersects with the bounding ball...
        // [this_rp_dual_plane_dist]: signed distance, + or -
        // [curr_ccd + dist_thresh]: always positive
        if (this_rp_dual_plane_dist < curr_ccd + dist_thresh) {
            
            // compute distance between dual plane and [cc];
            // this distance is unsigned, because anway the
            // dual plane cannot intersect power cell;
            float cc_dp_dist = fabs(dot(dual_plane_normal, curr_cc_pos) - dual_plane_offset);
            
            // if the dual plane is truly near the power cell,
            // we have to investigate further of it;
            if (cc_dp_dist < dist_thresh) {
                validity_mat[local_x][local_y] = true;
                return;
            }
            // if the dual plane was not near the power cell
            // in terms of current [cc], move onto next [cc];
            else
                continue;
        }
        // if the dual plane does not intersect with the bounding ball,
        // there is no chance anymore...
        else
            return;
    }
}

__global__ void translateCUDA(
    const int grid_offset_x,
    const int grid_offset_y,
    const int* rp_id,
    const int num_point_pair,
    torch::PackedTensorAccessor32<int, 2> point_pair
) {
    auto idx = cg::this_grid().thread_rank();
	if (idx >= num_point_pair)
		return;

    int local_point_pair_x = point_pair[idx][0];
    int local_point_pair_y = point_pair[idx][1];
    
    int point_pair_x = grid_offset_x + local_point_pair_x;
    int point_pair_y = grid_offset_y + local_point_pair_y;

    point_pair[idx][0] = rp_id[point_pair_x];
    point_pair[idx][1] = rp_id[point_pair_y];
}

torch::Tensor DIFFDT::sso(
    int num_points,
    int dim,
    const float* positions,
    const float* weights,
    const float dist_thresh,
    
    // info about real points;
    const RPI* rp_info,

    // info about circumcenters;
    const int* cc_pt_id,
    const float* cc_pos,

    int max_num_point_per_batch
) {
    assert(max_num_point_per_batch > 0 && max_num_point_per_batch <= 30000);

    int num_entire_point_to_process = rp_info->num_rp;      // have to fill in [num_rp] x [num_rp] boolean matrix
    int num_batch = (num_entire_point_to_process + max_num_point_per_batch - 1) / 
                        max_num_point_per_batch;
    int num_point_to_process_per_batch = min(num_entire_point_to_process, max_num_point_per_batch);

    // allocate memory for boolean matrix using tensor;
    auto tensor_bool_options = torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA, 0);
    auto validity_tensor = torch::zeros({num_point_to_process_per_batch, num_point_to_process_per_batch}, 
                                            tensor_bool_options);

    // place to store the result;
    auto tensor_int_options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0);
    auto point_pair_tensor = torch::zeros({0, 2}, tensor_int_options);
    
    // kernel config;
    dim3 grid_size((num_point_to_process_per_batch + BLOCK_X - 1) / BLOCK_X, 
                    (num_point_to_process_per_batch + BLOCK_Y - 1) / BLOCK_Y, 1);
    dim3 block_size(BLOCK_X, BLOCK_Y, 1);

    for (int i = 0; i < num_batch; i++) {
        for (int j = 0; j < num_batch; j++) {
            torch::fill(validity_tensor, false);

            int grid_offset_x = i * num_point_to_process_per_batch;
            int grid_offset_y = j * num_point_to_process_per_batch;

            // launch kernel;
            ssoKernelCUDA<<<grid_size, block_size>>>(
                grid_offset_x,
                grid_offset_y,

                rp_info->num_rp,
                rp_info->rp_id,
                rp_info->rp_cc_beg,
                rp_info->rp_cc_end,
                rp_info->rp_cc_id,
                rp_info->rp_cc_d,

                dim,
                positions,
                weights,

                cc_pos,
                cc_pt_id,

                dist_thresh,

                num_point_to_process_per_batch,
                num_point_to_process_per_batch,
                validity_tensor.packed_accessor32<bool, 2>()
            );
            cudaDeviceSynchronize();

            // get indices of entries that are true;
            torch::Tensor desired_point_pairs = torch::nonzero(validity_tensor);
            desired_point_pairs = desired_point_pairs.to(torch::kInt32);
            translateCUDA<<<desired_point_pairs.size(0), BLOCK_SIZE>>>(
                grid_offset_x,
                grid_offset_y,
                rp_info->rp_id,
                desired_point_pairs.size(0),
                desired_point_pairs.packed_accessor32<int, 2>()
            );
            cudaDeviceSynchronize();
            point_pair_tensor = torch::cat({point_pair_tensor, desired_point_pairs}, 0);
        }
    }
    return point_pair_tensor;
}