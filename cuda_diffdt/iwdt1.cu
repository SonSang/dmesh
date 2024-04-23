#include "iwdt1.h"
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

__forceinline__ __device__ void lineseg_line_distance(
    float3 lineseg_beg,
    float3 lineseg_end,
    bool lineseg_inf,

    float3 line_origin,
    float3 line_direction,
    float3 line_axis_0,
    float3 line_axis_1,

    float &dist,
    float &t,
    float eps=1e-6f
) {
    // compute distance between lineseg and line;
    float3 diff_0 = lineseg_beg - line_origin;
    float3 diff_1 = lineseg_end - line_origin;

    float lineseg0_x = dot(diff_0, line_axis_0);
    float lineseg0_y = dot(diff_0, line_axis_1);
    float lineseg1_x = dot(diff_1, line_axis_0);
    float lineseg1_y = dot(diff_1, line_axis_1);
    
    float2 lineseg0_xy = make_float2(lineseg0_x, lineseg0_y);
    float2 lineseg1_xy = make_float2(lineseg1_x, lineseg1_y);
    
    float2 lineseg_dir_xy = lineseg1_xy - lineseg0_xy;

    float nom = dot(lineseg0_xy, lineseg_dir_xy);
    float denom = dot(lineseg_dir_xy, lineseg_dir_xy);
    denom = max(denom, eps);
    
    t = -(nom / denom);
    if (lineseg_inf)
        t = max(t, 0.);
    else
        t = clamp(t, 0., 1.);
        
    float2 foot = lineseg0_xy + t * lineseg_dir_xy;
    dist = sqrt(dot(foot, foot));
}

// ========================================================================
// Kernel function for IWDT1
// ========================================================================
__global__ void iwdt1CUDA(
    int block_size_x,

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

    // faces;
    torch::PackedTensorAccessor64<int, 2> faces_points_id,
    torch::PackedTensorAccessor64<float, 2> faces_dl_origin,
    torch::PackedTensorAccessor64<float, 2> faces_dl_direction,
    torch::PackedTensorAccessor64<float, 2> faces_dl_axis_0,
    torch::PackedTensorAccessor64<float, 2> faces_dl_axis_1,

    // outputs;
    torch::PackedTensorAccessor64<int, 2> per_face_dl_nearest_pd_edge_idx
) {
    auto block = cg::this_thread_block();
    auto face_id = block.group_index().x * block_size_x + block.thread_index().x;
    auto k = block.thread_index().y;

    int num_faces = faces_points_id.size(0);
	if (face_id >= num_faces)
		return;

    // get information of dual line of current face;
    float3 curr_face_dl_origin = make_float3(
        faces_dl_origin[face_id][0],
        faces_dl_origin[face_id][1],
        faces_dl_origin[face_id][2]
    );
    float3 curr_face_dl_direction = make_float3(
        faces_dl_direction[face_id][0],
        faces_dl_direction[face_id][1],
        faces_dl_direction[face_id][2]
    );
    float3 curr_face_dl_axis_0 = make_float3(
        faces_dl_axis_0[face_id][0],
        faces_dl_axis_0[face_id][1],
        faces_dl_axis_0[face_id][2]
    );
    float3 curr_face_dl_axis_1 = make_float3(
        faces_dl_axis_1[face_id][0],
        faces_dl_axis_1[face_id][1],
        faces_dl_axis_1[face_id][2]
    );

    // process;
    per_face_dl_nearest_pd_edge_idx[face_id][k] = -1;

    int curr_point_id = faces_points_id[face_id][k];
    int curr_point_rp_id = pd_per_point_rp_id[curr_point_id];
    if (curr_point_rp_id == -1) {
        // point does not have a power cell;
        printf("Error (IWDT1 CUDA): Point does not have a power cell!\n");
        return;
    }

    // get pd edges of this power cell;
    int curr_point_pd_edge_beg = pd_per_rp_edge_beg_array[curr_point_rp_id];
    int curr_point_pd_edge_end = pd_per_rp_edge_end_array[curr_point_rp_id];
        
    // find the nearest pd edge;
    float min_dist = 1e10;
    int min_dist_pd_edge_id = -1;
    
    for (int i = curr_point_pd_edge_beg; i < curr_point_pd_edge_end; i++) {
        int curr_pd_edge_id = pd_per_rp_edge_index_array[i];

        // get information about current pd edge;
        int curr_pd_edge_vertex_id[2] = {
            pd_edge_vertex_id[curr_pd_edge_id][0],
            pd_edge_vertex_id[curr_pd_edge_id][1]
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
                pd_edge_origin[curr_pd_edge_id][0],
                pd_edge_origin[curr_pd_edge_id][1],
                pd_edge_origin[curr_pd_edge_id][2]
            );
            float3 curr_pd_edge_direction = make_float3(
                pd_edge_direction[curr_pd_edge_id][0],
                pd_edge_direction[curr_pd_edge_id][1],
                pd_edge_direction[curr_pd_edge_id][2]
            );

            curr_pd_edge_vertex_pos[1] = 
                curr_pd_edge_origin + curr_pd_edge_direction;
        }

        // compute distance between lineseg and line;
        float dist, t;
        lineseg_line_distance(
            curr_pd_edge_vertex_pos[0],
            curr_pd_edge_vertex_pos[1],
            curr_pd_edge_vertex_id[1] == -1,
            curr_face_dl_origin,
            curr_face_dl_direction,
            curr_face_dl_axis_0,
            curr_face_dl_axis_1,

            dist,
            t
        );
            
        if (dist < min_dist || min_dist_pd_edge_id == -1) {
            min_dist = dist;
            min_dist_pd_edge_id = curr_pd_edge_id;
        }
    }

    // save result;
    per_face_dl_nearest_pd_edge_idx[face_id][k] = min_dist_pd_edge_id;
}

void DIFFDT::iwdt1(
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

    // outputs;
    torch::Tensor& per_face_dl_nearest_pd_edge_idx
) {
    int num_faces = faces_points_id.size(0);

    int block_size_x = BLOCK_SIZE / 3;
    int block_size_y = 3;

    dim3 grid_size((num_faces + block_size_x - 1) / block_size_x, 1, 1);
    dim3 block_size(block_size_x, block_size_y, 1);

    // launch kernel;
    iwdt1CUDA<<<grid_size, block_size>>>(
        block_size_x,

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
        pd_per_rp_edge_beg_array_beg.packed_accessor64<int, 1>(),
        pd_per_rp_edge_end_array_end.packed_accessor64<int, 1>(),
        pd_per_rp_edge_dist_array.packed_accessor64<float, 1>(),

        // faces;
        faces_points_id.packed_accessor64<int, 2>(),
        faces_dl_origin.packed_accessor64<float, 2>(),
        faces_dl_direction.packed_accessor64<float, 2>(),
        faces_dl_axis_0.packed_accessor64<float, 2>(),
        faces_dl_axis_1.packed_accessor64<float, 2>(),

        // outputs;
        per_face_dl_nearest_pd_edge_idx.packed_accessor64<int, 2>()
    );
}