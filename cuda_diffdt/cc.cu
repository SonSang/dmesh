#include "cc.h"
#include "cuda_math.h"
#include "common.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
namespace cg = cooperative_groups;

#define CC_EPS 1e-6

// ========================================================================
// Kernel function for computing circumcenter with determinant
// ========================================================================
__forceinline__ __device__ float det3(float3 a, float3 b, float3 c) {
    return a.x * (b.y * c.z - b.z * c.y) -
           a.y * (b.x * c.z - b.z * c.x) +
           a.z * (b.x * c.y - b.y * c.x);
}

__forceinline__ __device__ double det3d(double3 a, double3 b, double3 c) {
    return a.x * (b.y * c.z - b.z * c.y) -
           a.y * (b.x * c.z - b.z * c.x) +
           a.z * (b.x * c.y - b.y * c.x);
}

__forceinline__ __device__ float det4(float4 a, float4 b, float4 c, float4 d) {
    return a.x * det3(make_float3(b.y, b.z, b.w), make_float3(c.y, c.z, c.w),
                      make_float3(d.y, d.z, d.w)) -
            a.y * det3(make_float3(b.x, b.z, b.w), make_float3(c.x, c.z, c.w),
                    make_float3(d.x, d.z, d.w)) +
            a.z * det3(make_float3(b.x, b.y, b.w), make_float3(c.x, c.y, c.w),
                    make_float3(d.x, d.y, d.w)) -
            a.w * det3(make_float3(b.x, b.y, b.z), make_float3(c.x, c.y, c.z),
                    make_float3(d.x, d.y, d.z));
}

__forceinline__ __device__ double det4d(double4 a, double4 b, double4 c, double4 d) {
    return a.x * det3d(make_double3(b.y, b.z, b.w), make_double3(c.y, c.z, c.w),
                      make_double3(d.y, d.z, d.w)) -
            a.y * det3d(make_double3(b.x, b.z, b.w), make_double3(c.x, c.z, c.w),
                    make_double3(d.x, d.z, d.w)) +
            a.z * det3d(make_double3(b.x, b.y, b.w), make_double3(c.x, c.y, c.w),
                    make_double3(d.x, d.y, d.w)) -
            a.w * det3d(make_double3(b.x, b.y, b.z), make_double3(c.x, c.y, c.z),
                    make_double3(d.x, d.y, d.z));
}

__forceinline__ __device__ void det4_deriv(
    float4 a, float4 b, float4 c, float4 d, 
    float4& dDet_da, float4& dDet_db, float4& dDet_dc, float4& dDet_dd) {
    
    dDet_da.x = det3(make_float3(b.y, b.z, b.w), 
                    make_float3(c.y, c.z, c.w),
                    make_float3(d.y, d.z, d.w));
    dDet_da.y = -det3(make_float3(b.x, b.z, b.w),
                    make_float3(c.x, c.z, c.w),
                    make_float3(d.x, d.z, d.w));
    dDet_da.z = det3(make_float3(b.x, b.y, b.w),
                    make_float3(c.x, c.y, c.w),
                    make_float3(d.x, d.y, d.w));
    dDet_da.w = -det3(make_float3(b.x, b.y, b.z),
                    make_float3(c.x, c.y, c.z),
                    make_float3(d.x, d.y, d.z));

    dDet_db.x = -det3(make_float3(a.y, a.z, a.w),
                    make_float3(c.y, c.z, c.w),
                    make_float3(d.y, d.z, d.w));
    dDet_db.y = det3(make_float3(a.x, a.z, a.w),
                    make_float3(c.x, c.z, c.w),
                    make_float3(d.x, d.z, d.w));
    dDet_db.z = -det3(make_float3(a.x, a.y, a.w),
                    make_float3(c.x, c.y, c.w),
                    make_float3(d.x, d.y, d.w));
    dDet_db.w = det3(make_float3(a.x, a.y, a.z),
                    make_float3(c.x, c.y, c.z),
                    make_float3(d.x, d.y, d.z));

    dDet_dc.x = det3(make_float3(a.y, a.z, a.w),
                    make_float3(b.y, b.z, b.w),
                    make_float3(d.y, d.z, d.w));
    dDet_dc.y = -det3(make_float3(a.x, a.z, a.w),
                    make_float3(b.x, b.z, b.w),
                    make_float3(d.x, d.z, d.w));
    dDet_dc.z = det3(make_float3(a.x, a.y, a.w),
                    make_float3(b.x, b.y, b.w),
                    make_float3(d.x, d.y, d.w));
    dDet_dc.w = -det3(make_float3(a.x, a.y, a.z),
                    make_float3(b.x, b.y, b.z),
                    make_float3(d.x, d.y, d.z));

    dDet_dd.x = -det3(make_float3(a.y, a.z, a.w),
                    make_float3(b.y, b.z, b.w),
                    make_float3(c.y, c.z, c.w));
    dDet_dd.y = det3(make_float3(a.x, a.z, a.w),
                    make_float3(b.x, b.z, b.w),
                    make_float3(c.x, c.z, c.w));
    dDet_dd.z = -det3(make_float3(a.x, a.y, a.w),
                    make_float3(b.x, b.y, b.w),
                    make_float3(c.x, c.y, c.w));
    dDet_dd.w = det3(make_float3(a.x, a.y, a.z),
                    make_float3(b.x, b.y, b.z),
                    make_float3(c.x, c.y, c.z));
}

__forceinline__ __device__ void det4d_deriv(
    double4 a, double4 b, double4 c, double4 d, 
    double4& dDet_da, double4& dDet_db, double4& dDet_dc, double4& dDet_dd) {
    
    dDet_da.x = det3d(make_double3(b.y, b.z, b.w), 
                    make_double3(c.y, c.z, c.w),
                    make_double3(d.y, d.z, d.w));
    dDet_da.y = -det3d(make_double3(b.x, b.z, b.w),
                    make_double3(c.x, c.z, c.w),
                    make_double3(d.x, d.z, d.w));
    dDet_da.z = det3d(make_double3(b.x, b.y, b.w),
                    make_double3(c.x, c.y, c.w),
                    make_double3(d.x, d.y, d.w));
    dDet_da.w = -det3d(make_double3(b.x, b.y, b.z),
                    make_double3(c.x, c.y, c.z),
                    make_double3(d.x, d.y, d.z));

    dDet_db.x = -det3d(make_double3(a.y, a.z, a.w),
                    make_double3(c.y, c.z, c.w),
                    make_double3(d.y, d.z, d.w));
    dDet_db.y = det3d(make_double3(a.x, a.z, a.w),
                    make_double3(c.x, c.z, c.w),
                    make_double3(d.x, d.z, d.w));
    dDet_db.z = -det3d(make_double3(a.x, a.y, a.w),
                    make_double3(c.x, c.y, c.w),
                    make_double3(d.x, d.y, d.w));
    dDet_db.w = det3d(make_double3(a.x, a.y, a.z),
                    make_double3(c.x, c.y, c.z),
                    make_double3(d.x, d.y, d.z));

    dDet_dc.x = det3d(make_double3(a.y, a.z, a.w),
                    make_double3(b.y, b.z, b.w),
                    make_double3(d.y, d.z, d.w));
    dDet_dc.y = -det3d(make_double3(a.x, a.z, a.w),
                    make_double3(b.x, b.z, b.w),
                    make_double3(d.x, d.z, d.w));
    dDet_dc.z = det3d(make_double3(a.x, a.y, a.w),
                    make_double3(b.x, b.y, b.w),
                    make_double3(d.x, d.y, d.w));
    dDet_dc.w = -det3d(make_double3(a.x, a.y, a.z),
                    make_double3(b.x, b.y, b.z),
                    make_double3(d.x, d.y, d.z));

    dDet_dd.x = -det3d(make_double3(a.y, a.z, a.w),
                    make_double3(b.y, b.z, b.w),
                    make_double3(c.y, c.z, c.w));
    dDet_dd.y = det3d(make_double3(a.x, a.z, a.w),
                    make_double3(b.x, b.z, b.w),
                    make_double3(c.x, c.z, c.w));
    dDet_dd.z = -det3d(make_double3(a.x, a.y, a.w),
                    make_double3(b.x, b.y, b.w),
                    make_double3(c.x, c.y, c.w));
    dDet_dd.w = det3d(make_double3(a.x, a.y, a.z),
                    make_double3(b.x, b.y, b.z),
                    make_double3(c.x, c.y, c.z));
}

__global__ void ccKernelCUDA(
	// point info;
    torch::PackedTensorAccessor64<double, 2> positions,
    torch::PackedTensorAccessor64<double, 1> weights,
    
    // tri info;
    torch::PackedTensorAccessor64<int, 2> tri_idx,
    torch::PackedTensorAccessor64<double, 2> tri_cc,
    torch::PackedTensorAccessor64<double, 2> tri_cc_unnorm
) {
    int num_points = positions.size(0);
    int dim = positions.size(1);
    int num_tri = tri_idx.size(0);

	auto idx = cg::this_grid().thread_rank();
	if (idx >= num_tri)
		return;

    // corresponding (dim + 1)-dimensional hyperplanes for each points; 
    double hp[4][5] = {};
    for (int i = 0; i < 4; i++) {
        int pt_id = tri_idx[idx][i];
        auto ptx = positions[pt_id][0];
        auto pty = positions[pt_id][1];
        auto ptz = positions[pt_id][2];
        auto ptw = weights[pt_id];

        hp[i][0] = 2 * ptx;
        hp[i][1] = 2 * pty;
        hp[i][2] = 2 * ptz;
        hp[i][3] = -1;
        hp[i][4] = ptw - (ptx * ptx + pty * pty + ptz * ptz);
    }

    // cross product;
    double dets[5];
    double4 rows[4];
    for (int i = 0; i < 5; i++) {
        // This is different from the original code
        // that computes determinant of 4x4 matrix
        for (int j = 0; j < 4; j++) {
            rows[j].x = hp[j][(i + 1) % 5];
            rows[j].y = hp[j][(i + 2) % 5];
            rows[j].z = hp[j][(i + 3) % 5];
            rows[j].w = hp[j][(i + 4) % 5];
        }

        dets[i] = det4d(rows[0], rows[1], rows[2], rows[3]);
    }

    // adjust last dimension not to be zero;
    if (dets[4] < 0 && dets[4] > -CC_EPS)
        dets[4] = -CC_EPS;
    else if (dets[4] >= 0 && dets[4] < CC_EPS)
        dets[4] = CC_EPS;

    for (int i = 0; i < 5; i++)
        tri_cc_unnorm[idx][i] = dets[i];

    // compute circumcenter;
    for (int i = 0; i < 3; i++)
        tri_cc[idx][i] = dets[i] / dets[4];
}

void DIFFDT::cc(
    // point info;
    torch::PackedTensorAccessor64<double, 2> positions,
    torch::PackedTensorAccessor64<double, 1> weights,
    
    // tri info;
    torch::PackedTensorAccessor64<int, 2> tri_idx,
    torch::PackedTensorAccessor64<double, 2> tri_cc,
    torch::PackedTensorAccessor64<double, 2> tri_cc_unnorm
) {
    // launch kernel;
    int num_tri = tri_idx.size(0);
    int grid_size = (num_tri + BLOCK_SIZE - 1) / BLOCK_SIZE;
    ccKernelCUDA<<<grid_size, BLOCK_SIZE>>>(
        positions, 
        weights, 
        
        tri_idx, 
        tri_cc, 
        tri_cc_unnorm
    );
    cudaDeviceSynchronize();
}

// ========================================================================
// BACKWARD
// ========================================================================

__global__ void ccKernelBackwardCUDA(
	// point info
	torch::PackedTensorAccessor64<double, 2> positions,
    torch::PackedTensorAccessor64<double, 1> weights,

    // tri info
    torch::PackedTensorAccessor64<int, 2> tri_idx,
    torch::PackedTensorAccessor64<double, 2> tri_cc_unnorm,
    torch::PackedTensorAccessor64<double, 2> grad_tri_cc,

    // point grad
    torch::PackedTensorAccessor64<double, 2> grad_positions,
    torch::PackedTensorAccessor64<double, 1> grad_weights
) {
    int num_points = positions.size(0);
    int dim = positions.size(1);
    int num_tri = tri_idx.size(0);

	auto idx = cg::this_grid().thread_rank();
	if (idx >= num_tri)
		return;
    
    double dL_dcc[3];
    dL_dcc[0] = grad_tri_cc[idx][0];
    dL_dcc[1] = grad_tri_cc[idx][1];
    dL_dcc[2] = grad_tri_cc[idx][2];

    /*
    Gradient of loss w.r.t. unnormalized circumcenter;
    */
    double dcc_dcc_unnorm[3][5] = {};    // initialize to zero;
    
    auto cc_unnorm_x = tri_cc_unnorm[idx][0];
    auto cc_unnorm_y = tri_cc_unnorm[idx][1];
    auto cc_unnorm_z = tri_cc_unnorm[idx][2];
    auto cc_unnorm_w = tri_cc_unnorm[idx][4];
    
    dcc_dcc_unnorm[0][0] = 1.0 / cc_unnorm_w;
    dcc_dcc_unnorm[0][1] = 0;
    dcc_dcc_unnorm[0][2] = 0;
    dcc_dcc_unnorm[0][3] = 0;
    dcc_dcc_unnorm[0][4] = -cc_unnorm_x / (cc_unnorm_w * cc_unnorm_w);
    
    dcc_dcc_unnorm[1][0] = 0;
    dcc_dcc_unnorm[1][1] = 1.0 / cc_unnorm_w;
    dcc_dcc_unnorm[1][2] = 0;
    dcc_dcc_unnorm[1][3] = 0;
    dcc_dcc_unnorm[1][4] = -cc_unnorm_y / (cc_unnorm_w * cc_unnorm_w);

    dcc_dcc_unnorm[2][0] = 0;
    dcc_dcc_unnorm[2][1] = 0;
    dcc_dcc_unnorm[2][2] = 1.0 / cc_unnorm_w;
    dcc_dcc_unnorm[2][3] = 0;
    dcc_dcc_unnorm[2][4] = -cc_unnorm_z / (cc_unnorm_w * cc_unnorm_w);

    double dL_dcc_unnorm[5] = {};    // initialize to zero;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 3; j++) {
            dL_dcc_unnorm[i] += dL_dcc[j] * dcc_dcc_unnorm[j][i];
        }
    }

    /*
    Gradient of loss w.r.t. hyperplane;
    */

    double hp[4][5];
    for (int i = 0; i < 4; i++) {
        int pt_id = tri_idx[idx][i];
        auto ptx = positions[pt_id][0];
        auto pty = positions[pt_id][1];
        auto ptz = positions[pt_id][2];
        auto ptw = weights[pt_id];

        hp[i][0] = 2 * ptx;
        hp[i][1] = 2 * pty;
        hp[i][2] = 2 * ptz;
        hp[i][3] = -1;
        hp[i][4] = ptw - (ptx * ptx + pty * pty + ptz * ptz);
    }

    double dL_dhp[4][5] = {};    // initialize to zero;
    double4 rows[4];
    double4 dDet_drows[4];
    for (int i = 0; i < 5; i++) {
        // 4 x 4 submatrix;
        for (int j = 0; j < 4; j++) {
            rows[j].x = hp[j][(i + 1) % 5];
            rows[j].y = hp[j][(i + 2) % 5];
            rows[j].z = hp[j][(i + 3) % 5];
            rows[j].w = hp[j][(i + 4) % 5];
        }

        // derivative of determinant of 4 x 4 submatrix;
        det4d_deriv(rows[0], rows[1], rows[2], rows[3], 
            dDet_drows[0], dDet_drows[1], dDet_drows[2], dDet_drows[3]);

        double dL_dcc_unnorm_i = dL_dcc_unnorm[i];
        for (int j = 0; j < 4; j++) {
            dL_dhp[j][(i + 1) % 5] += dL_dcc_unnorm_i * dDet_drows[j].x;
            dL_dhp[j][(i + 2) % 5] += dL_dcc_unnorm_i * dDet_drows[j].y;
            dL_dhp[j][(i + 3) % 5] += dL_dcc_unnorm_i * dDet_drows[j].z;
            dL_dhp[j][(i + 4) % 5] += dL_dcc_unnorm_i * dDet_drows[j].w;
        }
    }

    /*
    Gradient of loss w.r.t. point poisitions and weights
    */
    double dL_dpositions[4][3] = {};    // initialize to zero;
    double dL_dweights[4] = {};    // initialize to zero;
    for (int i = 0; i < 4; i++) {
        int pt_id = tri_idx[idx][i];
        for (int j = 0; j < 3; j++) {
            dL_dpositions[i][j] += dL_dhp[i][j] * 2;
            dL_dpositions[i][j] += dL_dhp[i][4] * (-2 * positions[pt_id][j]);
        }
        dL_dweights[i] = dL_dhp[i][4];

        atomicAdd(&(grad_positions[pt_id][0]), dL_dpositions[i][0]);
        atomicAdd(&(grad_positions[pt_id][1]), dL_dpositions[i][1]);
        atomicAdd(&(grad_positions[pt_id][2]), dL_dpositions[i][2]);
        atomicAdd(&(grad_weights[pt_id]), dL_dweights[i]);
    }
}

void DIFFDT::cc_backward(
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
) {
    // launch kernel;
    int num_tri = tri_idx.size(0);
    int grid_size = (num_tri + BLOCK_SIZE - 1) / BLOCK_SIZE;
    ccKernelBackwardCUDA<<<grid_size, BLOCK_SIZE>>>(
        positions, 
        weights, 
        
        tri_idx,
        tri_cc_unnorm,
        grad_tri_cc,

        grad_positions,
        grad_weights
    );
    cudaDeviceSynchronize();
}