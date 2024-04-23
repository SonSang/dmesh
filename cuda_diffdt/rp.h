#ifndef CUDA_DIFFDT_REAL_POINT_H_INCLUDED
#define CDUA_DIFFDT_REAL_POINT_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace DIFFDT
{
    /*
    ================================================================
    In WDT, some points cannot have power cells because they have 
    relatively smaller weights than their neighboring points. Real
    points are the points that have their own power cells.
    ================================================================
    */
    typedef struct RealPointsInfo
    {
        int num_rp;                 // number of real points;
        int* rp_id;                 // id of real points;
        int* rp_cc_beg;             // begin index of real points in [rp_cc] and [rp_ccd];
        int* rp_cc_end;             // end index of real points in [rp_cc] and [rp_ccd];
        int* rp_cc_id;              // for a real point [i], [rp_cc[rp_beg[i]]] ~ [rp_cc[rp_end[i]]] 
                                    // are indices of circumcenters that include [i];
        float* rp_cc_d;             // for a real point [i], [rp_ccd[rp_beg[i]]] ~ [rp_ccd[rp_end[i]]] 
                                    // are distances from [i] to circumcenters that include [i];
    } RPI;
}


#endif