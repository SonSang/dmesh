#include <torch/extension.h>
#include "diffdt.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("delaunay_triangulation", &ComputeDT);
    m.def("compute_circumcenters", &ComputeCircumcentersCUDA);
    m.def("compute_circumcenters_backward", &ComputeCircumcentersBackwardCUDA);
    m.def("compute_sso", &ComputeSSOCUDA);
    m.def("compute_iwdt0", &ComputeIWDT0CUDA);
    m.def("compute_iwdt1", &ComputeIWDT1CUDA);
}