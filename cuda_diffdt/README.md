This directory contains the CUDA implementation for differentiable Weighted Delaunay Triangulation (WDT) algorithm, which is central to DMesh.

Before moving on, we note that we use the following external code.

* cuda_math.h: This source code was copied from NVIDIA's [cuda-samples](https://github.com/NVIDIA/cuda-samples.git) repository. Specifically, this file corresponds to [this file](https://github.com/NVIDIA/cuda-samples/blob/master/Common/helper_math.h).

Our own implementation is in the following files.

* cc.h, cc.cu: Compute the circumcenters of given tetrahedra.
* iwdt0.h, iwdt0.cu: Compute the delta value for faces that exist in WDT.
* iwdt1.h, iwdt1.cu: Compute the delta value for faces that do not exist in WDT.
* rp.h: Define struct for real points.
* sso.h, sso.cu: "Select Simplex One", find one-simplexes that are likely included in WDT.