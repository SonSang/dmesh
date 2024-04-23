This directory contains the Python implementation & wrapper for CUDA implementation of differentiable Weighted Delaunay Triangulation (WDT) algorithm, which is central to DMesh. Please see cuda_diffdt directory for the CUDA implementation.

* cgalwdt.py: Python wrapper for CGAL's WDT algorithm.
* iwdt0.py: Python wrapper for iwdt0 function. Contains python implementation for comparison.
* iwdt1.py: Python wrapper for iwdt1 function. Contains python implementation for comparison.
* pd.py: Python code to pre-process Power Diagram information to call CUDA functions. 
* sso.py: Python wrapper for sso function. Contains python implementation for comparison.
* torch_func.py: Python implementation for circumcenter computation.