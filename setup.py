from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="dmesh",
    packages=['diffdt'],
    ext_modules=[
        CUDAExtension(
            name="diffdt._C",
            sources=[
                "cuda_diffdt/cc.cu",
                "cuda_diffdt/sso.cu",
                "cuda_diffdt/iwdt0.cu",
                "cuda_diffdt/iwdt1.cu",
                "diffdt/diffdt.cu",
                "diffdt/ext.cpp"],
            extra_compile_args={"nvcc": [
                "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/"),
                "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "cgal_wrapper/"),
                "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda_diffdt/"),
            ]},
            library_dirs=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "cgal_wrapper/"),
                        os.path.join(os.path.dirname(os.path.abspath(__file__)), "external/oneTBB/install/lib/")],
            libraries=["cgal_wrapper", "gmp", "mpfr", "tbb", "tbbmalloc", "tbbmalloc_proxy"],
        )
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)