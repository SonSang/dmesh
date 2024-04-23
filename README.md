# DMesh: A Differentiable Representation for General Meshes

By computing existence probability for faces in a mesh, DMesh offers a way to represent a general triangular mesh in a differentiable manner. Please refer to our [arxiv](https://arxiv.org/abs/2404.13445), [full paper](https://www.cs.umd.edu/~shh1295/dmesh/full.pdf), and [project website](https://sonsang.github.io/dmesh-project) for more details.

![Teaser image](<static/teaser.png>)

## Installation

Please clone this repository recursively to include all submodules.

```bash
git clone https://github.com/SonSang/dmesh.git --recursive
```

### Dependencies

We use Python version 3.9, and recommend using Anaconda to manage the environment. 
After creating a new environment, please run following command to install the required python packages.

```bash
pip install -r requirements.txt
```

We also need additional external libraries to run DMesh. Please install them by following the instructions below.

#### Pytorch

Please install Pytorch that aligns with your NVIDIA GPU. Currently, our code requires NVIDIA GPU to run, because our main algorithm is written in CUDA. You can find instructions [here](https://pytorch.org/get-started/locally/). We tested with Pytorch version 1.13.1 and 2.2.1.


#### pytorch3d (0.7.6)

Please follow detailed instructions [here](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md). In short, you can install (the latest) pytorch3d by running the following command.

```bash
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

#### CGAL (5.6)

We use [CGAL](https://github.com/CGAL/cgal) to run the Weighted Delaunay Triangulation (WDT) algorithm, which forms the basis of our approach. If you cloned this repository recursively, you should already have the latest CGAL source code in the `external/cgal` directory. Please follow the instructions below to build and install CGAL.

```bash
cd external/cgal
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

You might need to install some additional dependencies to build CGAL. Please refer to the [official documentation](https://doc.cgal.org/latest/Manual/thirdparty.html) and install essential third party libraries, such as Boost, GMP, and MPFR, to build CGAL and CGAL-dependent code successfully. If you are using Ubuntu, you can install GMP and MPFR with following commands.

```bash
sudo apt-get install libgmp3-dev
sudo apt-get install libmpfr-dev
```

#### OneTBB (2021.11.0)

We use [OneTBB](https://github.com/oneapi-src/oneTBB) to (possibly) accelerate CGAL's WDT algorithm using multi-threading. Even though it is not used in the current implementation, we include it here for future work. If you cloned this repository recursively, you should already have the latest OneTBB source code in the `external/oneTBB` directory. Please follow the instructions below to build and install OneTBB.

```bash
cd external/oneTBB
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=../install -DTBB_TEST=OFF ..
cmake --build .
cmake --install .
```

You would be able to find OneTBB install files in `external/oneTBB/install` directory.

#### Nvdiffrast

We use [nvdiffrast](https://github.com/NVlabs/nvdiffrast) for differentiable rasterization. Please follow the instructions below to build and install nvdiffrast.

```bash
sudo apt-get install libglvnd0 libgl1 libglx0 libegl1 libgles2 libglvnd-dev libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev
cd external/nvdiffrast
pip install -e .
```

Please see the [official documentation](https://nvlabs.github.io/nvdiffrast/) if you encounter any issues during the installation.

#### DMeshRenderer

We implemented our own renderers, named [DMeshRenderer](https://github.com/SonSang/dmesh_renderer) for multi-view reconstruction. Before installing it, please install [GLM](https://github.com/g-truc/glm) library. If you use Ubuntu, you can install it by running the following command.

```bash
sudo apt-get install libglm-dev
```

Then, please follow the instructions below to build and install DMeshRenderer.

```bash
cd external/dmesh_renderer
pip install -e .
```

### Build CGAL-dependent code

Run following command to build CGAL-dependent code.

```bash
cd cgal_wrapper
cmake -DCMAKE_BUILD_TYPE=Release .
make
```

You would be able to find `libcgal_diffdt.a` file in `cgal_wrapper/` directory.

### Build DMesh

Finally, run following command to build DMesh.

```bash
pip install -e .
```

## Dataset

Now, it's ready to run downstream applications. For our reconstruction experiments, we mainly used 4 closed surface models from [Thingi10K](https://ten-thousand-models.appspot.com/) dataset, 4 open surface models from [DeepFashion3D](https://github.com/GAP-LAB-CUHK-SZ/deepFashion3D) dataset, and 3 mixed surface model from [Objaverse](https://objaverse.allenai.org/objaverse-1.0/) dataset and [Adobe Stock](https://stock.adobe.com/). For models from DeepFashion3D dataset, we used the ground truth mesh provided by [NeuralUDF](https://github.com/xxlong0/NeuralUDF) repository. Additionally, we used 3 models from [Stanford Dataset](https://graphics.stanford.edu/data/3Dscanrep/) for the first mesh conversion experiment.

Except for 2 mixed surface models (plant, raspberry) from Adobe Stock, you can download the dataset from [Google Drive](https://drive.google.com/drive/folders/1gCTai2NeaHzgGXLb9RFfbCyBQeiJvH9p?usp=sharing). Please place it under `dataset` folder. For reference, we provide links for the [plant](https://stock.adobe.com/3d-assets/dieffenbachia-camilla/317765746?prev_url=detail) and [raspberry](https://stock.adobe.com/3d-assets/raspberry-plant/379986186?prev_url=detail) model in Adobe Stock.

## Usage

Here we provide some examples to use DMesh. All of the examples use config files in `exp/config` folder. You can modify the config files to change the input/output paths, hyperparameters, etc. By default, all the results are stored in `exp_result`. If you want to run every experiment sequentially, plase use the following command.

```bash
bash run_all.sh
```

### Example 1: Mesh to DMesh

First, we convert a ground truth mesh to DMesh, by restoring the connectivity of the given mesh.

Run following command to convert Stanford Bunny model into DMesh.

```bash
python exp/1_mesh_to_dmesh.py --config=exp/config/exp_1/bunny.yaml
```

### Example 2: Point cloud reconstruction

Next, we reconstruct a 3D mesh from a point cloud using DMesh, by minimizing (expected) Chamfer Distance.

Run following command to reconstruct a Lucy model from a point cloud.

```bash
python exp/2_pc_recon.py --config=exp/config/exp_2/thingi32/252119.yaml
```

### Example 3: Multi-view image reconstruction

Finally, we reconstruct a 3D mesh from multi-view (diffuse, depth) images using DMesh, by minimizing the rendering loss.

Run following command to reconstruct a cloth model from multi-view images.

```bash
python exp/3_mv_recon.py --config=exp/config/exp_3/deepfashion3d/448.yaml
```

## Discussions and Future Work

As discussed in the paper, our approach is a quite versatile approach to represent triangular mesh. However, because there is no additional constraint, our method does not guarantee to generate manifold mesh. Therefore, the orientation is not aligned well in the current reconstruction results, with small geometric artifacts. Also, when it comes to multi-view reconstruction, we still have a lot of room for improvement, because the differentiable renderers are premature. In the near future, we are aiming at overcoming these issues.

## Citation

```bibtex
@misc{son2024dmesh,
      title={DMesh: A Differentiable Representation for General Meshes}, 
      author={Sanghyun Son and Matheus Gadelha and Yang Zhou and Zexiang Xu and Ming C. Lin and Yi Zhou},
      year={2024},
      eprint={2404.13445},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgement

As described above, we used [CGAL](https://github.com/CGAL/cgal) for implementing our core algorithm. For implementing multi-view reconstruction code, we brought implementations of [nvdiffrast](https://github.com/NVlabs/nvdiffrast), [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [Continuous Remeshing For Inverse Rendering](https://github.com/Profactor/continuous-remeshing). We appreciate these great works.