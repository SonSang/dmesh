# exp 1
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python exp/1_mesh_to_dmesh.py --config=exp/config/exp_1/bunny.yaml --no-log-time
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python exp/1_mesh_to_dmesh.py --config=exp/config/exp_1/dragon.yaml --no-log-time
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python exp/1_mesh_to_dmesh.py --config=exp/config/exp_1/happy.yaml --no-log-time 

# exp 2
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python exp/2_pc_recon.py --config=exp/config/exp_2/deepfashion3d/30.yaml --no-log-time
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python exp/2_pc_recon.py --config=exp/config/exp_2/deepfashion3d/164.yaml --no-log-time
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python exp/2_pc_recon.py --config=exp/config/exp_2/deepfashion3d/320.yaml --no-log-time
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python exp/2_pc_recon.py --config=exp/config/exp_2/deepfashion3d/448.yaml --no-log-time

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python exp/2_pc_recon.py --config=exp/config/exp_2/objaverse/bigvegas.yaml --no-log-time
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python exp/2_pc_recon.py --config=exp/config/exp_2/objaverse/plant.yaml --no-log-time
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python exp/2_pc_recon.py --config=exp/config/exp_2/objaverse/raspberry.yaml --no-log-time

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python exp/2_pc_recon.py --config=exp/config/exp_2/thingi32/64444.yaml --no-log-time
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python exp/2_pc_recon.py --config=exp/config/exp_2/thingi32/252119.yaml --no-log-time
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python exp/2_pc_recon.py --config=exp/config/exp_2/thingi32/313444.yaml --no-log-time
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python exp/2_pc_recon.py --config=exp/config/exp_2/thingi32/527631.yaml --no-log-time

# exp 3
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python exp/3_mv_recon.py --config=exp/config/exp_3/deepfashion3d/30.yaml --no-log-time
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python exp/3_mv_recon.py --config=exp/config/exp_3/deepfashion3d/164.yaml --no-log-time
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python exp/3_mv_recon.py --config=exp/config/exp_3/deepfashion3d/320.yaml --no-log-time
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python exp/3_mv_recon.py --config=exp/config/exp_3/deepfashion3d/448.yaml --no-log-time

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python exp/3_mv_recon.py --config=exp/config/exp_3/objaverse/bigvegas.yaml --no-log-time
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python exp/3_mv_recon.py --config=exp/config/exp_3/objaverse/plant.yaml --no-log-time
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python exp/3_mv_recon.py --config=exp/config/exp_3/objaverse/raspberry.yaml --no-log-time

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python exp/3_mv_recon.py --config=exp/config/exp_3/thingi32/64444.yaml --no-log-time
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python exp/3_mv_recon.py --config=exp/config/exp_3/thingi32/252119.yaml --no-log-time
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python exp/3_mv_recon.py --config=exp/config/exp_3/thingi32/313444.yaml --no-log-time
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python exp/3_mv_recon.py --config=exp/config/exp_3/thingi32/527631.yaml --no-log-time