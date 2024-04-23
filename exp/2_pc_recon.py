import torch as th
import numpy as np
import argparse
import yaml
import time
import os
import trimesh

from typing import List
from exp.recon import BaseRecon, DEVICE, DOMAIN
from exp.utils.utils import triangle_area, sample_bary, run_knn, setup_logdir, import_mesh, sample_points_on_mesh, nd_chamfer_dist, point_tri_distance

PROB_THRESH = 1e-3

# number of k-nearest neighbors for (probabilistic) CD loss computation
# from ground truth sample points to our sample points;
MIN_GT_LOSS_K = 1
MAX_GT_LOSS_K = 100

class PCRecon(BaseRecon):

    def __init__(self, 
                 
                logdir: str,

                # target points (and normals);
                target_point_positions: th.Tensor,
                target_point_normals: th.Tensor,

                # init method;
                init_args: dict,

                # num sample points for loss computation;
                recon_num_sample_points: int,

                # weight of normal error in loss computation;
                recon_normal_loss_weight: float,

                # LR;
                lr: float,
                lr_schedule: str,
                update_bound: float,

                # steps;
                num_epochs: int,
                
                num_phase_0_steps: int,
                num_phase_0_real_warmup_steps: int,
                num_phase_0_real_freeze_steps: int,

                num_phase_1_steps: int,
                num_phase_1_real_warmup_steps: int,
                num_phase_1_real_freeze_steps: int,

                save_steps: int,
                
                # IWDT;
                iwdt_sigmoid_k: float = 1000.0,
                iwdt_knn_k: int = 8,
                iwdt_knn_interval: int = 10,

                # refinement;
                max_real: float = 3e-3,
                real_dmin_k: float = 100.0,
                refine_ud_thresh: float = 0.01,

                # regularizer;
                weight_regularizer_coef: float = 1e-5,
                quality_regularizer_coef: float = 1e-3,

                # real regularizer;
                max_real_regularizer_coef: float = 0.1,
                min_real_regularizer_coef: float = 0.01,
                real_regularizer_step: int = 100,

                # use weight;
                use_weight: bool = True,
                
                # epoch args;
                epoch_args: dict = {},):

        self.target_point_positions = target_point_positions
        self.target_point_normals = target_point_normals

        self.recon_num_sample_points = recon_num_sample_points
        self.recon_normal_loss_weight = recon_normal_loss_weight
        self.recon_gt_loss_k = MAX_GT_LOSS_K
        self.recon_our_loss_k = MAX_GT_LOSS_K

        self.init_args = init_args

        super().__init__(
            
            logdir, 
            
            lr, 
            lr_schedule, 
            update_bound,
            
            num_epochs,
            num_phase_0_steps,
            num_phase_0_real_warmup_steps,
            num_phase_0_real_freeze_steps,
            
            num_phase_1_steps,
            num_phase_1_real_warmup_steps,
            num_phase_1_real_freeze_steps,
            
            save_steps,
            
            iwdt_sigmoid_k, 
            iwdt_knn_k, 
            iwdt_knn_interval,
            
            max_real,
            real_dmin_k,
            refine_ud_thresh,
            
            weight_regularizer_coef,
            quality_regularizer_coef,

            max_real_regularizer_coef,
            min_real_regularizer_coef,
            real_regularizer_step,

            use_weight,
            epoch_args
        )

    def init_points(self):
        
        init_method = self.init_args.get("method", "sample")

        if init_method == "sample":

            # select sample points from target points;
            num_sample_points = int(float(self.init_args.get("num_sample_points", 1000)))
            if num_sample_points > len(self.target_point_positions):
                num_sample_points = len(self.target_point_positions)

            sample_point_indices = th.randperm(
                len(self.target_point_positions)
            )[:num_sample_points]

            sample_point_positions = self.target_point_positions[sample_point_indices]

            self.sample_points_init(sample_point_positions)

        elif init_method == "random":

            grid_res = self.init_args.get("grid_res", 10)

            self.random_points_init(grid_res)

        else:

            raise ValueError("Invalid init method.")

    def sample_points_soft_faces(self,
                    point_positions: th.Tensor,
                    point_reals: th.Tensor,
                    
                    faces: th.Tensor,
                    faces_iwdt_prob: th.Tensor,
                    faces_real_prob: th.Tensor,):

        '''
        Sample points from probabilistic faces.
        '''
        
        faces_prob = faces_iwdt_prob * faces_real_prob
            
        with th.no_grad():

            total_face_idx = []

            '''
            Stratified sampling.

            0: Sample from every face that has at least one point with high real.
            1: Sample from every face that has all of its points with high real.
            '''

            for strata in range(2):

                tmp_faces_real_prob = point_reals[faces.to(dtype=th.long)]
                tmp_faces_real_prob_max = th.max(tmp_faces_real_prob, dim=-1)[0]
                tmp_faces_real_prob_min = th.min(tmp_faces_real_prob, dim=-1)[0]

                if strata == 0:
                    tmp_faces_real_prob = tmp_faces_real_prob_max
                else:
                    tmp_faces_real_prob = tmp_faces_real_prob_min

                tmp_faces_prob = faces_iwdt_prob * tmp_faces_real_prob

                faces_validity = tmp_faces_prob > PROB_THRESH

                valid_faces = faces[faces_validity]
                valid_faces_idx = th.nonzero(faces_validity).squeeze(-1)
                valid_faces_area = triangle_area(point_positions, valid_faces.to(dtype=th.long))
                valid_faces_prob = tmp_faces_prob[faces_validity]

                # sample more points from faces with higher probability and larger area;
                valid_faces_multinomial_prob = valid_faces_area * valid_faces_prob

                total_faces_area = th.sum(valid_faces_area)
                assert th.all(valid_faces_area >= 0.0), "face area should be non-negative."
                assert total_faces_area > 0, "sum of face areas should be positive."

                num_sample_points = self.recon_num_sample_points
                num_sample_faces = num_sample_points

                # sample points based on face area;
                face_idx = th.multinomial(
                    valid_faces_multinomial_prob,
                    num_sample_faces // 2, 
                    replacement=True
                )
                face_idx = valid_faces_idx[face_idx]

                total_face_idx.append(face_idx)

            face_idx = th.cat(total_face_idx, dim=0)
            num_sample_points = len(face_idx)

            # sample barycentric coordinates;
            face_bary = sample_bary(num_sample_points, DEVICE)

        final_faces = faces[face_idx].to(dtype=th.long)
        final_faces_prob = faces_prob[face_idx]
        
        # compute sample positions;
        sample_point_positions = th.sum(
            point_positions[final_faces] * face_bary.unsqueeze(-1),
            dim=-2
        )
        sample_point_normals = th.nn.functional.normalize(
            th.cross(
                point_positions[final_faces[:, 1]] - point_positions[final_faces[:, 0]],
                point_positions[final_faces[:, 2]] - point_positions[final_faces[:, 0]],
                dim=-1
            ), dim=-1
        )

        sample_face_idx = face_idx
        sample_face_probs = final_faces_prob
        # assert th.all(sample_face_probs >= PROB_THRESH), "."

        return sample_point_positions, sample_point_normals, sample_face_idx, sample_face_probs
    
    def soft_chamfer_loss_from_gt(self,
                            our_sample_point_pos: th.Tensor,
                            our_sample_point_normals: th.Tensor,
                            our_sample_face_prob: th.Tensor,
                            our_sample_face_idx: th.Tensor,
                            gt_loss_k: int,
                            recon_normal_loss_weight: float):
        
        '''
        Compute probabilistic Chamfer distance from ground truth 
        sample points to our sample points.

        @ gt_loss_k: Number of k-nearest neighbors for loss computation.
        '''
        assert th.all(our_sample_face_prob >= 0.0), "probability should be non-negative."

        num_our_sample_points = len(our_sample_point_pos)

        curr_gt_loss_k = gt_loss_k
        if curr_gt_loss_k > num_our_sample_points:
            curr_gt_loss_k = num_our_sample_points

        gt_sample_point_pos = self.target_point_positions
        gt_sample_point_normals = self.target_point_normals

        gt_knn_idx, _ = run_knn(
            gt_sample_point_pos,
            our_sample_point_pos,
            curr_gt_loss_k
        )

        '''
        ==============================
        '''

        '''
        Point version.
        '''

        gt_knn_pos = our_sample_point_pos[gt_knn_idx]       # [# gt sample points, # gt loss k, 3]
        gt_knn_normals = our_sample_point_normals[gt_knn_idx]       # [# gt sample points, # gt loss k, 3]
        gt_knn_face_idx = our_sample_face_idx[gt_knn_idx]   # [# gt sample points, # gt loss k]

        # position distance;
        dist = th.norm(gt_knn_pos - gt_sample_point_pos.unsqueeze(1), dim=-1)  # [# gt sample points, # gt loss k]
        
        # normal distance;
        n_dist_0 = 1.0 - th.sum(gt_knn_normals * gt_sample_point_normals.unsqueeze(1), dim=-1)  # [# gt sample points, # gt loss k]
        n_dist_1 = 1.0 - th.sum(-gt_knn_normals * gt_sample_point_normals.unsqueeze(1), dim=-1)  # [# gt sample points, # gt loss k]
        n_dist = th.min(n_dist_0, n_dist_1) # [# gt sample points, # gt loss k]

        # reorder [gt_knn_idx] based on (dist + n_dist),
        # because [gt_knn_idx] is sorted based on dist only;
        
        f_dist = dist + (n_dist * recon_normal_loss_weight)

        # F_EPS = 1e-2
        # f_dist = dist.clone()
        # f_dist[dist < F_EPS] = (dist[dist < F_EPS] * (1.0 - recon_normal_loss_weight)) + (n_dist[dist < F_EPS] * recon_normal_loss_weight * F_EPS)

        f_dist, f_dist_idx = th.sort(f_dist, dim=1, descending=False)
        gt_knn_idx = th.gather(gt_knn_idx, 1, f_dist_idx)

        gt_knn_pos = our_sample_point_pos[gt_knn_idx]       # [# gt sample points, # gt loss k, 3]
        gt_knn_normals = our_sample_point_normals[gt_knn_idx]       # [# gt sample points, # gt loss k, 3]
        gt_knn_face_idx = our_sample_face_idx[gt_knn_idx]   # [# gt sample points, # gt loss k]
        prob_mat = our_sample_face_prob[gt_knn_idx]         # [# gt sample points, # gt loss k]

        '''
        Sorting: If a sample point from gt mesh finds a near point from a certain face,
        we do not consider another point from the same face in computing the loss.
        '''

        # =========

        sorted_indices = th.argsort(gt_knn_face_idx, dim=1, stable=True)

        # Step 2: Rearrange A using sorted indices
        sorted_A = th.gather(prob_mat, 1, sorted_indices)

        # Step 3: Identify duplicates in sorted B
        sorted_B = th.gather(gt_knn_face_idx, 1, sorted_indices)
        duplicate_mask = sorted_B[:, 1:] == sorted_B[:, :-1]
        # Pad the mask to match the shape of A and B
        padded_mask = th.cat([th.zeros(duplicate_mask.shape[0], 1, dtype=th.bool, device=DEVICE), duplicate_mask], dim=1)

        # Step 4: Revert A to the original order, applying the duplicate mask
        # First, set duplicates in sorted_A to 0
        sorted_A[padded_mask] = 0.0
        
        # Then, invert the sorted indices to get the original order
        inverse_indices = th.argsort(sorted_indices, dim=1)
        original_order_A = th.gather(sorted_A, 1, inverse_indices)

        prob_mat = original_order_A
        
        # =========

        n_prob_mat = 1.0 - prob_mat
        n_prob_mat_prod = th.cumprod(n_prob_mat, dim=-1)

        prob_mat[:, 1:] = prob_mat[:, 1:].clone() * n_prob_mat_prod[:, :-1]

        # expected_dist = th.sum(prob_mat * dist, dim=-1)         # [# gt sample points,]
        # expected_n_dist = th.sum(prob_mat * n_dist, dim=-1)     # [# gt sample points,]

        loss = th.sum(prob_mat * f_dist, dim=-1)         # [# gt sample points,]
        loss = loss.mean()
        
        # loss = expected_dist.mean() + (expected_n_dist.mean() * recon_normal_loss_weight)

        # if the last column of [n_prob_mat_prod] is too small, decrease gt_loss_k;
        # if the last column of [n_prob_mat_prod] is too large, increase gt_loss_k;
        if gt_loss_k < num_our_sample_points:
        
            thresh = 1e-4
            if th.all(n_prob_mat_prod[:, -1] < thresh):
                gt_loss_k = gt_loss_k - 1
            elif th.any(n_prob_mat_prod[:, -1] > thresh):
                gt_loss_k = gt_loss_k + 1
            
            gt_loss_k = min(max(gt_loss_k, MIN_GT_LOSS_K), MAX_GT_LOSS_K)
        
        return loss, gt_loss_k
        
    def soft_chamfer_loss_from_ours(self,
                                our_sample_point_pos: th.Tensor,
                                our_sample_point_normals: th.Tensor,
                                our_sample_face_prob: th.Tensor,
                                our_loss_k: int,
                                recon_normal_loss_weight: float):
        
        assert th.all(our_sample_face_prob >= 0.0), "probability should be non-negative."

        gt_sample_point_pos = self.target_point_positions
        gt_sample_point_normals = self.target_point_normals

        num_gt_sample_points = len(gt_sample_point_pos)

        curr_our_loss_k = our_loss_k
        if curr_our_loss_k > num_gt_sample_points:
            curr_our_loss_k = num_gt_sample_points

        with th.no_grad():
            our_knn_idx = run_knn(
                our_sample_point_pos,
                gt_sample_point_pos,
                curr_our_loss_k
            )[0]

        our_knn_pos = gt_sample_point_pos[our_knn_idx]     # [# our sample points, # our loss k, 3]
        our_knn_normal = gt_sample_point_normals[our_knn_idx]     # [# our sample points, # our loss k, 3]

        dist = th.norm(our_knn_pos - our_sample_point_pos.unsqueeze(1), dim=-1)  # [# our sample points, # our loss k]
        
        n_dist_0 = 1.0 - th.sum(our_knn_normal * our_sample_point_normals.unsqueeze(1), dim=-1)  # [# our sample points, # our loss k]
        n_dist_1 = 1.0 - th.sum(-our_knn_normal * our_sample_point_normals.unsqueeze(1), dim=-1)  # [# our sample points, # our loss k]
        n_dist = th.min(n_dist_0, n_dist_1) # [# our sample points, # our loss k]
        
        f_dist = dist + (n_dist * recon_normal_loss_weight)
        f_dist, f_dist_idx = th.min(f_dist, dim=-1)

        loss = f_dist * our_sample_face_prob                    # [# our sample points,]
        loss = loss.mean()

        # adjust next [our_loss_k];
        if our_loss_k < num_gt_sample_points:

            sorted_f_dist_idx = th.sort(f_dist_idx)[0]
            percentile_id = int(len(sorted_f_dist_idx) * 0.9)    # 90th percentile
            if percentile_id >= len(sorted_f_dist_idx):
                percentile_id = len(sorted_f_dist_idx) - 1
            max_f_dist_idx = sorted_f_dist_idx[percentile_id]    # 90th percentile

            # if we can find the target nearest point for 90% of our sample points
            # using only half of the current [our_loss_k], we decrease [our_loss_k];
            if max_f_dist_idx < (our_loss_k - 1) * 0.5:
                our_loss_k -= 1
            else:
                our_loss_k += 1
            
            our_loss_k = min(max(our_loss_k, MIN_GT_LOSS_K), MAX_GT_LOSS_K)
        
        return loss, our_loss_k

    def compute_recon_loss_0(self,
                        epoch: int, step: int, num_steps: int, 
                        point_positions: th.Tensor, 
                        point_weights: th.Tensor,
                        point_reals: th.Tensor,
                        
                        soft_faces: th.Tensor, 
                        soft_faces_iwdt_prob: th.Tensor, 
                        soft_faces_real_prob: th.Tensor,

                        hard_faces: th.Tensor):

        soft_loss, soft_loss_log = self.compute_soft_recon_loss(
            point_positions, 
            point_weights,
            point_reals,

            soft_faces, 
            soft_faces_iwdt_prob, 
            soft_faces_real_prob,
            
            self.recon_normal_loss_weight
        )

        loss = soft_loss
        loss_log = { **soft_loss_log }
        loss_log['recon_normal_loss_weight'] = self.recon_normal_loss_weight

        return loss, loss_log
        
    def compute_recon_loss_1(self, 
                            epoch: int, step: int, num_steps: int, 
                            
                            point_positions: th.Tensor, 
                            point_weights: th.Tensor,
                            point_reals: th.Tensor,

                            faces: th.Tensor, 
                            faces_iwdt_prob: th.Tensor, 
                            faces_real_prob: th.Tensor,
                            tets: th.Tensor):

        soft_loss, soft_loss_log = self.compute_soft_recon_loss(
            point_positions, 
            point_weights,
            point_reals,

            faces, 
            faces_iwdt_prob, 
            faces_real_prob,
            
            self.recon_normal_loss_weight
        )

        loss = soft_loss
        loss_log = { **soft_loss_log }
        loss_log['recon_normal_loss_weight'] = self.recon_normal_loss_weight

        return loss, loss_log
    
    def compute_soft_recon_loss(self,
                        point_positions: th.Tensor, 
                        point_weights: th.Tensor,
                        point_reals: th.Tensor,
                        
                        faces: th.Tensor, 
                        faces_iwdt_prob: th.Tensor, 
                        faces_real_prob: th.Tensor,
                        
                        recon_normal_loss_weight: float):
        
        '''
        Sample points from current mesh.
        '''
        
        gt_loss_k = self.recon_gt_loss_k
        our_loss_k = self.recon_our_loss_k

        our_sample_point_positions, our_sample_point_normals, our_sample_face_idx, our_sample_face_probs = \
            self.sample_points_soft_faces(
                point_positions,
                point_reals,

                faces,
                faces_iwdt_prob,
                faces_real_prob
            )
        
        '''
        Compute loss based on Chamfer distance.
        '''

        gt_loss, n_gt_loss_k = self.soft_chamfer_loss_from_gt(
            our_sample_point_positions,
            our_sample_point_normals,
            our_sample_face_probs,
            our_sample_face_idx,
            gt_loss_k,
            recon_normal_loss_weight
        )
        self.recon_gt_loss_k = n_gt_loss_k
        
        our_loss, n_our_loss_k = self.soft_chamfer_loss_from_ours(
            our_sample_point_positions,
            our_sample_point_normals,
            our_sample_face_probs,
            our_loss_k,
            recon_normal_loss_weight
        )
        self.recon_our_loss_k = n_our_loss_k

        loss = gt_loss + our_loss

        with th.no_grad():
            log = {
                "soft_gt_loss": gt_loss.item(),
                "soft_our_loss": our_loss.item(),
                "soft_loss": loss.item(),
                "soft_gt_loss_k": gt_loss_k,
                "soft_our_loss_k": our_loss_k
            }

        return loss, log

    def compute_eval_loss(self, positions: th.Tensor, faces: th.Tensor):
        '''
        Compute evaluation loss.
        '''

        with th.no_grad():

            # sample points on our mesh;
            our_sample_positions, _ = \
                sample_points_on_mesh(
                    positions,
                    faces.to(dtype=th.long),
                    len(self.target_point_positions),
                    DEVICE
                )

            # compute loss;
            loss_gt, loss_ours = nd_chamfer_dist(self.target_point_positions, our_sample_positions)
            
        return loss_gt + loss_ours

    def optimize_epoch_start(self, epoch: int):

        # reset gt_loss_k;

        super().optimize_epoch_start(epoch)

        self.recon_gt_loss_k = MAX_GT_LOSS_K

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="exp/config/exp_2/objaverse/plant.yaml")
    # parser.add_argument("--init", type=str, choices=["sample", "random"], default="sample")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-log-time", action='store_true')
    args = parser.parse_args()

    # load settings from yaml file;
    with open(args.config, "r") as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
    logdir = settings['log_dir']
    if not args.no_log_time:
        logdir = logdir + time.strftime("/%Y_%m_%d_%H_%M_%S")
    logdir = setup_logdir(logdir)

    DEVICE = settings['device']

    # settings['args']['init_args']['method'] = args.init
    settings['args']['seed'] = args.seed

    # save settings;
    with open(os.path.join(logdir, "config.yaml"), "w") as f:
        yaml.dump(settings, f)
    th.random.manual_seed(args.seed)

    '''
    Ground truth mesh
    '''
    mesh_path = settings['mesh']
    verts, faces = import_mesh(mesh_path, DEVICE, scale=DOMAIN)
    
    print("===== Ground truth mesh =====")
    print("Number of vertices: ", verts.shape[0])
    print("Number of faces: ", faces.shape[0])
    print("=============================")

    # save gt mesh;
    mesh = trimesh.base.Trimesh(vertices=verts.cpu().numpy(), faces=faces.cpu().numpy())
    mesh.export(os.path.join(logdir, "gt_mesh.obj"))

    '''
    Arguments
    '''
    # LR;
    lr = float(settings['args']['lr'])
    lr_schedule = settings['args']['lr_schedule']
    update_bound = float(settings['args']['update_bound'])
    assert lr_schedule in ["constant", "linear", "exp"], \
        "lr_schedule should be either constant, linear, or exp."

    # Regularizer;
    weight_regularizer_coef = float(settings['args']['weight_regularizer_coef'])
    quality_regularizer_coef = float(settings['args']['quality_regularizer_coef'])
    
    # Real regularizer;
    max_real_regularizer_coef = float(settings['args']['max_real_regularizer_coef'])
    min_real_regularizer_coef = float(settings['args']['min_real_regularizer_coef'])
    real_regularizer_step = int(float(settings['args']['real_regularizer_step']))

    # Target points and normals;
    num_target_sample_points = int(float(settings['args']['num_target_sample_points']))
    num_recon_sample_points = int(float(settings['args']['num_recon_sample_points']))
    recon_normal_loss_weight = float(settings['args']['recon_normal_loss_weight'])

    '''
    Sample points from gt mesh
    '''
    target_sample_point_positions, target_sample_point_normals = \
        sample_points_on_mesh(
            verts,
            faces,
            num_target_sample_points,
            DEVICE
        )
    # save gt samples;
    th.save(target_sample_point_positions, os.path.join(logdir, "gt_samples.pth"))
    th.save(target_sample_point_normals, os.path.join(logdir, "gt_samples_normal.pth"))

    # Init method;
    init_args = settings['args']['init_args']

    # optimization params;
    num_epochs = settings['args']['num_epochs']
    num_phase_0_steps = settings['args']['num_phase_0_steps']
    num_phase_1_steps = settings['args']['num_phase_1_steps']
    num_phase_0_real_warmup_steps = settings['args']['num_phase_0_real_warmup_steps']
    num_phase_1_real_warmup_steps = settings['args']['num_phase_1_real_warmup_steps']
    num_phase_0_real_freeze_steps = settings['args']['num_phase_0_real_freeze_steps']
    num_phase_1_real_freeze_steps = settings['args']['num_phase_1_real_freeze_steps']
    save_steps = settings['args']['save_steps']

    # IWDT;
    iwdt_sigmoid_k = float(settings['args']['iwdt_sigmoid_k'])
    max_real = float(settings['args']['max_real'])
    refine_ud_thresh = float(settings['args']['refine_ud_thresh'])

    # use weight;
    use_weight = settings['args']['use_weight']

    # epoch args;
    epoch_args = settings['args']['epoch_args']
    
    optimizer = PCRecon(
        logdir,
        target_sample_point_positions,
        target_sample_point_normals,
        init_args,
        num_recon_sample_points,
        recon_normal_loss_weight,
        
        lr,
        lr_schedule,
        update_bound,
        
        num_epochs,
        
        num_phase_0_steps,
        num_phase_0_real_warmup_steps,
        num_phase_0_real_freeze_steps,
        
        num_phase_1_steps,
        num_phase_1_real_warmup_steps,
        num_phase_1_real_freeze_steps,
        
        save_steps,
        
        iwdt_sigmoid_k=iwdt_sigmoid_k,
        max_real=max_real,
        refine_ud_thresh=refine_ud_thresh,
        
        weight_regularizer_coef=weight_regularizer_coef,
        quality_regularizer_coef=quality_regularizer_coef,
        
        max_real_regularizer_coef=max_real_regularizer_coef,
        min_real_regularizer_coef=min_real_regularizer_coef,
        real_regularizer_step=real_regularizer_step,
        
        use_weight=use_weight,
        epoch_args=epoch_args
    )

    optimizer.optimize()