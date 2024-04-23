import torch as th
import os
import time
from tqdm import tqdm

from diffdt.wp import WPoints
from diffdt.pd import PDStruct
from diffdt.sso import PseudoSSO
from diffdt import DiffDT

from exp.utils.dmesh import DMesh
from exp.utils.nn import pd_knn_faces
from exp.utils.utils import *

from torch.utils.tensorboard import SummaryWriter

DEVICE = 'cuda:0'
SIGMOID_MAX = 10.0          # maximum abs value for sigmoid;
MIN_LR = 1e-6               # minimum learning rate;
DOMAIN = 1.0                # domain for optimization;

class BaseRecon:

    def __init__(self, 

                logdir: str,

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
                max_real_regularizer_coef: float = 3.0,
                min_real_regularizer_coef: float = 0.3,
                real_regularizer_step: int = 100,
                
                # use weight;
                use_weight: bool = True,
                
                # epoch args;
                epoch_args: dict = {},):
        
        '''
        Points that we optimize.
        '''
        self.point_positions = None
        self.point_weights = None
        self.point_reals = None

        '''
        Logdir
        '''
        self.logdir = logdir
        self.writer = SummaryWriter(logdir)

        '''
        LR
        '''
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.update_bound = update_bound

        '''
        Steps
        '''
        self.num_epochs = num_epochs
        self.num_phase_0_steps = num_phase_0_steps
        self.num_phase_1_steps = num_phase_1_steps
        
        # warm up step: after these number of steps, fix the real values
        # that are beyond the threshold;
        self.num_phase_0_real_warmup_steps = num_phase_0_real_warmup_steps
        self.num_phase_1_real_warmup_steps = num_phase_1_real_warmup_steps
        # freeze step: at every [num_phase_0_real_freeze_steps] steps, we freeze
        # the real values that are beyond the threshold;
        self.num_phase_0_real_freeze_steps = num_phase_0_real_freeze_steps
        self.num_phase_1_real_freeze_steps = num_phase_1_real_freeze_steps

        self.save_steps = save_steps

        '''
        IWDT
        '''
        self.iwdt_sigmoid_k = iwdt_sigmoid_k
        self.iwdt_knn_k = iwdt_knn_k
        self.iwdt_knn_interval = iwdt_knn_interval

        '''
        Reals
        '''
        # real range = [0, max_real]
        self.max_real = max_real
        self.real_dmin_k = real_dmin_k                  # k for differentiable min;
        self.refine_ud_thresh = refine_ud_thresh
        
        '''
        Regularizer
        '''
        self.weight_regularizer_coef = weight_regularizer_coef
        self.quality_regualizer_coef = quality_regularizer_coef


        '''
        Real regularizer
        '''
        self.max_real_regularizer_coef = max_real_regularizer_coef
        self.min_real_regularizer_coef = min_real_regularizer_coef
        self.real_regularizer_step = real_regularizer_step

        '''
        Use weight
        '''
        self.use_weight = use_weight

        '''
        Args during optim
        '''
        self.default_args = {
            "lr": lr,
            "lr_schedule": lr_schedule,
            "update_bound": update_bound,

            "num_phase_0_steps": num_phase_0_steps,
            "num_phase_0_real_warmup_steps": num_phase_0_real_warmup_steps,
            "num_phase_0_real_freeze_steps": num_phase_0_real_freeze_steps,

            "num_phase_1_steps": num_phase_1_steps,
            "num_phase_1_real_warmup_steps": num_phase_1_real_warmup_steps,
            "num_phase_1_real_freeze_steps": num_phase_1_real_freeze_steps,

            "save_steps": save_steps,

            "iwdt_sigmoid_k": iwdt_sigmoid_k,
            "iwdt_knn_k": iwdt_knn_k,
            "iwdt_knn_interval": iwdt_knn_interval,

            "max_real": max_real,
            "real_dmin_k": real_dmin_k,
            "refine_ud_thresh": refine_ud_thresh,

            "weight_regularizer_coef": weight_regularizer_coef,
            "quality_regularizer_coef": quality_regularizer_coef,

            "max_real_regularizer_coef": max_real_regularizer_coef,
            "min_real_regularizer_coef": min_real_regularizer_coef,
            "real_regularizer_step": real_regularizer_step,
        }

        self.epoch_args = epoch_args

        self.global_optim_start_time = time.time()

        self.remove_invisible = False

    '''
    Initialization and refinement
    '''
    def init_points(self):

        raise NotImplementedError()

    def random_points_init(self, grid_res: int):
        '''
        Random initialization with points on grid.
        '''

        positions, _, _ = grid_points(grid_res, -DOMAIN * 1.25, DOMAIN * 1.25, DEVICE)
        weights = th.ones_like(positions[:, 0])
        reals = th.ones_like(positions[:, 0]) * self.max_real

        self.point_positions = positions
        self.point_weights = weights
        self.point_reals = reals

    def sample_points_init(self, sample_points: th.Tensor):
        '''
        Initialization with sample points on target geometry.
        '''

        num_sample_points = sample_points.shape[0]

        # alpha shape initialization;

        seed_points = sample_points.to(dtype=th.float32, device=DEVICE)
        seed_weights = th.ones((num_sample_points,), dtype=th.float32, device=DEVICE)
        seed_wp = WPoints(seed_points, seed_weights)

        # compute alpha shape;

        wdt_result = DiffDT.CGAL_WDT(
            seed_wp,
            True,
            False,
        )

        pd = PDStruct.forward(seed_wp, wdt_result.tets_point_id)

        add_points = pd.pd_vertices.positions

        # remove additional points outside domain;
        add_points_valid_0 = th.logical_and(add_points[:, 0] >= -DOMAIN, add_points[:, 0] <= DOMAIN)
        add_points_valid_1 = th.logical_and(add_points[:, 1] >= -DOMAIN, add_points[:, 1] <= DOMAIN)
        add_points_valid_2 = th.logical_and(add_points[:, 2] >= -DOMAIN, add_points[:, 2] <= DOMAIN)
        add_points_valid = th.logical_and(th.logical_and(add_points_valid_0, add_points_valid_1), add_points_valid_2)
        add_points = add_points[add_points_valid]

        # merge points;
        point_positions = th.cat([seed_points, add_points], dim=0)
        point_weights = th.ones((point_positions.shape[0],), dtype=th.float32, device=DEVICE)
        point_uds = th.zeros((point_positions.shape[0],), dtype=th.float32, device=DEVICE)  # unsigned distance;
        
        # initialize uds by computing distance to gt sample points;
        point_uds = run_knn(point_positions, sample_points, 1)[1]
        point_uds = point_uds.squeeze(-1)

        if self.refine_ud_thresh > 0:
            point_reals = (point_uds < self.refine_ud_thresh).to(dtype=th.float32) * self.max_real
        else:
            point_reals = th.ones_like(point_uds, dtype=th.float32) * self.max_real

        self.point_positions = point_positions
        self.point_weights = point_weights
        self.point_reals = point_reals
        
    def load_points_init(self, path: str):
        dmesh = DMesh.load(path, th.float32, DEVICE)
        self.point_positions = dmesh.point_positions
        self.point_weights = dmesh.point_weights
        self.point_reals = dmesh.point_real * self.max_real

    def refine_points(self, num_seeds):

        points_positions = self.point_positions.clone()
        points_weights = self.point_weights.clone()
        points_reals = self.point_reals.clone() / self.max_real

        dmesh = DMesh(points_positions, points_weights, points_reals)
            
        _, real_faces, _ = dmesh.get_faces(self.remove_invisible)

        # sample on current mesh;
        num_seed_points = num_seeds
        num_gt_points = 1_000_000
        sample_points, _ = \
            sample_points_on_mesh(
                points_positions,
                real_faces.to(dtype=th.long),
                num_gt_points,      # large number of points, as we use them for distance computation;
                DEVICE
            )
        
        seed_idx = th.randperm(num_gt_points)[:num_seed_points]
        seed_points = sample_points[seed_idx]
        seed_weights = th.ones((num_seed_points,), dtype=th.float32, device=DEVICE)
        seed_wp = WPoints(seed_points, seed_weights)

        # compute alpha shape;
        wdt_result = DiffDT.CGAL_WDT(
            seed_wp,
            True,
            False,
        )

        pd = PDStruct.forward(seed_wp, wdt_result.tets_point_id)

        add_points = pd.pd_vertices.positions

        # remove additional points outside domain;
        add_points_valid_0 = th.logical_and(add_points[:, 0] >= -DOMAIN, add_points[:, 0] <= DOMAIN)
        add_points_valid_1 = th.logical_and(add_points[:, 1] >= -DOMAIN, add_points[:, 1] <= DOMAIN)
        add_points_valid_2 = th.logical_and(add_points[:, 2] >= -DOMAIN, add_points[:, 2] <= DOMAIN)
        add_points_valid = th.logical_and(th.logical_and(add_points_valid_0, add_points_valid_1), add_points_valid_2)
        add_points = add_points[add_points_valid]

        # merge points;
        point_positions = th.cat([seed_points, add_points], dim=0)
        point_weights = th.ones((point_positions.shape[0],), dtype=th.float32, device=DEVICE)
        point_uds = th.zeros((point_positions.shape[0],), dtype=th.float32, device=DEVICE)  # unsigned distance;
        
        # initialize uds by computing distance to gt sample points;
        point_uds = run_knn(point_positions, sample_points, 1)[1]
        point_uds = point_uds.squeeze(-1)

        if self.refine_ud_thresh > 0:
            point_reals = (point_uds < self.refine_ud_thresh).to(dtype=th.float32) * self.max_real
        else:
            point_reals = th.ones_like(point_uds, dtype=th.float32) * self.max_real
        
        return point_positions, point_weights, point_reals
        
    def refresh_optimizer(self):
        point_positions = self.point_positions.clone()
        point_weights = self.point_weights.clone()
        point_reals = self.point_reals.clone()
        
        point_positions.requires_grad = True
        point_weights.requires_grad = True
        point_reals.requires_grad = True

        optimizer = th.optim.Adam([point_positions, point_weights, point_reals], lr=self.lr)
    
        return optimizer, point_positions, point_weights, point_reals

    '''
    Saving
    '''
    def save(self, 
            save_dir: str,
            points_positions: th.Tensor,
            points_weights: th.Tensor,
            points_reals: th.Tensor,
            time: float):
        
        with th.no_grad():
            save_path = save_dir
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            save_data = DMesh(points_positions, points_weights, points_reals)
            save_data.save(os.path.join(save_path, "points.pth"))

            save_mesh = save_data.get_mesh(self.remove_invisible)
            save_mesh.export(os.path.join(save_path, "mesh.obj"))

            time_path = os.path.join(save_path, "time_sec.txt")
            with open(time_path, "w") as f:
                f.write("{}".format(time))        

    def save_step(self, 
            epoch: int,
            phase: int,
            step: int, 
            points_positions: th.Tensor,
            points_weights: th.Tensor,
            points_reals: th.Tensor,
            time: float):

        save_path = os.path.join(self.writer.log_dir, "save/epoch_{}/phase_{}/step_{}".format(epoch, phase, step))
        self.save(save_path, points_positions, points_weights, points_reals, time)
    
    '''
    Losses
    '''
    def compute_recon_loss_0(self, epoch: int, step: int, num_steps: int, point_positions: th.Tensor, point_weights: th.Tensor, point_reals: th.Tensor,
                            soft_faces: th.Tensor, soft_faces_iwdt_prob: th.Tensor, soft_faces_real_prob: th.Tensor,
                            hard_faces: th.Tensor):
        '''
        Compute reconstruction loss (phase 0).

        @ soft faces: Probabilistic faces, with existence probability.
        @ hard faces: Faces that exist in the current (non-differentiable) mesh.
        '''
        raise NotImplementedError()

    def compute_recon_loss_1(self, epoch: int, step: int, num_steps: int, point_positions: th.Tensor, point_weights: th.Tensor, point_reals: th.Tensor, faces: th.Tensor, faces_iwdt_prob: th.Tensor, faces_real_prob: th.Tensor, tets: th.Tensor):
        '''
        Compute reconstruction loss (phase 1).
        '''
        raise NotImplementedError()

    def compute_weight_regularizer(self, pd: PDStruct):

        pd_edge_vertex_0 = pd.pd_edges.vertex_id[:, 0]
        pd_edge_vertex_1 = pd.pd_edges.vertex_id[:, 1]
        finite_edge = (pd_edge_vertex_1 != -1)

        finite_pd_edge_vertex_0 = pd_edge_vertex_0[finite_edge]
        finite_pd_edge_vertex_1 = pd_edge_vertex_1[finite_edge]

        finite_pd_edge_vertex_0_pos = pd.pd_vertices.positions[finite_pd_edge_vertex_0.to(dtype=th.long)]
        finite_pd_edge_vertex_1_pos = pd.pd_vertices.positions[finite_pd_edge_vertex_1.to(dtype=th.long)]
        finite_pd_edge_len = th.norm(finite_pd_edge_vertex_0_pos - finite_pd_edge_vertex_1_pos, dim=-1)
        finite_pd_edge_len = th.clamp(finite_pd_edge_len, max=1.0)

        coef = self.weight_regularizer_coef if self.use_weight else 0.0
        
        return th.sum(finite_pd_edge_len) * coef

    def compute_avg_real_regularizer(self, faces: th.Tensor, faces_prob: th.Tensor, p_r: th.Tensor, coef: float):
        '''
        This regularizer encourages the real values of the faces to be similar.
        '''
        # make real values of points in the same faces similar;
        faces_reals = p_r[faces.to(dtype=th.long)]
        avg_faces_real = th.mean(faces_reals, dim=-1, keepdim=True).detach()
        faces_reals_diff = avg_faces_real - faces_reals     # [F, 3]
        faces_reals_var = (faces_reals_diff.abs()).mean(dim=-1)
        faces_reals_var = faces_reals_var * faces_prob.detach()

        faces_reals_var = (faces_reals_var.sum() / faces_prob.detach().sum())

        reg = faces_reals_var.sum() * coef

        return reg
    
    def compute_max_real_regularizer(self, faces: th.Tensor, faces_prob: th.Tensor, p_r: th.Tensor, coef: float):
        '''
        It encourages the real values of points connected to the points
        that have high real values to have similar, high real values.
        '''

        faces_reals = p_r[faces.to(dtype=th.long)]
        
        # make real values of points connected to the points that have high real values similar;
        max_faces_real = th.max(faces_reals, dim=-1)[0]
        
        h_faces = faces[(max_faces_real > 0.8)]     # faces connected to high real points;
        h_faces_prob = faces_prob[(max_faces_real > 0.8)]

        h_faces_reals = p_r[h_faces.to(dtype=th.long)]
        h_faces_reals_diff = 1.0 - h_faces_reals
        h_faces_reals_diff = h_faces_reals_diff.mean(dim=-1)
        h_faces_reals_diff = h_faces_reals_diff * h_faces_prob.detach()
        mean_h_faces_reals_diff = h_faces_reals_diff.sum() / faces_prob.detach().sum()

        # final loss;
        reg = mean_h_faces_reals_diff * coef

        return reg

    def compute_quality_regularizer(self, point_positions: th.Tensor, faces: th.Tensor, faces_prob: th.Tensor):
        '''
        This regularizer improves the aspect ratio of faces.
        '''

        faces_aspect_ratio = triangle_aspect_ratio(point_positions, faces)
        faces_aspect_ratio = th.clamp(faces_aspect_ratio, min=1.0, max=20.0)
        
        # consider length of the longest edge;
        faces_edge_len_0 = th.norm(point_positions[faces[:, 1]] - point_positions[faces[:, 0]], dim=-1)
        faces_edge_len_1 = th.norm(point_positions[faces[:, 2]] - point_positions[faces[:, 1]], dim=-1)
        faces_edge_len_2 = th.norm(point_positions[faces[:, 0]] - point_positions[faces[:, 2]], dim=-1)
        faces_edge_len = th.stack([faces_edge_len_0, faces_edge_len_1, faces_edge_len_2], dim=-1)
        faces_max_edge_len = th.max(faces_edge_len, dim=-1)[0]

        faces_aspect_ratio = faces_aspect_ratio * faces_max_edge_len * faces_prob
        reg = (faces_aspect_ratio.sum() / (faces_prob.sum() + 1e-6).detach())
        reg = reg * self.quality_regualizer_coef
        
        return reg

    def compute_eval_loss(self, positions: th.Tensor, faces: th.Tensor):
        '''
        Compute evaluation loss.
        '''
        raise NotImplementedError()

    '''
    Epoch args
    '''
    def find_epoch_args(self, epoch: int):

        '''
        Find args for current epoch.
        '''

        t_epoch_args = {}
        found = False
        if self.epoch_args is None:
            return t_epoch_args, found
        
        for epoch_name in self.epoch_args.keys():
            id = epoch_name.split("_")[-1]
            if id.isdigit():
                id = int(id)
                if epoch == id:
                    t_epoch_args = self.epoch_args[epoch_name]
                    found = True
                    break

        return t_epoch_args, found

    def optimize_epoch_start(self, epoch: int):

        '''
        Set args based on epoch.
        '''

        # bugfix
        with th.no_grad():
            if self.point_reals is not None:
                normalized_p_r = self.point_reals.clone() / self.max_real

        t_epoch_args, found = self.find_epoch_args(epoch)
        if not found:
            print("Warning: no args found for epoch {}.".format(epoch))
                
        keys = self.default_args.keys()

        for k in keys:
            if k in t_epoch_args:
                val = t_epoch_args[k]
                if k in ["lr", "update_bound", 
                        
                        "iwdt_sigmoid_k", "max_real", 
                        "real_dmin_k", "refine_ud_thresh",
                        
                        "weight_regularizer_coef",
                        "quality_regularizer_coef",

                        "max_real_regularizer_coef_ratio",
                        "min_real_regularizer_coef_ratio",]:
                    val = float(val)
                elif k in ["lr_schedule"]:
                    val = str(val)
                else:
                    val = int(float(val))
                self.__setattr__(k, val)
            else:
                self.__setattr__(k, self.default_args[k])

        with th.no_grad():
            if self.point_reals is not None:
                self.point_reals = normalized_p_r * self.max_real

    def get_refine_num_seeds(self, epoch: int):

        t_epoch_args, found = self.find_epoch_args(epoch)
        refine_num_seeds = int(t_epoch_args.get("refine", 1000))

        return refine_num_seeds

    '''
    Updates during optimization
    '''
    def update_iwdt1_faces(self, 
                        iwdt1_faces: th.Tensor,
                        pd: PDStruct,
                        normalized_p_r: th.Tensor,
                        use_knn_faces: bool = False):
        '''
        Update list of faces for iwdt1 using current faces in WDT.
        If [use_knn_faces] is True, then we also collect knn faces for update.
        '''
        wdt_faces = pd.pd_edges.points_id

        iwdt1_faces = th.cat([iwdt1_faces, wdt_faces], dim=0)
        if use_knn_faces:
            min_p_r_list = [0.0, 0.2, 0.4, 0.6, 0.8]
            for min_p_r in min_p_r_list:
                knn_faces = pd_knn_faces(pd, self.iwdt_knn_k, normalized_p_r, min_p_r)
                iwdt1_faces = th.cat([iwdt1_faces, knn_faces], dim=0)
            
        iwdt1_faces = th.sort(iwdt1_faces, dim=-1)[0]
        iwdt1_faces = th.unique(iwdt1_faces, dim=0)
        
        return iwdt1_faces

    def update_lr(self, step: int, num_steps: int, optimizer: th.optim.Optimizer):
        
        if self.lr_schedule == "linear":
            lr = self.lr * (1.0 - (step / num_steps))
        elif self.lr_schedule == "exp":
            min_log_lr = np.log(MIN_LR)
            max_log_lr = np.log(self.lr)
            curr_log_lr = max_log_lr + (min_log_lr - max_log_lr) * (step / num_steps)
            lr = np.exp(curr_log_lr)
        elif self.lr_schedule == "constant":
            lr = self.lr

        lr = max(lr, MIN_LR)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    '''
    Optimizations
    '''

    def optimize(self):

        for epoch in range(self.num_epochs):
            print("===== Epoch: {}".format(epoch))
            self.optimize_epoch_start(epoch)

            if epoch > 0:
                refine_num_seeds = self.get_refine_num_seeds(epoch)
                print("===== Refining...")
                self.point_positions, self.point_weights, self.point_reals = \
                    self.refine_points(refine_num_seeds)
            else:
                self.init_points()

            num_phase_0_steps = self.num_phase_0_steps
            num_phase_0_real_warmup_steps = self.num_phase_0_real_warmup_steps
            num_phase_0_real_freeze_steps = self.num_phase_0_real_freeze_steps

            num_phase_1_steps = self.num_phase_1_steps
            num_phase_1_real_warmup_steps = self.num_phase_1_real_warmup_steps
            num_phase_1_real_freeze_steps = self.num_phase_1_real_freeze_steps

            save_steps = self.save_steps

            print("===== Starting Phase 0...")
            self.optimize_phase_0(epoch, 
                                    num_phase_0_steps,  
                                    num_phase_0_real_warmup_steps,
                                    num_phase_0_real_freeze_steps,
                                    save_steps)

            print("===== Starting Phase 1...")
            self.optimize_phase_1(epoch,
                                    num_phase_1_steps, 
                                    num_phase_1_real_warmup_steps,
                                    num_phase_1_real_freeze_steps,
                                    save_steps)

    def optimize_phase_0(self, epoch: int, num_steps: int, num_real_warmup_steps: int, num_real_freeze_steps: int, save_steps: int):
        
        
        '''
        In phase 0, point positions, weights, and reals are optimized.
        In every step, we do IWDT to gather faces and compute their existence probability.

        @ epoch: Current epoch number.
        @ num_steps: Number of steps to optimize.
        @ num_real_warmup_steps: After these number of steps, fix the real values that are beyond the threshold.
        @ save_steps: Save the current state every [save_steps] steps.
        '''

        phase = 0

        '''
        IWDT settings
        '''

        iwdt_sigmoid_k = self.iwdt_sigmoid_k
        iwdt_knn_interval = self.iwdt_knn_interval
        iwdt_delta_thresh = SIGMOID_MAX / iwdt_sigmoid_k    # threshold for delta;
        pseudo_sso = PseudoSSO()
        
        # faces that will be fed into iwdt1;
        # it is dynamically updated during optimization;
        iwdt1_faces: th.Tensor = th.empty((0, 3), dtype=th.int32, device=DEVICE)
        
        '''
        Real settings
        '''
        max_real = self.max_real
        real_dmin_k = self.real_dmin_k

        # idx of points with frozen real values;
        frozen_real_1_pid = th.empty((0,), dtype=th.long, device=DEVICE)
        
        '''
        Refresh optimizer and variables
        '''
        optimizer, p_pos, p_w, p_r = self.refresh_optimizer()

        bar = tqdm(range(num_steps))

        optim_start_time = self.global_optim_start_time

        min_loss = float("inf")
        min_eval_loss = float("inf")

        wdt_parallel = False

        for step in bar:

            lr = self.update_lr(step, num_steps, optimizer)

            # change [p_r]'s domain to [0, 1];
            normalized_p_r = p_r / max_real
            
            '''
            1. Gather faces and compute their existence probability.
            '''

            '''
            Gather possible faces.
            '''

            # points;
            points = WPoints(p_pos, p_w)

            # run WDT;
            start_time = time.time()
            wdt_result = DiffDT.CGAL_WDT(
                points,
                True,
                wdt_parallel,
            )
            end_time = time.time()
            wdt_time = end_time - start_time

            # construct PD;
            start_time = time.time()
            pd = PDStruct.forward(points, wdt_result.tets_point_id)
            end_time = time.time()
            pd_time = end_time - start_time

            # update iwdt1 faces;
            start_time = time.time()
            use_knn_faces = (step % iwdt_knn_interval == 0)
            iwdt1_faces = self.update_iwdt1_faces(iwdt1_faces, pd, normalized_p_r, use_knn_faces)
            end_time = time.time()
            iwdt1_faces_time = end_time - start_time

            # sso;
            start_time = time.time()
            sso = pseudo_sso.get_sso(pd)
            end_time = time.time()
            sso_time = end_time - start_time

            # iwdt0: get deltas for existing faces;
            start_time = time.time()
            iwdt0_result = DiffDT.iwdt0(
                pd,
                sso,
                DEVICE
            )
            end_time = time.time()
            iwdt0_time = end_time - start_time

            # iwdt1: get deltas for non-existing faces;
            start_time = time.time()
            iwdt1_result = DiffDT.iwdt1(
                pd,
                iwdt1_faces.to(dtype=th.int32),
                DEVICE
            )
            end_time = time.time()
            iwdt1_time = end_time - start_time

            '''
            Compute soft faces and their existence probability.
            '''
            start_time = time.time()

            e_faces = pd.pd_edges.points_id             # existing faces
            n_faces = iwdt1_result.faces                # non-existing faces

            e_faces_d = iwdt0_result.pd_edge_delta      # existing faces delta
            n_faces_d = iwdt1_result.per_face_delta     # non-existing faces delta

            soft_faces = th.cat([e_faces, n_faces], dim=0)
            soft_faces_d = th.cat([e_faces_d, n_faces_d], dim=0)
            soft_faces_d = th.clamp(soft_faces_d, max=iwdt_delta_thresh)  # clamp delta;

            soft_faces_iwdt_prob = th.sigmoid(soft_faces_d * iwdt_sigmoid_k)

            soft_faces_verts_real = normalized_p_r[soft_faces.to(dtype=th.long)]
            soft_faces_real_prob = dmin(soft_faces_verts_real, k=real_dmin_k)

            soft_faces_prob = soft_faces_iwdt_prob * soft_faces_real_prob
            
            end_time = time.time()
            faces_prob_time = end_time - start_time

            # compute statistics about iwdt_deltas for existing faces;
            with th.no_grad():
                e_faces_verts_real = normalized_p_r[e_faces.to(dtype=th.long)]
                e_faces_min_verts_real = th.min(e_faces_verts_real, dim=-1)[0]
                e_faces_on_mesh_bool = (e_faces_min_verts_real > 0.5)

                e_faces_d_on_mesh = e_faces_d[e_faces_on_mesh_bool]
                e_faces_d_on_mesh_med = th.median(e_faces_d_on_mesh)

            '''
            Compute hard faces.
            '''
            hard_faces = DMesh.eval_tet_faces(
                p_pos,
                normalized_p_r,
                pd.tets_point_id,
                self.remove_invisible
            )[1]

            '''
            2. Compute losses
            '''

            # recon loss
            start_time = time.time()
            recon_loss, recon_log = self.compute_recon_loss_0(
                epoch, step, num_steps,
                p_pos,
                p_w,
                normalized_p_r,

                soft_faces,
                soft_faces_iwdt_prob,
                soft_faces_real_prob,

                hard_faces,
            )
            end_time = time.time()
            recon_loss_time = end_time - start_time

            # regularizers
            start_time = time.time()
            weight_reg = self.compute_weight_regularizer(pd)
            end_time = time.time()
            weight_reg_time = end_time - start_time

            # schedule real regularizer coef;
            if step < self.real_regularizer_step:
                # interpolate
                real_reg_coef = self.max_real_regularizer_coef + \
                    (self.min_real_regularizer_coef - self.max_real_regularizer_coef) * \
                    (step / self.real_regularizer_step)
            else:
                real_reg_coef = self.min_real_regularizer_coef
            
            start_time = time.time()
            avg_real_reg = self.compute_avg_real_regularizer(soft_faces, soft_faces_iwdt_prob, normalized_p_r, real_reg_coef)
            max_real_reg = self.compute_max_real_regularizer(soft_faces, soft_faces_iwdt_prob, normalized_p_r, real_reg_coef)
            real_reg = avg_real_reg + max_real_reg
            end_time = time.time()
            real_reg_time = end_time - start_time

            start_time = time.time()
            quality_reg = self.compute_quality_regularizer(p_pos, soft_faces.to(dtype=th.long), soft_faces_prob)
            end_time = time.time()
            quality_reg_time = end_time - start_time

            loss = recon_loss + weight_reg + real_reg + quality_reg
            
            '''
            Update points.
            '''
            with th.no_grad():
                prev_p_pos = p_pos.clone()
                prev_p_w = p_w.clone()
                prev_p_r = p_r.clone()
                prev_normalized_p_r = normalized_p_r.clone()
                
            optimizer.zero_grad()

            start_time = time.time()
            loss.backward()
            end_time = time.time()
            loss_backward_time = end_time - start_time
            
            # clip grads;
            with th.no_grad():
                p_pos_grad = p_pos.grad if p_pos.grad is not None else th.zeros_like(p_pos)
                p_w_grad = p_w.grad if p_w.grad is not None else th.zeros_like(p_w)
                p_r_grad = p_r.grad if p_r.grad is not None else th.zeros_like(p_r)

                p_pos_grad_norm = th.norm(p_pos_grad, dim=-1) + 1e-6
                p_w_grad_norm = th.abs(p_w_grad) + 1e-6
                p_r_grad_norm = th.abs(p_r_grad) + 1e-6

                max_grad_norm = self.update_bound / self.lr

                p_pos_idx = p_pos_grad_norm > max_grad_norm
                p_w_idx = p_w_grad_norm > max_grad_norm
                p_r_idx = p_r_grad_norm > max_grad_norm

                p_pos_grad[p_pos_idx] = (p_pos_grad[p_pos_idx] / p_pos_grad_norm[p_pos_idx].unsqueeze(-1)) * max_grad_norm
                p_w_grad[p_w_idx] = (p_w_grad[p_w_idx] / p_w_grad_norm[p_w_idx]) * max_grad_norm
                p_r_grad[p_r_idx] = (p_r_grad[p_r_idx] / p_r_grad_norm[p_r_idx]) * max_grad_norm
                
                # fix for nan grads;
                p_pos_grad_nan_idx = th.any(th.isnan(p_pos_grad), dim=-1)
                p_w_grad_nan_idx = th.isnan(p_w_grad)
                p_r_grad_nan_idx = th.isnan(p_r_grad)
                
                p_pos_grad[p_pos_grad_nan_idx] = 0.0
                p_w_grad[p_w_grad_nan_idx] = 0.0
                p_r_grad[p_r_grad_nan_idx] = 0.0

                if p_pos.grad is not None:
                    p_pos.grad.data = p_pos_grad
                if p_w.grad is not None:
                    p_w.grad.data = p_w_grad
                if p_r.grad is not None:
                    p_r.grad.data = p_r_grad
                
                p_pos_nan_grad_ratio = th.count_nonzero(p_pos_grad_nan_idx) / p_pos_grad_nan_idx.shape[0]
                p_w_nan_grad_ratio = th.count_nonzero(p_w_grad_nan_idx) / p_w_grad_nan_idx.shape[0]
                p_r_nan_grad_ratio = th.count_nonzero(p_r_grad_nan_idx) / p_r_grad_nan_idx.shape[0]
                
            optimizer.step()

            '''
            Prev mesh we got.
            '''
            with th.no_grad():
                # previous (non-differentiable) mesh we got;
                prev_num_rp = pd.rp_point_id.shape[0]
                _, prev_mesh_faces, _ = DMesh.eval_tet_faces(
                    prev_p_pos, 
                    prev_normalized_p_r, 
                    pd.tets_point_id,
                    self.remove_invisible
                )
                
                prev_num_points_on_mesh = th.unique(prev_mesh_faces).shape[0]
                prev_num_faces_on_mesh = prev_mesh_faces.shape[0]

                # compute aspect ratio of faces on the mesh;
                prev_faces_aspect_ratio = triangle_aspect_ratio(prev_p_pos, prev_mesh_faces.to(dtype=th.long))
                prev_faces_aspect_ratio = prev_faces_aspect_ratio.mean()

            '''
            Bounding.
            '''
            with th.no_grad():
                p_pos.data = p_pos.data.clamp_(
                    prev_p_pos - self.update_bound,
                    prev_p_pos + self.update_bound
                )
                p_w.data = p_w.data.clamp_(
                    prev_p_w - self.update_bound,
                    prev_p_w + self.update_bound
                )
                p_r.data = p_r.data.clamp_(
                    prev_p_r - self.update_bound,
                    prev_p_r + self.update_bound
                )
                p_w.data = th.clamp(p_w.data, min=0.0, max=1.0)
                p_r.data = th.clamp(p_r.data, min=0.0, max=self.max_real)

                if not self.use_weight:
                    p_w.data = th.ones_like(p_w.data)

                # freeze real values;
                if step >= num_real_warmup_steps and \
                    (step - num_real_warmup_steps) % num_real_freeze_steps == 0:

                    # points on current mesh;
                    n_frozen_real_1_pid = th.unique(prev_mesh_faces)

                    frozen_real_1_pid = th.cat([frozen_real_1_pid, n_frozen_real_1_pid], dim=0)
                    frozen_real_1_pid = th.unique(frozen_real_1_pid)

                p_r[frozen_real_1_pid] = max_real

                # update points;
                self.point_positions = p_pos.clone()
                self.point_weights = p_w.clone()
                self.point_reals = p_r.clone()
                
                assert th.any(th.isnan(p_pos)) == False, "p_pos contains nan."
                assert th.any(th.isnan(p_w)) == False, "p_w contains nan."
                assert th.any(th.isnan(p_r)) == False, "p_r contains nan."

                assert th.any(th.isinf(p_pos)) == False, "p_pos contains inf."
                assert th.any(th.isinf(p_w)) == False, "p_w contains inf."
                assert th.any(th.isinf(p_r)) == False, "p_r contains inf."


            '''
            Logging
            '''

            with th.no_grad():
                self.writer.add_scalar(f"epoch_{epoch}_p0_loss/loss", loss, step)
                self.writer.add_scalar(f"epoch_{epoch}_p0_loss/recon_loss", recon_loss, step)
                self.writer.add_scalar(f"epoch_{epoch}_p0_loss/weight_reg", weight_reg, step)
                self.writer.add_scalar(f"epoch_{epoch}_p0_loss/real_reg", real_reg, step)
                self.writer.add_scalar(f"epoch_{epoch}_p0_loss/quality_reg", quality_reg, step)
                
                self.writer.add_scalar(f"epoch_{epoch}_p0_info/lr", lr, step)
                self.writer.add_scalar(f"epoch_{epoch}_p0_info/iwdt_sigmoid_k", iwdt_sigmoid_k, step)
                self.writer.add_scalar(f"epoch_{epoch}_p0_info/iwdt_delta_thresh", iwdt_delta_thresh, step)
                self.writer.add_scalar(f"epoch_{epoch}_p0_info/wdt_parallel", wdt_parallel, step)
                self.writer.add_scalar(f"epoch_{epoch}_p0_info/num_frozen_real_1", frozen_real_1_pid.shape[0], step)
                self.writer.add_scalar(f"epoch_{epoch}_p0_info/median_iwdt_delta_efaces", e_faces_d_on_mesh_med, step)
                
                self.writer.add_scalar(f"epoch_{epoch}_p0_coef/real_reg", real_reg_coef, step)

                self.writer.add_scalar(f"epoch_{epoch}_p0_mesh/num_rp", prev_num_rp, step)
                self.writer.add_scalar(f"epoch_{epoch}_p0_mesh/num_faces_on_mesh", prev_num_faces_on_mesh, step)
                self.writer.add_scalar(f"epoch_{epoch}_p0_mesh/num_points_on_mesh", prev_num_points_on_mesh, step)
                self.writer.add_scalar(f"epoch_{epoch}_p0_mesh/faces_aspect_ratio", prev_faces_aspect_ratio, step)

                # nan grad;
                self.writer.add_scalar(f"epoch_{epoch}_p0_nan/p_pos_nan_grad_ratio", p_pos_nan_grad_ratio, step)
                self.writer.add_scalar(f"epoch_{epoch}_p0_nan/p_w_nan_grad_ratio", p_w_nan_grad_ratio, step)
                self.writer.add_scalar(f"epoch_{epoch}_p0_nan/p_r_nan_grad_ratio", p_r_nan_grad_ratio, step)

                # time;
                self.writer.add_scalar(f"epoch_{epoch}_p0_time/wdt_time", wdt_time, step)
                self.writer.add_scalar(f"epoch_{epoch}_p0_time/pd_time", pd_time, step)
                self.writer.add_scalar(f"epoch_{epoch}_p0_time/iwdt1_faces_time", iwdt1_faces_time, step)
                self.writer.add_scalar(f"epoch_{epoch}_p0_time/sso_time", sso_time, step)
                self.writer.add_scalar(f"epoch_{epoch}_p0_time/iwdt0_time", iwdt0_time, step)
                self.writer.add_scalar(f"epoch_{epoch}_p0_time/iwdt1_time", iwdt1_time, step)
                self.writer.add_scalar(f"epoch_{epoch}_p0_time/faces_prob_time", faces_prob_time, step)
                self.writer.add_scalar(f"epoch_{epoch}_p0_time/recon_loss_time", recon_loss_time, step)
                self.writer.add_scalar(f"epoch_{epoch}_p0_time/weight_reg_time", weight_reg_time, step)
                self.writer.add_scalar(f"epoch_{epoch}_p0_time/real_reg_time", real_reg_time, step)
                self.writer.add_scalar(f"epoch_{epoch}_p0_time/quality_reg_time", quality_reg_time, step)
                self.writer.add_scalar(f"epoch_{epoch}_p0_time/loss_backward_time", loss_backward_time, step)
                
                # recon;
                for rk, rv in recon_log.items():
                    self.writer.add_scalar(f"epoch_{epoch}_p0_recon/{rk}", rv, step)

                bar.set_description("loss: {:.4f}".format(loss))

            '''
            Saving
            '''
            if step % save_steps == 0 or step == num_steps - 1:
                self.save_step(
                    epoch,
                    phase,
                    step,
                    prev_p_pos,
                    prev_p_w,
                    prev_normalized_p_r,
                    time.time() - optim_start_time
                )

                self.save(
                    os.path.join(
                        self.writer.log_dir, 
                        f"save/epoch_{epoch}/phase_{phase}/last"
                    ),
                    prev_p_pos,
                    prev_p_w,
                    prev_normalized_p_r,
                    time.time() - optim_start_time
                )

                eval_loss = self.compute_eval_loss(
                    prev_p_pos,
                    prev_mesh_faces
                )

                self.writer.add_scalar(f"epoch_{epoch}_p0_eval/eval_loss", eval_loss, step)

                if eval_loss < min_eval_loss:
                    min_eval_loss = eval_loss
                    self.save(
                        os.path.join(
                            self.writer.log_dir, 
                            f"save/epoch_{epoch}/phase_{phase}/min_eval_loss"
                        ),
                        prev_p_pos,
                        prev_p_w,
                        prev_normalized_p_r,
                        time.time() - optim_start_time
                    )

            if loss < min_loss:
                min_loss = loss
                self.save(
                    os.path.join(
                        self.writer.log_dir, 
                        f"save/epoch_{epoch}/phase_{phase}/min_loss"
                    ),
                    prev_p_pos,
                    prev_p_w,
                    prev_normalized_p_r,
                    time.time() - optim_start_time
                )

    def optimize_phase_1(self, epoch: int, num_steps: int, num_real_warmup_steps: int, num_real_freeze_steps: int, save_steps: int):
        
        '''
        In phase 1, only point reals are optimized while faces are fixed.
        
        @ epoch: Current epoch number.
        @ num_steps: Number of steps to optimize.
        @ save_steps: Save the current state every [save_steps] steps.
        '''

        phase = 1

        '''
        Real settings
        '''
        max_real = self.max_real
        real_dmin_k = self.real_dmin_k

        # idx of points with frozen real values;
        frozen_real_1_pid = th.empty((0,), dtype=th.long, device=DEVICE)
        
        '''
        Refresh optimizer and variables
        '''
        optimizer, p_pos, p_w, p_r = self.refresh_optimizer()

        bar = tqdm(range(num_steps))

        optim_start_time = self.global_optim_start_time

        min_loss = float("inf")
        min_eval_loss = float("inf")

        '''
        Gather fixed faces.
        '''

        # points;
        points = WPoints(p_pos, p_w)

        # run WDT;
        wdt_result = DiffDT.CGAL_WDT(
            points,
            True,
            False,
        )
        end_time = time.time()
        curr_faces = extract_faces(wdt_result.tets_point_id)
        curr_faces_iwdt_prob = th.ones((curr_faces.shape[0],), dtype=th.float32, device=DEVICE)

        for step in bar:

            lr = self.update_lr(step, num_steps, optimizer)

            # change [p_r]'s domain to [0, 1];
            normalized_p_r = p_r / max_real

            '''
            Evaluate probability based on reals.
            '''
            curr_faces_verts_real = normalized_p_r[curr_faces.to(dtype=th.long)]
            curr_faces_real_prob = dmin(curr_faces_verts_real, k=real_dmin_k)

            '''
            2. Compute losses
            '''
            # recon loss
            start_time = time.time()
            recon_loss, recon_log = self.compute_recon_loss_1(
                epoch, step, num_steps,
                
                p_pos,
                p_w,
                normalized_p_r,

                curr_faces, 
                curr_faces_iwdt_prob, 
                curr_faces_real_prob,

                wdt_result.tets_point_id
            )
            end_time = time.time()
            recon_loss_time = end_time - start_time

            # regularizers
            start_time = time.time()
            real_reg_coef = self.min_real_regularizer_coef
            max_real_regularizer = self.compute_max_real_regularizer(curr_faces, curr_faces_iwdt_prob, normalized_p_r, real_reg_coef)
            real_regularizer = max_real_regularizer
            end_time = time.time()
            real_loss_time = end_time - start_time

            loss = recon_loss + real_regularizer
            
            '''
            Update points.
            '''
            with th.no_grad():
                prev_p_pos = p_pos.clone()
                prev_p_w = p_w.clone()
                prev_p_r = p_r.clone()
                prev_normalized_p_r = normalized_p_r.clone()
                
            optimizer.zero_grad()

            start_time = time.time()
            loss.backward()
            end_time = time.time()
            loss_backward_time = end_time - start_time
            
            # clip grads;
            with th.no_grad():
                p_r_grad = p_r.grad if p_r.grad is not None else th.zeros_like(p_r)
                p_r_grad_norm = th.abs(p_r_grad) + 1e-6

                max_grad_norm = self.update_bound / self.lr

                p_r_idx = p_r_grad_norm > max_grad_norm
                p_r_grad[p_r_idx] = (p_r_grad[p_r_idx] / p_r_grad_norm[p_r_idx]) * max_grad_norm
                
                # fix for nan grads;
                p_r_grad_nan_idx = th.isnan(p_r_grad)
                p_r_grad[p_r_grad_nan_idx] = 0.0

                if p_r.grad is not None:
                    p_r.grad.data = p_r_grad
                
                p_r_nan_grad_ratio = th.count_nonzero(p_r_grad_nan_idx) / p_r_grad_nan_idx.shape[0]
                
            optimizer.step()

            '''
            Prev mesh we got.
            '''
            with th.no_grad():
                # previous (non-differentiable) mesh we got;
                _, prev_mesh_faces, _ = DMesh.eval_tet_faces(
                    prev_p_pos, 
                    prev_normalized_p_r, 
                    wdt_result.tets_point_id,
                    self.remove_invisible
                )
                
                prev_num_points_on_mesh = th.unique(prev_mesh_faces).shape[0]
                prev_num_faces_on_mesh = prev_mesh_faces.shape[0]

            '''
            Bounding.
            '''
            with th.no_grad():
                p_pos.data = prev_p_pos.data
                p_w.data = prev_p_w.data
                
                p_r.data = p_r.data.clamp_(
                    prev_p_r - self.update_bound,
                    prev_p_r + self.update_bound
                )
                p_r.data = th.clamp(p_r.data, min=0.0, max=self.max_real)

                # freeze;
                if step >= num_real_warmup_steps and \
                    (step - num_real_warmup_steps) % num_real_freeze_steps == 0:

                    # points on current mesh;
                    n_frozen_real_1_pid = th.unique(prev_mesh_faces)

                    frozen_real_1_pid = th.cat([frozen_real_1_pid, n_frozen_real_1_pid], dim=0)
                    frozen_real_1_pid = th.unique(frozen_real_1_pid)

                p_r[frozen_real_1_pid] = max_real

                # update points;
                self.point_positions = p_pos.clone()
                self.point_weights = p_w.clone()
                self.point_reals = p_r.clone()
                
                assert th.any(th.isnan(p_r)) == False, "p_r contains nan."
                assert th.any(th.isinf(p_r)) == False, "p_r contains inf."

            '''
            Logging
            '''

            with th.no_grad():
                self.writer.add_scalar(f"epoch_{epoch}_p1_loss/loss", loss, step)
                self.writer.add_scalar(f"epoch_{epoch}_p1_loss/recon_loss", recon_loss, step)
                self.writer.add_scalar(f"epoch_{epoch}_p1_loss/real_regularizer", real_regularizer, step)
                
                self.writer.add_scalar(f"epoch_{epoch}_p1_info/lr", lr, step)
                self.writer.add_scalar(f"epoch_{epoch}_p1_info/num_frozen_real_1", frozen_real_1_pid.shape[0], step)

                self.writer.add_scalar(f"epoch_{epoch}_p1_mesh/num_faces_on_mesh", prev_num_faces_on_mesh, step)
                self.writer.add_scalar(f"epoch_{epoch}_p1_mesh/num_points_on_mesh", prev_num_points_on_mesh, step)

                # nan grad;
                self.writer.add_scalar(f"epoch_{epoch}_p1_nan/p_r_nan_grad_ratio", p_r_nan_grad_ratio, step)

                # time;
                self.writer.add_scalar(f"epoch_{epoch}_p1_time/recon_loss_time", recon_loss_time, step)
                self.writer.add_scalar(f"epoch_{epoch}_p1_time/real_loss_time", real_loss_time, step)
                self.writer.add_scalar(f"epoch_{epoch}_p1_time/loss_backward_time", loss_backward_time, step)
                
                # recon;
                for rk, rv in recon_log.items():
                    self.writer.add_scalar(f"epoch_{epoch}_p1_recon/{rk}", rv, step)

                bar.set_description("loss: {:.4f}".format(loss))

            '''
            Saving
            '''
            if step % save_steps == 0 or step == num_steps - 1:
                self.save_step(
                    epoch,
                    phase,
                    step,
                    prev_p_pos,
                    prev_p_w,
                    prev_normalized_p_r,
                    time.time() - optim_start_time
                )

                self.save(
                    os.path.join(
                        self.writer.log_dir, 
                        f"save/epoch_{epoch}/phase_{phase}/last"
                    ),
                    prev_p_pos,
                    prev_p_w,
                    prev_normalized_p_r,
                    time.time() - optim_start_time
                )

                eval_loss = self.compute_eval_loss(
                    prev_p_pos,
                    prev_mesh_faces
                )

                self.writer.add_scalar(f"epoch_{epoch}_p1_eval/eval_loss", eval_loss, step)

                if eval_loss < min_eval_loss:
                    min_eval_loss = eval_loss
                    self.save(
                        os.path.join(
                            self.writer.log_dir, 
                            f"save/epoch_{epoch}/phase_{phase}/min_eval_loss"
                        ),
                        prev_p_pos,
                        prev_p_w,
                        prev_normalized_p_r,
                        time.time() - optim_start_time
                    )

            if loss < min_loss:
                min_loss = loss
                self.save(
                    os.path.join(
                        self.writer.log_dir, 
                        f"save/epoch_{epoch}/phase_{phase}/min_loss"
                    ),
                    prev_p_pos,
                    prev_p_w,
                    prev_normalized_p_r,
                    time.time() - optim_start_time
                )
