import torch as th
import numpy as np
import argparse
import yaml
import os
import trimesh
import time

from exp.recon import BaseRecon, DEVICE, DOMAIN, WPoints, PDStruct, DiffDT, sample_points_on_mesh, run_knn
from exp.utils.utils import setup_logdir, import_mesh
from exp.utils.mv import make_star_cameras, calc_face_normals, GTInitializer, AlphaRenderer, LIGHT_DIR, compute_verts_depth, compute_faces_intense
from exp.utils.tetra import TetraSet

from dmesh_renderer import TriRenderSettings, TriRenderer, TetRenderSettings, TetRenderer

from PIL import Image

class MVRecon(BaseRecon):

    def __init__(self, 
                 
                mesh: trimesh.base.Trimesh,

                logdir: str,

                # target images;
                mv: th.Tensor,
                proj: th.Tensor,
                image_size: int,
                batch_size: int,

                target_diffuse_map: th.Tensor,
                target_depth_map: th.Tensor,

                # alpha thresholds used for face culling;
                alpha_thresh: float,
                min_alpha_thresh: float,
                max_alpha_thresh: float,

                # use depth;
                use_depth: bool,
                depth_coef: float,

                # init method;
                init_args: dict,

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

        self.mesh = mesh
        '''
        Rendering settings.
        '''
        self.mv = mv
        self.proj = proj
        self.image_size = image_size
        self.batch_size = batch_size

        self.target_diffuse_map = target_diffuse_map
        self.target_depth_map = target_depth_map

        self.alpha_thresh = alpha_thresh
        self.min_alpha_thresh = min_alpha_thresh
        self.max_alpha_thresh = max_alpha_thresh

        self.use_depth = use_depth
        self.depth_coef = depth_coef

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

        for i in range(len(self.target_diffuse_map)):
            self.writer.add_image(f"gt/diffuse_{i}", self.target_diffuse_map[i], 0, dataformats="HWC")
            self.writer.add_image(f"gt/depth_{i}", self.target_depth_map[i], 0, dataformats="HWC")
        self.writer.flush()

        self.remove_invisible = True

    def init_points(self):

        if self.init_args['method'] == "sample":

            mesh_points = th.tensor(self.mesh.vertices, dtype=th.float32, device=DEVICE)
            mesh_faces = th.tensor(self.mesh.faces, dtype=th.int32, device=DEVICE)

            # sample on current mesh;
            num_seed_points = int(float(self.init_args.get("num_sample_points", 1000)))
            num_gt_points = 1_000_000
            sample_points, _ = \
                sample_points_on_mesh(
                    mesh_points,
                    mesh_faces.to(dtype=th.long),
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

            self.point_positions = point_positions
            self.point_weights = point_weights
            self.point_reals = point_reals
        
        elif self.init_args['method'] == "random":
        
            grid_res = self.init_args.get("grid_res", 10)
            self.random_points_init(grid_res)

        elif self.init_args['method'] == "load":

            path = self.init_args.get("load_path", None)
            try:
                self.load_points_init(path)
            except:
                print(f"Warning: failed to load points. Path: {path}")
                exit(-1)

        else:

            raise ValueError("Invalid init method.")

    '''
    Losses
    '''

    def soft_render(self,
                    p_r: th.Tensor,
                    verts: th.Tensor,
                    faces: th.Tensor,
                    faces_iwdt_prob: th.Tensor,
                    faces_alpha: th.Tensor,

                    mv: th.Tensor,
                    proj: th.Tensor,):

        batch_size = mv.shape[0]
        # if proj.ndim == 2:
        #     proj = proj.unsqueeze(0).expand((batch_size, -1, -1))

        curr_alpha_thresh = self.alpha_thresh

        '''
        Compute face alphas for selecting which faces to render.
        '''
        with th.no_grad():
            faces_verts_real = p_r[faces.to(dtype=th.long)]      # [F, 3]
            max_faces_verts_real = th.max(faces_verts_real, dim=-1)[0]  # [F]
            min_faces_verts_real = th.min(faces_verts_real, dim=-1)[0]  # [F]

            # valid faces based on maximum real value;
            s_faces_alpha_0 = faces_iwdt_prob * max_faces_verts_real  # [F]

            # valid faces based on minimum real value;
            s_faces_alpha_1 = faces_iwdt_prob * min_faces_verts_real  # [F]

            s_faces_alpha = th.max(s_faces_alpha_0, s_faces_alpha_1)  # [F]
            # s_faces_alpha = th.min(s_faces_alpha_0, s_faces_alpha_1)  # [F]

        while True:

            '''
            Extract valid faces based on alpha threshold.
            '''
            faces_validity = (s_faces_alpha > curr_alpha_thresh)

            valid_faces = faces[faces_validity]
            valid_faces_alpha = faces_alpha[faces_validity]
            
            '''
            Setup rendering.
            '''

            # verts
            ren_verts = verts

            # faces
            ren_faces = valid_faces

            # verts color
            ren_verts_color = th.ones_like(ren_verts)                   # [V, 3]

            # faces opacity
            ren_faces_opacity = valid_faces_alpha                             # [F]

            # verts depth;
            ren_verts_depth = compute_verts_depth(ren_verts, mv, proj)  # [B, V]

            # faces intense;
            lightdir = th.tensor([LIGHT_DIR], dtype=th.float32, device=DEVICE)                  # [1, 3] 
            lightdir = lightdir.expand((batch_size, -1))                                         # [B, 3]
            ren_faces_intense = compute_faces_intense(ren_verts, ren_faces.to(dtype=th.long), mv, lightdir)       # [B, F]

            render_settings = TriRenderSettings(
                image_height=image_size,
                image_width=image_size,
                bg=th.zeros((3,), dtype=th.float32, device=DEVICE),
            )

            '''
            Render.
            '''

            renderer = TriRenderer(render_settings)

            try:
                soft_color, soft_depth = renderer.forward(
                    ren_verts,
                    ren_faces,
                    ren_verts_color,
                    ren_faces_opacity,
                    mv,
                    proj,
                    ren_verts_depth,
                    ren_faces_intense
                )
            except:
                # increase alpha threshold if possible;
                if curr_alpha_thresh < self.max_alpha_thresh:
                    curr_alpha_thresh = min(curr_alpha_thresh * 2.0, self.max_alpha_thresh)
                    continue
                else:
                    print("Warning: alpha threshold exceeds maximum threshold.")
                    exit(-1)

            break

        self.alpha_thresh = curr_alpha_thresh
             
        soft_color = soft_color.transpose(2, 1)     # [B, H, 3, W]
        soft_color = soft_color.transpose(3, 2)     # [B, H, W, 3]
        soft_depth = soft_depth.transpose(2, 1)     # [B, H, 1, W]
        soft_depth = soft_depth.transpose(3, 2)     # [B, H, W, 1]
        soft_depth = (soft_depth + 1.0) / 2.0     # normalize to [0, 1]
        soft_depth = 1.0 - soft_depth

        return soft_color, soft_depth

    def hard_render(self,
                    verts: th.Tensor,
                    faces: th.Tensor,
                    
                    mv: th.Tensor,
                    proj: th.Tensor):
        
        renderer = AlphaRenderer(mv, proj, [self.image_size, self.image_size])

        # duplicate vertices;
        faces_normals = calc_face_normals(verts, faces, True)           # [F, 3]

        num_faces = faces.shape[0]
        tmp = th.arange(0, num_faces, dtype=th.long, device=DEVICE)

        n_verts = []
        n_verts_normals = []
        n_faces = []
        for i in range(3):
            n_verts.append(verts[faces[:, i]])          # [F, 3]
            n_verts_normals.append(faces_normals)       # [F, 3]
            n_faces.append(tmp.unsqueeze(-1) + (i * num_faces))  # [F, 1]

        n_verts = th.cat(n_verts, dim=0)                    # [3F, 3]
        n_verts_normals = th.cat(n_verts_normals, dim=0)    # [3F, 3]
        n_faces = th.cat(n_faces, dim=-1)                   # [F, 3]
        n_faces = n_faces.squeeze(-1)

        # render;
        col, _ = renderer.forward(n_verts, n_verts_normals, n_faces)    # [..., 5]

        # compute loss;
        diffuse = col[...,:-2]
        depth = col[..., [-2, -2, -2]]

        return diffuse, depth

    def compute_recon_loss_0(self,
                        epoch: int, step: int, num_steps: int, 
                        point_positions: th.Tensor, 
                        point_weights: th.Tensor,
                        point_reals: th.Tensor,
                        
                        soft_faces: th.Tensor, 
                        soft_faces_iwdt_prob: th.Tensor, 
                        soft_faces_real_prob: th.Tensor,

                        hard_faces: th.Tensor):

        batch_size = self.batch_size

        '''
        1. Random sample mv and proj.
        '''
        mv = self.mv
        proj = self.proj

        num_views = mv.shape[0]
        rand_idx = th.randperm(num_views)[:batch_size]

        mv = mv[rand_idx]
        proj = proj[rand_idx]
        
        b_target_diffuse_map = self.target_diffuse_map[rand_idx]
        b_target_depth_map = self.target_depth_map[rand_idx]

        '''
        2. Render soft faces.
        '''

        soft_diffuse, soft_depth = self.soft_render(
                                                point_reals,
                                                point_positions, 
                                                soft_faces,
                                                soft_faces_iwdt_prob,
                                                soft_faces_iwdt_prob * soft_faces_real_prob,
                                                mv, proj)
        
        '''
        3. Render hard faces.
        '''

        hard_diffuse, hard_depth = self.hard_render(point_positions, hard_faces.to(dtype=th.long), mv, proj)
        
        '''
        4. Compute loss.
        '''
        # get hard loss coef;
        hard_loss_coef = 0.1

        # diffuse loss;
        soft_diffuse_loss = th.abs((soft_diffuse - b_target_diffuse_map)).mean()
        hard_diffuse_loss = th.abs((hard_diffuse - b_target_diffuse_map)).mean() * hard_loss_coef

        # depth loss;
        soft_depth_loss = th.abs((soft_depth - b_target_depth_map)).mean() * self.depth_coef
        hard_depth_loss = th.abs((hard_depth - b_target_depth_map)).mean() * self.depth_coef * hard_loss_coef

        if not self.use_depth:
            soft_depth_loss = soft_depth_loss * 0
            hard_depth_loss = hard_depth_loss * 0
            
        # total loss;
        soft_loss = soft_diffuse_loss + soft_depth_loss
        hard_loss = hard_diffuse_loss + hard_depth_loss

        loss = soft_loss + hard_loss

        if step % 20 == 0:
            # save images;
            self.writer.add_image(f"epoch_{epoch}_p0_render/soft_diffuse", soft_diffuse[0], step, dataformats="HWC")
            self.writer.add_image(f"epoch_{epoch}_p0_render/hard_diffuse", hard_diffuse[0], step, dataformats="HWC")
            self.writer.add_image(f"epoch_{epoch}_p0_render/soft_depth", soft_depth[0], step, dataformats="HWC")
            self.writer.add_image(f"epoch_{epoch}_p0_render/hard_depth", hard_depth[0], step, dataformats="HWC")

        with th.no_grad():
            log = {
                "soft_diffuse_loss": soft_diffuse_loss.item(),
                "hard_diffuse_loss": hard_diffuse_loss.item(),
                "soft_depth_loss": soft_depth_loss.item(),
                "hard_depth_loss": hard_depth_loss.item(),
                "alpha_thresh": self.alpha_thresh,
                "hard_loss_coef": hard_loss_coef
            }

        return loss, log

    def compute_recon_loss_1(self, 
                            epoch: int, step: int, num_steps: int, 
                            
                            point_positions: th.Tensor, 
                            point_weights: th.Tensor,
                            point_reals: th.Tensor,

                            faces: th.Tensor, 
                            faces_iwdt_prob: th.Tensor, 
                            faces_real_prob: th.Tensor,
                            
                            tets: th.Tensor):

        # num_batch = len(self.mv)
        # mv = self.mv
        # proj = self.proj.unsqueeze(0).expand((num_batch, -1, -1))

        batch_size = self.batch_size
        # batch_size = len(self.mv)

        '''
        1. Random sample mv and proj.
        '''
        mv = self.mv
        proj = self.proj

        num_views = mv.shape[0]
        rand_idx = th.randperm(num_views)[:batch_size]

        mv = mv[rand_idx]
        proj = proj[rand_idx]
        # proj = self.proj.unsqueeze(0).expand((batch_size, -1, -1))
        
        b_target_diffuse_map = self.target_diffuse_map[rand_idx]
        b_target_depth_map = self.target_depth_map[rand_idx]

        # verts
        ren_verts = point_positions.clone()

        # faces
        ren_faces = faces.clone()

        # verts color
        ren_verts_color = th.ones_like(ren_verts)                   # [V, 3]

        # faces opacity
        faces_alpha = faces_iwdt_prob * faces_real_prob
        ren_faces_opacity = faces_alpha                             # [F]

        # verts depth;
        ren_verts_depth = compute_verts_depth(ren_verts, mv, proj)  # [B, V]

        # faces intense;
        lightdir = th.tensor([LIGHT_DIR], dtype=th.float32, device=DEVICE)                  # [1, 3] 
        lightdir = lightdir.expand((batch_size, -1))                                         # [B, 3]
        ren_faces_intense = compute_faces_intense(ren_verts, ren_faces.to(dtype=th.long), mv, lightdir)       # [B, F]

        # tetra set;
        tetset = TetraSet(ren_verts, tets.to(dtype=th.long))

        render_settings = TetRenderSettings(
            image_height=image_size,
            image_width=image_size,
            bg=th.zeros((3,), dtype=th.float32, device=DEVICE),
            ray_random_seed=th.randint(1, 10000, (1,)).cpu().item(),
        )

        renderer = TetRenderer(render_settings)
        diffuse, depth, _ = renderer.forward(
            ren_verts,
            ren_faces,
            ren_verts_color,
            ren_faces_opacity,
            mv,
            proj,
            ren_verts_depth,
            ren_faces_intense,
            tets,
            tetset.face_tet,
            tetset.tet_faces,
        )
        diffuse = diffuse.transpose(2, 1)     # [B, H, 3, W]
        diffuse = diffuse.transpose(3, 2)     # [B, H, W, 3]
        depth = depth.transpose(2, 1)     # [B, H, 1, W]
        depth = depth.transpose(3, 2)     # [B, H, W, 1]
        depth = (depth + 1.0) / 2.0     # normalize to [0, 1]
        depth = 1.0 - depth

        # diffuse loss;
        diffuse_loss = th.mean(th.abs(diffuse - b_target_diffuse_map))
        depth_loss = th.mean(th.abs(depth - b_target_depth_map)) * self.depth_coef
        if not self.use_depth:
            depth_loss = depth_loss * 0

        loss = diffuse_loss + depth_loss
        with th.no_grad():
            log = {
                "diffuse_loss": diffuse_loss.item(),
                "depth_loss": depth_loss.item()
            }

        if step % 100 == 0:
            # save images;
            self.writer.add_image(f"epoch_{epoch}_p1_render/diffuse", diffuse[0], step, dataformats="HWC")
            self.writer.add_image(f"epoch_{epoch}_p1_render/depth", depth[0], step, dataformats="HWC")

        return loss, log
    
    def compute_eval_loss(self, positions: th.Tensor, faces: th.Tensor):
        '''
        Compute evaluation loss.
        '''

        mv = self.mv
        proj = self.proj

        diffuse, depth = self.hard_render(positions, faces.to(dtype=th.long), mv, proj)
        
        '''
        5. Compute loss.
        '''
        diffuse_loss = th.abs((diffuse - self.target_diffuse_map)).mean()
        depth_loss = th.abs((depth - self.target_depth_map)).mean()

        if not self.use_depth:
            depth_loss = depth_loss * 0

        depth_loss = depth_loss * self.depth_coef

        loss = diffuse_loss + depth_loss

        return loss

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="exp/config/exp_3/objaverse/plant.yaml")
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
    Ground truth renderings, used for mv recon.
    '''
    num_viewpoints = int(float(settings['args']['num_viewpoints']))
    image_size = int(float(settings['args']['image_size']))
    batch_size = int(float(settings['args']['batch_size']))

    mv, proj = make_star_cameras(num_viewpoints, num_viewpoints, distance=2.0, r=0.6, n=1.0, f=3.0)
    proj = proj.unsqueeze(0).expand(mv.shape[0], -1, -1)
    renderer = AlphaRenderer(mv, proj, [image_size, image_size])

    gt_manager = GTInitializer(verts, faces, DEVICE)
    gt_manager.render(renderer)
    
    gt_diffuse_map = gt_manager.diffuse_images()
    gt_depth_map = gt_manager.depth_images()
    gt_shil_map = gt_manager.shillouette_images()

    # save gt images;
    image_save_path = os.path.join(logdir, "gt_images")
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)
        
    def save_image(img, path):
        img = img.cpu().numpy()
        img = img * 255.0
        img = img.astype(np.uint8)
        img = Image.fromarray(img)
        img.save(path)

    for i in range(len(gt_diffuse_map)):
        save_image(gt_diffuse_map[i], os.path.join(image_save_path, "diffuse_{}.png".format(i)))
        save_image(gt_depth_map[i], os.path.join(image_save_path, "depth_{}.png".format(i)))
        save_image(gt_shil_map[i], os.path.join(image_save_path, "shil_{}.png".format(i)))

    # alpha thresholds;
    alpha_thresh = float(settings['args']['alpha_thresh'])
    min_alpha_thresh = float(settings['args']['min_alpha_thresh'])
    max_alpha_thresh = float(settings['args']['max_alpha_thresh'])

    # use depth;
    use_depth = settings['args']['use_depth']
    depth_coef = float(settings['args']['depth_coef'])

    '''
    Arguments: default values;
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

    # Init method;
    init_args = settings['args']['init_args']

    # params;
    iwdt_sigmoid_k = float(settings['args']['iwdt_sigmoid_k'])
    max_real = float(settings['args']['max_real'])
    refine_ud_thresh = float(settings['args']['refine_ud_thresh'])

    use_weight = settings['args']['use_weight']

    # steps;
    num_epochs = settings['args']['num_epochs']
    
    num_phase_0_steps = settings['args']['num_phase_0_steps']
    num_phase_0_real_warmup_steps = settings['args']['num_phase_0_real_warmup_steps']
    num_phase_0_real_freeze_steps = settings['args']['num_phase_0_real_freeze_steps']

    num_phase_1_steps = settings['args']['num_phase_1_steps']
    num_phase_1_real_warmup_steps = settings['args']['num_phase_1_real_warmup_steps']
    num_phase_1_real_freeze_steps = settings['args']['num_phase_1_real_freeze_steps']

    save_steps = settings['args']['save_steps']
    
    '''
    Epoch args.
    '''
    epoch_args = settings['args']['epoch_args']

    optimizer = MVRecon(
        logdir=logdir,

        mesh=mesh,

        mv=mv,
        proj=proj,
        image_size=image_size,
        batch_size=batch_size,
        target_diffuse_map=gt_diffuse_map,
        target_depth_map=gt_depth_map,

        alpha_thresh=alpha_thresh,
        min_alpha_thresh=min_alpha_thresh,
        max_alpha_thresh=max_alpha_thresh,

        use_depth=use_depth,
        depth_coef=depth_coef,

        init_args=init_args,
        
        lr=lr,
        lr_schedule=lr_schedule,
        update_bound=update_bound,

        iwdt_sigmoid_k=iwdt_sigmoid_k,
        max_real=max_real,
        refine_ud_thresh=refine_ud_thresh,
        
        num_epochs=num_epochs,

        num_phase_0_steps=num_phase_0_steps,
        num_phase_0_real_warmup_steps=num_phase_0_real_warmup_steps,
        num_phase_0_real_freeze_steps=num_phase_0_real_freeze_steps,

        num_phase_1_steps=num_phase_1_steps,
        num_phase_1_real_warmup_steps=num_phase_1_real_warmup_steps,
        num_phase_1_real_freeze_steps=num_phase_1_real_freeze_steps,

        save_steps=save_steps,
        
        weight_regularizer_coef=weight_regularizer_coef,
        quality_regularizer_coef=quality_regularizer_coef,

        epoch_args=epoch_args,

        max_real_regularizer_coef=max_real_regularizer_coef,
        min_real_regularizer_coef=min_real_regularizer_coef,
        real_regularizer_step=real_regularizer_step,

        use_weight=use_weight
    )

    optimizer.optimize()