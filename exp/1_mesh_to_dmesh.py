'''
Script to convert a mesh to a DMesh using WDT.
'''
import argparse
import yaml
import trimesh
import os
import igl

from tqdm import tqdm
from fast_pytorch_kmeans import KMeans

import torch as th
from torch.utils.tensorboard import SummaryWriter

from diffdt import DiffDT
from diffdt.wp import WPoints
from diffdt.pd import PDStruct
from exp.utils.nn import pd_knn_faces
from exp.utils.tensor import *
from exp.utils.utils import *
from exp.utils.dmesh import DMesh

device = 'cuda:0'

class GtmeshOptimizer:

    def __init__(self, gt_verts: th.Tensor, gt_faces: th.Tensor, logdir: str):
        self.gt_verts = gt_verts

        # preprocess faces;
        self.gt_faces = gt_faces.to(dtype=th.int32)
        self.gt_faces = th.sort(self.gt_faces, dim=-1)[0]
        self.gt_faces = th.unique(self.gt_faces, dim=0)
        oriented_gt_faces, _ = igl.bfs_orient(self.gt_faces.cpu().numpy())
    
        # save gt mesh;

        mesh = trimesh.base.Trimesh(
            vertices=self.gt_verts.detach().cpu().numpy(),
            faces=oriented_gt_faces
        )
        mesh_path = os.path.join(logdir, "gt_mesh.obj")
        mesh.export(mesh_path)

        '''
        Points that we optimize for WDT.
        The first N points are the ground truth vertices with weight 1.
        The rest of the points are used to remove undesirable faces.
        '''
        self.points: WPoints = self.init_points()
        
        self.lr = 1e-4
        self.update_bound = 1e-5
        self.pd_knn_k = 8
        
        self.ratio_add_points = 0.1
        self.num_random_add_points = 1000

        self.writer = SummaryWriter(logdir)

    def num_gt_verts(self):
        return self.gt_verts.shape[0]

    def num_points(self):
        return self.points.positions.shape[0]

    def init_points(self):
        '''
        Initialize weighted points, which are comprised of only
        [gt_verts]. 
        '''
        point_positions = self.gt_verts.clone()
        point_weights = th.ones((self.num_gt_verts(),), dtype=th.float32, device=device)
        return WPoints(point_positions, point_weights)

    def refresh_optimizer(self):
        point_positions = self.points.positions.clone()
        point_weights = self.points.weights.clone()
        point_positions.requires_grad = True
        point_weights.requires_grad = True
        optimizer = th.optim.Adam([point_positions, point_weights], lr=self.lr)
        points = WPoints(point_positions, point_weights)
        return optimizer, points

    def get_real_faces(self, points: WPoints):
        '''
        Run WDT for current points and retrieve real faces.
        '''
        wdt_result = DiffDT.CGAL_WDT(
            points,
            True,
            False,
        )
        curr_faces = extract_faces(wdt_result.tets_point_id)

        '''
        Find faces in [curr_faces] that should not exist.
        '''
        # find out faces in [curr_faces] that are comprised of only
        # ground truth vertices;
        is_real_face = th.all(curr_faces < self.num_gt_verts(), dim=-1)
        real_faces = curr_faces[is_real_face]

        return real_faces, wdt_result
    
    def update_iwdt1_faces(self, 
                        iwdt1_faces: th.Tensor,
                        real_faces: th.Tensor,
                        pd: PDStruct = None,
                        use_knn_faces: bool = False):
        '''
        Update list of faces for iwdt1 using current solid faces.
        If [use_knn_faces] is True, then we also collect knn faces for update.
        '''
        if real_faces is not None:
            iwdt1_faces = th.cat([iwdt1_faces, real_faces], dim=0)
            iwdt1_faces = th.sort(iwdt1_faces, dim=-1)[0]
            iwdt1_faces = th.unique(iwdt1_faces, dim=0)

        if use_knn_faces:
            assert pd is not None, "PDStruct is not provided"
            knn_faces = pd_knn_faces(pd, self.pd_knn_k)
            iwdt1_faces = th.cat([iwdt1_faces, knn_faces], dim=0)
            iwdt1_faces = th.sort(iwdt1_faces, dim=-1)[0]
            iwdt1_faces = th.unique(iwdt1_faces, dim=0)

        # only consider real points;
        iwdt1_faces = iwdt1_faces[th.all(iwdt1_faces < self.num_gt_verts(), dim=-1)]

        return iwdt1_faces

    def refresh_points(self):
        '''
        Run WDT on current points, and add points on undesriable
        faces to remove them. Also, remove redundant points that 
        do not have power cells.
        '''
        print("===================================== Refreshing points...")
        with th.no_grad():
            curr_real_faces, wdt_result = self.get_real_faces(self.points)

            '''
            Remove redundant points.
            '''
            num_points_before = self.points.positions.shape[0]
            
            rp_idx = th.unique(wdt_result.tets_point_id)
            assert th.count_nonzero(rp_idx < self.num_gt_verts()) == self.num_gt_verts(), \
                "Ground truth points are included in redundant points"
            
            n_points_positions = self.points.positions[rp_idx.to(dtype=th.long)]
            n_points_weights = self.points.weights[rp_idx.to(dtype=th.long)]
            
            self.points.positions = n_points_positions
            self.points.weights = n_points_weights

            num_points_after = self.points.positions.shape[0]

            print("Number of points before removal: ", num_points_before)
            print("Number of points after removal: ", num_points_after)
            print()
            
            '''
            Find faces in [curr_faces] that should not exist.
            '''
            # get undesirable faces by removing [gt_faces] from [curr_real_faces];
            undesirable_real_faces = tensor_subtract_1(curr_real_faces, self.gt_faces)

            print("Number of desirable faces: ", curr_real_faces.shape[0] - undesirable_real_faces.shape[0])
            print("Number of undesirable faces: ", undesirable_real_faces.shape[0])

            # get points on undesirable faces;
            # [# F, 3, 3]
            undesirable_real_faces_points_positions = self.points.positions[undesirable_real_faces.to(dtype=th.long)]   

            # add random points on undesirable faces;
            # first sample random barycentric coords;
            num_new_points = undesirable_real_faces.shape[0]
            new_points_bary = th.zeros((num_new_points, 3), dtype=th.float32, device=device)
            new_points_bary[:, 0] = th.rand((num_new_points,), dtype=th.float32, device=device)
            new_points_bary[:, 1] = th.rand((num_new_points,), dtype=th.float32, device=device) * (1 - new_points_bary[:, 0])
            new_points_bary[:, 2] = 1 - new_points_bary[:, 0] - new_points_bary[:, 1]
            
            new_points_positions = undesirable_real_faces_points_positions[:, 0] * new_points_bary[:, 0].unsqueeze(-1) + \
                undesirable_real_faces_points_positions[:, 1] * new_points_bary[:, 1].unsqueeze(-1) + \
                undesirable_real_faces_points_positions[:, 2] * new_points_bary[:, 2].unsqueeze(-1)
            new_points_weights = th.ones((num_new_points,), dtype=th.float32, device=device)

            # use k-means clustering to select [ratio_add_points * new_points_positions] points;
            num_new_points_kmeans = int(self.ratio_add_points * num_new_points)
            while True:
                try:
                    if num_new_points_kmeans > 0:    
                        kmeans = KMeans(n_clusters=num_new_points_kmeans)
                        kmeans.fit(new_points_positions, None)
                        new_points_kmeans_positions = kmeans.centroids
                        new_points_kmeans_weights = th.ones((num_new_points_kmeans,), dtype=th.float32, device=device)
                    else:
                        new_points_kmeans_positions = th.empty((0, 3), dtype=th.float32, device=device)
                        new_points_kmeans_weights = th.empty((0,), dtype=th.float32, device=device)
                except Exception as e:
                    num_new_points_kmeans = num_new_points_kmeans // 2
                    continue

                break

            # select [num_random_add_points] random points;
            if self.num_random_add_points < num_new_points:
                random_select = th.randperm(num_new_points, device=device)[:self.num_random_add_points]
                new_points_positions = new_points_positions[random_select]
                new_points_weights = th.ones((self.num_random_add_points,), dtype=th.float32, device=device)

            # combine all new points;
            new_points_positions = th.cat([new_points_positions, new_points_kmeans_positions], dim=0)
            new_points_weights = th.cat([new_points_weights, new_points_kmeans_weights], dim=0)

            print("Number of points before addition: ", self.num_points())

            # add new points to [self.points];
            self.points.positions = th.cat([self.points.positions, new_points_positions], dim=0)
            self.points.weights = th.cat([self.points.weights, new_points_weights], dim=0)

            print("Number of points after addition: ", self.num_points())
        
        print("=====================================")

    def save(self, 
            save_dir: str,
            points_positions: th.Tensor,
            points_weights: th.Tensor,
            time: float):
        
        with th.no_grad():
            save_path = save_dir
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # only gt vertices have real flags;
            points_reals = th.ones((len(points_positions),), dtype=th.float32, device=device)
            points_reals[self.num_gt_verts():] = 0.0
            save_data = DMesh(points_positions, points_weights, points_reals)
            save_data.save(os.path.join(save_path, "points.pth"))

            time_path = os.path.join(save_path, "time_sec.txt")
            with open(time_path, "w") as f:
                f.write("{}".format(time))

            # save mesh;
            mesh = save_data.get_mesh(False)
            mesh_path = os.path.join(save_path, "mesh.obj")
            mesh.export(mesh_path)

    def save_step(self, 
            step: int, 
            points_positions: th.Tensor,
            points_weights: th.Tensor,
            time: float):

        save_path = os.path.join(self.writer.log_dir, "save/step_{}".format(step))
        self.save(save_path, points_positions, points_weights, time)
        
    def optimize(self, 
                num_steps: int, 
                save_steps: int,
                refresh_points_interval: int = 5000,
                max_perturb: float = 3e-3):
        '''
        Optimize the points to fit the ground truth mesh.
        '''

        refresh_points_interval: int = refresh_points_interval
        iwdt1_knn_interval: int = 100

        sso_dist_thresh = 3e-4
        delta_thresh = 3e-4

        # faces that will be fed into iwdt1;
        # it is dynamically updated during optimization;
        iwdt1_faces: th.Tensor = None
        
        optimizer, points = self.refresh_optimizer()
        bar = tqdm(range(num_steps))

        start_time = time.time()
        best_recovery_ratio = 0.0
        best_false_positive_ratio = 1.0
        
        for step in bar:

            # refresh points: add points to remove undesirable faces
            # and remove redundant points without power cells;
            if step % refresh_points_interval == 0:
                with th.no_grad():
                    # save before refreshing points;
                    self.save(
                        os.path.join(self.writer.log_dir, "save/step_{}_before_refresh".format(step)),
                        points.positions,
                        points.weights,
                        time.time() - start_time
                    )
                    self.refresh_points()
                    optimizer, points = self.refresh_optimizer()

                    # save after refreshing points;
                    self.save(
                        os.path.join(self.writer.log_dir, "save/step_{}_after_refresh".format(step)),
                        points.positions,
                        points.weights,
                        time.time() - start_time
                    )

                    # since points are added, we need to clear iwdt1 faces;
                    # since we need to evaluate every face in [self.gt_faces],
                    # we initialize it with [self.gt_faces];
                    iwdt1_faces = self.gt_faces.clone()

            # run WDT;
            curr_real_faces, wdt_result = self.get_real_faces(points)

            # construct PD;
            pd = PDStruct.forward(points, wdt_result.tets_point_id)

            # update iwdt1 faces;
            if step % iwdt1_knn_interval == 0:
                iwdt1_faces = self.update_iwdt1_faces(iwdt1_faces, None, pd, True)
            iwdt1_faces = self.update_iwdt1_faces(iwdt1_faces, curr_real_faces)

            # sso;
            sso = DiffDT.sso(
                pd,
                sso_dist_thresh,
            )

            # iwdt0: get scores for existing faces;
            iwdt0_result = DiffDT.iwdt0(
                pd,
                sso,
                device
            )

            # iwdt1: get scores for non-existing faces;
            iwdt1_result = DiffDT.iwdt1(
                pd,
                iwdt1_faces.to(dtype=th.int32),
                device
            )

            '''
            Compute loss.
            '''
            loss = 0

            e_faces = pd.pd_edges.points_id             # existing faces
            e_faces_d = iwdt0_result.pd_edge_delta      # existing faces delta

            n_faces = iwdt1_result.faces                # non-existing faces
            n_faces_d = iwdt1_result.per_face_delta     # non-existing faces delta

            # pick faces that we care about;
            # n_faces are already comprised of only ground truth vertices;
            validity = th.all(e_faces < self.num_gt_verts(), dim=-1)
            e_faces = e_faces[validity]                 # existing faces comprised of only ground truth vertices
            e_faces_d = e_faces_d[validity]
            assert th.all(n_faces < self.num_gt_verts()), \
                "Non-existing faces are not comprised of only ground truth vertices"

            # classify query faces;
            e_faces_desired, _ = tensor_intersect_idx(
                e_faces,
                self.gt_faces
            )

            n_faces_desired, _ = tensor_intersect_idx(
                n_faces,
                self.gt_faces
            )

            '''
            Desirable faces: Have to maximize scores for them.
            '''
            # desirable existing faces;
            desired_e_faces = e_faces[e_faces_desired]
            # clamp because they are already satisfied;
            desired_e_faces_d = th.clamp(e_faces_d[e_faces_desired], max=delta_thresh)
            loss = loss - th.sum(desired_e_faces_d)
            
            # desirable non-existing faces;
            desired_n_faces = n_faces[n_faces_desired]
            desired_n_faces_d = n_faces_d[n_faces_desired]      # do not clamp, since we want to maximize them
            loss = loss - th.sum(desired_n_faces_d)
            
            '''
            Undesirable faces: Have to minimize scores for them.
            '''
            # undesirable existing faces;
            undesired_e_faces = e_faces[~e_faces_desired]
            undesired_e_faces_d = e_faces_d[~e_faces_desired]    # do not clamp, since we want to minimize them
            loss = loss + th.sum(undesired_e_faces_d)

            # undesirable non-existing faces;
            undesired_n_faces = n_faces[~n_faces_desired]
            # clamp because they are already satisfied;
            undesired_n_faces_d = th.clamp(n_faces_d[~n_faces_desired], min=-delta_thresh)
            loss = loss + th.sum(undesired_n_faces_d)

            '''
            Update points.
            '''
            with th.no_grad():
                point_positions_before_update = points.positions.clone()
                point_weights_before_update = points.weights.clone()
            optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(
                [points.positions, points.weights], 
                self.update_bound / self.lr)
            optimizer.step()

            '''
            Bounding.
            '''
            with th.no_grad():
                points.positions.data = points.positions.data.clamp_(
                    point_positions_before_update - self.update_bound,
                    point_positions_before_update + self.update_bound
                )
                points.weights.data = points.weights.data.clamp_(
                    point_weights_before_update - self.update_bound,
                    point_weights_before_update + self.update_bound
                )
                points.weights.data = th.clamp(points.weights.data, min=0.0, max=1.0)

                # do not perturb ground truth vertices;
                points.positions.data[:self.num_gt_verts()] = points.positions.data[:self.num_gt_verts()].clamp_(
                    self.gt_verts - max_perturb,
                    self.gt_verts + max_perturb
                )
                points.weights.data[:self.num_gt_verts()] = th.ones_like(points.weights[:self.num_gt_verts()])

                # update points;
                self.points.positions = points.positions.clone()
                self.points.weights = points.weights.clone()

                assert th.all(self.points.weights[:self.num_gt_verts()] == 1.0), \
                    "Ground truth vertices are not weighted 1.0"

            '''
            Logging
            '''
            with th.no_grad():
                self.writer.add_scalar("loss/loss", loss, step)
                
                num_desired_faces = len(self.gt_faces)
                num_desired_e_faces = desired_e_faces.shape[0]
                recovery_ratio = num_desired_e_faces / num_desired_faces

                num_e_faces = e_faces.shape[0]
                num_undesired_e_faces = undesired_e_faces.shape[0]
                if num_e_faces > 0:
                    false_positive_ratio = num_undesired_e_faces / num_e_faces
                else:
                    false_positive_ratio = 1.0

                self.writer.add_scalar("info/recovery_ratio", recovery_ratio, step)
                self.writer.add_scalar("info/false_positive_ratio", false_positive_ratio, step)
                self.writer.add_scalar("info/num_faces", num_e_faces, step)

                bar.set_description("recovery_ratio: {:.4f}, false_positive_ratio: {:.4f}".format(
                    recovery_ratio, false_positive_ratio
                ))

            '''
            Saving
            '''
            if step % save_steps == 0 or step == num_steps - 1:
                self.save_step(
                    step,
                    point_positions_before_update,
                    point_weights_before_update,
                    time.time() - start_time
                )

                self.save(
                    os.path.join(self.writer.log_dir, "save/last"),
                    point_positions_before_update,
                    point_weights_before_update,
                    time.time() - start_time
                )

            if recovery_ratio > best_recovery_ratio:
                best_recovery_ratio = recovery_ratio
                self.save(
                    os.path.join(self.writer.log_dir, "save/best_recovery_ratio"),
                    point_positions_before_update,
                    point_weights_before_update,
                    time.time() - start_time
                )
                with open(os.path.join(self.writer.log_dir, "save/best_recovery_ratio.txt"), "w") as f:
                    f.write("recovery ratio: {}\n".format(recovery_ratio))
                    f.write("false positive ratio: {}\n".format(false_positive_ratio))

            if false_positive_ratio < best_false_positive_ratio:
                best_false_positive_ratio = false_positive_ratio
                self.save(
                    os.path.join(self.writer.log_dir, "save/best_false_positive_ratio"),
                    point_positions_before_update,
                    point_weights_before_update,
                    time.time() - start_time
                )
                with open(os.path.join(self.writer.log_dir, "save/best_false_positive.txt"), "w") as f:
                    f.write("recovery ratio: {}\n".format(recovery_ratio))
                    f.write("false positive ratio: {}\n".format(false_positive_ratio))

            if recovery_ratio > 0.999 and false_positive_ratio < 0.01:
                print("Perfect recovery ratio and false positive ratio achieved!")
                print("Saving...")
                self.save(
                    os.path.join(self.writer.log_dir, "save/perfect"),
                    point_positions_before_update,
                    point_weights_before_update,
                    time.time() - start_time
                )
                print("Saved!")
                break

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="exp/config/exp_1/bunny.yaml")
    parser.add_argument("--no-log-time", action='store_true')
    args = parser.parse_args()

    # load settings from yaml file;
    with open(args.config, "r") as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
    logdir = settings['log_dir']
    if not args.no_log_time:
        logdir = logdir + time.strftime("/%Y_%m_%d_%H_%M_%S")
    logdir = setup_logdir(logdir)

    # save settings;
    with open(os.path.join(logdir, "config.yaml"), "w") as f:
        yaml.dump(settings, f)

    th.random.manual_seed(1)
    device = settings['device']

    '''
    Ground truth mesh
    '''
    mesh_path = settings['mesh']
    verts, faces = import_mesh(mesh_path, device, scale=0.8)
    
    print("===== Ground truth mesh =====")
    print("Number of vertices: ", verts.shape[0])
    print("Number of faces: ", faces.shape[0])
    print("=============================")

    num_step = settings['args']['num_step']
    save_step = settings['args']['save_step']
    refresh_points_step = settings['args']['refresh_points_step']
    gt_max_perturb = float(settings['args']['gt_max_perturb'])

    optimizer = GtmeshOptimizer(verts, faces, logdir)
    optimizer.optimize(num_step,
                    save_step,
                    refresh_points_step,
                    gt_max_perturb)