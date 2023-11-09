# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Implementation of K-Planes (https://sarafridov.github.io/K-Planes/).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import cv2
import random
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.losses import MSELoss, distortion_loss, interlevel_loss
from nerfstudio.model_components.ray_samplers import (
    ProposalNetworkSampler,
    UniformLinDispPiecewiseSampler,
    UniformSampler,
)
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
    FeatureRenderer
)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, misc

from .kplanes_field import KPlanesDensityField, KPlanesField
from .LimitGradLayer import LimitGradLayer
from .decoder import ImageDecoder

@dataclass
class KPlanesModelConfig(ModelConfig):
    """K-Planes Model Config"""

    _target: Type = field(default_factory=lambda: KPlanesModel)

    near_plane: float = 2.0
    """How far along the ray to start sampling."""

    far_plane: float = 6.0
    """How far along the ray to stop sampling."""

    grid_base_resolution: List[int] = field(default_factory=lambda: [128, 128, 128])
    """Base grid resolution."""

    grid_feature_dim: int = 32
    """Dimension of feature vectors stored in grid."""

    multiscale_res: List[int] = field(default_factory=lambda: [1, 2, 4])
    """Multiscale grid resolutions."""

    is_contracted: bool = False
    """Whether to use scene contraction (set to true for unbounded scenes)."""

    concat_features_across_scales: bool = True
    """Whether to concatenate features at different scales."""

    linear_decoder: bool = False
    """Whether to use a linear decoder instead of an MLP."""

    linear_decoder_layers: Optional[int] = 1
    """Number of layers in linear decoder"""

    # proposal sampling arguments
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""

    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""

    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"num_output_coords": 8, "resolution": [64, 64, 64]},
            {"num_output_coords": 8, "resolution": [128, 128, 128]},
        ]
    )
    """Arguments for the proposal density fields."""

    num_proposal_samples: Optional[Tuple[int]] = (256, 128)
    """Number of samples per ray for each proposal network."""

    num_samples: Optional[int] = 48
    """Number of samples per ray used for rendering."""

    single_jitter: bool = False
    """Whether use single jitter or not for the proposal networks."""

    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps."""

    proposal_update_every: int = 5
    """Sample every n steps after the warmup."""

    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""

    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""

    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""

    appearance_embedding_dim: int = 0
    """Dimension of appearance embedding. Set to 0 to disable."""

    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""

    background_color: Literal["random", "last_sample", "black", "white"] = "white"
    """The background color as RGB."""

    loss_coefficients: Dict[str, float] = to_immutable_dict(
        {
            "img": 1.0,
            "interlevel": 1.0,
            "distortion": 0.001,
            "plane_tv": 0.0001,
            "plane_tv_proposal_net": 0.0001,
            "l1_time_planes": 0.0001,
            "l1_time_planes_proposal_net": 0.0001,
            "time_smoothness": 0.1,
            "time_smoothness_proposal_net": 0.001,
        }
    )
    """Loss coefficients."""


class KPlanesModel(Model):
    config: KPlanesModelConfig
    """K-Planes model

    Args:
        config: K-Planes configuration to instantiate model
    """

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.is_contracted:
            scene_contraction = SceneContraction(order=float("inf"))
        else:
            scene_contraction = None

        # Fields
        self.field = KPlanesField(
            self.scene_box.aabb,
            num_images=self.num_train_data,
            grid_base_resolution=self.config.grid_base_resolution,
            grid_feature_dim=self.config.grid_feature_dim,
            concat_across_scales=self.config.concat_features_across_scales,
            multiscale_res=self.config.multiscale_res,
            spatial_distortion=scene_contraction,
            appearance_embedding_dim=self.config.appearance_embedding_dim,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            linear_decoder=self.config.linear_decoder,
            linear_decoder_layers=self.config.linear_decoder_layers,
        )

        self.img_save_counter = 0
        self.vol_tvs = None
        self.cosine_idx = 0
        self.vol_tv_mult = 0.0001
        self.conv_vol_tv_mult = 0.0001
        self.mask_layer = LimitGradLayer.apply
        self.conv_switch = 500

        self.prev_image = None
        
        rot_angs = torch.nn.Parameter(torch.zeros(3,153))
        pos_diff = torch.nn.Parameter(torch.zeros(153,3))
        self.pos_idx = 0
        self.camera_pose_delts = torch.nn.ParameterList([rot_angs,pos_diff])

        self.decoder = ImageDecoder((45,80),(720,1280))
        
        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        '''
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = KPlanesDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                linear_decoder=self.config.linear_decoder,
                **prop_net_args,
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = KPlanesDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    linear_decoder=self.config.linear_decoder,
                    **prop_net_args,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])
        '''
        # Samplers
        def update_schedule(step):
            return np.clip(
                np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
                1,
                self.config.proposal_update_every,
            )

        if self.config.is_contracted:
            initial_sampler = UniformLinDispPiecewiseSampler(single_jitter=self.config.single_jitter)
        else:
            initial_sampler = UniformSampler(single_jitter=self.config.single_jitter)

        self.proposal_sampler = UniformSampler(single_jitter=self.config.single_jitter)
        '''
        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_samples,
            num_proposal_samples_per_ray=self.config.num_proposal_samples,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )
        '''

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # renderers
        self.renderer_rgb = FeatureRenderer(background_color=self.config.background_color)
        #self.renderer_accumulation = AccumulationRenderer()
        #self.renderer_depth = DepthRenderer()

        # losses
        self.rgb_loss = MSELoss()
        self.similarity_loss = torch.nn.CosineSimilarity(dim=1)
        self.grid_similarity_loss = torch.nn.CosineSimilarity(dim=0)        
        self.conv_mlp_loss = torch.nn.CrossEntropyLoss()
        self.mask_layer = LimitGradLayer.apply
        
        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.temporal_distortion = len(self.config.grid_base_resolution) == 4  # for viewer

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {
            #"proposal_networks": list(self.proposal_networks.parameters()),
            "fields": list(self.field.parameters()),
            "decoder": list(self.decoder.parameters()),
            #"pose_delts": self.camera_pose_delts,
        }
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)
                bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        return callbacks

    def quat_mult(self,q1,q2):
        a = (q1[:,0]*q2[:,0] - q1[:,1]*q2[:,1] - q1[:,2]*q2[:,2] - q1[:,3]*q2[:,3]).unsqueeze(-1)
        b = (q1[:,0]*q2[:,1] - q1[:,1]*q2[:,0] - q1[:,2]*q2[:,3] - q1[:,3]*q2[:,2]).unsqueeze(-1)
        c = (q1[:,0]*q2[:,2] - q1[:,1]*q2[:,3] - q1[:,2]*q2[:,0] - q1[:,3]*q2[:,1]).unsqueeze(-1)
        d = (q1[:,0]*q2[:,3] - q1[:,1]*q2[:,2] - q1[:,2]*q2[:,1] - q1[:,3]*q2[:,0]).unsqueeze(-1)        

        p = torch.cat([a,b,c,d],dim=1)

        if torch.sum(torch.abs(p[:,0])) > 0.00001:
            print('WHATTTTTT')
            exit(-1)
        
        return p

    def get_rot_mat_torch(self,angs): #roll,yaw,pitch):
        #roll = torch.clip(roll,-3,3) * torch.pi / 180
        #yaw = yaw * torch.pi / 180
        #pitch = torch.clip(pitch,-75,-40) * torch.pi / 180
        roll = torch.zeros_like(angs[0])
        yaw = angs[1]
        pitch = angs[2]
        rz = torch.stack([torch.stack([torch.cos(roll), -torch.sin(roll), torch.zeros_like(roll)]),
                          torch.stack([torch.sin(roll), torch.cos(roll), torch.zeros_like(roll)]),
                          torch.stack([torch.zeros_like(roll),torch.zeros_like(roll),torch.ones_like(roll)])])
        ry = torch.stack([torch.stack([torch.cos(yaw), torch.zeros_like(roll), torch.sin(yaw)]),
                          torch.stack([torch.zeros_like(roll), torch.ones_like(roll), torch.zeros_like(roll)]),
                          torch.stack([-torch.sin(yaw), torch.zeros_like(roll), torch.cos(yaw)])])
        rx = torch.stack([torch.stack([torch.ones_like(roll), torch.zeros_like(roll), torch.zeros_like(roll)]),
                          torch.stack([torch.zeros_like(roll), torch.cos(pitch), -torch.sin(pitch)]),
                          torch.stack([torch.zeros_like(roll), torch.sin(pitch), torch.cos(pitch)])])
        

        rz = rz.permute(2,0,1)
        ry = ry.permute(2,0,1)
        rx = rx.permute(2,0,1)

        R = torch.matmul(rz.squeeze(),torch.matmul(ry.squeeze(),rx.squeeze()))

        return R
    
    def get_outputs(self, ray_bundle: RayBundle):
        density_fns = self.density_fns
        if ray_bundle.times is not None:
            density_fns = [functools.partial(f, times=ray_bundle.times) for f in density_fns]

        orig_shape = None
        #print("BUNDLE: {}".format(ray_bundle.shape))        
        if len(ray_bundle.shape) > 1:
            orig_shape = ray_bundle.shape
            ray_bundle = ray_bundle.reshape(orig_shape[0]*orig_shape[1])
            #print("NEW BUNDLE: {}".format(ray_bundle.shape))

        '''
        rot_angs = self.camera_pose_delts[0]
        pos_diff = self.camera_pose_delts[1]
        
        R = self.get_rot_mat_torch(torch.clip(rot_angs,-0.01,0.01))
        selected_R = R[ray_bundle.camera_indices.squeeze()]
        new_dirs = torch.matmul(selected_R,ray_bundle.directions.unsqueeze(-1)).squeeze()
        selected_delts = pos_diff[ray_bundle.camera_indices.squeeze()]
        new_origins = ray_bundle.origins + torch.clip(selected_delts,-0.25,0.25)
        ray_bundle.origins = new_origins
        ray_bundle.directions = new_dirs #/ torch.norm(new_dirs,2,1).unsqueeze(-1)
        self.rot_angs = rot_angs[:,ray_bundle.camera_indices.squeeze()].transpose(-1,-2)
        self.pos_diff = pos_diff[ray_bundle.camera_indices.squeeze(),:]
        

        self.pos_idx = (self.pos_idx + 1) % 1000
        if self.pos_idx == 0:
            print(rot_angs)
            print(pos_diff)
        '''
        weights_list = []
        ray_samples_list = []
        #ray_samples, weights_list, ray_samples_list = self.proposal_sampler(
        ray_samples = self.proposal_sampler(            
            ray_bundle, 8 #, density_fns=density_fns
        )
        field_outputs = self.field(ray_samples)
        
        #weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights = field_outputs[FieldHeadNames.DENSITY]
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        #depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        #accumulation = self.renderer_accumulation(weights=weights)

        self.vol_tvs = field_outputs["vol_tvs"]
        if orig_shape is None:
            rgb_image = rgb.reshape((45,80,512)) #.transpose(2,0,1)
        else:
            rgb_image = rgb.reshape(orig_shape + (512,))
            
        rgb_image = rgb_image.permute(2,0,1)
        reconst_image = self.decoder(rgb_image.unsqueeze(0)).permute(0,2,3,1)

        self.img_save_counter = (self.img_save_counter + 1) % 50
        if self.img_save_counter == 0:
            #with torch.no_grad():
            #    zero_reconst_image = self.decoder(torch.zeros_like(rgb_image).unsqueeze(0)).permute(0,2,3,1)
            reconst_image_np = reconst_image.squeeze().detach().cpu().numpy()
            reconst_image_np = (reconst_image_np*255).astype(np.uint8)
            reconst_image_np = reconst_image_np[:,:,[2,1,0]]
            #zero_reconst_image_np = zero_reconst_image.squeeze().detach().cpu().numpy()
            #zero_reconst_image_np = (zero_reconst_image_np*255).astype(np.uint8)
            #zero_reconst_image_np = zero_reconst_image_np[:,:,[2,1,0]]
            #print("SIZE: {}".format(reconst_image_np.shape))
            #print("MEAN: {}".format(reconst_image_np.mean()))            
            cv2.imwrite("test_feature_img.png",reconst_image_np)
            #cv2.imwrite("test_feature_img_zero.png",zero_reconst_image_np)        

        outputs = {
            "rgb": reconst_image.reshape(-1,3), #rgb,
            #"accumulation": accumulation,
            #"depth": depth,
        }

        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        #for i in range(self.config.num_proposal_iterations):
        #    outputs[f"prop_depth_{i}"] = self.renderer_depth(
        #        weights=weights_list[i], ray_samples=ray_samples_list[i]
        #    )

        return outputs

    def get_metrics_dict(self, outputs, batch):
        #image = batch["image"].to(self.device)
        image = batch["full_image"].to(self.device)        

        metrics_dict = {
            "psnr": self.psnr(outputs["rgb"], image.reshape(-1,3))
        }
        if self.training:
            #metrics_dict["interlevel"] = interlevel_loss(outputs["weights_list"], outputs["ray_samples_list"])
            #metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])

            #prop_grids = [p.grids.plane_coefs for p in self.proposal_networks]
            field_grids = [g.plane_coefs for g in self.field.grids]

            metrics_dict["plane_tv"] = space_tv_loss(field_grids)
            #metrics_dict["plane_tv_proposal_net"] = space_tv_loss(prop_grids)

            if len(self.config.grid_base_resolution) == 4:
                metrics_dict["l1_time_planes"] = l1_time_planes(field_grids)
                #metrics_dict["l1_time_planes_proposal_net"] = l1_time_planes(prop_grids)
                metrics_dict["time_smoothness"] = time_smoothness(field_grids)
                #metrics_dict["time_smoothness_proposal_net"] = time_smoothness(prop_grids)

        return metrics_dict

    def get_dist_loss(self, outputs):
        weights = outputs['weights_list'][-1]
        ray_samples = outputs['ray_samples_list'][-1]

        num_samples = weights.shape[1]

        num_rays = 1000
        dist_loss = 0.0
        for i in range(num_samples-1):
            for j in range(num_samples-1):
                if i != j:
                    dist_loss += weights[:num_rays,i,0]*weights[:num_rays,j,0]*torch.abs((ray_samples.spacing_starts[:num_rays,i,0] +
                                                                                  ray_samples.spacing_starts[:num_rays,i+1,0]) -
                                                                                 (ray_samples.spacing_starts[:num_rays,j,0] +
                                                                                  ray_samples.spacing_starts[:num_rays,j+1,0])) / 2

        for i in range(0,num_samples-1):
            dist_loss += (weights[:num_rays,i,0]**2)*(ray_samples.spacing_starts[:num_rays,i+1,0] - ray_samples.spacing_starts[:num_rays,i,0]) / 3

        return dist_loss
                    
    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        image = batch["full_image"].to(self.device)

        #if self.prev_image is None:
        #    self.prev_image = image
        #else:
        #    print("DIFF: {}".format(torch.sum(torch.abs(image - self.prev_image))))
            
        #mask_image = batch["time_mask"].to(image.dtype)

        #image_mask = torch.sum(mask_image,-1).unsqueeze(-1).to(image.dtype)
        #print("SUM: {}".format(torch.sum(image_mask)))
        #exit(-1)
        #image_mask_bool = (image_mask > 10).to(image.dtype)
        #non_zero_idxs = torch.nonzero(image_mask_bool)[:,0]

        #image_diff = torch.abs(image - outputs["rgb"].detach()).mean(1).unsqueeze(-1)#[non_zero_idxs]

        #mask_image = mask_image / 255.
        #image = image*(1 - image_mask_bool) + mask_image*image_mask_bool
        #image = (1 - image_mask) + red_image*image_mask
        loss_dict = {"rgb": self.rgb_loss(image.reshape(-1,3), outputs["rgb"])}
        #loss_dict = {"rgb": self.rgb_loss(self.field.masks*image, outputs["rgb"])}        
        self.cosine_idx += 1

        if self.training:
            for key in self.config.loss_coefficients:
                if key in metrics_dict:
                    loss_dict[key] = metrics_dict[key].clone()

            loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)

            outputs_lst = self.vol_tvs
            vol_tvs = 0.0
            time_mask_loss = 0.0
            #time_mask_alt = image_mask_bool.unsqueeze(-1).expand(-1,10,-1)
            num_comps = outputs_lst[0][0].shape[-1]

            #for output_idx,_outputs in enumerate(outputs_lst):
            #    time_mask_loss += self.rgb_loss((_outputs[2][:,num_comps:]).reshape(-1,10,1),time_mask_alt)
            #    time_mask_loss += self.rgb_loss((_outputs[4][:,num_comps:]).reshape(-1,10,1),time_mask_alt)
            #    time_mask_loss += self.rgb_loss((_outputs[5][:,num_comps:]).reshape(-1,10,1),time_mask_alt)


            
            #field_grids = [g.plane_coefs for g in self.field.grids]
            #grid_norm = 0.0
            local_vol_tvs = 0.0

            for odx,_outputs in enumerate(outputs_lst):
                #continue
                #for tdx0,tdx1,tdx2,tdx3 in [(0,2,4,6),(1,2,5,7),(3,4,5,8)]:
                #    spatial = _outputs[tdx0].reshape(-1,48,32)
                #    temporal = (_outputs[tdx3]*_outputs[tdx1][:,:num_comps]*_outputs[tdx2][:,:num_comps]).reshape(-1,48,32).transpose(-1,-2)
                #    local_vol_tvs += torch.abs(torch.matmul(spatial,temporal)).mean()
                local_vol_tvs += torch.abs(self.similarity_loss(_outputs[0],_outputs[2]*_outputs[4]*_outputs[6])).mean()
                local_vol_tvs += torch.abs(self.similarity_loss(_outputs[1],_outputs[2]*_outputs[5]*_outputs[7])).mean()
                local_vol_tvs += torch.abs(self.similarity_loss(_outputs[3],_outputs[4]*_outputs[5]*_outputs[8])).mean()
                #local_vol_tvs += torch.abs(self.similarity_loss(_outputs[0],_outputs[6])).mean()
                #local_vol_tvs += torch.abs(self.similarity_loss(_outputs[1],_outputs[7])).mean()
                #local_vol_tvs += torch.abs(self.similarity_loss(_outputs[3],_outputs[8])).mean()

                
            #for grid_idx,grids in enumerate(field_grids):
            #    continue
            #    grid_norm += torch.abs(1 - torch.norm(grids[0],2,0)).mean()
            #    grid_norm += torch.abs(1 - torch.norm(grids[1],2,0)).mean()
            #    grid_norm += torch.abs(1 - torch.norm(grids[3],2,0)).mean()

            #loss_dict["camera_delts"] = (self.mask_layer(torch.abs(self.rot_angs)*image_diff,image_mask_bool)).mean()
            #loss_dict["camera_delts"] += (self.mask_layer(torch.abs(self.pos_diff)*image_diff,image_mask_bool)).mean()
            #loss_dict["camera_delts"] = (torch.abs(self.rot_angs[non_zero_idxs])*image_diff).mean()
            #loss_dict["camera_delts"] += (torch.abs(self.pos_diff[non_zero_idxs])*image_diff).mean()
            #loss_dict["camera_delts"] = (torch.abs(self.rot_angs*image_diff)).mean()
            #loss_dict["camera_delts"] += (torch.abs(self.pos_diff*image_diff)).mean()
            loss_dict["local_vol_tvs"] = local_vol_tvs / (3*len(outputs_lst))
            #loss_dict["grid_norm"] = 0.01*grid_norm / (6*len(outputs_lst))
            
            #loss_dict["time_masks"] = time_mask_loss
            
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)

        rgb = outputs["rgb"]
        #acc = colormaps.apply_colormap(outputs["accumulation"])
        #depth = colormaps.apply_depth_colormap(outputs["depth"], accumulation=outputs["accumulation"])

        combined_rgb = torch.cat([image, rgb], dim=1)
        #combined_acc = torch.cat([acc], dim=1)
        #combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        # all of these metrics will be logged as scalars
        metrics_dict = {
            "psnr": float(self.psnr(image, rgb).item()),
            "ssim": float(self.ssim(image, rgb)),
            "lpips": float(self.lpips(image, rgb))
        }
        images_dict = {"img": combined_rgb} #, "accumulation": combined_acc, "depth": combined_depth}

        #for i in range(self.config.num_proposal_iterations):
        #    key = f"prop_depth_{i}"
        #    prop_depth_i = colormaps.apply_depth_colormap(
        #        outputs[key],
        #        accumulation=outputs["accumulation"],
        #    )
        #    images_dict[key] = prop_depth_i

        return metrics_dict, images_dict


def compute_plane_tv(t: torch.Tensor, only_w: bool = False) -> float:
    """Computes total variance across a plane.

    Args:
        t: Plane tensor
        only_w: Whether to only compute total variance across w dimension

    Returns:
        Total variance
    """
    _, h, w = t.shape
    w_tv = torch.square(t[..., :, 1:] - t[..., :, : w - 1]).mean()
    
    if only_w:
        #w_tv2 = torch.square(t[..., 1:, 1:] - t[..., : h - 1, : w - 1]).mean()
        #w_tv3 = torch.square(t[..., : h - 1, 1:] - t[..., 1:, : w - 1]).mean()    
        
        return w_tv #+ 0.5*(w_tv2 + w_tv3)

    h_tv = torch.square(t[..., 1:, :] - t[..., : h - 1, :]).mean()
    #h_tv2 = torch.square(t[..., 1:, 1:] - t[..., : h - 1, : w - 1]).mean()
    #h_tv3 = torch.square(t[..., 1:, : w - 1] - t[..., : h - 1, 1:]).mean()    
    return h_tv + w_tv #+ 0.5*(w_tv2 + w_tv3) + 0.5*(h_tv2 + h_tv3)


def space_tv_loss(multi_res_grids: List[torch.Tensor]) -> float:
    """Computes total variance across each spatial plane in the grids.

    Args:
        multi_res_grids: Grids to compute total variance over

    Returns:
        Total variance
    """

    total = 0.0
    num_planes = 0
    num_comps = multi_res_grids[0][0].shape[0]
    for grids in multi_res_grids:
        if len(grids) == 3:
            spatial_planes = {0, 1, 2}
        else:
            spatial_planes = {0, 1, 3, 6, 7, 8} #3}

        for grid_id, grid in enumerate(grids):
            if grid_id in spatial_planes:
                total += compute_plane_tv(grid)
            else:
                # Space is the last dimension for space-time planes.
                total += compute_plane_tv(grid[:num_comps], only_w=True)
            num_planes += 1
    return total / num_planes


def l1_time_planes(multi_res_grids: List[torch.Tensor]) -> float:
    """Computes the L1 distance from the multiplicative identity (1) for spatiotemporal planes.

    Args:
        multi_res_grids: Grids to compute L1 distance over

    Returns:
         L1 distance from the multiplicative identity (1)
    """
    time_planes = [2, 4, 5]  # These are the spatiotemporal planes
    #time_planes = [4, 5, 7, 8, 10, 11]  # These are the spatiotemporal planes    
    total = 0.0
    num_comps = multi_res_grids[0][0].shape[0]
    num_planes = 0
    for grids in multi_res_grids:
        for grid_id in time_planes:
            #total += torch.abs(1 - grids[grid_id]).mean()
            total += torch.abs(grids[grid_id][:num_comps]).mean()            
            num_planes += 1

    return total / num_planes


def compute_plane_smoothness(t: torch.Tensor) -> float:
    """Computes smoothness across the temporal axis of a plane

    Args:
        t: Plane tensor

    Returns:
        Time smoothness
    """
    _, h, w = t.shape
    # Convolve with a second derivative filter, in the time dimension which is dimension 2
    first_difference = t[..., 1:, :] - t[..., : h - 1, :]  # [c, h-1, w]
    second_difference = first_difference[..., 1:, :] - first_difference[..., : h - 2, :]  # [c, h-2, w]
    #first_difference2 = t[..., 1:, 1:] - t[..., : h - 1, : w - 1]  # [c, h-1, w-1]
    #second_difference2 = first_difference2[..., 1:, 1:] - first_difference2[..., : h - 2, : w - 2]  # [c, h-2, w-2]
    #first_difference3 = t[..., 1:, : w - 1] - t[..., : h - 1, 1:]  # [c, h-1, w-1]
    #second_difference3 = first_difference3[..., 1:, : w - 2] - first_difference3[..., : h - 2, 1:]  # [c, h-2, w-2]
    # Take the L2 norm of the result
    return torch.square(second_difference).mean() #+ torch.square(second_difference2).mean() + torch.square(second_difference3).mean()


def time_smoothness(multi_res_grids: List[torch.Tensor]) -> float:
    """Computes smoothness across each time plane in the grids.

    Args:
        multi_res_grids: Grids to compute time smoothness over

    Returns:
        Time smoothness
    """
    total = 0.0
    num_planes = 0
    num_comps = multi_res_grids[0][0].shape[0]    
    for grids in multi_res_grids:
        time_planes = [2, 4, 5]  # These are the spatiotemporal planes
        #time_planes = [4, 5, 7, 8, 10, 11]  # These are the spatiotemporal planes            
        for grid_id in time_planes:
            total += compute_plane_smoothness(grids[grid_id][:num_comps])
            num_planes += 1

    return total / num_planes
