# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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
Base Model implementation which takes in RayBundles
"""

from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
from torch import nn
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.model_components.scene_colliders import NearFarCollider


# Model related configs
@dataclass
class ModelConfig(InstantiateConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: Model)
    """target class to instantiate"""
    enable_collider: bool = True
    """Whether to create a scene collider to filter rays."""
    collider_params: Optional[Dict[str, float]] = to_immutable_dict({"near_plane": 2.0, "far_plane": 6.0})
    """parameters to instantiate scene collider with"""
    loss_coefficients: Dict[str, float] = to_immutable_dict({"rgb_loss_coarse": 1.0, "rgb_loss_fine": 1.0})
    """parameters to instantiate density field with"""
    eval_num_rays_per_chunk: int = 4096
    """specifies number of rays per chunk during eval"""
    prompt: Optional[str] = None
    """A prompt to be used in text to NeRF models"""


class Model(nn.Module):
    """Model class
    Where everything (Fields, Optimizers, Samplers, Visualization, etc) is linked together. This should be
    subclassed for custom NeRF model.

    Args:
        config: configuration for instantiating model
        scene_box: dataset scene box
    """

    config: ModelConfig

    def __init__(
        self,
        config: ModelConfig,
        scene_box: SceneBox,
        num_train_data: int,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.scene_box = scene_box
        self.render_aabb: Optional[SceneBox] = None  # the box that we want to render - should be a subset of scene_box
        self.num_train_data = num_train_data
        self.kwargs = kwargs
        self.collider = None

        self.populate_modules()  # populate the modules
        self.callbacks = None
        # to keep track of which device the nn.Module is on
        self.device_indicator_param = nn.Parameter(torch.empty(0))

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.device_indicator_param.device

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns a list of callbacks that run functions at the specified training iterations."""
        return []

    def populate_modules(self):
        """Set the necessary modules to get the network working."""
        # default instantiates optional modules that are common among many networks
        # NOTE: call `super().populate_modules()` in subclasses

        if self.config.enable_collider:
            assert self.config.collider_params is not None
            self.collider = NearFarCollider(
                near_plane=self.config.collider_params["near_plane"], far_plane=self.config.collider_params["far_plane"]
            )

    @abstractmethod
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """

    @abstractmethod
    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """

    def forward(self, ray_bundle: RayBundle) -> Dict[str, Union[torch.Tensor, List]]:
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """

        if self.collider is not None:
            ray_bundles = []
            cidx = [0,0,1,2]

            for idx,rb in enumerate(ray_bundle):
                #rb = self.colliders[cidx[idx]](rb)
                rb = self.collider(rb)                
                ray_bundles.append(rb)

            #ray_bundle = self.collider(ray_bundle)            
        return self.get_outputs(ray_bundles)

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """

        return {}

    @abstractmethod
    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        actual_height = 720 #// 2
        actual_width = 1280 #// 2
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle[0][0].origins.shape[:2]
        #num_rays_per_chunk = image_height * image_width
        num_rays = len(camera_ray_bundle)
        #print("OUTS BUN: {}".format(num_rays))
        dim_delts = [0,4,6,7]
        #dim_delts = [0,0,0,0]        
        #dwh_delts = [0,2,3,4]
        #dwh_delts = [0,0,0,0]
        ogg_height_chunks = 144
        height_chunks = ogg_height_chunks
        #height_chunks = 240        
        num_heights = actual_height // height_chunks
        #num_heights += (num_heights - 1)
        ogg_width_chunks = 128
        width_chunks = ogg_width_chunks
        #width_chunks = 320
        num_widths = actual_width // width_chunks
        #num_widths += (num_widths - 1)
        outputs_lists = defaultdict(list)
        dim_adder = 0
        width_chunks += dim_adder
        height_chunks += dim_adder

        for i in range(num_heights): #,height_chunks,image_height): #, num_rays, num_rays_per_chunk):
            first_round = True
            for j in range(num_widths): #,width_chunks,image_width):
                #start_idx = i
                #end_idx = i + num_rays_per_chunk
                #ray_bundle = camera_ray_bundle #camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
                '''
                new_bundle = []
                curr_height_chunks = height_chunks
                curr_width_chunks = width_chunks
                og_height_chunks = height_chunks #// 2
                og_width_chunks = width_chunks #// 2
                
                for idx, ray_bundle in enumerate(camera_ray_bundle):
                    curr_height_chunks += dim_delts[idx]
                    curr_width_chunks += dim_delts[idx]                    
                    curr_ray_bundle = ray_bundle[i*og_height_chunks:i*og_height_chunks + curr_height_chunks,
                                                 j*og_width_chunks:j*og_width_chunks + curr_width_chunks]
                    #print(curr_ray_bundle.shape)
                    #print(curr_ray_bundle.shape, (i*curr_height_chunks,(i+1)*curr_height_chunks), (j*curr_width_chunks,(j+1)*curr_width_chunks))
                    new_bundle.append(curr_ray_bundle)
                    #print(ray_bundle.shape,curr_ray_bundle.shape)
                    curr_height_chunks = og_height_chunks // 2
                    curr_width_chunks = og_width_chunks // 2
                    og_height_chunks = og_height_chunks // 2
                    og_width_chunks = og_width_chunks // 2
                '''

                #exit(-1)
                curr_bundle = camera_ray_bundle[i*num_widths + j]
                #print("OUTTTIEEE: {}".format(ray_bundle.shape))
                #print("OUTTTIEEE SHAPPPEEE: {}".format(ray_bundle.origins.shape))            
                outputs = self.forward(ray_bundle=curr_bundle) #ray_bundle)

                for output_name, output in outputs.items():  # type: ignore
                    if not torch.is_tensor(output):
                        # TODO: handle lists of tensors as well
                        continue
                    if first_round:
                        outputs_lists[output_name].append([])
                    outputs_lists[output_name][i].append(output)
                first_round = False
        #image_height *= 8
        #image_width *= 8
        outputs = {}

        w_quarter = width_chunks // 4
        h_quarter = height_chunks // 4
        for output_name, outputs_list in outputs_lists.items():
            sub_lst = []
            for sidx, sub_output in enumerate(outputs_list):
                new_outs = []
                for oidx, outs in enumerate(sub_output):
                    new_out = outs.reshape(height_chunks, width_chunks, -1)
                    if dim_adder > 0:
                        new_out = new_out[(dim_adder//2):-(dim_adder//2),(dim_adder//2):-(dim_adder//2)]
                    '''
                    if oidx > 0:
                        new_out = new_out[:,w_quarter:]
                    if oidx < len(sub_output) - 1:
                        new_out = new_out[:,:-w_quarter]

                    if sidx > 0:
                        new_out = new_out[h_quarter:]
                    if sidx < len(outputs_list) - 1:
                        new_out = new_out[:-h_quarter]
                    '''
                    new_outs.append(new_out)
                sub_lst.append(torch.cat(new_outs,dim=1))
            #outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
            outputs[output_name] = torch.cat(sub_lst,dim=0)

        return outputs

    @abstractmethod
    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.
        TODO: This shouldn't return a loss

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """

    def load_model(self, loaded_state: Dict[str, Any]) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: dictionary of pre-trained model states
        """
        state = {key.replace("module.", ""): value for key, value in loaded_state["model"].items()}
        self.load_state_dict(state)  # type: ignore

    def update_to_step(self, step: int) -> None:
        """Called when loading a model from a checkpoint. Sets any model parameters that change over
        training to the correct value, based on the training step of the checkpoint.

        Args:
            step: training step of the loaded checkpoint
        """
