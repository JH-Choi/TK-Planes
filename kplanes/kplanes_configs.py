from __future__ import annotations

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.dataparsers.dnerf_dataparser import DNeRFDataParserConfig
from nerfstudio.data.dataparsers.okutama_dataparser import OkutamaDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import CosineDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from .kplanes import KPlanesModelConfig


kplanes_method = MethodSpecification(
    config=TrainerConfig(
        method_name="kplanes",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        steps_per_eval_all_images=30000,
        max_num_iterations=30001,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=BlenderDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=KPlanesModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                grid_base_resolution=[128, 128, 128],
                grid_feature_dim=32,
                multiscale_res=[1, 2, 4],
                proposal_net_args_list=[
                    {"num_output_coords": 8, "resolution": [128, 128, 128]},
                    {"num_output_coords": 8, "resolution": [256, 256, 256]}
                ],
                loss_coefficients={
                    "interlevel": 1.0,
                    "distortion": 0.01,
                    "plane_tv": 0.01,
                    "plane_tv_proposal_net": 0.0001,
                },
                background_color="white",
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-12),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=512, max_steps=30000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-12),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=512, max_steps=30000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="K-Planes NeRF model for static scenes"
)

kplanes_olddynamic_method = MethodSpecification(
    config=TrainerConfig(
        method_name="kplanes-olddynamic",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        steps_per_eval_all_images=30000,
        max_num_iterations=30001,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=DNeRFDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_res_scale_factor=0.5,  # DNeRF train on 400x400
            ),
            model=KPlanesModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                grid_base_resolution=[128, 128, 128, 25],  # time-resolution should be half the time-steps
                grid_feature_dim=32,
                multiscale_res=[1, 2, 4],
                proposal_net_args_list=[
                    # time-resolution should be half the time-steps
                    {"num_output_coords": 8, "resolution": [128, 128, 128, 25]},
                    {"num_output_coords": 8, "resolution": [256, 256, 256, 25]},
                ],
                loss_coefficients={
                    "interlevel": 1.0,
                    "distortion": 0.01,
                    "plane_tv": 0.1,
                    "plane_tv_proposal_net": 0.0001,
                    "l1_time_planes": 0.001,
                    "l1_time_planes_proposal_net": 0.0001,
                    "time_smoothness": 0.1,
                    "time_smoothness_proposal_net": 0.001,
                },
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-12),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=512, max_steps=30000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-12),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=512, max_steps=30000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="K-Planes NeRF model for dynamic scenes"
)

kplanes_dynamic_method = MethodSpecification(
    config=TrainerConfig(
        method_name="kplanes-dynamic",
        gradient_accumulation_steps=1,
        steps_per_eval_batch=150000,
        steps_per_save=1000,
        steps_per_eval_all_images=150000,
        max_num_iterations=150001,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=OkutamaDataParserConfig(),
                #dataparser=DNeRFDataParserConfig(),                                
                train_num_rays_per_batch=4196 + 1024,
                eval_num_rays_per_batch=256,
                camera_res_scale_factor=1,  # DNeRF train on 400x400
                patch_size=128
            ),
            model=KPlanesModelConfig(
                eval_num_rays_per_chunk=1 << 11,
                grid_base_resolution=[16, 16, 8, 77],  # time-resolution should be half the time-steps
                grid_feature_dim=256,
                near_plane=0,
                far_plane=10,
                num_samples=8,
                concat_features_across_scales=False,
                multiscale_res=[1,2,4],
                is_contracted=False,
                use_proposal_weight_anneal=False,
                proposal_net_args_list=[
                    # time-resolution should be half the time-steps
                    {"num_output_coords": 32, "resolution": [64, 64, 32, 77]},
                    {"num_output_coords": 32, "resolution": [128, 128, 64, 77]},
                ],
                num_proposal_samples=(64,64),
                loss_coefficients={
                    "interlevel": 1.0,
                    "distortion": 0.01,
                    "plane_tv": 0.000001,
                    "plane_tv_proposal_net": 0.000001,
                    "l1_time_planes": 0.000001,
                    "l1_time_planes_proposal_net": 0.000001,
                    "time_smoothness": 0.000001,
                    "time_smoothness_proposal_net": 0.000001,
                },
            ),
        ),
        optimizers={
            #"proposal_networks": {
            #    "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-12),
            #    "scheduler": CosineDecaySchedulerConfig(warm_up_end=512, max_steps=100000),
            #},
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-12),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=512, max_steps=150000),
            },
            "decoder": {
                "optimizer": AdamOptimizerConfig(lr=3e-5, eps=1e-12),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=512, max_steps=150000),
            },            
            #"pose_delts": {
            #    "optimizer": AdamOptimizerConfig(lr=1e-6, eps=1e-12),
            #    "scheduler": CosineDecaySchedulerConfig(warm_up_end=512, max_steps=100000),
            #},
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 12),
        vis="viewer",
    ),
    description="K-Planes NeRF model for dynamic scenes with Okutama"
)
