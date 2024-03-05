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
Encoding functions
"""

import itertools
from abc import abstractmethod
from typing import Literal, Optional, Sequence, List

import random
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int, Shaped, Bool
from torch import Tensor, nn

from nerfstudio.field_components.base_field_component import FieldComponent
from nerfstudio.utils.math import components_from_spherical_harmonics, expected_sin
from nerfstudio.utils.printing import print_tcnn_speed_warning
from .LimitGradLayer import LimitGradLayer
try:
    import tinycudann as tcnn

    TCNN_EXISTS = True
except ModuleNotFoundError:
    TCNN_EXISTS = False


class Encoding(FieldComponent):
    """Encode an input tensor. Intended to be subclassed

    Args:
        in_dim: Input dimension of tensor
    """

    def __init__(self, in_dim: int) -> None:
        if in_dim <= 0:
            raise ValueError("Input dimension should be greater than zero")
        super().__init__(in_dim=in_dim)

    @abstractmethod
    def forward(self, in_tensor: Shaped[Tensor, "*bs input_dim"]) -> Shaped[Tensor, "*bs output_dim"]:
        """Call forward and returns and processed tensor

        Args:
            in_tensor: the input tensor to process
        """
        raise NotImplementedError


class Identity(Encoding):
    """Identity encoding (Does not modify input)"""

    def get_out_dim(self) -> int:
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        return self.in_dim

    def forward(self, in_tensor: Shaped[Tensor, "*bs input_dim"]) -> Shaped[Tensor, "*bs output_dim"]:
        return in_tensor


class ScalingAndOffset(Encoding):
    """Simple scaling and offset to input

    Args:
        in_dim: Input dimension of tensor
        scaling: Scaling applied to tensor.
        offset: Offset applied to tensor.
    """

    def __init__(self, in_dim: int, scaling: float = 1.0, offset: float = 0.0) -> None:
        super().__init__(in_dim)

        self.scaling = scaling
        self.offset = offset

    def get_out_dim(self) -> int:
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        return self.in_dim

    def forward(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        return self.scaling * in_tensor + self.offset


class NeRFEncoding(Encoding):
    """Multi-scale sinusoidal encodings. Support ``integrated positional encodings`` if covariances are provided.
    Each axis is encoded with frequencies ranging from 2^min_freq_exp to 2^max_freq_exp.

    Args:
        in_dim: Input dimension of tensor
        num_frequencies: Number of encoded frequencies per axis
        min_freq_exp: Minimum frequency exponent
        max_freq_exp: Maximum frequency exponent
        include_input: Append the input coordinate to the encoding
    """

    def __init__(
        self,
        in_dim: int,
        num_frequencies: int,
        min_freq_exp: float,
        max_freq_exp: float,
        include_input: bool = False,
        implementation: Literal["tcnn", "torch"] = "torch",
    ) -> None:
        super().__init__(in_dim)

        self.num_frequencies = num_frequencies
        self.min_freq = min_freq_exp
        self.max_freq = max_freq_exp
        self.include_input = include_input

        self.tcnn_encoding = None
        if implementation == "tcnn" and not TCNN_EXISTS:
            print_tcnn_speed_warning("NeRFEncoding")
        elif implementation == "tcnn":
            encoding_config = {"otype": "Frequency", "n_frequencies": num_frequencies}
            assert min_freq_exp == 0, "tcnn only supports min_freq_exp = 0"
            assert max_freq_exp == num_frequencies - 1, "tcnn only supports max_freq_exp = num_frequencies - 1"
            self.tcnn_encoding = tcnn.Encoding(
                n_input_dims=in_dim,
                encoding_config=encoding_config,
            )

    def get_out_dim(self) -> int:
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        out_dim = self.in_dim * self.num_frequencies * 2
        if self.include_input:
            out_dim += self.in_dim
        return out_dim

    def pytorch_fwd(
        self,
        in_tensor: Float[Tensor, "*bs input_dim"],
        covs: Optional[Float[Tensor, "*bs input_dim input_dim"]] = None,
    ) -> Float[Tensor, "*bs output_dim"]:
        """Calculates NeRF encoding. If covariances are provided the encodings will be integrated as proposed
            in mip-NeRF.

        Args:
            in_tensor: For best performance, the input tensor should be between 0 and 1.
            covs: Covariances of input points.
        Returns:
            Output values will be between -1 and 1
        """
        scaled_in_tensor = 2 * torch.pi * in_tensor  # scale to [0, 2pi]
        freqs = 2 ** torch.linspace(self.min_freq, self.max_freq, self.num_frequencies).to(in_tensor.device)
        scaled_inputs = scaled_in_tensor[..., None] * freqs  # [..., "input_dim", "num_scales"]
        scaled_inputs = scaled_inputs.view(*scaled_inputs.shape[:-2], -1)  # [..., "input_dim" * "num_scales"]

        if covs is None:
            encoded_inputs = torch.sin(torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1))
        else:
            input_var = torch.diagonal(covs, dim1=-2, dim2=-1)[..., :, None] * freqs[None, :] ** 2
            input_var = input_var.reshape((*input_var.shape[:-2], -1))
            encoded_inputs = expected_sin(
                torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1), torch.cat(2 * [input_var], dim=-1)
            )

        if self.include_input:
            encoded_inputs = torch.cat([encoded_inputs, in_tensor], dim=-1)
        return encoded_inputs

    def forward(
        self, in_tensor: Float[Tensor, "*bs input_dim"], covs: Optional[Float[Tensor, "*bs input_dim input_dim"]] = None
    ) -> Float[Tensor, "*bs output_dim"]:
        if self.tcnn_encoding is not None:
            return self.tcnn_encoding(in_tensor)
        return self.pytorch_fwd(in_tensor, covs)


class RFFEncoding(Encoding):
    """Random Fourier Feature encoding. Supports integrated encodings.

    Args:
        in_dim: Input dimension of tensor
        num_frequencies: Number of encoding frequencies
        scale: Std of Gaussian to sample frequencies. Must be greater than zero
        include_input: Append the input coordinate to the encoding
    """

    def __init__(self, in_dim: int, num_frequencies: int, scale: float, include_input: bool = False) -> None:
        super().__init__(in_dim)

        self.num_frequencies = num_frequencies
        if not scale > 0:
            raise ValueError("RFF encoding scale should be greater than zero")
        self.scale = scale
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        b_matrix = torch.normal(mean=0, std=self.scale, size=(self.in_dim, self.num_frequencies))
        self.register_buffer(name="b_matrix", tensor=b_matrix)
        self.include_input = include_input

    def get_out_dim(self) -> int:
        return self.num_frequencies * 2

    def forward(
        self,
        in_tensor: Float[Tensor, "*bs input_dim"],
        covs: Optional[Float[Tensor, "*bs input_dim input_dim"]] = None,
    ) -> Float[Tensor, "*bs output_dim"]:
        """Calculates RFF encoding. If covariances are provided the encodings will be integrated as proposed
            in mip-NeRF.

        Args:
            in_tensor: For best performance, the input tensor should be between 0 and 1.
            covs: Covariances of input points.

        Returns:
            Output values will be between -1 and 1
        """
        scaled_in_tensor = 2 * torch.pi * in_tensor  # scale to [0, 2pi]
        scaled_inputs = scaled_in_tensor @ self.b_matrix  # [..., "num_frequencies"]

        if covs is None:
            encoded_inputs = torch.sin(torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1))
        else:
            input_var = torch.sum((covs @ self.b_matrix) * self.b_matrix, -2)
            encoded_inputs = expected_sin(
                torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1), torch.cat(2 * [input_var], dim=-1)
            )

        if self.include_input:
            encoded_inputs = torch.cat([encoded_inputs, in_tensor], dim=-1)

        return encoded_inputs


class HashEncoding(Encoding):
    """Hash encoding

    Args:
        num_levels: Number of feature grids.
        min_res: Resolution of smallest feature grid.
        max_res: Resolution of largest feature grid.
        log2_hashmap_size: Size of hash map is 2^log2_hashmap_size.
        features_per_level: Number of features per level.
        hash_init_scale: Value to initialize hash grid.
        implementation: Implementation of hash encoding. Fallback to torch if tcnn not available.
        interpolation: Interpolation override for tcnn hashgrid. Not supported for torch unless linear.
    """

    def __init__(
        self,
        num_levels: int = 16,
        min_res: int = 16,
        max_res: int = 1024,
        log2_hashmap_size: int = 19,
        features_per_level: int = 2,
        hash_init_scale: float = 0.001,
        implementation: Literal["tcnn", "torch"] = "torch",
        interpolation: Optional[Literal["Nearest", "Linear", "Smoothstep"]] = None,
    ) -> None:
        super().__init__(in_dim=3)
        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.hash_table_size = 2**log2_hashmap_size

        levels = torch.arange(num_levels)
        growth_factor = np.exp((np.log(max_res) - np.log(min_res)) / (num_levels - 1))
        self.scalings = torch.floor(min_res * growth_factor**levels)

        self.hash_offset = levels * self.hash_table_size
        self.hash_table = torch.rand(size=(self.hash_table_size * num_levels, features_per_level)) * 2 - 1
        self.hash_table *= hash_init_scale
        self.hash_table = nn.Parameter(self.hash_table)

        self.tcnn_encoding = None
        if implementation == "tcnn" and not TCNN_EXISTS:
            print_tcnn_speed_warning("HashEncoding")
        elif implementation == "tcnn":
            encoding_config = {
                "otype": "HashGrid",
                "n_levels": self.num_levels,
                "n_features_per_level": self.features_per_level,
                "log2_hashmap_size": self.log2_hashmap_size,
                "base_resolution": min_res,
                "per_level_scale": growth_factor,
            }
            if interpolation is not None:
                encoding_config["interpolation"] = interpolation

            self.tcnn_encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config=encoding_config,
            )

        if self.tcnn_encoding is None:
            assert (
                interpolation is None or interpolation == "Linear"
            ), f"interpolation '{interpolation}' is not supported for torch encoding backend"

    def get_out_dim(self) -> int:
        return self.num_levels * self.features_per_level

    def hash_fn(self, in_tensor: Int[Tensor, "*bs num_levels 3"]) -> Shaped[Tensor, "*bs num_levels"]:
        """Returns hash tensor using method described in Instant-NGP

        Args:
            in_tensor: Tensor to be hashed
        """

        # min_val = torch.min(in_tensor)
        # max_val = torch.max(in_tensor)
        # assert min_val >= 0.0
        # assert max_val <= 1.0

        in_tensor = in_tensor * torch.tensor([1, 2654435761, 805459861]).to(in_tensor.device)
        x = torch.bitwise_xor(in_tensor[..., 0], in_tensor[..., 1])
        x = torch.bitwise_xor(x, in_tensor[..., 2])
        x %= self.hash_table_size
        x += self.hash_offset.to(x.device)
        return x

    def pytorch_fwd(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        """Forward pass using pytorch. Significantly slower than TCNN implementation."""

        assert in_tensor.shape[-1] == 3
        in_tensor = in_tensor[..., None, :]  # [..., 1, 3]
        scaled = in_tensor * self.scalings.view(-1, 1).to(in_tensor.device)  # [..., L, 3]
        scaled_c = torch.ceil(scaled).type(torch.int32)
        scaled_f = torch.floor(scaled).type(torch.int32)

        offset = scaled - scaled_f

        hashed_0 = self.hash_fn(scaled_c)  # [..., num_levels]
        hashed_1 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_2 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_3 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_4 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1))
        hashed_5 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_f[..., 2:3]], dim=-1))
        hashed_6 = self.hash_fn(scaled_f)
        hashed_7 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1))

        f_0 = self.hash_table[hashed_0]  # [..., num_levels, features_per_level]
        f_1 = self.hash_table[hashed_1]
        f_2 = self.hash_table[hashed_2]
        f_3 = self.hash_table[hashed_3]
        f_4 = self.hash_table[hashed_4]
        f_5 = self.hash_table[hashed_5]
        f_6 = self.hash_table[hashed_6]
        f_7 = self.hash_table[hashed_7]

        f_03 = f_0 * offset[..., 0:1] + f_3 * (1 - offset[..., 0:1])
        f_12 = f_1 * offset[..., 0:1] + f_2 * (1 - offset[..., 0:1])
        f_56 = f_5 * offset[..., 0:1] + f_6 * (1 - offset[..., 0:1])
        f_47 = f_4 * offset[..., 0:1] + f_7 * (1 - offset[..., 0:1])

        f0312 = f_03 * offset[..., 1:2] + f_12 * (1 - offset[..., 1:2])
        f4756 = f_47 * offset[..., 1:2] + f_56 * (1 - offset[..., 1:2])

        encoded_value = f0312 * offset[..., 2:3] + f4756 * (
            1 - offset[..., 2:3]
        )  # [..., num_levels, features_per_level]

        return torch.flatten(encoded_value, start_dim=-2, end_dim=-1)  # [..., num_levels * features_per_level]

    def forward(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        if self.tcnn_encoding is not None:
            return self.tcnn_encoding(in_tensor)
        return self.pytorch_fwd(in_tensor)


class TensorCPEncoding(Encoding):
    """Learned CANDECOMP/PARFAC (CP) decomposition encoding used in TensoRF

    Args:
        resolution: Resolution of grid.
        num_components: Number of components per dimension.
        init_scale: Initialization scale.
    """

    def __init__(self, resolution: int = 256, num_components: int = 24, init_scale: float = 0.1) -> None:
        super().__init__(in_dim=3)

        self.resolution = resolution
        self.num_components = num_components

        # TODO Learning rates should be different for these
        self.line_coef = nn.Parameter(init_scale * torch.randn((3, num_components, resolution, 1)))

    def get_out_dim(self) -> int:
        return self.num_components

    def forward(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        line_coord = torch.stack([in_tensor[..., 2], in_tensor[..., 1], in_tensor[..., 0]])  # [3, ...]
        line_coord = torch.stack([torch.zeros_like(line_coord), line_coord], dim=-1)  # [3, ...., 2]

        # Stop gradients from going to sampler
        line_coord = line_coord.view(3, -1, 1, 2).detach()

        line_features = F.grid_sample(self.line_coef, line_coord, align_corners=True)  # [3, Components, -1, 1]

        features = torch.prod(line_features, dim=0)
        features = torch.moveaxis(features.view(self.num_components, *in_tensor.shape[:-1]), 0, -1)

        return features  # [..., Components]

    @torch.no_grad()
    def upsample_grid(self, resolution: int) -> None:
        """Upsamples underyling feature grid

        Args:
            resolution: Target resolution.
        """

        self.line_coef.data = F.interpolate(
            self.line_coef.data, size=(resolution, 1), mode="bilinear", align_corners=True
        )

        self.resolution = resolution


class TensorVMEncoding(Encoding):
    """Learned vector-matrix encoding proposed by TensoRF

    Args:
        resolution: Resolution of grid.
        num_components: Number of components per dimension.
        init_scale: Initialization scale.
    """

    plane_coef: Float[Tensor, "3 num_components resolution resolution"]
    line_coef: Float[Tensor, "3 num_components resolution 1"]

    def __init__(
        self,
        resolution: int = 128,
        num_components: int = 24,
        init_scale: float = 0.1,
    ) -> None:
        super().__init__(in_dim=3)

        self.resolution = resolution
        self.num_components = num_components

        self.plane_coef = nn.Parameter(init_scale * torch.randn((3, num_components, resolution, resolution)))
        self.line_coef = nn.Parameter(init_scale * torch.randn((3, num_components, resolution, 1)))

    def get_out_dim(self) -> int:
        return self.num_components * 3

    def forward(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        """Compute encoding for each position in in_positions

        Args:
            in_tensor: position inside bounds in range [-1,1],

        Returns: Encoded position
        """
        plane_coord = torch.stack([in_tensor[..., [0, 1]], in_tensor[..., [0, 2]], in_tensor[..., [1, 2]]])  # [3,...,2]
        line_coord = torch.stack([in_tensor[..., 2], in_tensor[..., 1], in_tensor[..., 0]])  # [3, ...]
        line_coord = torch.stack([torch.zeros_like(line_coord), line_coord], dim=-1)  # [3, ...., 2]

        # Stop gradients from going to sampler
        plane_coord = plane_coord.view(3, -1, 1, 2).detach()
        line_coord = line_coord.view(3, -1, 1, 2).detach()

        plane_features = F.grid_sample(self.plane_coef, plane_coord, align_corners=True)  # [3, Components, -1, 1]
        line_features = F.grid_sample(self.line_coef, line_coord, align_corners=True)  # [3, Components, -1, 1]

        features = plane_features * line_features  # [3, Components, -1, 1]
        features = torch.moveaxis(features.view(3 * self.num_components, *in_tensor.shape[:-1]), 0, -1)

        return features  # [..., 3 * Components]

    @torch.no_grad()
    def upsample_grid(self, resolution: int) -> None:
        """Upsamples underlying feature grid

        Args:
            resolution: Target resolution.
        """
        plane_coef = F.interpolate(
            self.plane_coef.data, size=(resolution, resolution), mode="bilinear", align_corners=True
        )
        line_coef = F.interpolate(self.line_coef.data, size=(resolution, 1), mode="bilinear", align_corners=True)

        self.plane_coef, self.line_coef = torch.nn.Parameter(plane_coef), torch.nn.Parameter(line_coef)
        self.resolution = resolution


class TriplaneEncoding(Encoding):
    """Learned triplane encoding

    The encoding at [i,j,k] is an n dimensional vector corresponding to the element-wise product of the
    three n dimensional vectors at plane_coeff[i,j], plane_coeff[i,k], and plane_coeff[j,k].

    This allows for marginally more expressivity than the TensorVMEncoding, and each component is self standing
    and symmetrical, unlike with VM decomposition where we needed one component with a vector along all the x, y, z
    directions for symmetry.

    This can be thought of as 3 planes of features perpendicular to the x, y, and z axes, respectively and intersecting
    at the origin, and the encoding being the element-wise product of the element at the projection of [i, j, k] on
    these planes.

    The use for this is in representing a tensor decomp of a 4D embedding tensor: (x, y, z, feature_size)

    This will return a tensor of shape (bs:..., num_components)

    Args:
        resolution: Resolution of grid.
        num_components: The number of scalar triplanes to use (ie: output feature size)
        init_scale: The scale of the initial values of the planes
        product: Whether to use the element-wise product of the planes or the sum
    """

    plane_coef: Float[Tensor, "3 num_components resolution resolution"]

    def __init__(
        self,
        resolution: int = 32,
        num_components: int = 64,
        init_scale: float = 0.1,
        reduce: Literal["sum", "product"] = "sum",
    ) -> None:
        super().__init__(in_dim=3)

        self.resolution = resolution
        self.num_components = num_components
        self.init_scale = init_scale
        self.reduce = reduce

        self.plane_coef = nn.Parameter(
            self.init_scale * torch.randn((3, self.num_components, self.resolution, self.resolution))
        )

    def get_out_dim(self) -> int:
        return self.num_components

    def forward(self, in_tensor: Float[Tensor, "*bs 3"]) -> Float[Tensor, "*bs num_components featuresize"]:
        """Sample features from this encoder. Expects in_tensor to be in range [0, resolution]"""

        original_shape = in_tensor.shape
        in_tensor = in_tensor.reshape(-1, 3)

        plane_coord = torch.stack([in_tensor[..., [0, 1]], in_tensor[..., [0, 2]], in_tensor[..., [1, 2]]], dim=0)

        # Stop gradients from going to sampler
        plane_coord = plane_coord.detach().view(3, -1, 1, 2)
        plane_features = F.grid_sample(
            self.plane_coef, plane_coord, align_corners=True
        )  # [3, num_components, flattened_bs, 1]

        if self.reduce == "product":
            plane_features = plane_features.prod(0).squeeze(-1).T  # [flattened_bs, num_components]
        else:
            plane_features = plane_features.sum(0).squeeze(-1).T

        return plane_features.reshape(*original_shape[:-1], self.num_components)

    @torch.no_grad()
    def upsample_grid(self, resolution: int) -> None:
        """Upsamples underlying feature grid

        Args:
            resolution: Target resolution.
        """
        plane_coef = F.interpolate(
            self.plane_coef.data, size=(resolution, resolution), mode="bilinear", align_corners=True
        )

        self.plane_coef = torch.nn.Parameter(plane_coef)
        self.resolution = resolution


class ComplexAct(nn.Module):
    def __init__(self, act, use_phase=False):
        super().__init__()
        # act can be either a function from nn.functional or a nn.Module if the
        # activation has learnable parameters
        self.act = act
        self.use_phase = use_phase

    def forward(self, z):
        if self.use_phase:
            return self.act(torch.abs(z)) * torch.exp(1.j * torch.angle(z)) 
        else:
            return self.act(z.real) + 1.j * self.act(z.imag)
        
class KPlanesEncoding(Encoding):
    """Learned K-Planes encoding

    A plane encoding supporting both 3D and 4D coordinates. With 3D coordinates this is similar to
    :class:`TriplaneEncoding`. With 4D coordinates, the encoding at point ``[i,j,k,q]`` is
    a n-dimensional vector computed as the elementwise product of 6 n-dimensional vectors at
    ``planes[i,j]``, ``planes[i,k]``, ``planes[i,q]``, ``planes[j,k]``, ``planes[j,q]``,
    ``planes[k,q]``.

    Unlike :class:`TriplaneEncoding` this class supports different resolution along each axis.

    This will return a tensor of shape (bs:..., num_components)

    Args:
        resolution: Resolution of the grid. Can be a sequence of 3 or 4 integers.
        num_components: The number of scalar planes to use (ie: output feature size)
        init_a: The lower-bound of the uniform distribution used to initialize the spatial planes
        init_b: The upper-bound of the uniform distribution used to initialize the spatial planes
        reduce: Whether to use the element-wise product of the planes or the sum
    """

    def __init__(
        self,
        resolution: Sequence[int] = (128, 128, 128),
        num_components: int = 64,
            select_dim: int = 64,
        init_a: float = 0.1,
        init_b: float = 0.5,
        reduce: Literal["sum", "product"] = "product",
    ) -> None:
        super().__init__(in_dim=len(resolution))

        self.resolution = resolution
        self.resolution_dyn = [resolution[0] // 4, resolution[1] // 4, resolution[2] // 4, resolution[3]]
        self.num_components = num_components
        self.select_dim = select_dim
        self.select_dim_dyn = select_dim // 4
        self.reduce = reduce
        if self.in_dim not in {3, 4}:
            raise ValueError(
                f"The dimension of coordinates must be either 3 (static scenes) "
                f"or 4 (dynamic scenes). Found resolution with {self.in_dim} dimensions."
            )
        has_time_planes = self.in_dim == 4
        self.proc_coef_counter = 0
        self.proc_coef_limit = 1000
        self.run_coef_proc = True
        self.dynamo = False
        self.proc_func = torch.nn.Identity()
        #self.proc_func = torch.nn.Tanh()
        #self.proc_func = F.normalize        
        self.print_idx = 0
        self.mask_layer = LimitGradLayer.apply
        self.coo_combs = list(itertools.combinations(range(self.in_dim), 2))
        # Unlike the Triplane encoding, we use a parameter list instead of batching all planes
        # together to support uneven resolutions (especially useful for time).
        # Dynamic models (in_dim == 4) will have 6 planes:
        # (y, x), (z, x), (t, x), (z, y), (t, y), (t, z)
        # static models (in_dim == 3) will only have the 1st, 2nd and 4th planes.
        self.coo_combs = [(0,1),(0,2),(0,3), (1,2),(1,3),(2,3), (0,1),(0,2),(1,2)]
        self.plane_coefs = nn.ParameterList()
        self.feature_coefs = []
        #self.feature_coefs.append(nn.Parameter(torch.empty([len(self.coo_combs),self.select_dim[0], self.num_components])))
        #self.feature_coefs.append(nn.Parameter(torch.empty([len(self.coo_combs),self.select_dim[1], self.num_components])))
        self.feature_coefs.append(nn.Parameter(torch.empty([len(self.coo_combs),self.select_dim, self.num_components]),requires_grad=False))        
        #self.feature_coefs.append(nn.Parameter(torch.empty([self.select_dim, self.num_components])))

        self.feature_coefs = nn.ParameterList(self.feature_coefs)
        nn.init.normal_(self.feature_coefs[0], 0, 1)

        #self.og_static = self.feature_coefs[0].clone().detach()
        #self.og_dynamic = self.feature_coefs[1].clone().detach()
        #print(self.feature_coefs[0])
        #exit(-1)
        self.feature_softmax_layer = nn.Softmax(dim=1)
        #self.feature_coefs = nn.ParameterList()        
        #self.coo_combs = [(0,1),(0,2),(1,2), (0,1),(0,3),(1,3), (0,2),(0,3),(2,3), (1,2),(1,3),(2,3)]

        for coo_idx,coo_comb in enumerate(self.coo_combs):

            if 3 in coo_comb or coo_idx > 5:
                #num_comps = self.select_dim#*self.num_components
                num_comps = self.num_components
                curr_resolution = self.resolution_dyn
            else:
                #num_comps = self.select_dim
                num_comps = self.num_components
                curr_resolution = self.resolution                


            
            new_plane_coef = nn.Parameter(
                torch.empty([num_comps] + [curr_resolution[cc] for cc in coo_comb[::-1]])
            )
            #new_feature_coef = nn.Parameter(
            #    torch.empty([self.select_dim, self.num_components])
            #)                
            #print(coo_comb[::-1])
            if has_time_planes and 3 in coo_comb:  # Time planes initialized to 1
                #nn.init.ones_(new_plane_coef)
                #nn.init.dirac_(new_plane_coef)
                #with torch.no_grad():
                #    new_plane_coef = new_plane_coef*100
                #nn.init.uniform_(new_plane_coef, a=init_a, b=init_b)
                nn.init.uniform_(new_plane_coef, a=0.2, b=0.9)
                #nn.init.normal_(new_plane_coef, 0, 0.5)
            #elif coo_idx > 5:
            #    nn.init.uniform_(new_plane_coef, a=-0.1, b=0.1)
            else:
                nn.init.uniform_(new_plane_coef, a=0.2, b=0.9) #init_a, b=init_b)
            #nn.init.uniform_(new_feature_coef, a=-0.1, b=0.1)
            self.plane_coefs.append(new_plane_coef)
            #self.feature_coefs.append(new_feature_coef)

        bias_bool = False

        total_comps = (self.select_dim)* self.num_components
        out_comps = self.num_components

        self.output_head = nn.Identity()
        '''
        self.output_head = nn.Sequential(
            nn.Linear(total_comps, total_comps*2, bias=bias_bool),
            #nn.LayerNorm(total_comps*4),
            #nn.ReLU(),
            #nn.Linear(total_comps*2, total_comps*2, bias=bias_bool),
            #nn.LayerNorm(total_comps*4),            
            #nn.ReLU(),
            #nn.Linear(total_comps*4, total_comps*4, bias=bias_bool),
            #nn.LayerNorm(total_comps*4),
            nn.ReLU(),            
            nn.Linear(total_comps*2, out_comps, bias=bias_bool))
        '''

    def process_feature_coefs(self):
        with torch.no_grad():
            for gidx,grid in enumerate(self.plane_coefs):
                normed_grid = torch.nn.functional.normalize(grid,p=2,dim=0)
                normed_grid = normed_grid.reshape(normed_grid.shape[0],-1)
                comp_matrix = torch.matmul(normed_grid.permute(1,0),normed_grid)
                sorted_vecs = torch.sort(torch.sum(comp_matrix,0) / comp_matrix.shape[0],descending=True)
                sorted_indices = sorted_vecs.indices
                bins = self.calc_bins(sorted_vecs.values)
                reorged_grid = grid.reshape(grid.shape[0],-1).permute(1,0)
                reorged_grid = reorged_grid[sorted_indices]
                for i in range(self.select_dim):
                    start = bins[i][0]
                    end = bins[i][1] + 1
                    self.feature_coefs[0][gidx,i] = reorged_grid[start:end].mean(dim=0)
                    reorged_grid[start:end] = reorged_grid[start:end].mean(dim=0)
                reset_indices = torch.sort(sorted_indices).indices
                reorged_grid = reorged_grid[reset_indices]
                self.plane_coefs[gidx] = (reorged_grid.permute(1,0)).reshape(grid.shape)
                
    def calc_bins(self,x):
        bin_num = self.select_dim
        iters = 50
        diff = x.shape[0] // (bin_num - 1)
        bins = [[i,min(x.shape[0]-1,i+diff - 1)] for i in range(0,x.shape[0],diff)]
        for _ in range(iters):
            x_dists = []
            for bin in bins:
                x_mean = x[bin[0]:bin[1]].mean()
                x_dist = torch.sum(torch.abs(x[bin[0]:bin[1]] - x_mean))
                x_dists.append(x_dist)
                
        for idx,x_dist in enumerate(x_dists):
            if idx < len(x_dists) - 1:
                if (x_dist < x_dists[idx+1] and (bins[idx+1][1] - bins[idx+1][0]) > 1):
                    bins[idx][1] += 1
                    bins[idx+1][0] += 1
                elif x_dist > x_dists[idx+1] and (bins[idx][1] - bins[idx][0]) > 1:
                    bins[idx][1] -= 1
                    bins[idx+1][0] -= 1
        return bins

    def process_feature_coefs2(self):
        bin_num = self.select_dim
        num_iters = 3
        with torch.no_grad():
            
            for gidx,grid in enumerate(self.plane_coefs):
                print('GRID {}/{} PROCESSING'.format(gidx+1,len(self.plane_coefs)))
                grid_r = grid.reshape(grid.shape[0],-1).permute(1,0)
                sets = [set() for _ in range(bin_num)]
                #means = torch.zeros(bin_num).to(grid.device)
                means = [None for _ in range(bin_num)]
                bins = self.calc_bins2(grid_r)
                
                for bidx,bin in enumerate(bins):
                    curr_set = sets[bidx]
                    mean_val = 0
                    for xidx in range(bin[0],bin[1]+1):
                        curr_set.add(xidx)
                        mean_val += grid_r[xidx]

                    mean_val = mean_val / len(curr_set)
                    means[bidx] = mean_val

                for nitdx in range(num_iters):
                    print("{} kmeans iter of {}".format(1+nitdx,num_iters))
                    new_sets = [set() for _ in range(bin_num)]
                    for curr_set in sets:
                        for xidx in curr_set:
                            best_set = None
                            best_dist = None
                            
                            for midx,curr_mean in enumerate(means):
                                curr_dist = torch.sum(torch.sqrt((grid_r[xidx] - curr_mean)**2))
                                if best_dist is None or curr_dist < best_dist:
                                    best_set = midx
                                    best_dist = curr_dist

                            new_sets[best_set].add(xidx)
                    means = [None for _ in range(bin_num)]
                    for cidx,curr_set in enumerate(new_sets):
                        mean_val = 0
                        for xidx in curr_set:
                            mean_val += grid_r[xidx]
                        if len(curr_set) > 0:
                            mean_val = mean_val / len(curr_set)
                        means[cidx] = mean_val
                    sets = new_sets

                num_zero_sets = 0
                for cidx,curr_set in enumerate(sets):
                    mean_val = means[cidx]
                    if len(curr_set) == 0:
                        num_zero_sets += 1
                    for xidx in curr_set:
                        grid_r[xidx] = mean_val

                if num_zero_sets > 0:
                    print("{} ZERO SETS OUT OF {}".format(num_zero_sets,bin_num))
                self.plane_coefs[gidx] = grid_r.permute(1,0).reshape(grid.shape)

    def process_feature_coefs3(self):
        num_iters = 3
        with torch.no_grad():
            
            for gidx,grid in enumerate(self.plane_coefs):
                bin_num = self.select_dim
                if gidx in [2,4,5,6,7,8]:
                    bin_num = self.select_dim_dyn
                #print('GRID {}/{} PROCESSING'.format(gidx+1,len(self.plane_coefs)))
                grid_r = grid.reshape(grid.shape[0],-1).permute(1,0)
                sets = [set() for _ in range(bin_num)]
                means = torch.zeros((bin_num,grid_r.shape[-1])).to(grid.device)
                bins = self.calc_bins2(grid_r,bin_num)
                init_indices = [i for i in range(grid_r.shape[0])]
                random.shuffle(init_indices)
                for bidx,bin in enumerate(bins):
                    sets[bidx].update(init_indices[bin[0]:bin[1]+1])
                    means[bidx] = grid_r[bin[0]:bin[1]+1].mean(0)

                for nitdx in range(num_iters):
                    #print("{} kmeans iter of {}".format(1+nitdx,num_iters))
                    new_sets = [set() for _ in range(bin_num)]
                    for setidx, curr_set in enumerate(sets):
                        curr_set_list = list(curr_set)
                        dists = torch.sqrt(torch.sum((grid_r[curr_set_list].unsqueeze(1) - means.unsqueeze(0))**2,dim=-1))
                        best_sets = torch.argmin(dists,dim=-1)
                        for sidx, xidx in enumerate(curr_set_list):
                            #dists = torch.sqrt(torch.sum((grid_r[xidx] - means)**2,dim=1))
                            #best_set = torch.argmin(dists)
                            new_sets[best_sets[sidx]].add(xidx)

                    for cidx,curr_set in enumerate(new_sets):
                        if len(curr_set) > 0:
                            means[cidx] = grid_r[list(curr_set)].mean(0)
                        else:
                            means[cidx] = 0
                    sets = new_sets

                #num_zero_sets = 0
                for cidx,curr_set in enumerate(sets):
                    grid_r[list(curr_set)] = means[cidx]
                    #if len(curr_set) == 0:
                    #    num_zero_sets += 1

                #if num_zero_sets > 0:
                    #print("{} ZERO SETS OUT OF {}".format(num_zero_sets,bin_num))
                self.plane_coefs[gidx] = grid_r.permute(1,0).reshape(grid.shape)

    def calc_bins2(self,x,bin_num):
        diff = x.shape[0] // (bin_num)
        bins = []
        for i in range(bin_num):
            if i < bin_num - 1:
                bins.append([i*diff,(i+1)*diff])
            else:
                bins.append([i*diff,x.shape[0]-1])
        return bins
                
    def get_out_dim(self) -> int:
        return total_comps

    def forward(self, in_tensor: Float[Tensor, "*bs input_dim"],
                time_mask: Bool[Tensor, "*bs input_dim"],
                static_mask: Int[Tensor, "*bs input_dim"],
                dynamic_mask: Int[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        """Sample features from this encoder. Expects ``in_tensor`` to be in range [-1, 1]"""
        original_shape = in_tensor.shape

        self.proc_coef_counter = (self.proc_coef_counter + 1) % self.proc_coef_limit

        if self.proc_coef_counter == 0 and self.run_coef_proc:
            self.process_feature_coefs3()
            #print('DONE PROCESSING')
        
        assert any(self.coo_combs)
        output = 1.0 if self.reduce == "product" else 0.0  # identity for corresponding op
        #time_output = 1.0 if self.reduce == "product" else 0.0  # identity for corresponding op
        outputs = []
        multi_outputs = []
        for ci, coo_comb in enumerate(self.coo_combs):
            grid = self.plane_coefs[ci].unsqueeze(0)  # [1, feature_dim, reso1, reso2]
            #if 3 in coo_comb:
            #    grid_type = grid.type()
            #    grid = torch.fft.fft2(grid)
            #    grid = self.time_freq_conv[0](grid) + grid
            #    grid = torch.fft.ifft2(grid).type(grid_type)
            coords = in_tensor[..., coo_comb].view(1, 1, -1, 2)  # [1, 1, flattened_bs, 2]
            if 3 in coo_comb and not self.dynamo and False:
                #new_time = (torch.rand(1) - 0.5) * 2
                new_time = torch.clip((torch.normal(coords[0,0,0,0],0.1) - 0.5) * 2,-1,1)
                coords[:,:,:,1] = new_time
            #elif 3 in coo_comb and self.dynamo:
            #    print(coords)
            #    exit(-1)

            interp = F.grid_sample(
               grid, coords, align_corners=True, padding_mode="border"
            )  # [1, output_dim, 1, flattened_bs]
            #if 3 in coo_comb:
            #    print(interp.shape)

            if 3 in coo_comb:
                #num_comps = self.select_dim#*self.num_components
                num_comps = self.num_components                                
            else:
                #num_comps = self.select_dim
                num_comps = self.num_components                                
                
            interp = interp.view(num_comps, -1).T  # [flattened_bs, output_dim]

            #if 3 in coo_comb:
            #    print(interp.shape)            
            #    exit(-1)

            #interp = torch.sigmoid(interp.unsqueeze(-1)) * self.feature_coefs[ci].unsqueeze(0)
            #interp = torch.tanh(interp.unsqueeze(-1)) * self.feature_coefs[ci].unsqueeze(0)            
            interp = interp.reshape(interp.shape[0],-1)

            #if 3 in coo_comb:
            #    interp = torch.fft.rfft(interp)
            if self.reduce == "product":
                #if 3 in coo_comb:
                #    interp = 1 + interp
                    
                #output = output * interp
                outputs.append(interp)
                
                #if 3 in coo_comb:
                #    time_output = time_output * interp
                #else:
                #    output = output * interp
            else:
                output = output + interp                
                #if 3 in coo_comb:
                #    time_output = time_output + interp
                #else:
                #    output = output + interp                    

        vol_tv = 0.0

        self.print_idx = (self.print_idx + 1) % 1000

        #xyz_static = outputs[0]*outputs[1]*outputs[3]
        #xyz_temporal = (outputs[2]*outputs[4]*outputs[5]*
        #                outputs[6]*outputs[7]*outputs[8])

        selection_func = torch.nn.Identity()
        #selection_func = torch.sigmoid
        #time_selection_func = torch.tanh
        select_kwargs = {}        
        #selection_func = torch.softmax
        #select_kwargs = {"dim":1}        
        #test_vec = torch.ones_like(self.feature_coefs[0][0])
        #test_vec[-5] = 0

        #               selection_func(outputs[1],**select_kwargs) * 
        #               selection_func(outputs[3],**select_kwargs)).unsqueeze(-1)) * self.feature_coefs[0].unsqueeze(0)
        #xyz_static = (selection_func(outputs[0]*outputs[1]*outputs[3],**select_kwargs).unsqueeze(-1)) * (self.feature_coefs[0].unsqueeze(0))

        '''
        if self.feature_coefs[0].shape[-1] == 32:
            feat_coef = self.feature_coefs[0].detach().cpu().numpy()
            np.save("feat_coef.npy",feat_coef)
            fmin = np.min(feat_coef)
            fmax = np.max(feat_coef)
            feat_coef = (feat_coef - fmin) / (fmax - fmin)
            feat_coef = feat_coef * 255
            feat_coef = feat_coef.astype(np.uint8)
            cv2.imwrite("feat_coef_test.png",feat_coef)
        '''
        
        xyz_static = (outputs[0]*outputs[1]*outputs[3])        
        xyz_temporal = (outputs[2]*outputs[4]*outputs[5]*outputs[6]*outputs[7]*outputs[8])
        #xyz_select = selection_func(xyz_static * xyz_temporal *100000,**select_kwargs)
        xyz_select = selection_func(xyz_static * xyz_temporal,**select_kwargs)
        xyz_select = torch.cat([xyz_static,xyz_temporal],dim=-1)

        #xyz_temporal = time_selection_func(xyz_temporal)

        # [(0,1),(0,2),(0,3), (1,2),(1,3),(2,3)]
        #xyz_alt = (outputs[0] * outputs[5]) + (outputs[1] * outputs[4]) + (outputs[3] * outputs[2])
        #xyz_select = selection_func(xyz_alt,**select_kwargs)        
        #curr_coeffy = torch.nn.functional.normalize(self.feature_coefs[0],p=2,dim=1)
        #curr_coeffy = self.feature_coefs[0]
        #print(torch.max(selection_func(xyz_static * xyz_temporal, **select_kwargs)))
        #exit(-1)
        #xyz_max = torch.amax(xyz_select,dim=1).unsqueeze(-1)
        #xyz_mask = xyz_select == xyz_max
        #xyz_select[xyz_mask] = 1
        #xyz_select[~xyz_mask] = 0
        
        #xyz = (xyz_select.unsqueeze(-1)) * (curr_coeffy.unsqueeze(0))
        #xyz = xyz.reshape(xyz.shape[0],-1)        
        #xyz = torch.sum(xyz,dim=1)
        #print(xyz.shape)
        #exit(-1)
        xyz = xyz_select
        #xyz = (selection_func(xyz_static,**select_kwargs).unsqueeze(-1)) * (self.feature_coefs[0].unsqueeze(0))        
        #xyz = xyz.reshape(xyz.shape[0],-1)

        output = xyz
        vol_tv = None
        if xyz.requires_grad:
            vol_tv = (self.feature_coefs,xyz_select)

        #print("STATIC DIFF: {}".format(torch.sum(torch.abs(self.feature_coefs[0].cpu() - self.og_static))))
        #print("DYNAMIC DIFF: {}".format(torch.sum(torch.abs(self.feature_coefs[1].cpu() - self.og_dynamic))))        
        
        # Typing: output gets converted to a tensor after the first iteration of the loop
        assert isinstance(output, Tensor)

        return output, vol_tv
        #print(output.shape)
        #exit(-1)
        #return output.reshape(*original_shape[:-1], (self.num_components * self.select_dim)), vol_tv


class SHEncoding(Encoding):
    """Spherical harmonic encoding

    Args:
        levels: Number of spherical harmonic levels to encode.
    """

    def __init__(self, levels: int = 4, implementation: Literal["tcnn", "torch"] = "torch") -> None:
        super().__init__(in_dim=3)

        if levels <= 0 or levels > 4:
            raise ValueError(f"Spherical harmonic encoding only supports 1 to 4 levels, requested {levels}")

        self.levels = levels

        self.tcnn_encoding = None
        if implementation == "tcnn" and not TCNN_EXISTS:
            print_tcnn_speed_warning("SHEncoding")
        elif implementation == "tcnn":
            encoding_config = {
                "otype": "SphericalHarmonics",
                "degree": levels,
            }
            self.tcnn_encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config=encoding_config,
            )

    def get_out_dim(self) -> int:
        return self.levels**2

    @torch.no_grad()
    def pytorch_fwd(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        """Forward pass using pytorch. Significantly slower than TCNN implementation."""
        return components_from_spherical_harmonics(levels=self.levels, directions=in_tensor)

    def forward(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        if self.tcnn_encoding is not None:
            return self.tcnn_encoding(in_tensor)
        return self.pytorch_fwd(in_tensor)
