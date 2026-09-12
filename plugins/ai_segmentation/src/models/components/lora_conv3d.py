import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAConv3d(nn.Conv3d):
    """Conv3d with an optional low-rank update added to the base kernel."""

    def __init__(self, *args, lora_rank=0, lora_alpha=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha if lora_alpha is not None else self.lora_rank)
        self.lora_scaling = (self.lora_alpha / self.lora_rank) if self.lora_rank > 0 else 0.0

        if self.lora_rank > 0:
            in_channels_per_group = self.in_channels // self.groups
            self.lora_A = nn.Parameter(
                torch.empty(self.lora_rank, in_channels_per_group, *self.kernel_size)
            )
            self.lora_B = nn.Parameter(torch.empty(self.out_channels, self.lora_rank))
            self.reset_lora_parameters()
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def reset_lora_parameters(self):
        if self.lora_rank == 0:
            return

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def _lora_delta_weight(self):
        if self.lora_rank == 0:
            return None

        delta_weight = torch.einsum("or,rijkl->oijkl", self.lora_B, self.lora_A)
        return delta_weight * self.lora_scaling

    def forward(self, input_tensor):
        weight = self.weight
        delta_weight = self._lora_delta_weight()
        if delta_weight is not None:
            weight = weight + delta_weight.to(dtype=weight.dtype, device=weight.device)

        return F.conv3d(
            input_tensor,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
    

class DoRAConv3d(nn.Conv3d):
    """Conv3d with DoRA: frozen/base direction + trainable magnitude + LoRA update."""

    def __init__(self, *args, lora_rank=0, lora_alpha=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha if lora_alpha is not None else self.lora_rank)
        self.lora_scaling = (self.lora_alpha / self.lora_rank) if self.lora_rank > 0 else 0.0

        if self.lora_rank > 0:
            in_channels_per_group = self.in_channels // self.groups
            self.lora_A = nn.Parameter(
                torch.empty(self.lora_rank, in_channels_per_group, *self.kernel_size)
            )
            self.lora_B = nn.Parameter(torch.empty(self.out_channels, self.lora_rank))
            self.magnitude = nn.Parameter(self._current_weight_norm())
            self.reset_lora_parameters()
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.register_parameter("magnitude", None)

    def _current_weight_norm(self):
        with torch.no_grad():
            weight_matrix = self.weight.detach().view(self.out_channels, -1)
            return torch.norm(weight_matrix, p=2, dim=1).clamp_min(1e-6)

    def reset_magnitude_from_weight(self):
        if self.lora_rank == 0:
            return
        self.magnitude.data.copy_(self._current_weight_norm().to(self.magnitude.device))

    def reset_lora_parameters(self):
        if self.lora_rank == 0:
            return

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def _lora_delta_weight(self):
        delta_weight = torch.einsum("or,rijkl->oijkl", self.lora_B, self.lora_A)
        return delta_weight * self.lora_scaling

    def forward(self, input_tensor):
        if self.lora_rank == 0:
            return F.conv3d(
                input_tensor,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

        delta_weight = self._lora_delta_weight().to(
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        full_weight = self.weight + delta_weight

        weight_norm = full_weight.view(self.out_channels, -1).norm(
            p=2,
            dim=1,
        ).clamp_min(1e-6)

        direction = full_weight / weight_norm.view(self.out_channels, 1, 1, 1, 1)
        dora_weight = direction * self.magnitude.view(self.out_channels, 1, 1, 1, 1)

        return F.conv3d(
            input_tensor,
            dora_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )    