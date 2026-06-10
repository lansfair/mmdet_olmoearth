from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmengine.model.weight_init import xavier_init
from mmdet.registry import MODELS
from torch import Tensor


@MODELS.register_module()
class DINOv3MultiLevelNeck(BaseModule):
    """Build a feature pyramid from DINOv3 intermediate features.

    DINOv3 ViT features selected from intermediate layers have the same patch
    stride. This neck follows MMSeg's MultiLevelNeck convention: each selected
    feature is projected to a common channel width, resized by its configured
    scale factor, then smoothed with a 3x3 convolution.
    """

    def __init__(
        self,
        in_channels: list[int],
        out_channels: int,
        scales: list[float],
        conv_cfg: dict | None = None,
        norm_cfg: dict | None = None,
        act_cfg: dict | None = None,
        init_cfg: dict | None = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        if not isinstance(in_channels, list):
            raise TypeError("in_channels must be a list")
        if len(scales) == 0:
            raise ValueError("scales must not be empty")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scales = scales
        self.num_outs = len(scales)
        self.lateral_convs = nn.ModuleList()
        self.convs = nn.ModuleList()

        for in_channel in in_channels:
            self.lateral_convs.append(
                ConvModule(
                    in_channel,
                    out_channels,
                    kernel_size=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
            )
        for _ in scales:
            self.convs.append(
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
            )

    def init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                xavier_init(module, distribution="uniform")

    def forward(self, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        if not isinstance(inputs, (tuple, list)):
            inputs = (inputs,)
        if len(inputs) != len(self.in_channels):
            raise ValueError(
                "DINOv3MultiLevelNeck expects "
                f"{len(self.in_channels)} input features, got {len(inputs)}"
            )

        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        if len(laterals) == 1:
            laterals = [laterals[0] for _ in range(self.num_outs)]

        outs = []
        for i, scale in enumerate(self.scales):
            resized = laterals[i]
            if scale != 1:
                resized = F.interpolate(
                    resized,
                    scale_factor=scale,
                    mode="bilinear",
                    align_corners=False,
                    recompute_scale_factor=True,
                )
            outs.append(self.convs[i](resized))
        return tuple(outs)
