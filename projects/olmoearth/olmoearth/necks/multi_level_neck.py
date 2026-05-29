from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmdet.registry import MODELS
from torch import Tensor


@MODELS.register_module()
class OlmoEarthMultiLevelNeck(BaseModule):
    """Build multiple detection feature levels from one OLMoEarth map."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scales: list[float],
        conv_cfg: dict | None = None,
        norm_cfg: dict | None = None,
        act_cfg: dict | None = dict(type="ReLU"),
        init_cfg: dict | None = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.scales = scales
        self.convs = nn.ModuleList(
            [
                ConvModule(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
                for _ in scales
            ]
        )

    def forward(self, inputs: tuple[Tensor]) -> tuple[Tensor, ...]:
        if len(inputs) != 1:
            raise ValueError(
                "OlmoEarthMultiLevelNeck expects one backbone feature map, "
                f"got {len(inputs)}"
            )
        feature = inputs[0]
        outs = []
        for scale, conv in zip(self.scales, self.convs):
            if scale == 1:
                resized = feature
            else:
                resized = F.interpolate(
                    feature,
                    scale_factor=scale,
                    mode="bilinear",
                    align_corners=False,
                    recompute_scale_factor=True,
                )
            outs.append(conv(resized))
        return tuple(outs)
