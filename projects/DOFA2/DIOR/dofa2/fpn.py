from typing import List

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmdet.registry import MODELS


@MODELS.register_module()
class DOFALearnedFPN(BaseModule):
    """Terratorch-style DOFA neck for MMDetection.

    Equivalent to:
    ReshapeTokensToImage -> LearnedInterpolateToPyramidal -> FeaturePyramidNetworkNeck
    after ``DOFAV1ViT`` has already reshaped token features to NCHW.
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels=256,
        num_outs=5,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        if len(in_channels) != 4:
            raise ValueError("DOFALearnedFPN expects exactly 4 DOFA feature maps.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels[0], in_channels[0] // 2, 2, 2),
            nn.BatchNorm2d(in_channels[0] // 2),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels[0] // 2, in_channels[0] // 4, 2, 2),
        )
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels[1], in_channels[1] // 2, 2, 2))
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        pyramid_channels = [
            in_channels[0] // 4,
            in_channels[1] // 2,
            in_channels[2],
            in_channels[3],
        ]

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channel in pyramid_channels:
            self.lateral_convs.append(
                ConvModule(
                    in_channel,
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False,
                ))
            self.fpn_convs.append(
                ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False,
                ))

    def forward(self, inputs):
        if len(inputs) != 4:
            raise ValueError(f"Expected 4 DOFA features, got {len(inputs)}.")

        inputs = [
            self.fpn1(inputs[0]),
            self.fpn2(inputs[1]),
            self.fpn3(inputs[2]),
            self.fpn4(inputs[3]),
        ]

        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest")

        outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(self.num_ins)
        ]
        while len(outs) < self.num_outs:
            outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        return tuple(outs)
