import torch.nn as nn
from mmcv.cnn import ConvModule, build_norm_layer
from mmdet.registry import MODELS


@MODELS.register_module()
class CopernicusFeature2Pyramid(nn.Module):
    """Build an FPN-like pyramid from Copernicus-FM same-stride features."""

    def __init__(
        self,
        embed_dim,
        out_channels,
        rescales=(4, 2, 1, 0.5),
        norm_cfg=dict(type="BN", requires_grad=True),
        conv_cfg=None,
        act_cfg=dict(type="ReLU"),
    ):
        super().__init__()
        self.rescales = tuple(rescales)
        ops = []
        for scale in self.rescales:
            if scale == 4:
                ops.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            embed_dim,
                            embed_dim,
                            kernel_size=2,
                            stride=2,
                        ),
                        build_norm_layer(norm_cfg, embed_dim)[1],
                        nn.GELU(),
                        nn.ConvTranspose2d(
                            embed_dim,
                            embed_dim,
                            kernel_size=2,
                            stride=2,
                        ),
                    )
                )
            elif scale == 2:
                ops.append(
                    nn.ConvTranspose2d(
                        embed_dim,
                        embed_dim,
                        kernel_size=2,
                        stride=2,
                    )
                )
            elif scale == 1:
                ops.append(nn.Identity())
            elif scale == 0.5:
                ops.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif scale == 0.25:
                ops.append(nn.MaxPool2d(kernel_size=4, stride=4))
            else:
                raise KeyError(f"invalid {scale} for feature2pyramid")
        self.ops = nn.ModuleList(ops)
        self.convs = nn.ModuleList(
            [
                ConvModule(
                    embed_dim,
                    out_channels,
                    kernel_size=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
                for _ in self.rescales
            ]
        )

    def forward(self, inputs):
        if len(inputs) != len(self.rescales):
            raise ValueError(
                "CopernicusFeature2Pyramid expects "
                f"{len(self.rescales)} feature maps, got {len(inputs)}"
            )
        return tuple(
            conv(op(feature))
            for feature, op, conv in zip(inputs, self.ops, self.convs)
        )
