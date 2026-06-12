import logging
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from timm.models.vision_transformer import VisionTransformer
from mmdet.registry import MODELS
from mmengine.runner.checkpoint import CheckpointLoader

from . import utils


class TransformerWeightGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, num_heads=4, num_layers=1):
        super(TransformerWeightGenerator, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            activation="gelu",
            norm_first=False,
            batch_first=False,
            dropout=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # Linear layer to map transformer output to desired weight shape
        self.fc_weight = nn.Linear(input_dim, output_dim)
        self.fc_bias = nn.Linear(input_dim, embed_dim)
        self.wt_num = 128
        self.weight_tokens = nn.Parameter(torch.empty([self.wt_num, input_dim]))
        self.bias_token = nn.Parameter(torch.empty([1, input_dim]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        nn.init.normal_(self.weight_tokens, std=0.02)
        nn.init.normal_(self.bias_token, std=0.02)

    def forward(self, x):
        # x should have shape [seq_len, batch, input_dim]
        pos_wave = x
        x = torch.cat([self.weight_tokens, pos_wave], dim=0)
        x = torch.cat([x, self.bias_token], dim=0)
        transformer_output = self.transformer_encoder(x)
        weights = self.fc_weight(transformer_output[self.wt_num : -1] + pos_wave)
        bias = self.fc_bias(
            transformer_output[-1]
        )  # Using the last output to generate bias
        return weights, bias
    

class FCResLayer(nn.Module):
    def __init__(self, linear_size=128):
        super(FCResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        # self.dropout1 = nn.Dropout()
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        # y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y
        return out
    

def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


class DynamicMLPOFAV2(nn.Module):
    """DOFA dynamic patch embedding with optional patch14-to-patch16 conversion."""

    def __init__(
        self,
        wv_planes=128,
        inter_dim=128,
        kernel_size=14,
        embed_dim=1024,
        convert_patch_14_to_16=False,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.wv_planes = wv_planes
        self.embed_dim = embed_dim
        self.inter_dim = inter_dim
        self.patch_size = (kernel_size, kernel_size)
        self.num_patches = -1
        self.convert_patch_14_to_16 = convert_patch_14_to_16
        self.weight_generator = TransformerWeightGenerator(
            wv_planes, kernel_size * kernel_size * embed_dim, embed_dim)
        self.scaler = 0.01
        self.fclayer = FCResLayer(wv_planes)
        self._init_weights()

    def weight_init(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    def _init_weights(self):
        self.weight_generator.apply(self.weight_init)
        self.fclayer.apply(self.weight_init)

    def forward(self, img_feat, wvs):
        inplanes = wvs.size(0)
        waves = get_1d_sincos_pos_embed_from_grid_torch(self.wv_planes, wvs * 1000)
        waves = self.fclayer(waves)
        weight, bias = self.weight_generator(waves)
        dynamic_weight = weight.view(
            inplanes, self.kernel_size, self.kernel_size, self.embed_dim)
        dynamic_weight = dynamic_weight.permute(3, 0, 1, 2)

        if bias is not None:
            bias = bias.view([self.embed_dim]) * self.scaler
        weights = dynamic_weight * self.scaler

        stride = self.kernel_size
        if self.convert_patch_14_to_16:
            if self.kernel_size != 14:
                raise ValueError("convert_patch_14_to_16 requires kernel_size=14.")
            stride = 16
            weights = F.interpolate(
                weights, size=(16, 16), mode="bicubic", align_corners=False)

        x = F.conv2d(img_feat, weights, bias=bias, stride=stride, padding=1, dilation=1)
        x = x.flatten(2).transpose(1, 2)
        return x, waves


@MODELS.register_module(force=True)
class DOFAV2ViT(BaseModule):
    """DOFA v2 ViT backbone for MMDetection.

    DOFAv2 follows Terratorch's patch14 setup and uses timm's dynamic ViT
    position embedding path. With ``convert_patch_14_to_16=True`` it resizes
    the learned dynamic patch kernels to 16x16 and uses stride 16, matching the
    Terratorch object-detection configs.
    """

    def __init__(
        self,
        arch="large",
        img_size=896,
        patch_size=14,
        out_indices=None,
        model_bands=("RED", "GREEN", "BLUE"),
        pos_interpolation_mode="bilinear",
        convert_patch_14_to_16=True,
        wv_planes=128,
        mlp_ratio=4.0,
        drop_path_rate=0.1,
        frozen_stages=True,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        arch_settings = utils.get_arch_setting(arch)
        self.effective_patch_size = 16 if convert_patch_14_to_16 else patch_size
        self.embed_dim = arch_settings["embed_dim"]
        self.depth = arch_settings["depth"]
        self.frozen_stages = frozen_stages
        self.pos_interpolation_mode = pos_interpolation_mode
        self.convert_patch_14_to_16 = convert_patch_14_to_16
        self.out_indices = tuple(out_indices or arch_settings["default_out_indices"])

        wavelengths = utils.get_wavelenghts(model_bands)
        self.register_buffer(
            "wavelengths",
            torch.tensor(wavelengths, dtype=torch.float32),
            persistent=False,
        )

        self.patch_embed = DynamicMLPOFAV2(
            wv_planes=wv_planes,
            inter_dim=wv_planes,
            kernel_size=patch_size,
            embed_dim=self.embed_dim,
            convert_patch_14_to_16=convert_patch_14_to_16,
        )
        self.patch_embed.num_patches = (img_size // self.effective_patch_size) ** 2

        model_args = dict(
            patch_size=patch_size,
            embed_dim=self.embed_dim,
            depth=self.depth,
            drop_path_rate=drop_path_rate,
            num_heads=arch_settings["num_heads"],
            mlp_ratio=mlp_ratio,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        self.model = VisionTransformer(**model_args)
        del self.model.patch_embed.proj

    def init_weights(self):
        retval = super().init_weights()
        checkpoint_path = None
        if isinstance(self.init_cfg, dict) and self.init_cfg.get("type") == "Pretrained":
            checkpoint_path = self.init_cfg.get("checkpoint")

        checkpoint = CheckpointLoader.load_checkpoint(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        checkpoint = {
            key.replace("module.", "", 1): value
            for key, value in checkpoint.items()
        }
        msg = self.load_state_dict(checkpoint, strict=False)
        for params in self.parameters():
            params.requires_grad = not self.frozen_stages
        logging.info(f"Loaded DOFAv2 checkpoint: {msg}")
        return retval

    def _patch_hw(self, x):
        input_h, input_w = x.shape[-2:]
        patch_size = self.effective_patch_size
        return (
            (input_h + 2 - patch_size) // patch_size + 1,
            (input_w + 2 - patch_size) // patch_size + 1,
        )

    def _tokens_to_image(self, x, hw_shape):
        x = x[:, 1:, :]
        x = x.reshape(x.shape[0], hw_shape[0], hw_shape[1], self.embed_dim)
        return x.permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        hw_shape = self._patch_hw(x)
        wavelengths = self.wavelengths.to(device=x.device, dtype=torch.float32)

        x, _ = self.patch_embed(x, wavelengths)
        if hw_shape[0] * hw_shape[1] != x.shape[1]:
            raise RuntimeError(f"Expected {hw_shape[0] * hw_shape[1]} DOFAv2 tokens, got {x.shape[1]}.")

        batch_size, _, channels = x.shape
        x = x.view(batch_size, hw_shape[0], hw_shape[1], channels)
        x = self.model._pos_embed(x)
        x = self.model.patch_drop(x)
        x = self.model.norm_pre(x)

        outs = []
        for i, block in enumerate(self.model.blocks):
            x = block(x)
            if i in self.out_indices:
                outs.append(self._tokens_to_image(x, hw_shape))
        return tuple(outs)
