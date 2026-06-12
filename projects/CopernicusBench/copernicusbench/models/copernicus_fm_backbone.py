import torch
from mmdet.registry import MODELS
from mmengine.logging import print_log
from mmengine.model import BaseModule
from mmengine.runner.checkpoint import CheckpointLoader

from .backbones.models_dwv_seg import (
    vit_base_patch16,
    vit_large_patch16,
    vit_small_patch16,
)


@MODELS.register_module()
class CopernicusFMBackbone(BaseModule):
    """Copernicus-FM ViT backbone registered for MMDetection."""

    arch_zoo = {
        "small": (vit_small_patch16, 384),
        "base": (vit_base_patch16, 768),
        "large": (vit_large_patch16, 1024),
    }
    language_key_aliases = {
        "s5p_co": "Sentinel 5P Carbon Monoxide",
        "s5p_no2": "Sentinel 5P Nitrogen Dioxide",
        "s5p_o3": "Sentinel 5P Ozone",
        "s5p_so2": "Sentinel 5P Sulfur Dioxide",
        "dem": "Copernicus Digital Elevation Model",
    }

    def __init__(
        self,
        arch="base",
        band_wavelengths=(665, 560, 490),
        band_bandwidths=(30, 35, 65),
        input_mode="spectral",
        kernel_size=16,
        key=None,
        language_embed=None,
        language_key=None,
        loc_option="lonlat",
        var_option="spectrum",
        patch_area=None,
        norm_eval=False,
        frozen_exclude=("all",),
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        if arch not in self.arch_zoo:
            raise ValueError(f"Unsupported Copernicus-FM arch: {arch}")

        builder, embed_dim = self.arch_zoo[arch]
        self.encoder = builder(loc_option=loc_option, var_option=var_option)
        self.out_channels = [embed_dim] * 4
        self.band_wavelengths = list(band_wavelengths)
        self.band_bandwidths = list(band_bandwidths)
        self.input_mode = input_mode
        self.kernel_size = kernel_size
        self.key = key
        self.language_embed = None
        self.language_embed_path = language_embed
        self.language_key = language_key
        self.patch_area = patch_area
        self.norm_eval = norm_eval
        self.frozen_exclude = tuple(frozen_exclude)
        if self.input_mode == "variable" and var_option == "language":
            if self.key is None:
                raise ValueError(
                    'key must be set when input_mode="variable" and '
                    'var_option="language".'
                )
            if self.language_embed_path is None:
                raise ValueError(
                    "language_embed must be set in the config when "
                    'input_mode="variable" and var_option="language".'
                )

    def init_weights(self):
        if (
            isinstance(self.init_cfg, dict)
            and self.init_cfg.get("type") == "Pretrained"
        ):
            self._load_pretrained(self.init_cfg["checkpoint"])
        elif self.init_cfg is not None:
            super().init_weights()
        if self.language_embed_path is not None:
            self.language_embed = self._load_language_embed(
                self.language_embed_path,
                self.language_key,
            )
        self._freeze()

    def _prepare_pretrained_state_dict(self, checkpoint):
        if isinstance(checkpoint, dict):
            for key in ("model", "state_dict", "model_state_dict", "net", "network"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break
        if not isinstance(checkpoint, dict):
            raise TypeError(
                "Unsupported Copernicus-FM checkpoint type: "
                f"{type(checkpoint)}"
            )

        prefixes = (
            "module.",
            "model.",
            "model.encoder.",
            "model.backbone.",
            "model.backbone.cfm.",
            "backbone.cfm.",
            "backbone.encoder.",
            "backbone.",
            "cfm.",
            "encoder.",
        )
        mae_only_prefixes = (
            "decoder_",
            "decoder.",
            "decoder_blocks.",
            "decoder_norm.",
            "decoder_pred_",
            "coord_fc_decoder.",
            "scale_fc_decoder.",
            "time_fc_decoder.",
            "teacher.",
            "student_proj.",
        )
        mae_only_keys = {
            "mask_token",
            "decoder_pos_embed",
            "coord_token_dec",
            "scale_token_dec",
            "time_token_dec",
        }

        encoder_state = self.encoder.state_dict()
        normalized = {}
        dropped_unexpected = []
        dropped_shape = []
        for raw_key, value in checkpoint.items():
            key = raw_key
            changed = True
            while changed:
                changed = False
                for prefix in prefixes:
                    if key.startswith(prefix):
                        key = key[len(prefix):]
                        changed = True
                        break
            if key in mae_only_keys or key.startswith(mae_only_prefixes):
                dropped_unexpected.append(raw_key)
                continue
            if key not in encoder_state:
                dropped_unexpected.append(raw_key)
                continue
            if value.shape != encoder_state[key].shape:
                dropped_shape.append((raw_key, tuple(value.shape), tuple(encoder_state[key].shape)))
                continue
            normalized[key] = value

        if dropped_shape:
            preview = "; ".join(
                f"{key}: {src} -> {dst}" for key, src, dst in dropped_shape[:10]
            )
            raise RuntimeError(
                "Shape mismatch when loading Copernicus-FM checkpoint. "
                "This usually means arch/model_size is wrong, e.g. loading "
                "a base checkpoint with arch=large. First mismatches: "
                f"{preview}"
            )
        if dropped_unexpected:
            print_log(
                "Ignored non-encoder Copernicus-FM checkpoint keys: "
                f"{len(dropped_unexpected)}",
                logger="current",
            )
        return normalized

    def _load_pretrained(self, checkpoint_path):
        checkpoint = CheckpointLoader.load_checkpoint(
            checkpoint_path,
            logger=None,
            map_location="cpu",
        )
        state_dict = self._prepare_pretrained_state_dict(checkpoint)
        incompatible = self.encoder.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            missing = ", ".join(incompatible.missing_keys[:20])
            raise RuntimeError(
                "Missing keys when loading Copernicus-FM backbone after "
                "checkpoint key normalization. Check that the config arch "
                "matches the pretraining model size and that this is a "
                "Copernicus-FM checkpoint. First missing keys: "
                f"{missing}"
            )
        if incompatible.unexpected_keys:
            unexpected = ", ".join(incompatible.unexpected_keys[:20])
            print_log(
                "Unexpected keys ignored when loading Copernicus-FM "
                f"backbone: {unexpected}",
                logger="current",
            )
        print_log(
            f"Loaded Copernicus-FM weights from {checkpoint_path}",
            logger="current",
        )
    def _freeze(self):
        if "all" in self.frozen_exclude:
            return
        for name, param in self.named_parameters():
            if not any(exclude in name for exclude in self.frozen_exclude):
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self._freeze()
        if mode and self.norm_eval:
            for module in self.modules():
                if isinstance(module, torch.nn.LayerNorm):
                    module.eval()
        return self

    def _load_language_embed(self, filename, key):
        if filename is None:
            return None
        embeds = torch.load(filename, map_location="cpu")
        candidate_keys = []
        if key is not None:
            candidate_keys.append(key)
        if self.key is not None:
            candidate_keys.append(self.key)
            candidate_keys.append(
                self.language_key_aliases.get(self.key, self.key)
            )
        for candidate_key in candidate_keys:
            if candidate_key in embeds:
                return embeds[candidate_key]
        available = ", ".join(str(k) for k in embeds.keys())
        requested = ", ".join(str(k) for k in candidate_keys)
        raise KeyError(
            "Unable to find language embedding key. Requested candidates: "
            f"{requested}. Available keys: {available}"
        )

    def _default_meta(self, x):
        meta = torch.full(
            (x.shape[0], 4),
            float("nan"),
            dtype=x.dtype,
            device=x.device,
        )
        if self.patch_area is not None:
            meta[:, 3] = self.patch_area
        return meta

    def forward(self, x, meta=None):
        if meta is None:
            meta = self._default_meta(x)
        else:
            meta = meta.to(device=x.device, dtype=x.dtype)
        feats = self.encoder(
            x,
            meta,
            self.key,
            self.band_wavelengths,
            self.band_bandwidths,
            self.language_embed,
            self.input_mode,
            self.kernel_size,
        )
        return tuple(feats)
