from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from mmengine.logging import MMLogger
from mmengine.model import BaseModule
from mmdet.registry import MODELS
from torch import Tensor


@MODELS.register_module()
class DINOv3ViTBackbone(BaseModule):
    """DINOv3 ViT backbone loaded from a local torch.hub repo.

    The backbone returns dense patch feature maps in MMDetection format. By
    default it returns the final feature map. Set ``out_indices`` to return
    multiple intermediate layers. ``return_tuple=False`` is useful when the
    next neck expects one tensor, e.g. ViTDet ``SimpleFPN``.
    """

    def __init__(
        self,
        repo_dir: str,
        model_name: str = "dinov3_vitl16",
        weights_path: str | None = None,
        patch_size: int = 16,
        out_channels: int = 1024,
        freeze: bool = True,
        out_indices: int | Sequence[int] | None = None,
        return_tuple: bool = True,
        hub_kwargs: dict[str, Any] | None = None,
        init_cfg: dict | None = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.repo_dir = str(repo_dir)
        self.model_name = model_name
        self.weights_path = str(weights_path) if weights_path else None
        self.patch_size = patch_size
        self.freeze = freeze
        self.out_indices = self._normalize_out_indices(out_indices)
        self.return_tuple = return_tuple
        self.hub_kwargs = hub_kwargs or {}
        self.model = self._load_model()
        self.out_channels = (
            [out_channels] * len(self.out_indices)
            if self.out_indices is not None
            else out_channels
        )
        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    @staticmethod
    def _normalize_out_indices(
        out_indices: int | Sequence[int] | None,
    ) -> tuple[int, ...] | None:
        if out_indices is None:
            return None
        if isinstance(out_indices, int):
            return (out_indices,)
        return tuple(out_indices)

    def _load_model(self):
        repo_dir = Path(self.repo_dir)
        if not repo_dir.exists():
            raise FileNotFoundError(
                f"DINOv3 repo_dir does not exist: {repo_dir}"
            )
        kwargs = dict(self.hub_kwargs)
        weights_path = None
        if self.weights_path is not None:
            weights_path = Path(self.weights_path)
            if not weights_path.exists():
                raise FileNotFoundError(
                    f"DINOv3 weights_path does not exist: {weights_path}"
                )
            kwargs["weights"] = str(weights_path)
            kwargs["pretrained"] = False
        model = torch.hub.load(
            str(repo_dir),
            self.model_name,
            source="local",
            **kwargs,
        )
        if weights_path is not None:
            checkpoint = torch.load(
                str(weights_path),
                map_location="cpu",
            )
            if isinstance(checkpoint, dict):
                checkpoint = checkpoint.get(
                    "state_dict",
                    checkpoint.get("model", checkpoint),
                )
            incompatible = model.load_state_dict(checkpoint, strict=True)
            logger = MMLogger.get_current_instance()
            logger.info(
                "Loaded DINOv3 backbone weights from local path: "
                f"{weights_path}. "
                f"missing_keys={len(incompatible.missing_keys)}, "
                f"unexpected_keys={len(incompatible.unexpected_keys)}"
            )
        return model

    def init_weights(self) -> None:
        """Keep DINOv3 weights loaded during model construction."""
        return

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def forward(self, inputs: Tensor) -> tuple[Tensor, ...]:
        if inputs.shape[1] != 3:
            raise ValueError(
                f"DINOv3ViTBackbone expects 3-channel RGB inputs, "
                f"got {inputs.shape[1]} channels"
            )
        height, width = inputs.shape[-2:]
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                "DINOv3 input size must be divisible by patch_size, "
                f"got {(height, width)} and patch_size={self.patch_size}"
            )
        if not hasattr(self.model, "get_intermediate_layers"):
            raise AttributeError(
                "DINOv3ViTBackbone requires get_intermediate_layers."
            )

        with torch.set_grad_enabled(not self.freeze):
            if self.out_indices is not None:
                features = self.model.get_intermediate_layers(
                    inputs,
                    n=self.out_indices,
                    reshape=True,
                    norm=True,
                    return_class_token=False,
                )
            else:
                features = self.model.get_intermediate_layers(
                    inputs,
                    n=1,
                    reshape=True,
                    return_class_token=False,
                )
                if isinstance(features, (tuple, list)):
                    features = features[-1]

        if isinstance(features, Tensor):
            feature = features.contiguous()
            return (feature,) if self.return_tuple else feature
        features = tuple(feature.contiguous() for feature in features)
        return features
