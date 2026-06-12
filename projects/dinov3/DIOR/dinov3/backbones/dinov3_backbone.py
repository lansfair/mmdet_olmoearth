from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from mmengine.model import BaseModule
from mmdet.registry import MODELS
from torch import Tensor


@MODELS.register_module()
class DINOv3ViTBackbone(BaseModule):
    """DINOv3 ViT backbone loaded from a local torch.hub repo.

    The backbone returns a single dense patch feature map in MMSegmentation
    format: ``(B, C, H // patch_size, W // patch_size)``.
    """

    def __init__(
        self,
        repo_dir: str,
        model_name: str = "dinov3_vitl16",
        weights_path: str | None = None,
        patch_size: int = 16,
        out_channels: int = 1024,
        freeze: bool = True,
        hub_kwargs: dict[str, Any] | None = None,
        init_cfg: dict | None = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.repo_dir = str(repo_dir)
        self.model_name = model_name
        self.weights_path = str(weights_path) if weights_path else None
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.freeze = freeze
        self.hub_kwargs = hub_kwargs or {}
        self.model = self._load_model()
        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    def _load_model(self):
        repo_dir = Path(self.repo_dir)
        if not repo_dir.exists():
            raise FileNotFoundError(
                f"DINOv3 repo_dir does not exist: {repo_dir}"
            )
        kwargs = dict(self.hub_kwargs)
        if self.weights_path is not None:
            weights_path = Path(self.weights_path)
            if not weights_path.exists():
                raise FileNotFoundError(
                    f"DINOv3 weights_path does not exist: {weights_path}"
                )
            kwargs.setdefault("weights", str(weights_path))
        return torch.hub.load(
            str(repo_dir),
            self.model_name,
            source="local",
            **kwargs,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def _tokens_to_feature_map(
        self,
        tokens: Tensor,
        height: int,
        width: int,
    ) -> Tensor:
        if tokens.ndim == 4:
            return tokens.contiguous()
        if tokens.ndim != 3:
            raise ValueError(
                "Expected DINOv3 tokens as BCHW or BNC, "
                f"got shape {tuple(tokens.shape)}"
            )
        batch_size, num_tokens, channels = tokens.shape
        grid_h = height // self.patch_size
        grid_w = width // self.patch_size
        expected_tokens = grid_h * grid_w
        if num_tokens != expected_tokens:
            raise ValueError(
                "DINOv3 patch token count does not match input size: "
                f"got {num_tokens}, expected {expected_tokens} "
                f"for image size {(height, width)} and patch_size={self.patch_size}"
            )
        return tokens.transpose(1, 2).reshape(
            batch_size,
            channels,
            grid_h,
            grid_w,
        ).contiguous()

    def _extract_tokens(self, inputs: Tensor) -> Tensor:
        if hasattr(self.model, "get_intermediate_layers"):
            features = self.model.get_intermediate_layers(
                inputs,
                n=1,
                reshape=True,
                return_class_token=False,
            )
            if isinstance(features, (tuple, list)):
                return features[-1]
            return features

        features = self.model.forward_features(inputs)
        if isinstance(features, dict):
            for key in (
                "x_norm_patchtokens",
                "x_prenorm",
                "patch_tokens",
                "tokens",
            ):
                if key in features:
                    return features[key]
        return features

    def forward(self, inputs: Tensor) -> tuple[Tensor]:
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
        with torch.set_grad_enabled(not self.freeze):
            tokens = self._extract_tokens(inputs)
        return (self._tokens_to_feature_map(tokens, height, width),)
    

@MODELS.register_module()
class DINOv3ViTBackbone2(BaseModule):
    """DINOv3 ViT backbone loaded from a local torch.hub repo.

    The backbone returns a single dense patch feature map in MMSegmentation
    format: ``(B, C, H // patch_size, W // patch_size)``.
    """

    def __init__(
        self,
        repo_dir: str,
        model_name: str = "dinov3_vitl16",
        weights_path: str | None = None,
        patch_size: int = 16,
        out_channels: int = 1024,
        freeze: bool = True,
        hub_kwargs: dict[str, Any] | None = None,
        init_cfg: dict | None = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.repo_dir = str(repo_dir)
        self.model_name = model_name
        self.weights_path = str(weights_path) if weights_path else None
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.freeze = freeze
        self.hub_kwargs = hub_kwargs or {}
        self.model = self._load_model()
        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    def __getattribute__(self, name):
        if name == 'init_weights':
            raise AttributeError(f"'{name}' is hidden")
        return super().__getattribute__(name)
    
    def _load_model(self):
        repo_dir = Path(self.repo_dir)
        if not repo_dir.exists():
            raise FileNotFoundError(
                f"DINOv3 repo_dir does not exist: {repo_dir}"
            )
        kwargs = dict(self.hub_kwargs)
        if self.weights_path is not None:
            weights_path = Path(self.weights_path)
            if not weights_path.exists():
                raise FileNotFoundError(
                    f"DINOv3 weights_path does not exist: {weights_path}"
                )
            kwargs.setdefault("weights", str(weights_path))
        return torch.hub.load(
            str(repo_dir),
            self.model_name,
            source="local",
            **kwargs,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def _tokens_to_feature_map(
        self,
        tokens: Tensor,
        height: int,
        width: int,
    ) -> Tensor:
        if tokens.ndim == 4:
            return tokens.contiguous()
        if tokens.ndim != 3:
            raise ValueError(
                "Expected DINOv3 tokens as BCHW or BNC, "
                f"got shape {tuple(tokens.shape)}"
            )
        batch_size, num_tokens, channels = tokens.shape
        grid_h = height // self.patch_size
        grid_w = width // self.patch_size
        expected_tokens = grid_h * grid_w
        if num_tokens != expected_tokens:
            raise ValueError(
                "DINOv3 patch token count does not match input size: "
                f"got {num_tokens}, expected {expected_tokens} "
                f"for image size {(height, width)} and patch_size={self.patch_size}"
            )
        return tokens.transpose(1, 2).reshape(
            batch_size,
            channels,
            grid_h,
            grid_w,
        ).contiguous()

    def _extract_tokens(self, inputs: Tensor) -> Tensor:
        if hasattr(self.model, "get_intermediate_layers"):
            features = self.model.get_intermediate_layers(
                inputs,
                n=1,
                reshape=True,
                return_class_token=False,
            )
            if isinstance(features, (tuple, list)):
                return features[-1]
            return features

        features = self.model.forward_features(inputs)
        if isinstance(features, dict):
            for key in (
                "x_norm_patchtokens",
                "x_prenorm",
                "patch_tokens",
                "tokens",
            ):
                if key in features:
                    return features[key]
        return features

    def forward(self, inputs: Tensor) -> tuple[Tensor]:
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
        with torch.set_grad_enabled(not self.freeze):
            tokens = self._extract_tokens(inputs)
        return (self._tokens_to_feature_map(tokens, height, width),)