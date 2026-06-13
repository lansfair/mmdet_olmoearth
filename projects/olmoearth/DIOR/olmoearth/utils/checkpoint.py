from __future__ import annotations

import json
import sys
import types
import warnings
from pathlib import Path
from typing import Any


def _unsupported_distributed_api(name: str):
    def _raise(*args, **kwargs):
        raise RuntimeError(
            f"OLMoEarth {name} requires newer PyTorch distributed APIs. "
            "This OpenMMLab integration only patches torch 2.3 import-time "
            "symbols for normal encoder forward. Do not call OLMoEarth "
            "apply_fsdp() or apply_ddp() in this setup."
        )

    return _raise


def _patch_torch_23_for_olmoearth_import() -> None:
    """Patch import-time PyTorch 2.3 gaps used by OLMoEarth.

    This is deliberately narrow: it only makes OLMoEarth importable for the
    normal encoder forward path used by MMDet. Distributed OLMoEarth FSDP/DDP
    helpers still fail loudly if called.
    """

    try:
        import torch
        import torch.distributed as dist
        import torch.distributed.fsdp as fsdp
    except Exception:
        return

    patched = False
    torch_version = tuple(
        int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2]
    )

    if torch_version >= (2, 7):
        return

    if not hasattr(fsdp, "MixedPrecisionPolicy"):

        class MixedPrecisionPolicy:
            def __init__(
                self,
                param_dtype=None,
                reduce_dtype=None,
                output_dtype=None,
                cast_forward_inputs=True,
                **kwargs,
            ) -> None:
                self.param_dtype = param_dtype
                self.reduce_dtype = reduce_dtype
                self.output_dtype = output_dtype
                self.cast_forward_inputs = cast_forward_inputs
                for key, value in kwargs.items():
                    setattr(self, key, value)

        fsdp.MixedPrecisionPolicy = MixedPrecisionPolicy
        patched = True

    if not hasattr(fsdp, "fully_shard"):
        fsdp.fully_shard = _unsupported_distributed_api("apply_fsdp()")
        patched = True

    if not hasattr(fsdp, "register_fsdp_forward_method"):
        fsdp.register_fsdp_forward_method = _unsupported_distributed_api(
            "register_fsdp_forward_method()"
        )
        patched = True

    if not hasattr(dist, "DeviceMesh"):
        try:
            from torch.distributed.device_mesh import DeviceMesh

            dist.DeviceMesh = DeviceMesh
        except Exception:

            class DeviceMesh:
                def __init__(self, *args, **kwargs) -> None:
                    _unsupported_distributed_api("DeviceMesh")(*args, **kwargs)

            dist.DeviceMesh = DeviceMesh
        patched = True

    try:
        import torch.distributed.tensor as dist_tensor

        if not hasattr(dist_tensor, "distribute_tensor"):

            def distribute_tensor(tensor, *args, **kwargs):
                return tensor

            dist_tensor.distribute_tensor = distribute_tensor
            patched = True
    except Exception:
        pass

    parent_name = "torch.distributed._composable"
    module_name = f"{parent_name}.replicate"
    if module_name not in sys.modules:
        parent_mod = sys.modules.get(parent_name)
        if parent_mod is None:
            parent_mod = types.ModuleType(parent_name)
            parent_mod.__path__ = []
            sys.modules[parent_name] = parent_mod
        replicate_mod = types.ModuleType(module_name)
        replicate_mod.replicate = _unsupported_distributed_api("apply_ddp()")
        sys.modules[module_name] = replicate_mod
        setattr(parent_mod, "replicate", replicate_mod)
        patched = True

    if patched:
        warnings.warn(
            "Applied a torch<2.7 import compatibility patch for OLMoEarth. "
            "Only normal encoder forward is supported; OLMoEarth FSDP/DDP "
            "helpers are intentionally disabled.",
            RuntimeWarning,
            stacklevel=2,
        )


def build_olmoearth_model(model_config_path: str | Path) -> Any:
    """Build an OLMoEarth model from the released config.json."""

    model_config_path = Path(model_config_path)
    if not model_config_path.exists():
        raise FileNotFoundError(
            f"model_config_path does not exist: {model_config_path}"
        )

    _patch_torch_23_for_olmoearth_import()

    from olmoearth_pretrain.config import Config
    from olmoearth_pretrain.model_loader import patch_legacy_encoder_config

    with open(model_config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    config_dict = patch_legacy_encoder_config(config_dict)
    model_config = Config.from_dict(config_dict["model"])
    return model_config.build()
