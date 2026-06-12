from __future__ import annotations

from mmdet.models.detectors import FasterRCNN
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from torch import Tensor


@MODELS.register_module()
class OlmoEarthFasterRCNN(FasterRCNN):
    """Faster R-CNN wrapper that forwards metainfo to OLMoEarth backbone."""

    def _set_olmoearth_metainfo(
        self,
        batch_data_samples: SampleList | None,
    ) -> None:
        if not hasattr(self.backbone, "set_batch_metainfo"):
            return
        if batch_data_samples is None:
            self.backbone.set_batch_metainfo(None)
            return
        self.backbone.set_batch_metainfo(
            [data_sample.metainfo for data_sample in batch_data_samples]
        )

    def loss(
        self,
        batch_inputs: Tensor,
        batch_data_samples: SampleList,
    ) -> dict:
        self._set_olmoearth_metainfo(batch_data_samples)
        return super().loss(batch_inputs, batch_data_samples)

    def predict(
        self,
        batch_inputs: Tensor,
        batch_data_samples: SampleList,
        rescale: bool = True,
    ) -> SampleList:
        self._set_olmoearth_metainfo(batch_data_samples)
        return super().predict(batch_inputs, batch_data_samples, rescale)

    def _forward(
        self,
        batch_inputs: Tensor,
        batch_data_samples: SampleList | None = None,
    ) -> tuple:
        self._set_olmoearth_metainfo(batch_data_samples)
        return super()._forward(batch_inputs, batch_data_samples)
