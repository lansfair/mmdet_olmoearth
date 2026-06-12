from __future__ import annotations

from dataclasses import dataclass


SENTINEL2_L2A_BANDS = [
    "B02",
    "B03",
    "B04",
    "B08",
    "B05",
    "B06",
    "B07",
    "B8A",
    "B11",
    "B12",
    "B01",
    "B09",
]

SENTINEL1_BANDS = ["vv", "vh"]

RGB_BANDS = ["B", "G", "R", "NIR"]

RGB_TO_SENTINEL2_L2A = {"R": "B04", "G": "B03", "B": "B02"}


@dataclass(frozen=True)
class ModalitySpec:
    name: str
    bands: tuple[str, ...]
    sample_field: str


MODALITY_SPECS = {
    "sentinel2_l2a": ModalitySpec(
        name="sentinel2_l2a",
        bands=tuple(SENTINEL2_L2A_BANDS),
        sample_field="sentinel2_l2a",
    ),
    "sentinel1": ModalitySpec(
        name="sentinel1",
        bands=tuple(SENTINEL1_BANDS),
        sample_field="sentinel1",
    ),
    "rgb": ModalitySpec(
        name="rgb",
        bands=tuple(RGB_BANDS),
        sample_field="rgb",
    ),
}


def get_modality_bands(modality: str) -> tuple[str, ...]:
    return MODALITY_SPECS[modality].bands


def get_sample_field(modality: str) -> str:
    return MODALITY_SPECS[modality].sample_field
