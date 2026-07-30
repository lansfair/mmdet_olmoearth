from mmdet.datasets import VOCDataset
from mmdet.registry import DATASETS


CLASSES = (
    'airplane',
    'airport',
    'baseballfield',
    'basketballcourt',
    'bridge',
    'chimney',
    'dam',
    'Expressway-Service-area',
    'Expressway-toll-station',
    'golffield',
    'groundtrackfield',
    'harbor',
    'overpass',
    'ship',
    'stadium',
    'storagetank',
    'tenniscourt',
    'trainstation',
    'vehicle',
    'windmill',
)

PALETTE = [
    (120, 166, 157),  # airplane
    (0, 60, 100),  # airport
    (0, 80, 100),  # baseballfield
    (106, 0, 228),  # basketballcourt
    (183, 130, 88),  # bridge
    (220, 20, 60),  # chimney
    (0, 0, 192),  # dam
    (165, 42, 42),  # Expressway-Service-area
    (182, 182, 255),  # Expressway-toll-station
    (0, 0, 230),  # golffield
    (163, 255, 0),  # groundtrackfield
    (197, 226, 255),  # harbor
    (0, 226, 252),  # overpass
    (0, 0, 142),  # ship
    (0, 82, 0),  # stadium
    (0, 182, 199),  # storagetank
    (3, 95, 161),  # tenniscourt
    (119, 11, 32),  # trainstation
    (153, 69, 1),  # vehicle
    (255, 77, 255),  # windmill
]


@DATASETS.register_module()
class DIORDataset(VOCDataset):
    METAINFO = {'classes': CLASSES, 'palette': PALETTE}
