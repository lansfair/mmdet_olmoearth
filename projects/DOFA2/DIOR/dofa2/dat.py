from mmdet.datasets import VOCDataset
from mmdet.registry import DATASETS


CLASSES = (
    'trainstation',            #  1
    'Expressway-Service-area', #  2
    'dam',                     #  3
    'harbor',                  #  4
    'airport',                 #  5
    'ship',                    #  6
    'windmill',                #  7
    'vehicle',                 #  8
    'airplane',                #  9
    'storagetank',             # 10
    'overpass',                # 11
    'Expressway-toll-station', # 12
    'golffield',               # 13
    'chimney',                 # 14
    'groundtrackfield',        # 15
    'stadium',                 # 16
    'tenniscourt',             # 17
    'baseballfield',           # 18
    'bridge',                  # 19
    'basketballcourt'          # 20
)

PALETTE = [
    (119, 11, 32),             # trainstation
    (165, 42, 42),             # Expressway-Service-area 
    (0, 0, 192),               # dam
    (197, 226, 255),           # harbor
    (0, 60, 100),              # airport
    (0, 0, 142),               # ship
    (255, 77, 255),            # windmill
    (153, 69, 1),              # vehicle 
    (120, 166, 157),           # airplane
    (0, 182, 199),             # storagetank
    (0, 226, 252),             # overpass
    (182, 182, 255),           # Expressway-toll-station
    (0, 0, 230),               # golffield
    (220, 20, 60),             # chimney
    (163, 255, 0),             # groundtrackfield
    (0, 82, 0),                # stadium
    (3, 95, 161),              # tenniscourt
    (0, 80, 100),              # baseballfield
    (183, 130, 88),            # bridge
    (106, 0, 228)              # basketballcourt
]


@DATASETS.register_module()
class DIORDataset(VOCDataset):
    METAINFO = {'classes': CLASSES, 'palette': PALETTE}
