_base_ = [
    '../../../../configs/_base_/models/faster-rcnn_r50_fpn.py',
    '../../../../configs/_base_/default_runtime.py',
    './dataset/dior.py'
]

custom_imports = dict(imports=['projects.DOFA2.DIOR.dofa2'])

DATA_SIZE = 896

# TerraTorch's detection data path converts uint8 RGB imagery to float tensors
# in [0, 1], while its Faster R-CNN transform uses identity normalization.
# MMDetection receives uint8 imagery, so dividing by 255 reproduces that input
# scale without applying ImageNet statistics.
BANDS_MEAN = [0.0, 0.0, 0.0]
BANDS_STD = [255.0, 255.0, 255.0]

BACKBONE_ARCH_EMBED_DIM = {'base': 768, 'large': 1024}
BACKBONE_OUT_INDICES = [5, 9, 15, 21]
BACKBONE_ARCH = 'large'

NECK_IN_CHANNELS = [BACKBONE_ARCH_EMBED_DIM[BACKBONE_ARCH]] * len(BACKBONE_OUT_INDICES)
NECK_OUT_CHANNELS = 256

MULTI_SCALES_STRIDES = [4, 8, 16, 32, 64]
NUM_CLASSES = 20

TRAIN_EPOCH = 15

CHECKPOINT = (
    '/mnt/ht2-nas2/EO_test/openmmlab-archive/pretrained/'
    'dofav2_vit_large_e150.pth'
)


model = dict(
    data_preprocessor=dict(
        mean=BANDS_MEAN,
        std=BANDS_STD,
        bgr_to_rgb=True,
        pad_size_divisor=32,
    ),
    backbone=dict(
        _delete_=True,
        type="DOFAV2ViT",
        arch=BACKBONE_ARCH,
        img_size=DATA_SIZE,
        patch_size=14,
        model_bands=["RED", "GREEN", "BLUE"],
        out_indices=BACKBONE_OUT_INDICES,
        pos_interpolation_mode="bicubic",
        # Keep DOFAv2's native patch-14 embedding. Kernel interpolation to
        # patch 16 is optional in TerraTorch and is disabled by default.
        convert_patch_14_to_16=False,
        drop_path_rate=0.1,
        freeze_backbone=False,
        init_cfg=dict(type='Pretrained', checkpoint=CHECKPOINT),
    ),
    neck=dict(
        _delete_=True,
        type="DOFALearnedFPN",
        in_channels=NECK_IN_CHANNELS,
        out_channels=NECK_OUT_CHANNELS,
        num_outs=5,
        # torchvision.ops.FeaturePyramidNetwork, used by the public
        # TerraTorch detection path, does not add normalization to FPN convs.
        norm_cfg=None,
    ),
    rpn_head=dict(
        in_channels=NECK_OUT_CHANNELS, 
        feat_channels=NECK_OUT_CHANNELS,
        anchor_generator=dict(strides=MULTI_SCALES_STRIDES)
    ),
    roi_head=dict(
        bbox_roi_extractor=dict(
            out_channels=NECK_OUT_CHANNELS, 
            featmap_strides=MULTI_SCALES_STRIDES
        ),
        bbox_head=dict(in_channels=NECK_OUT_CHANNELS, num_classes=NUM_CLASSES)
    )
)

# The upstream object-detection recipe exposes layers [5, 9, 15, 21] from
# ViT-L. Parameters after the last exposed block do not contribute to the
# detection loss, so multi-GPU training must allow unused parameters.
model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True,
)

# The public TerraTorch detection recipe warms up from one tenth of the peak
# learning rate and then follows a cosine decay. For a 15-epoch DIOR run, one
# warm-up epoch is the closest epoch-granularity equivalent to pct_start=0.05.
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=1,
    ),
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=1,
        end=TRAIN_EPOCH,
        eta_min=1e-7,
    ),
]
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-2),
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=TRAIN_EPOCH)
val_cfg = None
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_last=True,
    ),
)
auto_scale_lr = dict(enable=False, base_batch_size=16)
randomness = dict(seed=0)
