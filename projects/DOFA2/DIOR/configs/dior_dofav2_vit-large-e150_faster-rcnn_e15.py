_base_ = [
    '../../../../configs/_base_/models/faster-rcnn_r50_fpn.py',
    '../../../../configs/_base_/default_runtime.py',
    './dataset/dior.py'
]

custom_imports = dict(imports=['projects.DOFA2.DIOR.dofa2'])

DATA_SIZE = 896

# The upstream TerraTorch VHR10 data module first divides uint8 RGB by 255,
# then applies ImageNet normalization.  MMDetection consumes uint8 BGR and
# `bgr_to_rgb=True` converts it to RGB before applying these equivalent
# 0-255-domain statistics.
BANDS_MEAN = [123.675, 116.28, 103.53]
BANDS_STD = [58.395, 57.12, 57.375]

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

# Match the public TerraTorch DOFAv2 detection recipe directly.  Using a
# separate LinearLR followed by CosineAnnealingLR leaves the cosine scheduler
# based at the warm-up LR in MMEngine, so the LR can remain at 1e-5.
param_scheduler = [
    dict(
        type='OneCycleLR',
        eta_max=1e-4,
        total_steps=TRAIN_EPOCH,
        pct_start=0.05,
        anneal_strategy='cos',
        div_factor=10.0,
        final_div_factor=1000.0,
        by_epoch=True,
        begin=0,
        end=TRAIN_EPOCH,
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
