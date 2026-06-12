import os

_base_ = [
    '../../../../configs/_base_/models/faster-rcnn_r50_fpn.py',
    '../../../../configs/_base_/default_runtime.py',
    './dataset/dior.py'
]

custom_imports = dict(imports=['projects.DOFA2.DIOR.dofa2'])

DATA_SIZE = 800

BANDS_MEAN = [
    123.675, # R
    116.280, # G
    103.530  # B
]
BANDS_STD = [
    58.395, # R
    57.120, # G
    57.375  # B
]

BACKBONE_ARCH_EMBED_DIM = {'base': 768, 'large': 1024}
BACKBONE_OUT_INDICES = [5, 11, 17, 23]
BACKBONE_ARCH = 'large'

NECK_IN_CHANNELS = [BACKBONE_ARCH_EMBED_DIM[BACKBONE_ARCH]] * len(BACKBONE_OUT_INDICES)
NECK_OUT_CHANNELS = 256

MULTI_SCALES_STRIDES = [4, 8, 16, 32, 64]
NUM_CLASSES = 20

TRAIN_EPOCH = 15

CHECKPOINT = os.path.join(os.environ.get('MM_ARCHIVE_CKPT_HOME'), 'dofav2_vit_large_e150.pth')


model = dict(
    data_preprocessor=dict(
        mean=BANDS_MEAN, 
        std=BANDS_STD, 
        bgr_to_rgb=True
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
        convert_patch_14_to_16=True,
        drop_path_rate=0.1,
        frozen_stages=False,
        init_cfg=dict(type='Pretrained', checkpoint=CHECKPOINT)
    ),
    neck=dict(
        _delete_=True,
        type="DOFALearnedFPN",
        in_channels=NECK_IN_CHANNELS,
        out_channels=NECK_OUT_CHANNELS,
        num_outs=5,
        norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),
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

param_scheduler = [dict(type='CosineAnnealingLR', by_epoch=True, begin=0, end=TRAIN_EPOCH)]
optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-2))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=TRAIN_EPOCH, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
