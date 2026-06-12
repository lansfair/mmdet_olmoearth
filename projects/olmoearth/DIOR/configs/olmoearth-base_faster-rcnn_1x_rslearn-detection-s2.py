custom_imports = dict(
    imports=["projects.olmoearth.DIOR.olmoearth"],
    allow_failed_imports=False,
)

mm_archive_home = '/mnt/ht2-nas2/EO_test/openmmlab-archive'
data_root = f"{mm_archive_home}/dat/DIOR"
olmoearth_model_dir = f"{mm_archive_home}/pretrained/OlmoEarth-v1-Base"
model_config_path = f"{olmoearth_model_dir}/config.json"
weights_path = f"{olmoearth_model_dir}/weights.pth"

classes = ("object",)
num_classes = len(classes)
num_timesteps = 1
patch_size = 8
image_size = 128
out_channels = 768
fpn_channels = 256
featmap_strides = [patch_size, patch_size * 2, patch_size * 4, patch_size * 8]
anchor_sizes = [32, 64, 128, 256]

metainfo = dict(classes=classes)

train_pipeline = [
    dict(type="LoadOlmoEarthTifFromFile"),
    dict(
        type="OlmoEarthNormalize",
        modality="sentinel2_l2a",
        num_timesteps=num_timesteps,
    ),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="RandomFlip", prob=0.5),
    dict(
        type="PackDetInputs",
        meta_keys=(
            "img_id",
            "sample_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
            "flip",
            "flip_direction",
            "timestamps",
            "present_bands",
            "valid",
            "olmoearth_modality",
            "olmoearth_num_timesteps",
            "olmoearth_band_names",
            "rslearn",
        ),
    ),
]

test_pipeline = [
    dict(type="LoadOlmoEarthTifFromFile"),
    dict(
        type="OlmoEarthNormalize",
        modality="sentinel2_l2a",
        num_timesteps=num_timesteps,
    ),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PackDetInputs",
        meta_keys=(
            "img_id",
            "sample_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
            "timestamps",
            "present_bands",
            "valid",
            "olmoearth_modality",
            "olmoearth_num_timesteps",
            "olmoearth_band_names",
            "rslearn",
        ),
    ),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        type="OlmoEarthDetDataset",
        data_root=data_root,
        ann_file="train.json",
        data_prefix=dict(img=""),
        metainfo=metainfo,
        filter_cfg=dict(
            filter_empty_gt=False,
            filter_invalid=True,
            min_size=1,
        ),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="OlmoEarthDetDataset",
        data_root=data_root,
        ann_file="val.json",
        data_prefix=dict(img=""),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="OlmoEarthDetMetric",
    num_classes=num_classes,
    iou_thr=0.5,
    score_thresholds=[0.05, 0.1, 0.2, 0.3, 0.5],
)
test_evaluator = val_evaluator

model = dict(
    type="OlmoEarthFasterRCNN",
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=None,
        std=None,
        bgr_to_rgb=False,
        pad_size_divisor=32,
    ),
    backbone=dict(
        type="OlmoEarthBackbone",
        model_config_path=model_config_path,
        init_cfg=dict(type="Pretrained", checkpoint=weights_path),
        modality="sentinel2_l2a",
        patch_size=patch_size,
        num_timesteps=num_timesteps,
        out_channels=out_channels,
        pooling_type="mean",
    ),
    neck=dict(
        type="OlmoEarthMultiLevelNeck",
        in_channels=out_channels,
        out_channels=fpn_channels,
        scales=[1.0, 0.5, 0.25, 0.125],
    ),
    rpn_head=dict(
        type="RPNHead",
        in_channels=fpn_channels,
        feat_channels=fpn_channels,
        anchor_generator=dict(
            type="AnchorGenerator",
            scales=[1],
            base_sizes=anchor_sizes,
            ratios=[0.5, 1.0, 2.0],
            strides=featmap_strides,
        ),
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder",
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        loss_cls=dict(
            type="CrossEntropyLoss",
            use_sigmoid=True,
            loss_weight=1.0,
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=1.0),
    ),
    roi_head=dict(
        type="StandardRoIHead",
        bbox_roi_extractor=dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=2),
            out_channels=fpn_channels,
            featmap_strides=featmap_strides,
        ),
        bbox_head=dict(
            type="Shared2FCBBoxHead",
            in_channels=fpn_channels,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=num_classes,
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder",
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2],
            ),
            reg_class_agnostic=False,
            loss_cls=dict(
                type="CrossEntropyLoss",
                use_sigmoid=False,
                loss_weight=1.0,
            ),
            loss_bbox=dict(type="L1Loss", loss_weight=1.0),
        ),
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type="MaxIoUAssigner",
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1,
            ),
            sampler=dict(
                type="RandomSampler",
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
        ),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type="nms", iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            assigner=dict(
                type="MaxIoUAssigner",
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1,
            ),
            sampler=dict(
                type="RandomSampler",
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True,
            ),
            pos_weight=-1,
            debug=False,
        ),
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type="nms", iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type="nms", iou_threshold=0.5),
            max_per_img=100,
        ),
    ),
)

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.05),
    clip_grad=None,
)

param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type="MultiStepLR",
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1,
    ),
]

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=12, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

default_scope = "mmdet"
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=1),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="DetVisualizationHook"),
)
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="DetLocalVisualizer",
    vis_backends=vis_backends,
    name="visualizer",
)
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)
log_level = "INFO"
load_from = None
resume = False
