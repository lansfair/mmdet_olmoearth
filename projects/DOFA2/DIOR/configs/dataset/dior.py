DATASET_TYPE = 'DIORDataset'
DATA_ROOT = (
    '/mnt/ht2-nas2/EO_test/openmmlab-archive/dat/DIOR/'
)
DATA_SIZE = (896, 896)
BATCH_SIZE = 4
BACKEND_ARGS = None


train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=BACKEND_ARGS),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=DATA_SIZE, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
train_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=DATASET_TYPE,
        data_root=DATA_ROOT,
        data_prefix=dict(sub_data_root=''),
        img_subdir='Images/trainval/',
        ann_subdir='Annotations/trainval/',
        ann_file='ImageSets/Main/train.txt',
        filter_cfg=dict(
            filter_empty_gt=True,
            min_size=32,
            bbox_min_size=1,
        ),
        pipeline=train_pipeline,
        backend_args=BACKEND_ARGS,
    ),
)


test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=BACKEND_ARGS),
    dict(type='Resize', scale=DATA_SIZE, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
    ),
]
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=DATASET_TYPE,
        data_root=DATA_ROOT,
        data_prefix=dict(sub_data_root=''),
        img_subdir='Images/trainval/',
        ann_subdir='Annotations/trainval/',
        ann_file='ImageSets/Main/val.txt',
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=BACKEND_ARGS,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=DATASET_TYPE,
        data_root=DATA_ROOT,
        data_prefix=dict(sub_data_root=''),
        img_subdir='Images/test/',
        ann_subdir='Annotations/trainval/',
        ann_file='ImageSets/Main/test.txt',
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=BACKEND_ARGS,
    ),
)

# Pascal VOC2007 uses `11points` as default evaluate mode, while PASCAL
# VOC2012 defaults to use 'area'.
val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')
test_evaluator = val_evaluator
