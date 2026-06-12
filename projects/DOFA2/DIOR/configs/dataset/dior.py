import os

DATASET_TYPE = 'DIORDataset'

DATA_ROOT = os.environ.get('MM_ARCHIVE_DATA_HOME')

DATASET = 'DIOR'

DATA_SIZE = 800

BATCH_SIZE = 4


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=DATA_SIZE, keep_ratio=True),
    # dict(type='FilterAnnotations', min_gt_bbox_wh=(1., 1.)),
    dict(type='PackDetInputs')
]
train_dataloader = dict(
    dataset=dict(
        type=DATASET_TYPE,
        data_root=DATA_ROOT,
        data_prefix=dict(sub_data_root=DATASET),
        img_subdir='Images/trainval/',
        ann_subdir='Annotations/trainval/',
        ann_file=f'{DATASET}/ImageSets/Main/train.txt',
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline
    ),
    batch_size=BATCH_SIZE,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True)
)


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=DATA_SIZE, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs')
]
val_dataloader = dict(
    dataset=dict(
        type=DATASET_TYPE,
        data_root=DATA_ROOT,
        data_prefix=dict(sub_data_root=DATASET),
        img_subdir='Images/trainval/',
        ann_subdir='Annotations/trainval/',
        ann_file=f'{DATASET}/ImageSets/Main/val.txt',
        test_mode=True,
        pipeline=test_pipeline
    ),
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False)
)

test_dataloader = val_dataloader

# Pascal VOC2007 uses `11points` as default evaluate mode, while PASCAL
# VOC2012 defaults to use 'area'.
val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')
test_evaluator = val_evaluator