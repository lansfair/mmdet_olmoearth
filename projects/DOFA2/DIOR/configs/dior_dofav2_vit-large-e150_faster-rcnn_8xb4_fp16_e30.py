_base_ = ['./dior_dofav2_vit-large-e150_faster-rcnn_8xb4_fp16_e15.py']

# Keep the eight-GPU global batch size of 32 but double the duration.  This
# restores roughly the same number of optimizer updates as 15 epochs with the
# paper's global batch size of 16: 367 * 30 ~= 733 * 15.
TRAIN_EPOCH = 30

param_scheduler = [
    dict(
        type='OneCycleLR',
        eta_max=2e-4,
        total_steps=TRAIN_EPOCH,
        pct_start=2 / TRAIN_EPOCH,
        anneal_strategy='cos',
        div_factor=10.0,
        final_div_factor=1000.0,
        by_epoch=True,
        begin=0,
        end=TRAIN_EPOCH,
    ),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=TRAIN_EPOCH)
