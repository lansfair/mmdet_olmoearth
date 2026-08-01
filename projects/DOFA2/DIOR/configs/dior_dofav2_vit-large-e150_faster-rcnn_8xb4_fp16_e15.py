_base_ = ['./dior_dofav2_vit-large-e150_faster-rcnn_e15.py']

# Eight GPUs x four images per GPU. The four-GPU base recipe has a global
# batch size of 16 and lr=1e-4, so linear scaling gives lr=2e-4 here.
# FP16 matches TerraTorch's public `16-mixed` detection recipe and is supported
# by the MMCV CUDA RoIAlign kernel used in this environment.
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    dtype='float16',
    loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=1e-2),
)

# Keep the base recipe's two warm-up epochs and 15-epoch OneCycle schedule,
# while scaling the peak learning rate with the global batch size.
param_scheduler = [
    dict(
        type='OneCycleLR',
        eta_max=2e-4,
        total_steps=15,
        pct_start=2 / 15,
        anneal_strategy='cos',
        div_factor=10.0,
        final_div_factor=1000.0,
        by_epoch=True,
        begin=0,
        end=15,
    ),
]

auto_scale_lr = dict(enable=False, base_batch_size=32)
