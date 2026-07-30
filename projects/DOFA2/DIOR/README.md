# DOFA+ on DIOR

This project adapts the official DOFA+ large backbone to MMDetection for
DIOR object detection.

## Reference configuration

The public DOFA+ paper reports the following DIOR protocol:

- detector: Faster R-CNN;
- backbone: DOFA+ ViT-L, fully fine-tuned;
- epochs: 15;
- global batch size: 16;
- optimizer learning rate: 1e-4;
- evaluation metric: DIOR VOC-style mAP50.

The paper does not publish a standalone DIOR configuration file. Backbone
details therefore follow the official TerraTorch DOFAv2 object-detection
recipe: 896-pixel inputs, patch14-to-patch16 kernel conversion, transformer
outputs `[5, 9, 15, 21]`, a learned feature pyramid, and a conventional FPN.
The cosine schedule is retained from this repository's original migration
because the paper does not specify the DIOR scheduler.

With four GPUs, `batch_size=4` per GPU gives the paper's global batch size 16.
The backbone is not frozen.

## Fixed server paths

- Dataset: `/mnt/ht2-nas2/EO_test/openmmlab-archive/dat/DIOR/`
- Checkpoint:
  `/mnt/ht2-nas2/EO_test/openmmlab-archive/pretrained/dofav2_vit_large_e150.pth`

The dataset configuration uses the real DIOR split layout:

- train: `Images/trainval` + `ImageSets/Main/train.txt`;
- validation: `Images/trainval` + `ImageSets/Main/val.txt`;
- test: `Images/test` + `ImageSets/Main/test.txt`.

## Training and testing

Use MMDetection's existing distributed scripts from the repository root:

```bash
bash tools/dist_train.sh \
  projects/DOFA2/DIOR/configs/dior_dofav2_vit-large-e150_faster-rcnn_e15.py \
  4
```

```bash
bash tools/dist_test.sh \
  projects/DOFA2/DIOR/configs/dior_dofav2_vit-large-e150_faster-rcnn_e15.py \
  /path/to/best_checkpoint.pth \
  4
```
