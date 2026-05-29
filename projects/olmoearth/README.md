# OLMoEarth for MMDetection

This project migrates the rslearn OLMoEarth detection path into MMDetection.

The initial target is the rslearn detection stack:

- `rslearn.train.tasks.detection.DetectionTask`
- `rslearn.models.faster_rcnn.FasterRCNN`

## Alignment

- Inputs are Sentinel-2 L2A GeoTIFFs in OLMoEarth band order.
- `convert_rslearn_det.py` uses rslearn `ModelDataset` and `DetectionTask` to
  produce the same patch-relative `boxes`, `labels`, and `valid` semantics, then
  writes COCO JSON for MMDetection.
- `OlmoEarthBackbone` keeps the same OLMoEarth sample construction, timestamp
  handling, present-band masks, `fast_pass` logic, and PyTorch 2.3 CUDA bool-sort
  compatibility patch used by the MMSeg project.
- The detector head follows rslearn Faster R-CNN defaults: RPN IoU 0.7/0.3,
  RPN batch size 256, ROI IoU 0.5/0.5, ROI batch size 512, RoIAlign 7x7 with
  sampling ratio 2, 2000 RPN proposals, NMS 0.7/0.5, and 100 detections per
  image.
- OLMoEarth produces one dense feature map at `1 / patch_size`. The config uses
  `OlmoEarthMultiLevelNeck` to derive detection levels at strides
  `[patch_size, 2*patch_size, 4*patch_size, 8*patch_size]`.

## Convert

```bash
python projects/olmoearth/tools/convert_rslearn_det.py \
  --input-root /path/to/rslearn_dataset \
  --output-root data/rslearn_detection_coco \
  --image-layers sentinel2 \
  --target-layers label \
  --classes object \
  --property-name category
```

For point labels, pass `--box-size N` to match rslearn `DetectionTask` point to
box conversion.

## Train

Edit the paths and class names at the top of:

```text
projects/olmoearth/configs/olmoearth-base_faster-rcnn_1x_rslearn-detection-s2.py
```

Then run:

```bash
python tools/train.py \
  projects/olmoearth/configs/olmoearth-base_faster-rcnn_1x_rslearn-detection-s2.py
```
