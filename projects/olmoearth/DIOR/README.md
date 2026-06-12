# OLMoEarth for MMDetection

This project migrates the rslearn OLMoEarth detection path into MMDetection.
It also includes a conventional OpenMMLab RGB detection example for DIOR.

## Which OpenMMLab Project

Use the OLMoEarth project that matches the downstream task:

| Task | Project | Typical data |
| --- | --- | --- |
| Semantic segmentation | MMSegmentation | masks, valid masks, GeoTIFF manifests |
| Horizontal-box detection | MMDetection | rslearn detection manifest, VOC/XML DIOR |
| Oriented-box detection | MMRotate | DOTA txt, DIOR-R oriented XML |

For a broader Chinese walkthrough covering both MMSegmentation and
MMDetection/MMRotate, see
`projects/olmoearth/docs/olmoearth_openmmlab_migration_zh.md`.

The initial target is the rslearn detection stack:

- `rslearn.train.tasks.detection.DetectionTask`
- `rslearn.models.faster_rcnn.FasterRCNN`

## Alignment

- Inputs are Sentinel-2 L2A GeoTIFFs in OLMoEarth band order.
- `convert_rslearn_det.py` uses rslearn `ModelDataset` and `DetectionTask` to
  produce the same patch-relative `boxes`, `labels`, and `valid` semantics, then
  writes an OLMoEarth detection manifest rather than COCO.
- `OlmoEarthDetDataset` reads that manifest directly, preserving multi-timestep
  `img_paths`, timestamps, present bands, validity flags, and rslearn metadata.
- `OlmoEarthDetMetric` evaluates valid rslearn samples with IoU matching and
  reports F1, precision, and recall over score thresholds.
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

## rslearn Convert

```bash
python projects/olmoearth/tools/convert_rslearn_det.py \
  --input-root /path/to/rslearn_dataset \
  --output-root data/rslearn_detection_manifest \
  --image-layers sentinel2 \
  --target-layers label \
  --classes object \
  --property-name category
```

The converter shows progress. If `tqdm` is installed it uses a progress bar;
otherwise it prints periodic `current/total` updates.

For point labels, pass `--box-size N` to match rslearn `DetectionTask` point to
box conversion.

The converted layout is:

```text
data/rslearn_detection_manifest/
  train.json
  val.json
  test.json
  samples/<sample_id>/t00_sentinel2_l2a.tif
```

Each split JSON is a manifest with `metainfo` and `samples`. A sample stores
`img_paths`, `height`, `width`, `bboxes` in xyxy format, zero-based `labels`,
`valid`, timestamps, present bands, and rslearn metadata.

Before training, smoke-check the converted manifest:

```bash
python projects/olmoearth/tools/check_converted_det_dataset.py \
  --data-root data/rslearn_detection_manifest \
  --ann-file train.json
```

## rslearn Train

Edit the paths and class names at the top of:

```text
projects/olmoearth/configs/olmoearth-base_faster-rcnn_1x_rslearn-detection-s2.py
```

Then run:

```bash
python tools/train.py \
  projects/olmoearth/configs/olmoearth-base_faster-rcnn_1x_rslearn-detection-s2.py
```

## DIOR RGB Example

This example is for original DIOR horizontal boxes. DIOR is kept as a normal
OpenMMLab XML-style dataset. The only OLMoEarth-specific part is the pipeline
transform that maps RGB into normalized Sentinel-2 RGB slots before the
OLMoEarth backbone.

Expected layout:

```text
data/DIOR/
  JPEGImages/*.jpg
  Annotations/*.xml
  ImageSets/Main/train.txt
  ImageSets/Main/val.txt
```

If your DIOR annotation is DIOR-R oriented XML or DOTA-like txt, use the
MMRotate OLMoEarth project instead.

Run:

```bash
python tools/train.py \
  projects/olmoearth/configs/olmoearth-base_faster-rcnn_1x_dior-rgb.py
```
