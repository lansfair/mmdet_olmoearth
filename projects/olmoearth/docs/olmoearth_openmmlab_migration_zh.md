# OlmoEarth -> MMSeg / MMDet 迁移教程

这份文档总结当前项目里 OlmoEarth 迁移到 OpenMMLab 的做法。它不是
“把代码能跑起来”的流水账，而是解释为什么要做这些适配，以及哪些地方
必须对齐原论文，哪些地方只是工程便利。

## 教程目标

读完以后应该能回答三个问题：

1. OlmoEarth Backbone 的输入、输出和普通 ResNet/ViT 有什么不同。
2. 如何在 MMSegmentation / MMDetection 里接入 OlmoEarth 做遥感下游任务。
3. 为什么本项目选择 manifest、OpenMMLab Dataset、`init_cfg`、neck/head 适配，
   而不是训练时直接包一层 rslearn。

最终能跑通两条路径：

- 分割：PASTIS、MADOS、Sen1Floods11、AWF、Nandi、Crop-Type、Potsdam。
- 检测：rslearn detection manifest，以及 DIOR 这类常规 OpenMMLab RGB 数据集。

## 环境准备

推荐把 OpenMMLab 和 OlmoEarth 放在同一个环境里，避免权重和依赖在不同
Python 环境之间来回找。

核心依赖：

- Python 3.10 或 3.11。
- PyTorch。OLMoEarth 原项目偏新，服务器如果是 PyTorch 2.3，需要保留本项目里
  对 CUDA bool sort 的兼容补丁。
- OpenMMLab：`mmengine`、`mmcv`、`mmsegmentation`、`mmdetection`。
- `rasterio`：用于读写多波段 GeoTIFF。
- 本地 `olmoearth_pretrain`：用于构建模型、读取 modality 定义和归一化参数。
- `rslearn`：只在转换 rslearn 项目数据时需要，训练时不再依赖它。

权重布局推荐固定成：

```text
checkpoints/olmoearth/
  config.json
  weights.pth
```

在 OpenMMLab config 里，`config.json` 用于构建模型结构，`weights.pth` 通过
backbone 的 `init_cfg` 加载。不要把 OLMoEarth 的 `weights.pth` 放到
OpenMMLab 顶层 `load_from`，因为 `load_from` 表示加载完整 OpenMMLab 模型，
不是只加载 backbone。

## OlmoEarth 简介

普通视觉 backbone 通常接收：

```text
image: B x C x H x W
```

OlmoEarth 在预训练中接收的是带 modality、time、band、mask 的样本：

```text
sentinel2_l2a:      B x H x W x T x C
sentinel2_l2a_mask: B x H x W x T x bandset
timestamps:         B x T x 3
```

这带来几个迁移差异。

### 与 ResNet / 普通 ViT 的差异

| 问题 | 普通 ResNet / ViT | OlmoEarth |
| --- | --- | --- |
| 输入 | 3 通道 RGB 或固定多通道 | modality + 多时相 + 多波段 |
| 缺失波段 | 通常不表达 | 通过 mask 表达 online / missing |
| 时间信息 | 通常没有 | `timestamps` 是前向输入的一部分 |
| 输出 | 多尺度或单尺度 feature | dense token map，尺度约为 `1 / patch_size` |
| 下游适配 | 直接接 FPN/UPerNet | 需要 pooling、mask、neck 或 head 适配 |

一个容易误解的点是：OlmoEarth 可以输入单时相。`T` 不是必须大于 1；
只要 `timestamps`、图像张量和 mask 的 T 维一致即可。因此 RGB / DIOR /
Potsdam 这类单图数据可以走单时相适配，但这属于 out-of-domain 实验，不等于
复现论文里的多时相遥感设定。

## 迁移总原则

本项目最终采用四条原则。

### 1. 非侵入式 projects 迁移

不改 OpenMMLab 主干代码，所有新增逻辑放在：

```text
mmsegmentation/projects/olmoearth/
mmdetection/projects/olmoearth/
```

并通过：

```python
custom_imports = dict(
    imports=["projects.olmoearth.olmoearth"],
    allow_failed_imports=False,
)
```

注册 Dataset、Backbone、Transform、Metric、Hook。

### 2. 训练时不用 rslearn Dataset

迁移早期有一个诱惑：直接在 OpenMMLab Dataset 里包
`rslearn.train.dataset.ModelDataset`。最后没有这么做。

前后对比：

| 方案 | 优点 | 问题 |
| --- | --- | --- |
| 训练时直接包 rslearn | 最接近原项目数据读取 | 生命周期不符合 OpenMMLab，DataLoader/serialize/filter/debug 都绕进 rslearn |
| 预转换成 manifest | OpenMMLab 原生、可审计、可复现 | 多一个转换步骤 |

本项目选择第二种。转换脚本只负责把原任务语义物化出来，训练阶段只读
GeoTIFF 和 JSON manifest。

### 3. 原论文任务优先对齐语义，常规数据集走原生格式

分割里的 PASTIS/AWF/Nandi/MADOS/Sen1Floods11、检测里的 rslearn detection，
都保留原任务的 label、valid mask、时间戳和多波段输入。

DIOR、Potsdam 这种常规 OpenMMLab 数据集，则尽量走原生 Dataset：

- DIOR：`XMLDataset` + `RGBToOlmoEarthS2`。
- Potsdam：MMSeg Potsdam 数据布局 + RGB adapter。

不要为了“所有数据集统一”而把 DIOR/Potsdam 也强行转成 rslearn 格式。

### 4. `init_cfg` 加载 OLMoEarth 权重

OpenMMLab 原生语义是：

- `model.backbone.init_cfg.checkpoint`：初始化 backbone。
- `load_from`：加载完整 OpenMMLab checkpoint。
- `resume`：恢复训练状态。

因此本项目让 OLMoEarth backbone 自己用 `init_cfg` 加载 `weights.pth`，
这比在 config 里自定义 `model_path/model_id/checkpoint_path` 更符合框架。

## 迁移到 MMSegmentation

### 数据集准备

分割 manifest 的核心结构：

```json
{
  "metainfo": {
    "classes": ["class_0", "class_1"],
    "palette": [[0, 0, 0], [255, 255, 255]]
  },
  "samples": [
    {
      "sample_id": "train_000000",
      "img_paths": [
        "samples/train_000000/t00_sentinel2_l2a.tif",
        "samples/train_000000/t01_sentinel2_l2a.tif"
      ],
      "seg_map_path": "samples/train_000000/label.tif",
      "valid_mask_path": "samples/train_000000/valid_mask.tif",
      "timestamps": [[1, 4, 2020], [1, 5, 2020]],
      "olmoearth_modality": "sentinel2_l2a",
      "olmoearth_num_timesteps": 2
    }
  ]
}
```

转换前通常是：

- OLMoEarth / rslearn 项目的内部 dataset、raster layer、vector label。
- 或 OLMoEarth 已处理好的 `.pt/.pth/.npy` eval 张量。
- 或 OpenMMLab 常规图片目录。

转换后统一是：

- 原始、未归一化 GeoTIFF。
- label GeoTIFF。
- 可选 valid mask GeoTIFF。
- manifest JSON。

为什么不用 `.npz`：GeoTIFF 更容易用 GIS / raster 工具查看，也能保留多波段
描述；manifest 只记录路径和元数据，不把大数组塞进 JSON。

### Backbone 封装与注册

MMSeg 的模型看到的是 `B x C*T x H x W`。`OlmoEarthBackbone` 在内部还原成：

```text
B x H x W x T x C
```

然后构造 `MaskedOlmoEarthSample`，把 `timestamps`、`present_bands` 转成
OlmoEarth encoder 需要的 mask。

关键点：

- `fast_pass=None` 表示自动判断：没有 missing token 时可以走 fast path。
- PyTorch 2.3 CUDA 不支持 bool dtype stable sort，所以 backbone 里保留
  bool sort 兼容补丁。
- RGB 数据通过 `RGBToOlmoEarthS2` 映射到 B04/B03/B02，缺失的 S2 band 用
  missing mask 表达。

### Feature map 适配与 decode head

OlmoEarth 输出的是一个 dense feature map，空间尺度取决于 `patch_size`：

```text
输入 512 x 512, patch_size=4  -> feature 128 x 128
输入 512 x 512, patch_size=16 -> feature 32 x 32
```

因此有两类分割头：

- paper-style linear probe：冻结 backbone，只训练 patch-linear head。
- OpenMMLab style UPerNet：用 `MultiLevelNeck` 把一个 feature map 派生成多尺度，
  再接 UPerHead / auxiliary head。

前后对比：

| 目标 | 推荐做法 | 原因 |
| --- | --- | --- |
| 复现 OLMoEarth 线性探针 | offline embedding + patch-linear head | 最接近原评估，训练快 |
| 做 OpenMMLab 工程实验 | online backbone + UPerNet | 更像常规语义分割模型 |
| 高分辨率 RGB | patch_size=16 可省显存 | 但空间细节可能下降 |

### 训练流程

以 PASTIS 为例：

```bash
python projects/olmoearth/tools/convert_pastis.py \
  --input-root /path/to/pastis_r \
  --output-root data/olmoearth_mmseg/pastis

python projects/olmoearth/tools/check_converted_dataset.py \
  --data-root data/olmoearth_mmseg/pastis \
  --ann-file train.json

python tools/train.py \
  projects/olmoearth/configs/pastis/olmoearth-base_4xb4-50e_pastis-s2.py
```

以 Crop-Type offline probe 为例：

```bash
python projects/olmoearth/tools/extract_embeddings.py

python tools/train.py \
  projects/olmoearth/configs/crop_type/olmoearth-base_1xb8-50e_crop-type-s2-offline-linear.py
```

offline probe 慢变快的原因很简单：原来每个 epoch 都前向 OLMoEarth encoder；
现在先把 dense embedding 抽出来，训练时只读 embedding 并训练线性头。

## 迁移到 MMDetection

### 数据集准备

rslearn detection 不再转 COCO，而是转 OLMoEarth detection manifest：

```json
{
  "metainfo": {
    "format": "olmoearth_rslearn_detection_manifest",
    "classes": ["object"],
    "box_format": "xyxy",
    "label_offset": 0
  },
  "samples": [
    {
      "sample_id": "train_000000",
      "img_paths": ["samples/train_000000/t00_sentinel2_l2a.tif"],
      "height": 128,
      "width": 128,
      "bboxes": [[10.0, 12.0, 42.0, 50.0]],
      "labels": [0],
      "valid": 1,
      "timestamps": [[1, 1, 2024]],
      "present_bands": ["B02", "B03", "B04"]
    }
  ]
}
```

为什么不转 COCO：

| 信息 | COCO 能放吗 | manifest 做法 |
| --- | --- | --- |
| xyxy box | COCO 默认 xywh，需要转换 | 保持 rslearn 输出 xyxy |
| 多时相 `img_paths` | 只能塞自定义字段 | manifest 原生字段 |
| `valid` | COCO 没有标准语义 | manifest 原生字段 |
| `timestamps` | COCO 没有标准语义 | manifest 原生字段 |
| rslearn metadata | COCO 只能附加 | manifest 原生字段 |

检测转换：

```bash
python projects/olmoearth/tools/convert_rslearn_det.py \
  --input-root /path/to/rslearn_dataset \
  --output-root data/rslearn_detection_manifest \
  --image-layers sentinel2 \
  --target-layers label \
  --classes object \
  --property-name category
```

检查：

```bash
python projects/olmoearth/tools/check_converted_det_dataset.py \
  --data-root data/rslearn_detection_manifest \
  --ann-file train.json
```

### Backbone 接入与 neck/head 适配

MMDet 的 Faster R-CNN 需要多尺度 feature。OlmoEarth 只输出一个 dense map，
所以本项目用 `OlmoEarthMultiLevelNeck` 派生多个尺度：

```text
stride: patch_size, 2*patch_size, 4*patch_size, 8*patch_size
scale:  1.0,        0.5,          0.25,         0.125
```

这不是说 OlmoEarth 变成了 ResNet FPN，而是为了让 RPN / RoIHead 能按 MMDet
常规接口工作。

检测 head 参考 rslearn 的 torchvision Faster R-CNN 设置：

- RPN IoU：0.7 / 0.3。
- RPN batch size：256。
- RoI assign：0.5 / 0.5。
- RoI batch size：512。
- RoIAlign：7 x 7，sampling ratio 2。
- RPN proposals：2000。
- NMS：RPN 0.7，RCNN 0.5。
- max detections：100。

### DIOR 常规数据集示例

DIOR 不需要转 manifest。它是常规 VOC/XML 风格数据集，用 MMDet 原生
`XMLDataset` 更合理：

```text
data/DIOR/
  JPEGImages/*.jpg
  Annotations/*.xml
  ImageSets/Main/train.txt
  ImageSets/Main/val.txt
```

训练：

```bash
python tools/train.py \
  projects/olmoearth/configs/olmoearth-base_faster-rcnn_1x_dior-rgb.py
```

这里的关键不是改 Dataset，而是在 pipeline 里加入：

```python
dict(
    type="RGBToOlmoEarthS2",
    rgb_channel_order="BGR",
    input_value_range="0_255",
)
```

这样普通 RGB 图像会映射到 Sentinel-2 的 B04/B03/B02 槽位，其余 band 缺失。

## 评估与可视化

### 分割

分割使用 `OlmoEarthIoUMetric`：

- 输出 MMSeg 风格的 `aAcc`、`mIoU`、`mAcc`。
- 可选用 valid mask 过滤无效像素。
- 每类 IoU 表格仍按 OpenMMLab 日志打印。

可视化问题在多波段数据上很常见。默认 MMSeg visual hook 假设输入来自 RGB
文件路径，但 OLMoEarth 输入可能是多时相多波段张量。因此项目里提供
`OlmoEarthVisualizationHook`，直接从 batch tensor 生成可视化，避免读取错文件。

### 检测

rslearn manifest 检测使用 `OlmoEarthDetMetric`：

- 按 class 分组。
- 用 IoU 做预测框和 GT 框匹配。
- 在多个 score threshold 下报告 F1、precision、recall。
- 输出 best F1 对应的 TP/FP/FN，便于排查阈值问题。

DIOR 这类常规数据集继续用 `VOCMetric` 或数据集标准 metric。

## 常见问题与调试

### 权重加载失败

检查三件事：

1. `model.backbone.model_config_path` 指向 released `config.json`。
2. `model.backbone.init_cfg.checkpoint` 指向 released `weights.pth`。
3. 顶层 `load_from` 没有误填 OLMoEarth backbone 权重。

### 输入通道不匹配

报错类似：

```text
Expected 144 channels (12 bands x 12 timesteps), got 36
```

说明 config 的 `num_timesteps` 和 manifest 里的 `img_paths` 数量不一致，
或者 band order 不一致。

### `fast_pass=True` 报错或精度异常

不要固定 `fast_pass=True`。RGB adapter、缺失 band、多模态缺失 token 都应该
让 backbone 自动判断。固定 True 会跳过缺失 token 处理，可能直接错，也可能
悄悄改变语义。

### PyTorch 2.3 bool sort 报错

报错类似：

```text
Sort currently does not support bool dtype on CUDA.
```

本项目的 backbone 已把 bool mask 临时转成 `uint8` 再 sort，这是为了兼容
OlmoEarth 原始代码在较旧 PyTorch CUDA 上的问题。

### 大图慢或显存高

分割 offline embedding extractor 已支持滑窗：

```bash
python projects/olmoearth/tools/extract_embeddings.py \
  --tile-size 512 \
  --tile-overlap 64
```

检测和在线分割训练仍建议用 OpenMMLab 的 crop / resize / batch size / AMP
控制显存。

### position embedding

OlmoEarth 的 dense encoder 通过 `patch_size` 控制输出 stride。不要把普通 ViT
“固定 16x16 patch”的直觉直接套过来。对于高分辨率任务：

- `patch_size=4`：细节好，显存高。
- `patch_size=16`：显存低，输出更粗。

## 进阶方向

### 多波段与多模态

manifest 的好处是可以自然扩展：

- `img_paths` 支持多时相。
- `present_bands` 支持缺失 band。
- modality 可以从 `sentinel2_l2a` 扩展到 `sentinel1` 等。

但每个新 modality 都要确认：

- band order。
- normalization。
- `MaskedOlmoEarthSample` 字段名。
- mask 的 bandset 语义。

### 参数高效微调

现在的复现路径主要是冻结 backbone + probe，或全量训练。后续可以加：

- LoRA。
- Adapter。
- bias-only / norm-only tuning。

建议仍然放在 `projects/olmoearth`，不要改 OpenMMLab 主干。

### 小样本与滑窗预测

遥感常见问题是数据少、图大、类别稀疏。实用方向：

- offline embedding 缓存，减少重复 encoder forward。
- overlap sliding window，降低边界 artifact。
- 按 class frequency 调 loss 或 sampler。
- 对 valid mask 做严格检查，避免无效区域污染指标。

## 最重要的迁移经验

1. 先对齐数据语义，再对齐模型接口。
   只要 label、mask、timestamp、band order 错了，模型接得再优雅也没意义。

2. 原论文复现路径和 OpenMMLab 工程路径要分开。
   PASTIS/AWF/Nandi 需要保留 OLMoEarth 评估语义；DIOR/Potsdam 更适合走原生
   Dataset + RGB adapter。

3. manifest 是折中点。
   它比训练时包 rslearn 更 OpenMMLab，比强转 COCO 更能表达遥感多模态语义。

4. backbone 不只是 `forward(x)`。
   OlmoEarth 的 `forward` 还需要 timestamps、present_bands、mask、pooling、
   patch_size 和 PyTorch 版本兼容。

5. 能先检查数据，就不要先跑训练。
   先跑 manifest checker、pipeline checker、forward checker，再开长训练。
