# OlmoEarth -> MMSeg / MMDet / MMRotate 迁移教程

这份文档面向想学习“如何把一个非 OpenMMLab 模型迁移到 MMSeg / MMDet /
MMRotate，并尽量复现实验”的读者。OlmoEarth 是这里的案例；真正要学习的是一套
迁移方法：先拆原项目的数据、模型、训练、评估 contract，再决定哪些逻辑保留原实现，
哪些逻辑改写成 OpenMMLab 原生组件。

为了避免“内容都有但路线混乱”，本文按迁移工作的自然顺序组织：先讲通用方法论，
再讲 OlmoEarth 的特殊性，然后分别落到数据、模型、recipes、验证和 debug。

## 0. 读者定位与路线图

先确定这篇教程在教什么、读者该怎么选框架和 config。

### 教程目标

读完以后应该能回答五个问题：

1. OlmoEarth Backbone 的输入、输出和普通 ResNet/ViT 有什么不同。
2. 如何在 MMSegmentation / MMDetection / MMRotate 里接入 OlmoEarth 做遥感下游任务。
3. 为什么本项目选择 manifest、OpenMMLab Dataset、`init_cfg`、neck/head 适配，
   而不是训练时直接包一层 rslearn。
4. 如何判断一个数据集应该转 manifest，还是继续使用 OpenMMLab 原生 Dataset。
5. 如何区分“论文复现路径”和“OpenMMLab 工程实验路径”。

最终能跑通三类实验，并理解每一类为什么这样迁移：

- 分割：PASTIS、MADOS、Sen1Floods11、AWF、Nandi、Crop-Type、Potsdam。
- 水平框检测：rslearn detection manifest，以及原始 DIOR 这类常规 OpenMMLab RGB 数据集。
- 旋转框检测：DOTA、DIOR-R、DOTA-like DIOR，走 MMRotate 原生 rotated 组件。

对学习迁移的人来说，最重要的不是背这个项目里的类名，而是建立四个 contract：

| Contract | 要回答的问题 | 本项目里的例子 |
| --- | --- | --- |
| 数据 contract | OpenMMLab Dataset 最终给 pipeline 什么字段 | manifest、`img_paths`、`timestamps`、`present_bands` |
| 模型 contract | Backbone 接收什么 tensor，输出什么 feature | `B,C*T,H,W` 还原成 `B,H,W,T,C`，输出 dense map |
| 训练 contract | 哪些参数训练，哪些参数冻结，权重从哪里加载 | `init_cfg` 加载 OLMoEarth，linear probe 冻结 backbone |
| 评估 contract | metric 是否和原论文/数据集语义一致 | valid mask、ignore index、DOTA rotated mAP、rslearn F1 |

只要这四个 contract 写清楚，迁移其他 foundation model、遥感模型或普通视觉模型时，
也可以沿用同样的分析方法。

### 如何阅读这份教程

这份教程建议按三条路线阅读，不要一开始就把所有任务混在一起。

先用下面这个入口判断该进哪个仓库：

| 任务/标注 | 推荐框架 | 推荐数据入口 | 说明 |
| --- | --- | --- | --- |
| 语义分割 mask | MMSegmentation | manifest、GEO-Bench loader 或原生 Potsdam | 需要 decode head、IoU、valid-mask 语义 |
| 水平框检测 `xmin/ymin/xmax/ymax` | MMDetection | rslearn detection manifest、`XMLDataset` | 适合 Faster R-CNN/VOCMetric/水平 NMS |
| 旋转框检测 8 点框或 `robndbox` | MMRotate | `DOTADataset`、`DIORDataset` | 需要 qbox/rbox、rotated IoU、rotated NMS |

最容易混淆的是 DIOR：

| 你磁盘上的 DIOR 形态 | 应该用 |
| --- | --- |
| `Annotations/*.xml`，里面是 `bndbox/xmin/ymin/xmax/ymax` | MMDetection `XMLDataset` |
| `Annotations/Oriented Bounding Boxes/*.xml`，里面是 `robndbox` | MMRotate `DIORDataset` |
| `annfiles/*.txt`，每行 `x1 y1 ... x4 y4 class difficult` | MMRotate `DOTADataset` |

#### 路线 A：先跑通一个 OpenMMLab 实验

适合已经熟悉 OpenMMLab，但还不了解 OlmoEarth 的读者。推荐从 RGB 数据集开始：

```text
DIOR / Potsdam
  -> 原生 OpenMMLab Dataset
  -> RGBToOlmoEarthS2
  -> OlmoEarthBackbone
  -> 常规 head 和 metric
```

这条路线最快，因为不需要先理解 rslearn，也不需要写转换脚本。它的目标是先确认
环境、权重、registry、forward 都能跑通。

#### 路线 B：复现 OlmoEarth 论文下游任务

适合关心论文精度对齐的读者。推荐从 PASTIS 或 Crop-Type 开始：

```text
PASTIS / MADOS / Sen1Floods11
  -> OLMoEarth 已处理 eval tensor
  -> GeoTIFF + manifest
  -> 冻结 backbone + linear probe

AWF / Nandi
  -> rslearn dataset
  -> converter 物化 raster/label/valid/timestamp
  -> GeoTIFF + manifest
```

这条路线要优先对齐 label、valid mask、timestamp、band order 和 metric。只要这些
语义错了，即使模型代码能跑，精度也没有可比性。

#### 路线 C：迁移自己的遥感数据集

适合要把新数据接到 OlmoEarth 的读者。先回答下面几个问题：

```text
你的数据是 RGB 图片 + VOC/COCO/XML 水平框标注？
  -> 优先用 MMDetection 原生 Dataset，只加 RGB adapter。

你的数据是 RGB 图片 + DOTA txt / DIOR-R oriented XML？
  -> 优先用 MMRotate 原生 Dataset，只加 RGB adapter。

你的数据是多波段 GeoTIFF + mask/box？
  -> 可以直接写 manifest，或写轻量 Dataset。

你的数据依赖 rslearn raster/vector layer？
  -> 先写 converter，把 rslearn 输出物化成 GeoTIFF + manifest。

你的目标是论文复现？
  -> 保留 valid mask、timestamp、band order、ignore label。

你的目标是工程实验？
  -> 优先使用 OpenMMLab 原生 Dataset、metric 和可视化工具。
```

这个决策树比“所有数据都转成同一种格式”更重要。迁移不是追求形式统一，而是
尽量少损失原任务语义，同时尽量多复用 OpenMMLab 的训练生态。

#### Config 导航表

学习迁移时，建议先把每个 config 放进正确语境里：

| Config | 框架 | 任务 | 输入 | 迁移目的 | 是否论文复现路径 |
| --- | --- | --- | --- | --- | --- |
| `configs/pastis/olmoearth-base_4xb4-50e_pastis-s2.py` | MMSeg | 分割 | S2 manifest | 线性探针复现 | 是 |
| `configs/mados/olmoearth-base_4xb4-50e_mados-s2.py` | MMSeg | 分割 | S2 manifest | 线性探针复现 | 是 |
| `configs/sen1floods11/olmoearth-base_4xb4-50e_sen1floods11-s1.py` | MMSeg | 分割 | S1 manifest | 线性探针复现 | 是 |
| `configs/awf/olmoearth-base_4xb4-100e_awf-s2.py` | MMSeg | 分割 | rslearn 转 manifest | 对齐 rslearn 任务语义 | 接近论文任务 |
| `configs/nandi/olmoearth-base_4xb4-100e_nandi-s2.py` | MMSeg | 分割 | rslearn 转 manifest | 对齐 rslearn 任务语义 | 接近论文任务 |
| `configs/crop_type/olmoearth-base_1xb8-50e_crop-type-s2-linear.py` | MMSeg | 分割 | GEO-Bench S2 | online linear probe | 是 |
| `configs/crop_type/olmoearth-base_1xb8-50e_crop-type-s2-offline-linear.py` | MMSeg | 分割 | embedding GeoTIFF | offline linear probe | 是，但有离线缓存例外 |
| `configs/potsdam/olmoearth-base_upernet_4xb4-80k_potsdam-rgb-p4-512x512.py` | MMSeg | 分割 | RGB adapter | UPerNet 工程实验 | 否 |
| `configs/olmoearth-base_faster-rcnn_1x_rslearn-detection-s2.py` | MMDet | 水平框检测 | rslearn detection manifest | 对齐 rslearn detection | 接近原任务 |
| `configs/olmoearth-base_faster-rcnn_1x_dior-rgb.py` | MMDet | 水平框检测 | RGB adapter | OpenMMLab DIOR 示例 | 否 |
| `configs/olmoearth-base_oriented-rcnn_1x_dota-rgb.py` | MMRotate | 旋转框检测 | RGB adapter | DOTA rotated 示例 | 否 |
| `configs/olmoearth-base_oriented-rcnn_1x_dior-rgb.py` | MMRotate | 旋转框检测 | RGB adapter | DIOR-R rotated 示例 | 否 |

表里的“否”不是说实验没有价值，而是提醒：RGB adapter 或常规遥感图片数据集不等于
OLMoEarth 论文中的多光谱/多时相设置，结果不能直接当成论文复现数值。

## 1. 环境与复现前提

环境、权重和依赖是所有迁移工作的前置条件。

### 环境准备

推荐把 OpenMMLab 和 OlmoEarth 放在同一个环境里，避免权重和依赖在不同
Python 环境之间来回找。

核心依赖：

- Python 3.10 或 3.11。
- PyTorch。OLMoEarth 原项目偏新，服务器如果是 PyTorch 2.3，需要保留本项目里
  对 CUDA bool sort 的兼容补丁。
- OpenMMLab：`mmengine`、`mmcv`、`mmsegmentation`、`mmdetection`、`mmrotate`。
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

## 2. OpenMMLab 迁移方法论

这一部分抽象出可迁移到其他模型的通用方法，而不是只服务 OlmoEarth。

### 通用迁移模板

如果把 OlmoEarth 换成另一个非 OpenMMLab 模型，也可以按同样顺序迁移。

#### Step 1：拆原项目

先不要急着写 OpenMMLab config，先回答：

| 问题 | 为什么重要 |
| --- | --- |
| 原项目 Dataset 最终返回什么字段 | 决定是否转 manifest、写 Dataset、还是用原生 Dataset |
| 模型 `forward` 真实需要哪些输入 | 决定 backbone wrapper 要从 DataSample metainfo 里拿什么 |
| 权重文件是什么结构 | 决定用 `init_cfg`、自定义 `init_weights`，还是顶层 `load_from` |
| 训练时哪些模块冻结 | 决定 optimizer paramwise、`requires_grad` 和复现设置 |
| metric 如何计算 | 决定用 OpenMMLab 原生 metric 还是自定义 metric |

#### Step 2：定 OpenMMLab 边界

迁移时尽量让 OpenMMLab 管 OpenMMLab 擅长的部分：

| 交给 OpenMMLab | 从原项目保留 |
| --- | --- |
| runner、optimizer、hook、DDP、AMP、checkpoint | 模型结构、权重命名、核心数学算子 |
| Dataset 生命周期、sampler、pipeline、DataSample | 特殊输入语义，如 timestamp、mask、band order |
| 原生 metric，如 IoU/VOC/DOTA | 原论文特有 valid mask、F1、ignore 规则 |

如果一个逻辑影响论文精度，就不要为了“更像 OpenMMLab”随便改；如果一个逻辑只是
训练工程控制，就应该尽量交给 OpenMMLab。

#### Step 3：先建最小闭环

最小闭环不是完整训练，而是：

```text
config 可解析
  -> Dataset 能取一个样本
  -> pipeline 后 tensor shape 正确
  -> model 能算一次 loss
  -> metric 能跑一次 val
```

只有这个闭环稳定后，再讨论长训练、调参和复现实验。

#### Step 4：写清楚复现边界

每个 config 都应该能回答：

| 问题 | 例子 |
| --- | --- |
| 是否论文复现 | PASTIS/Crop-Type linear probe 是，Potsdam/DIOR RGB 不是 |
| 是否冻结 backbone | linear probe 冻结，UPerNet/检测实验可训练 |
| 输入是否同分布 | Sentinel-2 是，RGB adapter 不是 |
| metric 是否同原论文 | valid-mask IoU/F1 需要特别说明 |
| 是否有离线缓存 | embedding 有，online 训练没有 |

这样其他人学习迁移时，看到一个结果就知道它代表什么，也知道不能和哪些结果直接比较。

### 迁移总原则

本项目最终采用四条原则。

#### 1. 非侵入式 projects 迁移

不改 OpenMMLab 主干代码，所有新增逻辑放在：

```text
mmsegmentation/projects/olmoearth/
mmdetection/projects/olmoearth/
mmrotate/projects/olmoearth/
```

并通过：

```python
custom_imports = dict(
    imports=["projects.olmoearth.olmoearth"],
    allow_failed_imports=False,
)
```

注册 Dataset、Backbone、Transform、Metric、Hook。

#### 2. 训练时不用 rslearn Dataset

迁移早期有一个诱惑：直接在 OpenMMLab Dataset 里包
`rslearn.train.dataset.ModelDataset`。最后没有这么做。

前后对比：

| 方案 | 优点 | 问题 |
| --- | --- | --- |
| 训练时直接包 rslearn | 最接近原项目数据读取 | 生命周期不符合 OpenMMLab，DataLoader/serialize/filter/debug 都绕进 rslearn |
| 预转换成 manifest | OpenMMLab 原生、可审计、可复现 | 多一个转换步骤 |

本项目选择第二种。转换脚本只负责把原任务语义物化出来，训练阶段只读
GeoTIFF 和 JSON manifest。

#### 3. 原论文任务优先对齐语义，常规数据集走原生格式

分割里的 PASTIS/AWF/Nandi/MADOS/Sen1Floods11、检测里的 rslearn detection，
都保留原任务的 label、valid mask、时间戳和多波段输入。

DIOR、DOTA、Potsdam 这种常规 OpenMMLab 数据集，则尽量走原生 Dataset：

- 原始 DIOR 水平框：MMDetection `XMLDataset` + `RGBToOlmoEarthS2`。
- DIOR-R / DOTA-like DIOR：MMRotate `DIORDataset` 或 `DOTADataset` + `RGBToOlmoEarthS2`。
- DOTA：MMRotate `DOTADataset` + `RGBToOlmoEarthS2`。
- Potsdam：MMSeg Potsdam 数据布局 + RGB adapter。

不要为了“所有数据集统一”而把 DIOR/DOTA/Potsdam 也强行转成 rslearn 格式。

#### 4. `init_cfg` 加载 OLMoEarth 权重

OpenMMLab 原生语义是：

- `model.backbone.init_cfg.checkpoint`：初始化 backbone。
- `load_from`：加载完整 OpenMMLab checkpoint。
- `resume`：恢复训练状态。

因此本项目让 OLMoEarth backbone 自己用 `init_cfg` 加载 `weights.pth`，
这比在 config 里自定义 `model_path/model_id/checkpoint_path` 更符合框架。

### 模型迁移：OpenMMLab 通常要迁哪些东西

迁移一个 backbone 到 OpenMMLab，通常不是只写 `Backbone.forward`。完整链路至少
包含下面几类组件。

| 组件 | MMSeg 位置 | MMDet 位置 | MMRotate 位置 | 为什么需要 |
| --- | --- | --- | --- | --- |
| Dataset | `DATASETS` | `DATASETS` | `DATASETS` | 把 manifest/原生数据变成样本字典 |
| Transform | `TRANSFORMS` | `TRANSFORMS` | `TRANSFORMS` | 读 GeoTIFF、归一化、RGB adapter、crop/pad |
| Pack transform | `PackOlmoEarthSegInputs` | `PackDetInputs` | `PackDetInputs` | 把元数据放进 DataSample |
| Data preprocessor | `OlmoEarthSegDataPreProcessor` | `DetDataPreProcessor` | `DetDataPreProcessor` | pad batch、对齐 valid mask 或 box tensor |
| Backbone | `MODELS` | `MODELS` | `MODELS` | 构造 OLMoEarth sample 并调用 encoder |
| Segmentor/Detector wrapper | `OlmoEarthEncoderDecoder` | `OlmoEarthFasterRCNN` | `OlmoEarthFasterRCNN` | 把 DataSample metainfo 传给 backbone |
| Neck | `MultiLevelNeck` | `OlmoEarthMultiLevelNeck` | `OlmoEarthMultiLevelNeck` | 单尺度 dense map 转多尺度 |
| Head | linear/UPerHead | RPN/RoIHead | OrientedRPN/RotatedRoIHead | 接具体下游任务 |
| Metric | IoU/Accuracy | F1/VOCMetric | DOTAMetric | 对齐论文或数据集指标 |
| Hook/Tool | visualization/checker | checker | 原生 DOTA/DIOR 检查 + smoke train | 多波段可视化和数据预检 |

#### 哪些 import 原项目，哪些自己写

折中原则是：**数学定义和权重结构 import 原项目；框架生命周期自己写。**

直接 import 原项目的部分：

- `olmoearth_pretrain.config.Config`：保证模型结构和 released `config.json` 对齐。
- `patch_legacy_encoder_config`：兼容官方 config。
- `MaskedOlmoEarthSample` / `MaskValue`：保证 sample 和 mask 语义不变。
- `PoolingType` / `pool_unmasked_tokens`：保证 token pooling 逻辑不重写。
- OLMoEarth computed normalization 参数：保证输入尺度对齐预训练。

自己写 OpenMMLab 适配的部分：

- Dataset / manifest loader：OpenMMLab 需要自己的 `load_data_list/filter_data`。
- Transform：OpenMMLab pipeline 负责 image/label 同步增强和元数据传递。
- Backbone wrapper：把 `B,C*T,H,W` 还原成 OLMoEarth 的 `B,H,W,T,C`。
- Segmentor/Detector wrapper：OpenMMLab 默认不会把 timestamps 传给 backbone。
- Neck/head config：让 dense map 接 UPerNet/Faster R-CNN。
- Metric/checker：保留 valid mask、rslearn F1 这类非标准语义。

不建议 import 的部分：

- rslearn `ModelDataset` 作为训练 Dataset。
- OLMoEarth 原生 FSDP/DDP 封装。
- 原项目训练 loop、optimizer 封装、环境变量读取。

原因是这些东西属于训练框架生命周期。OpenMMLab 已经有 runner、sampler、
hook、DDP、AMP、checkpoint 语义；硬搬会让两个框架互相抢控制权。

## 3. OlmoEarth 案例难点

理解 OlmoEarth 为什么不能像普通 ResNet/ViT 一样直接塞进 OpenMMLab。

### OlmoEarth 简介

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

#### 与 ResNet / 普通 ViT 的差异

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

### RGB 输入如何适配 OlmoEarth

RGB 是本项目里最需要强调边界的适配。它的目的不是把 RGB 数据“变成真实
Sentinel-2”，而是让 Potsdam、DIOR 这类普通 RGB 遥感数据能走同一个
OlmoEarth backbone 接口。

#### 为什么需要 adapter

OlmoEarth 的 Sentinel-2 L2A 分支期望的是 12 个 band：

```text
B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12, B01, B09
```

普通 RGB 图像只有 3 个通道，没有近红外、红边、SWIR，也没有真实的 Sentinel-2
辐射尺度。如果直接把 RGB 当成前三个 S2 band 输入，会同时错两个东西：

- band 语义错：RGB 的 R/G/B 不等于 OLMoEarth band order 的前三个 B02/B03/B04。
- 数值尺度错：PNG/JPG 通常是 0-255，OLMoEarth S2 归一化按 0-10000 反射率尺度。

所以我们显式写了 `RGBToOlmoEarthS2`，让这种适配在 config 里可见，而不是
悄悄藏在 Dataset 里。

#### 映射规则

映射关系固定为：

```text
R -> Sentinel-2 B04
G -> Sentinel-2 B03
B -> Sentinel-2 B02
```

OpenMMLab 的 `LoadImageFromFile` 默认通过 cv2/mmcv 读图，常见输出是 BGR
顺序，所以 Potsdam 和 DIOR config 里写的是：

```python
dict(
    type="RGBToOlmoEarthS2",
    num_timesteps=1,
    rgb_channel_order="BGR",
    input_value_range="0_255",
)
```

如果你的 pipeline 已经把图像转成 RGB，就要改成：

```python
rgb_channel_order="RGB"
```

#### 数值怎么处理

`RGBToOlmoEarthS2` 先把输入转到近似 Sentinel-2 反射率尺度：

```text
0-255 RGB -> value * (10000 / 255)
0-1 RGB   -> value * 10000
s2        -> 不缩放，认为已经是 S2 尺度
```

然后只对 B04/B03/B02 三个槽位应用 OLMoEarth 的 Sentinel-2 computed
normalization：

```text
normalized = (value - (mean - 2 * std)) / ((mean + 2 * std) - (mean - 2 * std))
```

这和真实 Sentinel-2 输入使用的是同一套 OLMoEarth computed stats。区别是：
真实 S2 的 12 个 band 都有观测值；RGB adapter 只有 3 个 band 有观测值。

#### 缺失 band 怎么表达

adapter 会创建完整的 12-band Sentinel-2 L2A 通道布局：

```text
输出通道数 = 12 * num_timesteps
```

其中：

- B04/B03/B02 写入由 RGB 映射来的归一化值。
- 其他 Sentinel-2 band 填 0。
- `present_bands = ["B04", "B03", "B02"]`。

后面 `OlmoEarthBackbone` 会根据 `present_bands` 构造 bandset mask。也就是说，
缺失 band 不只是数值填 0，更重要的是 mask 会告诉 OLMoEarth：这些 band 没有
真实观测，不应该当成完整 Sentinel-2 样本。

#### RGB 适配前后对比

| 项目 | RGB 原图 | adapter 后 |
| --- | --- | --- |
| shape | `H x W x 3` | `H x W x 12*T` |
| 通道顺序 | RGB 或 BGR | OLMoEarth Sentinel-2 band order |
| 数值范围 | 0-255 或 0-1 | 先近似到 0-10000，再按 S2 stats 归一化 |
| 缺失 band | 无法表达 | 通过 `present_bands` 和 mask 表达 |
| 任务定位 | 常规 RGB 遥感 | OLMoEarth out-of-domain 兼容实验 |

#### 为什么不把 RGB 转成 tif manifest

Potsdam/DIOR/DOTA 本身就是 OpenMMLab 能直接读取的图片数据集。对它们来说，最干净
的做法是保留原生 Dataset：

```text
Potsdam: MMSeg PotsdamDataset / OlmoEarthPotsdamDataset
DIOR:    MMDet XMLDataset 或 MMRotate DIORDataset
DOTA:    MMRotate DOTADataset
```

然后只在 pipeline 里加 `RGBToOlmoEarthS2`。这样类别映射、split、metric、
可视化都继续按 OpenMMLab 原生态走，只有进入 backbone 前的通道适配是
OLMoEarth 特有的。

如果强行把 RGB 也预转换成 manifest，会多出一份数据副本，但没有新增真实
遥感信息，反而让常规数据集调试更绕。

#### RGB 适配的局限

这一路径不能声称复现 OLMoEarth 论文中的 Sentinel-2 多光谱性能。原因很直接：

- 没有 NIR、red edge、SWIR 等 band。
- 没有真实多时相观测，通常 `num_timesteps=1`。
- 数值尺度只是近似映射到 S2 反射率范围。
- OLMoEarth 的预训练分布是多模态遥感，不是普通 RGB PNG/JPG。

因此文档和 config 里都把它称为 RGB compatibility path。它适合做 Potsdam/DIOR
这类工程实验，不适合作为论文精度对齐的主路径。

## 4. 数据迁移

数据语义先对齐，后面的模型和指标才有意义。

### 数据集迁移：到底在迁什么

OpenMMLab 的 Dataset 并不是“读一个文件就完了”。它要和 sampler、pipeline、
data preprocessor、metric、visualizer、resume/debug 工具一起工作。因此数据集
迁移的目标不是复刻原项目的 Python Dataset，而是把原项目的任务语义变成
OpenMMLab 能稳定消费的标准样本字典。

#### 迁移前：原项目里数据通常长什么样

OLMoEarth/rslearn 侧的数据不是一个统一的图片目录。常见来源有三类：

| 来源 | 原始形态 | 里面真正重要的语义 |
| --- | --- | --- |
| OLMoEarth 预处理 eval 张量 | `.pt`、`.pth`、`.npy` 或项目内部 tensor | 已裁剪好的 image、label、valid mask、months |
| rslearn 项目数据 | dataset root + raster layers + vector layers + split tags | layer 名、band、时间范围、vector label、valid |
| OpenMMLab 常规数据 | PNG/JPG/TIF + label/XML/COCO | 图片路径、类别映射、mask 或 bbox |

如果直接把这些都塞进一个 Dataset，会出现两个问题。

第一，OpenMMLab 看不到稳定的数据边界。比如 rslearn 在 `__getitem__` 里才决定
读哪个 layer、怎么 crop、怎么把 vector 变 box。这样 OpenMMLab 的
`filter_data`、`serialize_data`、可视化、单样本 debug 都很难可靠工作。

第二，遥感任务有很多普通 COCO/VOC 不表达的信息：多时相路径、band order、
timestamp、present bands、valid mask、ignore label、rslearn metadata。强行塞进
COCO/VOC 字段会导致“能训练但语义不透明”。

#### 迁移后：manifest 是中间协议

本项目用 manifest 做中间协议。它不是新的深度学习框架，只是一个可审计的样本
清单：大数组放 GeoTIFF，JSON 只放路径和元数据。

分割样本：

```json
{
  "sample_id": "train_000000",
  "img_paths": [
    "samples/train_000000/t00_sentinel2_l2a.tif",
    "samples/train_000000/t01_sentinel2_l2a.tif"
  ],
  "seg_map_path": "samples/train_000000/label.tif",
  "valid_mask_path": "samples/train_000000/valid_mask.tif",
  "timestamps": [[1, 4, 2020], [1, 5, 2020]],
  "present_bands": ["B02", "B03", "B04", "B08"],
  "olmoearth_modality": "sentinel2_l2a",
  "olmoearth_num_timesteps": 2
}
```

检测样本：

```json
{
  "sample_id": "train_000000",
  "img_paths": ["samples/train_000000/t00_sentinel2_l2a.tif"],
  "height": 128,
  "width": 128,
  "bboxes": [[10.0, 12.0, 42.0, 50.0]],
  "labels": [0],
  "valid": 1,
  "timestamps": [[1, 1, 2024]],
  "present_bands": ["B02", "B03", "B04"],
  "rslearn": {"source_index": 0}
}
```

转换前后可以这样理解：

| 阶段 | 转换前 | 转换后 |
| --- | --- | --- |
| 图像 | 内部 raster item、tensor、PNG/JPG | 每个时相一个多波段 GeoTIFF |
| 标签 | tensor、vector layer、PNG mask、XML | `label.tif` 或 `bboxes + labels` |
| 时间 | rslearn time_range、month tensor、缺省值 | 显式 `timestamps: T x 3` |
| 缺失波段 | 原项目内部 mask | `present_bands` |
| 无效区域 | `valid`、`valid_mask`、ignore label | `valid_mask_path` 或 `valid` 字段 |
| 类别 | 原任务配置、property name | manifest `metainfo.classes` |

这个格式的优势是：

- 训练阶段不再依赖 rslearn 数据生命周期，OpenMMLab sampler/pipeline 可以正常工作。
- 每个样本是什么输入、多少时相、什么 label，一眼能从 JSON 看出来。
- GeoTIFF 能被 GIS/raster 工具查看，比 `.npz` 更适合遥感排错。
- 分割和检测共享同一套“路径 + 元数据”思想，但不强行使用同一种标注格式。
- 原论文任务保留 valid mask/timestamp/band order，常规数据集仍可使用原生格式。

#### 为什么有些数据集转 manifest，有些不转

这里的判断标准不是“统一”，而是“哪个格式最少损失语义”。

| 数据集类型 | 推荐方式 | 原因 |
| --- | --- | --- |
| PASTIS/MADOS/Sen1Floods11 | 转 manifest | 原任务有 OLMoEarth 处理后的 tensor/valid/ignore 语义 |
| AWF/Nandi | rslearn -> manifest | 需要物化 rslearn 的 raster/vector/task 输出 |
| rslearn detection | rslearn -> detection manifest | COCO 不能自然表达 `valid/timestamps/img_paths` |
| Crop-Type | 可直接读 GEO-Bench，也可抽 embedding | 原 loader 能清楚表达 band stats 和 label |
| Potsdam | 用 MMSeg Potsdam 布局 + RGB adapter | 它本来就是图片分割数据集 |
| DIOR 水平框 | 用 MMDet `XMLDataset` + RGB adapter | 它本来就是 VOC/XML 水平框检测数据集 |
| DIOR-R | 用 MMRotate `DIORDataset` + RGB adapter | 需要 oriented XML、rotated box coder 和 rotated mAP |
| DOTA / DOTA-like DIOR | 用 MMRotate `DOTADataset` + RGB adapter | 它本来就是 8 点框 txt 格式 |

换句话说：原始格式已经是 OpenMMLab 擅长的，就不要为了 OLMoEarth 强行转换；
原始格式依赖 rslearn/OLMoEarth 内部 task 语义的，就先转换成 manifest。

#### 数据流转总图

完整的数据迁移可以拆成六层。每层都应该能单独检查，不要等训练报错时才回头
猜是哪一层坏了。

```text
Layer 0: 原始数据
  rslearn dataset / OLMoEarth eval tensor / RGB 图片数据集

Layer 1: converter 或原生 Dataset 选择
  convert_xxx.py / XMLDataset / PotsdamDataset / GeoBench loader

Layer 2: 物化后的数据
  raw GeoTIFF + label GeoTIFF + valid mask GeoTIFF + manifest JSON

Layer 3: OpenMMLab data_info / results
  Dataset.load_data_list() 输出路径和元数据
  pipeline 读取数组并做增强、归一化、adapter

Layer 4: DataSample
  SegDataSample / DetDataSample 保存 gt、metainfo、valid mask

Layer 5: model forward
  wrapper 把 metainfo 交给 backbone
  backbone 构造 MaskedOlmoEarthSample
  head 计算 loss 或 prediction
```

每层的检查工具也不同：

| 层级 | 检查方法 | 典型问题 |
| --- | --- | --- |
| 原始数据 | 打开样本、统计类别、查看 split | 数据集路径错、split 空 |
| converter | 看输出文件数量和 summary | label 越界、valid 全 0 |
| manifest | `check_converted_dataset.py` / `check_converted_det_dataset.py` | 路径丢失、box 越界 |
| pipeline | `check_pipeline.py` 或构建 dataloader 取一个 batch | img/label/mask 尺寸不一致 |
| model | `check_forward.py` 或 1 iter train | channel、T、band order 不匹配 |
| metric | 跑一次 val/test | ignore_index、valid mask、类别数不一致 |

#### 三种格式不要混淆

迁移时最容易混淆的是 manifest sample、OpenMMLab `data_info/results` 和
`DataSample`。它们不是同一个东西。

| 阶段 | 谁产生 | 主要内容 | 是否包含真实数组 |
| --- | --- | --- | --- |
| manifest sample | converter | 路径、label 路径、bbox、timestamp、band 元数据 | 否 |
| data_info | Dataset | 解析后的绝对路径、类别、尺寸、instances | 否 |
| results | pipeline 中间态 | `img`、`gt_seg_map`、增强信息、归一化后数组 | 是 |
| DataSample | Pack transform | gt、metainfo、预测结果容器 | gt 是 tensor，metainfo 是字典 |

为什么要拆这么细：OpenMMLab 的 pipeline 会不断修改 `results`，比如 crop/flip/resize
会同步改 image 和 label；而 `DataSample.metainfo` 是最终传给模型和可视化的元数据。
OlmoEarth 的 timestamps、present_bands 必须从 manifest 一路保留到 metainfo，
否则 backbone 只能使用默认时间和默认全 band，精度语义就变了。

#### 数据集迁移时必须保留的字段

不同任务字段不完全一样，但下面这些字段最好在转换时显式写出来。

分割：

| 字段 | 是否必需 | 用途 |
| --- | --- | --- |
| `sample_id` | 推荐 | debug、可视化、错误定位 |
| `img_paths` | 必需 | 每个时相一个 GeoTIFF |
| `seg_map_path` | 必需 | segmentation label |
| `valid_mask_path` | 论文复现任务推荐 | 过滤无效像素 |
| `timestamps` | 推荐 | 构造 OLMoEarth temporal encoding |
| `present_bands` | RGB/缺 band 必需 | 构造 missing mask |
| `olmoearth_modality` | 推荐 | 明确走哪个 sample field |
| `olmoearth_num_timesteps` | 推荐 | 和 config 的 T 互相校验 |

检测：

| 字段 | 是否必需 | 用途 |
| --- | --- | --- |
| `sample_id` | 推荐 | debug、可视化、错误定位 |
| `img_paths` | 必需 | 单时相或多时相 GeoTIFF |
| `height/width` | 必需 | box 检查和 Dataset 信息 |
| `bboxes` | 必需 | xyxy 格式检测框 |
| `labels` | 必需 | 0-based 类别 id |
| `valid` | rslearn 任务必需 | 跳过无效样本或 metric 过滤 |
| `timestamps` | 推荐 | 构造 temporal encoding |
| `present_bands` | RGB/缺 band 必需 | 构造 missing mask |
| `rslearn` | 可选但推荐 | 保留 source_index/window 等调试信息 |

#### 迁移后为什么更好调试

使用 manifest 后，很多问题可以在训练前发现。

| 旧方式：训练时包 rslearn | 新方式：manifest |
| --- | --- |
| 数据读取逻辑藏在 `__getitem__` 内部 | JSON 可直接看到每个样本的输入和标签 |
| OpenMMLab 很难提前 filter/serialize | Dataset 可以正常走 OpenMMLab 生命周期 |
| 报错常发生在 DataLoader worker 内 | checker 可以先单进程检查路径和 label |
| 时间、band、valid 语义不透明 | 关键字段显式保存在 sample 里 |
| 换服务器/环境时 rslearn 依赖重 | 训练阶段只依赖 GeoTIFF + OpenMMLab |

## 5. 模型与 Forward 迁移

把 OpenMMLab 的 tensor/DataSample 变成 OlmoEarth 需要的 masked sample，再接回 head。

### 推理和训练的 forward 逻辑

#### MMSeg online forward

MMSeg online 路径可以按这个顺序理解：

```text
manifest sample
  -> OlmoEarthSegDataset.load_data_list()
  -> LoadOlmoEarthArrays
       img_paths: T 个 GeoTIFF
       stack: T,C,H,W
       flatten: H,W,C*T
       label/valid_mask/timestamps 一起放入 results
  -> Normalize / Crop / Pad / Flip
  -> PackOlmoEarthSegInputs
       inputs: C*T,H,W
       SegDataSample.metainfo: timestamps/present_bands/...
  -> OlmoEarthSegDataPreProcessor
       batch pad inputs/labels/valid_mask
  -> OlmoEarthEncoderDecoder.loss/predict
       set_batch_metainfo(data_samples.metainfo)
  -> OlmoEarthBackbone.forward(inputs)
       reshape: B,C*T,H,W -> B,H,W,T,C
       build bandset_mask from present_bands
       build timestamps tensor
       MaskedOlmoEarthSample(...)
       encoder(sample, fast_pass=auto, patch_size=...)
       pool_unmasked_tokens(...)
       output: (B,D,H/patch,W/patch,)
  -> decode_head / auxiliary_head
  -> loss or prediction
```

这里最关键的是 `OlmoEarthEncoderDecoder`。普通 `EncoderDecoder` 只会把 image
tensor 传给 backbone，不知道 timestamps 和 present bands。我们加 wrapper 的
目的就是把 `SegDataSample.metainfo` 临时塞给 backbone。

#### MMSeg offline embedding forward

offline probe 则把 encoder forward 前移到抽特征阶段：

```text
原始样本 -> extract_embeddings.py -> embedding.tif
embedding.tif -> OlmoEarthFeatureBackbone -> patch-linear head
```

优势是训练阶段不再反复跑 OLMoEarth encoder，更接近论文的线性探针评估方式。
代价是 embedding 固定，不能端到端微调 backbone。

这里要明确一个边界：`extract_embeddings.py` 不是标准 MMSeg Runner 训练流程。
它会复用 MMSeg 的 config、registry、Dataset 和 pipeline 来构建同一批样本，但
不会进入 `tools/train.py` 的 train loop，也不会走 optimizer、hook、scheduler。
它只是以 inference mode 调用 OLMoEarth backbone，把 dense feature 物化为
`embedding.tif`。随后 offline probe 再回到标准 MMSeg：`tools/train.py` 构建
`OlmoEarthFeatureBackbone`，读取固定 embedding，并训练 patch-linear head。

因此它是为了对齐 OLMoEarth 原论文线性探针和节省重复 encoder forward 的工程例外，
不是通用的 MMSeg 数据读取范式。只要改了 checkpoint、`patch_size`、输入 pipeline、
crop size、normalization 或 split，都应该重新抽 embedding。

#### MMDet forward

MMDet 检测路径类似，但后面接的是 detector：

```text
detection manifest / XMLDataset
  -> LoadOlmoEarthTifFromFile 或 LoadImageFromFile
  -> OlmoEarthNormalize 或 RGBToOlmoEarthS2
  -> LoadAnnotations
  -> PackDetInputs
       DetDataSample.metainfo: timestamps/present_bands/...
  -> DetDataPreProcessor
  -> OlmoEarthFasterRCNN.loss/predict
       set_batch_metainfo(data_samples.metainfo)
  -> OlmoEarthBackbone.forward
       output one dense feature map
  -> OlmoEarthMultiLevelNeck
       one map -> strides [p, 2p, 4p, 8p]
  -> RPNHead
  -> RoIHead
  -> bbox loss or predictions
```

为什么 MMDet 需要 neck：Faster R-CNN/FPN 系列默认消费多尺度 feature。OlmoEarth
不像 ResNet 那样天然输出 C2/C3/C4/C5，所以需要把单个 dense map 派生成多个尺度。
这不是让 OlmoEarth 真的变成 FPN backbone，而是满足 RPN/RoIHead 的接口假设。

#### MMRotate forward

MMRotate 的路径和 MMDet 相似，但 box 语义完全不同：

```text
DOTA / DIOR-R
  -> LoadImageFromFile
  -> LoadAnnotations(box_type="qbox")
  -> ConvertBoxType(qbox -> rbox)
  -> Resize / RandomFlip
  -> RGBToOlmoEarthS2
  -> PackDetInputs
       DetDataSample.metainfo: present_bands/...
       gt_instances.bboxes: rotated boxes
  -> OlmoEarthFasterRCNN.loss/predict
       set_batch_metainfo(data_samples.metainfo)
  -> OlmoEarthBackbone.forward
  -> OlmoEarthMultiLevelNeck
       one map -> strides [p, 2p, 4p, 8p, 16p]
  -> OrientedRPNHead
  -> RotatedSingleRoIExtractor + rotated bbox head
  -> rotated loss / nms_rotated / DOTAMetric
```

这里不能继续使用普通 MMDet 的水平框 `RPNHead + SingleRoIExtractor`，因为
DOTA/DIOR-R 的标注和评估都是旋转框。迁移到 MMRotate 的核心价值，就是复用
`qbox -> rbox`、rotated IoU、rotated NMS 和 DOTA metric。

#### Shape trace：MMSeg 多时相 Sentinel-2

假设一个 PASTIS 样本有 `T=12` 个时相，每个时相是 `C=12` 个 Sentinel-2 band，
crop 后大小为 `H=W=128`，batch size 为 `B=4`。

| 步骤 | 张量形状 | 说明 |
| --- | --- | --- |
| 单个 GeoTIFF | `C,H,W = 12,128,128` | 每个时相一个多 band tif |
| stack 多时相 | `T,C,H,W = 12,12,128,128` | `img_paths` 列表读入 |
| flatten 给 MMSeg | `H,W,C*T = 128,128,144` | pipeline 中间态 |
| Pack 后 | `C*T,H,W = 144,128,128` | `inputs` |
| batch 后 | `B,C*T,H,W = 4,144,128,128` | data preprocessor 输出 |
| backbone 内部 reshape | `B,H,W,T,C = 4,128,128,12,12` | 还原 OLMoEarth 输入 |
| encoder 输出 feature | `B,D,H/p,W/p` | `p=4` 时是 `4,768,32,32` |
| decode head logits | `B,num_classes,H,W` | 上采样回 label 尺寸 |

如果这里任何一行对不上，先不要怀疑模型，先检查 manifest 的 `img_paths` 数量、
config 的 `num_timesteps`、band order 和 pipeline 顺序。

#### Shape trace：MMDet DIOR RGB

假设 DIOR 图片 resize 后近似 `800 x 800`，batch size 为 `B=4`，`patch_size=8`。

| 步骤 | 张量形状 | 说明 |
| --- | --- | --- |
| LoadImageFromFile | `H,W,3` | OpenMMLab 常规 BGR/RGB 图片 |
| RGBToOlmoEarthS2 | `H,W,12` | 只写 B04/B03/B02，其他 band 缺失 |
| PackDetInputs | `12,H,W` | DetDataSample 保存 bbox 和 metainfo |
| batch 后 | `B,12,H,W` | DetDataPreProcessor 输出 |
| backbone 内部 reshape | `B,H,W,1,12` | 单时相 Sentinel-2 兼容输入 |
| encoder 输出 feature | `B,768,H/8,W/8` | 单个 dense feature map |
| MultiLevelNeck | 4 个尺度 | stride 为 `8,16,32,64` |
| RPN/RoIHead | proposals / boxes | 按 Faster R-CNN 逻辑训练或推理 |

DIOR 这条路径没有真实 NIR/SWIR/red-edge band。shape 看起来像 Sentinel-2，
但语义上是 RGB compatibility。

#### Shape trace：MMRotate DOTA / DIOR-R RGB

假设输入图片 resize 后为 `1024 x 1024`，`patch_size=8`。

| 步骤 | 数据形态 | 说明 |
| --- | --- | --- |
| LoadImageFromFile | `H,W,3` | 原生 RGB/BGR 图片 |
| LoadAnnotations | qbox，8 点四边形 | DOTA txt 或 DIOR-R XML 解析结果 |
| ConvertBoxType | rbox，`cx,cy,w,h,angle` | 旋转框检测头消费的 box 类型 |
| RGBToOlmoEarthS2 | `H,W,12` | 只写 B04/B03/B02，其他 band 缺失 |
| PackDetInputs | `12,H,W` + rotated gt | metainfo 保留 present_bands |
| backbone | `B,768,H/8,W/8` | OLMoEarth 单个 dense feature map |
| MultiLevelNeck | 5 个尺度 | stride 为 `8,16,32,64,128` |
| Oriented R-CNN | rotated proposals / rboxes | 用 rotated IoU、rotated NMS、DOTAMetric |

如果你的 annotation 是水平框 XML，就不应该走这条路径；用 MMDet 的 DIOR
配置即可。只有标注本身是旋转框，MMRotate 才是合理选择。

## 6. 复现实验 Recipes

这一部分从“怎么迁”落到“怎么跑、怎么确认结果语义”。

### 最小跑通命令

学习迁移时不要直接开长训练。建议按“config -> dataset -> forward -> train”的顺序
逐步加复杂度。

#### DIOR / MMDet RGB 路线

先确认 config 能解析：

```bash
python tools/misc/print_config.py \
  projects/olmoearth/configs/olmoearth-base_faster-rcnn_1x_dior-rgb.py
```

然后跑一个很短的训练，先看 dataloader、forward、loss 是否正常：

```bash
python tools/train.py \
  projects/olmoearth/configs/olmoearth-base_faster-rcnn_1x_dior-rgb.py \
  --cfg-options train_cfg.max_epochs=1 default_hooks.logger.interval=1
```

如果已经有 checkpoint，再测试：

```bash
python tools/test.py \
  projects/olmoearth/configs/olmoearth-base_faster-rcnn_1x_dior-rgb.py \
  work_dirs/olmoearth-base_faster-rcnn_dior-rgb/latest.pth
```

#### rslearn detection / MMDet manifest 路线

先转换：

```bash
python projects/olmoearth/tools/convert_rslearn_det.py \
  --input-root /path/to/rslearn_dataset \
  --output-root data/rslearn_detection_manifest \
  --image-layers sentinel2 \
  --target-layers label \
  --classes object \
  --property-name category
```

再检查 manifest：

```bash
python projects/olmoearth/tools/check_converted_det_dataset.py \
  --data-root data/rslearn_detection_manifest \
  --ann-file train.json
```

最后训练：

```bash
python tools/train.py \
  projects/olmoearth/configs/olmoearth-base_faster-rcnn_1x_rslearn-detection-s2.py
```

#### DOTA / DIOR-R / MMRotate RGB 路线

先看数据目录属于哪一种：

```bash
find /path/to/data -maxdepth 3 -type d | sort
find /path/to/data -maxdepth 4 -type f \( -name "*.txt" -o -name "*.xml" \) | head
```

如果是标准 DOTA split：

```text
trainval/images/*.png
trainval/annfiles/*.txt
```

跑：

```bash
python tools/misc/print_config.py \
  projects/olmoearth/configs/olmoearth-base_oriented-rcnn_1x_dota-rgb.py

python tools/train.py \
  projects/olmoearth/configs/olmoearth-base_oriented-rcnn_1x_dota-rgb.py \
  --cfg-options train_cfg.max_epochs=1 default_hooks.logger.interval=1
```

如果是 DIOR-R oriented XML：

```text
JPEGImages-trainval/*.jpg
JPEGImages-test/*.jpg
Annotations/Oriented Bounding Boxes/*.xml
ImageSets/Main/train.txt
ImageSets/Main/test.txt
```

跑：

```bash
python tools/train.py \
  projects/olmoearth/configs/olmoearth-base_oriented-rcnn_1x_dior-rgb.py \
  --cfg-options train_cfg.max_epochs=1 default_hooks.logger.interval=1
```

如果 DIOR 已经被转换成 DOTA-like txt，则用：

```bash
python tools/train.py \
  projects/olmoearth/configs/olmoearth-base_oriented-rcnn_1x_dior-dota-rgb.py \
  --cfg-options train_cfg.max_epochs=1 default_hooks.logger.interval=1
```

#### MMSeg manifest 路线

以 PASTIS 为例：

```bash
python projects/olmoearth/tools/convert_pastis.py \
  --input-root /path/to/pastis_r \
  --output-root data/olmoearth_mmseg/pastis

python projects/olmoearth/tools/check_converted_dataset.py \
  --data-root data/olmoearth_mmseg/pastis \
  --ann-file train.json

python projects/olmoearth/tools/check_pipeline.py \
  projects/olmoearth/configs/pastis/olmoearth-base_4xb4-50e_pastis-s2.py \
  --split train

python projects/olmoearth/tools/check_forward.py \
  projects/olmoearth/configs/pastis/olmoearth-base_4xb4-50e_pastis-s2.py \
  --split train \
  --device cuda
```

`check_pipeline.py` 用来确认数据增强和 pack 后的 tensor 对齐；`check_forward.py`
用来确认模型能计算一次 loss。它们比完整训练便宜得多。

### 迁移到 MMSegmentation

#### 数据集准备

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

#### Backbone 封装与注册

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

#### Feature map 适配与 decode head

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

#### 训练流程

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

注意：抽 embedding 的第一步虽然读取 MMSeg config 和 dataset，但不是 MMSeg
标准训练；它绕过 runner，只做一次离线特征物化。第二步训练 offline-linear config
时，才是标准 MMSeg 训练流程。

### 迁移到 MMDetection

#### 数据集准备

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

#### Backbone 接入与 neck/head 适配

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

#### DIOR 常规数据集示例

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

### 迁移到 MMRotate

#### 数据集准备

MMRotate 只负责旋转框。先判断标注格式：

| 格式 | 目录/文件特征 | config |
| --- | --- | --- |
| DOTA | `annfiles/*.txt`，每行 8 个点 | `olmoearth-base_oriented-rcnn_1x_dota-rgb.py` |
| DIOR-R XML | `Annotations/Oriented Bounding Boxes/*.xml`，含 `robndbox` | `olmoearth-base_oriented-rcnn_1x_dior-rgb.py` |
| DIOR 转 DOTA | DIOR 类别名 + DOTA txt 行格式 | `olmoearth-base_oriented-rcnn_1x_dior-dota-rgb.py` |

不要把水平框 DIOR XML 硬塞进 MMRotate。水平框 DIOR 更适合 MMDetection；
DIOR-R 或 DOTA-like DIOR 才适合 MMRotate。

#### Box flow

MMRotate pipeline 中 box 会经历：

```text
8 点 qbox
  -> ConvertBoxType(qbox -> rbox)
  -> rotated assigner / sampler
  -> rotated bbox coder
  -> nms_rotated
  -> DOTAMetric
```

OLMoEarth 只替换 backbone 输入和 feature 输出，不改这条 rotated box 语义链。
这也是迁移到 MMRotate 的关键：box、NMS、metric 继续使用框架原生实现。

#### Backbone 和 neck

MMRotate 的 Oriented R-CNN 默认需要多尺度特征。本项目使用：

```text
OlmoEarthBackbone -> OlmoEarthMultiLevelNeck -> OrientedRPNHead -> Rotated ROI Head
```

`OlmoEarthMultiLevelNeck` 从一个 dense feature map 派生 5 个尺度：

```text
stride: patch_size, 2p, 4p, 8p, 16p
scale:  1.0,        0.5, 0.25, 0.125, 0.0625
```

这是一种接口适配，不表示 OLMoEarth encoder 本身天然输出 FPN。

### 评估与可视化

#### 分割

分割使用 `OlmoEarthIoUMetric`：

- 输出 MMSeg 风格的 `aAcc`、`mIoU`、`mAcc`。
- 可选用 valid mask 过滤无效像素。
- 每类 IoU 表格仍按 OpenMMLab 日志打印。

可视化问题在多波段数据上很常见。默认 MMSeg visual hook 假设输入来自 RGB
文件路径，但 OLMoEarth 输入可能是多时相多波段张量。因此项目里提供
`OlmoEarthVisualizationHook`，直接从 batch tensor 生成可视化，避免读取错文件。

#### 检测

rslearn manifest 检测使用 `OlmoEarthDetMetric`：

- 按 class 分组。
- 用 IoU 做预测框和 GT 框匹配。
- 在多个 score threshold 下报告 F1、precision、recall。
- 输出 best F1 对应的 TP/FP/FN，便于排查阈值问题。

原始 DIOR 水平框这类常规 MMDetection 数据集继续用 `VOCMetric` 或数据集标准
metric。DOTA、DIOR-R 这类旋转框数据则交给 MMRotate 的 `DOTAMetric`。

## 7. Debug 与常见问题

把常见报错按原因和排查入口集中起来。

### 常见问题与调试

#### 权重加载失败

检查三件事：

1. `model.backbone.model_config_path` 指向 released `config.json`。
2. `model.backbone.init_cfg.checkpoint` 指向 released `weights.pth`。
3. 顶层 `load_from` 没有误填 OLMoEarth backbone 权重。

#### 输入通道不匹配

报错类似：

```text
Expected 144 channels (12 bands x 12 timesteps), got 36
```

说明 config 的 `num_timesteps` 和 manifest 里的 `img_paths` 数量不一致，
或者 band order 不一致。

#### `fast_pass=True` 报错或精度异常

不要固定 `fast_pass=True`。RGB adapter、缺失 band、多模态缺失 token 都应该
让 backbone 自动判断。固定 True 会跳过缺失 token 处理，可能直接错，也可能
悄悄改变语义。

#### PyTorch 2.3 bool sort 报错

报错类似：

```text
Sort currently does not support bool dtype on CUDA.
```

本项目的 backbone 已把 bool mask 临时转成 `uint8` 再 sort，这是为了兼容
OlmoEarth 原始代码在较旧 PyTorch CUDA 上的问题。

#### 大图慢或显存高

分割 offline embedding extractor 已支持滑窗：

```bash
python projects/olmoearth/tools/extract_embeddings.py \
  --tile-size 512 \
  --tile-overlap 64
```

检测和在线分割训练仍建议用 OpenMMLab 的 crop / resize / batch size / AMP
控制显存。

#### position embedding

OlmoEarth 的 dense encoder 通过 `patch_size` 控制输出 stride。不要把普通 ViT
“固定 16x16 patch”的直觉直接套过来。对于高分辨率任务：

- `patch_size=4`：细节好，显存高。
- `patch_size=16`：显存低，输出更粗。

#### 报错索引

| 报错或现象 | 大概率原因 | 先检查哪里 | 处理方式 |
| --- | --- | --- | --- |
| `Expected 144 channels, got 36` | `num_timesteps` 或 band 数不一致 | manifest `img_paths`、config `num_timesteps` | 让 T 和 `C*T` 对齐 |
| `Sort currently does not support bool dtype on CUDA` | PyTorch 2.3 bool sort 限制 | backbone 兼容补丁是否存在 | 保留 bool->uint8 sort patch |
| `LatentMIM.forward() got unexpected keyword fast_pass` | OLMoEarth 版本 forward 签名不同 | backbone 调用 encoder 的位置 | 做 signature 兼容或升级代码 |
| CUDA `device-side assert triggered` | label 越界或 ignore_index 错 | label 取值、num_classes、ignore_index | 跑 dataset checker，修类别映射 |
| mIoU 很低但 loss 正常下降 | ignore/valid mask 没对齐 | valid mask、metric `use_valid_mask` | 确认 pad/crop 同步作用到 mask |
| mAP/F1 全 0 | bbox 坐标或类别错 | manifest `bboxes/labels` | 检查 xyxy、0-based label、图片尺寸 |
| `No such file or directory` | `data_root`、路径空格或相对路径错 | config 顶部路径、manifest 路径 | 用绝对路径，manifest 路径相对 data_root |
| `KeyError: modality/band` | modality 名或 band order 错 | `olmoearth_modality`、`band_names` | 使用 OLMoEarth 定义的 band order |
| 显存异常低 | backbone 可能冻结或没跑 | optimizer params、日志、显存曲线 | 检查 `requires_grad` 和 offline/online 路径 |
| 显存异常高 | patch_size 小、crop 大、batch 大 | config crop、patch_size、batch size | 调大 patch_size 或减小 crop/batch |

排错顺序建议固定为：

```text
路径是否存在
  -> manifest 字段是否完整
  -> label/box 是否合法
  -> pipeline 输出 shape 是否对
  -> backbone 输入 channel/T 是否对
  -> metric 是否对齐 ignore/valid
```

## 8. 进阶方向

当复现闭环稳定后，再考虑更复杂的扩展。

### 进阶方向

#### 多波段与多模态

manifest 的好处是可以自然扩展：

- `img_paths` 支持多时相。
- `present_bands` 支持缺失 band。
- modality 可以从 `sentinel2_l2a` 扩展到 `sentinel1` 等。

但每个新 modality 都要确认：

- band order。
- normalization。
- `MaskedOlmoEarthSample` 字段名。
- mask 的 bandset 语义。

#### 参数高效微调

现在的复现路径主要是冻结 backbone + probe，或全量训练。后续可以加：

- LoRA。
- Adapter。
- bias-only / norm-only tuning。

建议仍然放在 `projects/olmoearth`，不要改 OpenMMLab 主干。

#### 小样本与滑窗预测

遥感常见问题是数据少、图大、类别稀疏。实用方向：

- offline embedding 缓存，减少重复 encoder forward。
- overlap sliding window，降低边界 artifact。
- 按 class frequency 调 loss 或 sampler。
- 对 valid mask 做严格检查，避免无效区域污染指标。

## 9. 经验总结与 Checklist

最后用经验和清单帮助读者迁移下一个模型或数据集。

### 最重要的迁移经验

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

### 迁移 Checklist

真正迁移一个新数据集时，可以按下面的清单逐项打勾。

#### 数据 Checklist

- 原始数据的 split 是否明确，train/val/test 是否为空。
- 图像是 RGB、单时相多波段，还是多时相多波段。
- band order 是否和 OLMoEarth modality 对齐。
- 原始数值范围是否明确：0-255、0-1、0-10000，还是已经归一化。
- label 是否是 0-based。
- ignore label 是否统一成 OpenMMLab 的 `ignore_index`。
- valid mask 是否需要参与 loss 或 metric。
- detection bbox 是否是 xyxy，且没有越界。
- class names 是否和 config 的 `num_classes` 对齐。
- manifest 路径是否都相对 `data_root`，或使用绝对路径。

#### Pipeline Checklist

- image、label、valid mask 是否一起 crop/flip/pad。
- RGB 数据是否显式使用 `RGBToOlmoEarthS2`。
- 多波段数据是否使用 OLMoEarth normalization，而不是 ImageNet mean/std。
- `PackOlmoEarthSegInputs` / `PackDetInputs` 是否保留 timestamps 和 present_bands。
- data preprocessor 的 pad size 是否和 head/backbone stride 兼容。

#### 模型 Checklist

- `model.backbone.model_config_path` 是否指向 OLMoEarth `config.json`。
- `model.backbone.init_cfg.checkpoint` 是否指向 OLMoEarth `weights.pth`。
- 顶层 `load_from` 是否只用于完整 OpenMMLab checkpoint。
- `num_timesteps` 是否和 manifest 的 `img_paths` 数量一致。
- `patch_size` 是否和期望输出 stride 一致。
- 检测任务是否需要 multi-level neck。
- 分割任务是 paper-style linear probe，还是 UPerNet 工程实验。
- backbone 是否应该冻结，是否和实验目标一致。

#### 评估 Checklist

- 分割是否需要 valid mask filtering。
- 检测是用数据集标准 mAP，还是 rslearn-style F1。
- `ignore_index` 是否和 label 里的 ignore 值一致。
- 可视化是否能处理多波段输入。
- 评价前是否先跑过一个小 batch 的 forward。

#### 复现实验 Checklist

- 是否使用论文对应的数据预处理，而不是为了方便换成 RGB adapter。
- 是否冻结了论文线性探针中应该冻结的部分。
- 是否保留了原任务的 timestamp、valid mask、ignore label。
- 是否使用和原任务一致的 metric。
- 是否记录了 patch_size、crop_size、batch size、学习率和随机种子。
