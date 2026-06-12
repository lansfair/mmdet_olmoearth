#! /usr/bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")" && pwd > /dev/null

# 自动读取归档目录绝对路径，非必要不修改。  
export MM_ARCHIVE_HOME=$(echo "$PWD" | cut -d '/' -f 1-5)

# 自动读取归档数据目录路径，非必要不修改。
export MM_ARCHIVE_DATA_HOME="$MM_ARCHIVE_HOME/dat"

# 自动读取归档预训练骨干网络权重目录路径，非必要不修改。
export MM_ARCHIVE_CKPT_HOME="$MM_ARCHIVE_HOME/pretrained"

# Miniconda3 安装路径，非必要不修改。
export CONDA_HOME="$HOME/miniconda3"

# Miniconda3 虚拟环境名称，确保此环境可以正常运行你的项目。
export CONDA_ENV_NAME='openmm'

# 指定训练使用的 GPU 索引，格式：0, 1, 2..., N。
export CUDA_VISIBLE_DEVICES='0'
                              
# 训练启动配置名称。
export CONFIG_NAME='olmoearth-base_faster-rcnn_1x_dior-rgb'


SRCDIR=$PWD
cd "$PWD/../../../"
export PYTHONPATH=".":"$PYTHONPATH"
PYINTERPRETER="$CONDA_HOME/envs/$CONDA_ENV_NAME/bin/python3"

MM_WORK_HOME="$(echo "$MM_ARCHIVE_HOME" | cut -d '/' -f 1-4)"
WORK_DIR="$MM_WORK_HOME/openmmlab_work_dirs/mmdet/"$(basename $SRCDIR)""
mkdir -p "$WORK_DIR"

$PYINTERPRETER "$PWD/tools/train.py" "$SRCDIR/configs/$CONFIG_NAME.py" --work-dir "$WORK_DIR"
