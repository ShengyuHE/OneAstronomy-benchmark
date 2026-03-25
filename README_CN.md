# AION 快速入门指南

## 1. 数据准备

- **下载数据**：请从以下地址获取数据集
- **谷歌网盘地址**：<https://drive.google.com/file/d/1kxW4odI3EqIojyCttJ2S1MEZV7gu7GPJ/view?usp=drive_link>
- 将下载后的解压缩放在项目目录的data目录中。

## 2. 环境搭建

### 2.1 容器安装（推荐）

- **拉取镜像**：`docker pull zwgcv/aion:v1.0`
- **创建容器**：`docker run -dit --gpus all --name aion_container -v /home/:/mnt zwgcv/aion:v1.0`
- **进入容器**：`docker exec -it aion_container_id /bin/bash`（将 `aion_container_id` 替换为你的容器ID）
- **查看容器ID**：通过 `docker ps -a` 命令查看
- **激活环境**：`conda activate aion`

### 2.2 直接安装

如果你已经有了PyTorch环境，可以直接通过以下命令安装AION：

- **安装AION**：`pip install polymathic-aion`

## 3. 模型预测

- **执行红移预测**：`python RedshiftPrediction.py`

## 4. 模型训练

- **启动训练**：`CUDA_VISIBLE_DEVICES=0,1 nohup torchrun --nproc_per_node=2 train_aion_ddp_token.py > training.log 2>&1 & disown`
  - 此命令将在后台运行训练过程，使用2个GPU，并将输出重定向到 training.log 文件

## 5. 项目结构

```
AION/
├── aion/                # 核心源代码目录
│   ├── codecs/          # 编码/解码相关代码
│   ├── fourm/           # 特定功能模块
│   ├── __init__.py
│   ├── modalities.py    # 模态定义
│   └── model.py         # 模型定义
├── data/                # 数据目录
│   ├── test/            # 测试数据
│   ├── train/           # 训练数据
│   └── best_model.pt    # 测试模型
├── docs/                # 文档目录
├── notebooks/           # Jupyter notebooks
├── scripts/             # 脚本目录
├── tests/               # 测试目录
├── outputs/             # 输出目录
├── weight/              # 模型权重目录
├── RedshiftPrediction.py # 红移预测脚本
├── train_aion_ddp_token.py # 训练脚本
├── download_model.py    # 模型下载脚本
├── README.md            # 详细文档
├── README_CN.md         # 中文详细文档
└── README_FIRST.md      # 快速入门指南
```

