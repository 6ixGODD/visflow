# VisFlow
[English](README.md) | 中文

## 概述

VisFlow 是一个专为计算机视觉任务设计的综合深度学习框架，为模型训练、评估和可视化提供了流畅的工作流程。基于 PyTorch 构建。

### 主要特性

- 🚀 **易用的命令行界面**: 通过简单命令训练模型和生成可视化
- 🎯 **广泛的模型支持**: 内置支持流行架构（ResNet、VGG、EfficientNet、ViT 等）
- 📊 **高级可视化**: 集成 Grad-CAM 支持，提供模型可解释性
- ⚙️ **灵活配置**: 基于 YAML 的配置系统，确保实验可重现
- 🔧 **可扩展架构**: 易于添加自定义模型和管道
- 📈 **全面的训练功能**: 早停、学习率调度、数据增强等

## 安装

```bash
pip install visflow
```

或从源码安装：

```bash
git clone https://github.com/6ixGODD/visflow.git
cd visflow
pip install -e .
```

## 快速开始

### 1. 训练模型

创建配置文件（例如 `config.yml`）：

```yaml
model:
  architecture: resnet18
  pretrained: true
  num_classes: 2

training:
  epochs: 10
  batch_size: 32
  learning_rate: 0.001
  optimizer: adam

data:
  train_dir: ./data/train
  val_dir: ./data/val
  test_dir: ./data/test
```

训练模型：

```bash
# 使用命令行
visflow train --config config.yml

# 或使用 Python 模块
python -m visflow train --config config.yml
```

### 2. 生成 Grad-CAM 可视化

```bash
visflow gradcam --ckpt-path model.pth --image-path image.jpg --output-dir ./output
```

### 3. 编程方式使用

```python
from visflow.resources.configs import TrainConfig
from visflow.pipelines.train import TrainPipeline

# 加载配置并开始训练
config = TrainConfig.from_yaml('config.yml')
pipeline = TrainPipeline(config)
pipeline()
```

## 支持的模型

VisFlow 支持广泛的预训练模型：

- **ResNet**: resnet18, resnet34, resnet50, resnet101, resnet152
- **VGG**: vgg11, vgg13, vgg16, vgg19（带/不带批量归一化）
- **EfficientNet**: efficientnet_b0 到 efficientnet_b7
- **Vision Transformer**: vit_b_16, vit_b_32, vit_l_16
- **Swin Transformer**: swin_t, swin_s, swin_b
- **ConvNeXt**: convnext_tiny, convnext_small, convnext_base
- **MobileNet**: mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
- **DenseNet**: densenet121, densenet169, densenet201
- 还有更多...

## 自定义模型

使用自定义模型扩展 VisFlow：

```python
from visflow.resources.models import BaseClassifier, register_model
import torch.nn as nn

@register_model('my_custom_model')
class MyCustomModel(BaseClassifier):
    def __init__(self, num_classes: int):
        super().__init__(num_classes)
        # 定义模型架构
        
    def forward(self, x):
        # 实现前向传播
        pass
        
    def last_conv(self):
        # Grad-CAM 支持所需
        pass
```

## 命令行工具

### 训练
```bash
visflow train [选项]

选项:
  -c, --config PATH   配置文件路径
  -v, --verbose       启用详细输出
```

### Grad-CAM 可视化
```bash
visflow gradcam [选项]

选项:
  -k, --ckpt-path PATH     模型检查点路径
  -i, --image-path PATH    输入图像路径
  -o, --output-dir PATH    输出目录
  -l, --target-layer TEXT  目标层名称
  -t, --target-class TEXT  目标类别
  -a, --alpha FLOAT        叠加透明度 (0-1)
  -c, --colormap TEXT      颜色映射 (jet/turbo/viridis/inferno/plasma)
  --heatmap-only           仅保存热力图
  --eigen-smooth           应用特征值平滑
  --aug-smooth             应用增强平滑
  -d, --device TEXT        设备 (cpu/cuda)
  -v, --verbose            启用详细输出
```

## 配置

VisFlow 使用 YAML 配置文件进行训练。完整示例：

```yaml
logging:
  backend: native
  loglevel: info

seed: 42

model:
  architecture: resnet18
  pretrained: true
  num_classes: 2

training:
  device: cuda
  batch_size: 32
  epochs: 10
  learning_rate: 0.001
  optimizer: adam
  lr_scheduler: step
  early_stopping: true
  early_stopping_patience: 5

data:
  train_dir: ./data/train
  val_dir: ./data/val
  test_dir: ./data/test
  num_workers: 4

augmentation:
  horizontal_flip:
    enabled: true
    p: 0.5
  color_jitter:
    enabled: true
    brightness: 0.2
    contrast: 0.2

output:
  output_dir: ./output
  experiment_name: my-experiment
```
详见[配置示例](.config.example.yml)。

## 许可证

MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。