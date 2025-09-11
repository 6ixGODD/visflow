<div align="center">
<h1>VisFlow</h1>

[![PyPI version](https://badge.fury.io/py/visflow.svg)](https://badge.fury.io/py/visflow)
[![Python Version](https://img.shields.io/pypi/pyversions/visflow)](https://pypi.org/project/visflow/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/visflow)](https://pepy.tech/project/visflow)

*A comprehensive computer vision framework for training, evaluation, and visualization*

[English](README.md) | 中文

</div>

### 🚀 特性

- **🎯 简易训练**: 通过简单的 YAML 配置进行模型训练
- **🔥 GradCAM 可视化**: 内置模型可解释性支持
- **🏗️ 多种架构**: 支持 torchvision 中的 50+ 预训练模型
- **🎨 可扩展**: 通过注册系统轻松添加自定义模型
- **⚡ CLI 和编程接口**: 支持命令行和 Python API 两种使用方式
- **📊 丰富的日志**: 美观的终端输出和进度跟踪

### 📦 安装

```bash
pip install visflow
```

### 🎯 快速开始

#### 训练模型

1. **创建配置文件** (`config.yml`):

```yaml
model:
  architecture: resnet18    # 模型架构
  pretrained: true         # 使用预训练权重
  num_classes: 2          # 分类数量

training:
  device: cuda            # 设备选择
  batch_size: 32         # 批次大小
  epochs: 10             # 训练轮数
  learning_rate: 0.001   # 学习率
  optimizer: adam        # 优化器

data:
  train_dir: ./data/train  # 训练数据目录
  val_dir: ./data/val     # 验证数据目录
  test_dir: ./data/test   # 测试数据目录

output:
  output_dir: ./output           # 输出目录
  experiment_name: my-experiment # 实验名称
```

2. **通过 CLI 训练**:
```bash
visflow train --config config.yml
```

3. **或通过 Python API 训练**:
```python
from visflow.resources.configs import TrainConfig
from visflow.pipelines.train import TrainPipeline

pipeline = TrainPipeline(TrainConfig.from_yaml('config.yml'))
pipeline()
```

#### GradCAM 可视化

```bash
visflow gradcam \
    --ckpt-path model.pth \
    --image-path image.jpg \
    --output-dir ./output \
    --target-layer layer4 \
    --colormap jet
```

### 🏗️ 支持的架构

Visflow 支持 torchvision 中的 50+ 架构：

- **ResNet 系列**: resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d 等
- **EfficientNet 系列**: efficientnet_b0 到 efficientnet_b7, efficientnet_v2_s/m/l
- **Vision Transformers**: vit_b_16, vit_b_32, vit_l_16, swin_t, swin_s, swin_b
- **MobileNet 系列**: mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
- **DenseNet 系列**: densenet121, densenet169, densenet201, densenet161
- **还有更多**: VGG, ConvNeXt, RegNet, MaxViT 等

### 🎨 自定义模型

轻松添加您自己的模型：

```python
from visflow.resources.models import BaseClassifier, register_model

@register_model('my_custom_model')
class MyCustomModel(BaseClassifier):
    def __init__(self, num_classes: int):
        super().__init__(num_classes=num_classes)
        # 您的模型实现
        
    def forward(self, x):
        # 前向传播实现
        pass
        
    def last_conv(self):
        # 返回最后一个卷积层用于 GradCAM
        return self.conv_layer
```

### 📖 CLI 参考

#### 训练命令
```bash
visflow train [OPTIONS]

选项:
  -c, --config PATH     训练配置文件路径 [必需]
  -v, --verbose         启用详细日志
  --help                显示帮助信息并退出
```

#### GradCAM 命令
```bash
visflow gradcam [OPTIONS]

选项:
  -k, --ckpt-path PATH      模型检查点路径 [必需]
  -i, --image-path PATH     输入图像路径 [必需]
  -o, --output-dir PATH     输出目录 [默认: ./output]
  -l, --target-layer TEXT   目标层名称
  -t, --target-class TEXT   目标类别（索引或名称）
  -c, --colormap TEXT       颜色映射 [默认: jet]
  --heatmap-only            仅保存热力图
  --eigen-smooth            应用特征值平滑
  --aug-smooth              应用增强平滑
  -d, --device TEXT         设备 (cpu/cuda)
  -v, --verbose             启用详细日志
```

### 📄 许可证

本项目采用 MIT 许可证 - 详情请见 [LICENSE](LICENSE) 文件。
