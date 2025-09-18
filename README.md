<div align="center">
<h1>VisFlow</h1>

[![PyPI version](https://badge.fury.io/py/visflow.svg)](https://badge.fury.io/py/visflow)
[![Python Version](https://img.shields.io/pypi/pyversions/visflow)](https://pypi.org/project/visflow/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*A comprehensive computer vision framework for training, evaluation, and visualization*

English | [中文](README.zh-CN.md)

</div>

---
### 🚀 Features

- **🎯 Easy Training**: Simple YAML configuration for model training
- **🔥 GradCAM Visualization**: Built-in support for model interpretability  
- **🏗️ Multiple Architectures**: Support for 50+ pre-trained models from torchvision
- **🎨 Extensible**: Easy to add custom models with the registration system
- **⚡ CLI & Programmatic**: Use via command line or Python API
- **📊 Rich Logging**: Beautiful terminal output with progress tracking

### 📦 Installation

```bash
pip install visflow
```

### 🎯 Quick Start

#### Training a Model

1. **Create a configuration file** (`config.yml`):

```yaml
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

data:
  train_dir: ./data/train
  val_dir: ./data/val
  test_dir: ./data/test

output:
  output_dir: ./output
  experiment_name: my-experiment
```

2. **Train via CLI**:
```bash
visflow train --config config.yml
```

3. **Or train via Python API**:
```python
from visflow.resources.configs import TrainConfig
from visflow.pipelines.train import TrainPipeline

pipeline = TrainPipeline(TrainConfig.from_yaml('config.yml'))
pipeline()
```

#### GradCAM Visualization

```bash
visflow gradcam \
    --ckpt-path model.pth \
    --image-path image.jpg \
    --output-dir ./output \
    --target-layer layer4 \
    --colormap jet
```

### 🏗️ Supported Architectures

Visflow supports 50+ architectures from torchvision:

- **ResNet**: resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, etc.
- **EfficientNet**: efficientnet_b0 through efficientnet_b7, efficientnet_v2_s/m/l
- **Vision Transformers**: vit_b_16, vit_b_32, vit_l_16, swin_t, swin_s, swin_b
- **MobileNet**: mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
- **DenseNet**: densenet121, densenet169, densenet201, densenet161
- **And many more**: VGG, ConvNeXt, RegNet, MaxViT, etc.

### 🎨 Custom Models

Easily add your own models:

```python
from visflow.resources.models import BaseClassifier, register_model

@register_model('my_custom_model')
class MyCustomModel(BaseClassifier):
    def __init__(self, num_classes: int):
        super().__init__(num_classes=num_classes)
        # Your model implementation
        
    def forward(self, x):
        # Forward pass implementation
        pass
        
    def gradcam_layer(self):
        # Return the last convolutional layer for GradCAM
        return self.conv_layer
```

### 📖 CLI Reference

#### Training Command
```bash
visflow train [OPTIONS]

Options:
  -c, --config PATH     Path to training configuration file [required]
  -v, --verbose         Enable verbose logging
  --help                Show this message and exit
```

#### GradCAM Command
```bash
visflow gradcam [OPTIONS]

Options:
  -k, --ckpt-path PATH      Path to model checkpoint [required]
  -i, --image-path PATH     Path to input image [required]
  -o, --output-dir PATH     Output directory [default: ./output]
  -l, --target-layer TEXT   Target layer name
  -t, --target-class TEXT   Target class (index or name)
  -c, --colormap TEXT       Colormap [default: jet]
  --heatmap-only            Save only heatmap
  --eigen-smooth            Apply eigen smoothing
  --aug-smooth              Apply augmented smoothing
  -d, --device TEXT         Device (cpu/cuda)
  -v, --verbose             Enable verbose logging
```

### 📋 Configuration Reference

<details>
<summary>Complete configuration example</summary>

```yaml
logging:
  backend: native  # Options: native, loguru
  loglevel: info   # Options: debug, info, warning, error, critical

seed: 42

model:
  architecture: resnet18
  pretrained: true
  num_classes: 2
  weights_path: ~  # Optional custom weights

training:
  device: cuda
  shuffle: true
  batch_size: 32
  weighted_sampling: false
  drop_last: false
  epochs: 10
  learning_rate: 0.001
  momentum: 0.9
  weight_decay: 0.0001
  optimizer: adam  # Options: sgd, adam, adamw
  lr_scheduler: ~  # Options: step, cosine, plateau
  early_stopping: true
  early_stopping_patience: 5
  label_smoothing: 0.0

testing:
  batch_size: 32

data:
  train_dir: ./data/train
  val_dir: ./data/val
  test_dir: ./data/test
  num_workers: 4
  pin_memory: false

resize:
  size: 224
  interpolation: bicubic
  antialias: true

normalization:
  enabled: true
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]

augmentation:
  horizontal_flip:
    enabled: true
    p: 0.5
  rotation:
    enabled: false
    degrees: 30
  color_jitter:
    enabled: false
    brightness: 0.2
    contrast: 0.2
    saturation: 0.2
    hue: 0.1
  # ... more augmentation options

output:
  output_dir: ./output
  experiment_name: vision-research
  checkpoint_frequency: 10
```

</details>

See the [example config file](.config.example.yml) for more details.

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
