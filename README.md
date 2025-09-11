# VisFlow

English | [中文](README.zh-CN.md)

## Overview

VisFlow is a comprehensive deep learning framework designed for computer vision tasks, providing streamlined workflows for model training, evaluation, and visualization. Built with PyTorch, it offers a flexible and extensible architecture for researchers and practitioners.

### Key Features

- 🚀 **Easy-to-use CLI Interface**: Train models and generate visualizations with simple commands
- 🎯 **Extensive Model Support**: Built-in support for popular architectures (ResNet, VGG, EfficientNet, ViT, etc.)
- 📊 **Advanced Visualization**: Integrated Grad-CAM support for model interpretability
- ⚙️ **Flexible Configuration**: YAML-based configuration system for reproducible experiments
- 🔧 **Extensible Architecture**: Easy to add custom models and pipelines
- 📈 **Comprehensive Training Features**: Early stopping, learning rate scheduling, data augmentation

## Installation

```bash
pip install visflow
```

Or install from source:

```bash
git clone https://github.com/6ixGODD/visflow.git
cd visflow
pip install -e .
```

## Quick Start

### 1. Training a Model

Create a configuration file (e.g., `config.yml`):

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

Train your model:

```bash
# Using CLI
visflow train --config config.yml

# Or using Python module
python -m visflow train --config config.yml
```

### 2. Generating Grad-CAM Visualizations

```bash
visflow gradcam --ckpt-path model.pth --image-path image.jpg --output-dir ./output
```

### 3. Programmatic Usage

```python
from visflow.resources.configs import TrainConfig
from visflow.pipelines.train import TrainPipeline

# Load configuration and start training
config = TrainConfig.from_yaml('config.yml')
pipeline = TrainPipeline(config)
pipeline()
```

## Supported Models

VisFlow supports a wide range of pre-trained models:

- **ResNet**: resnet18, resnet34, resnet50, resnet101, resnet152
- **VGG**: vgg11, vgg13, vgg16, vgg19 (with/without batch norm)
- **EfficientNet**: efficientnet_b0 through efficientnet_b7
- **Vision Transformer**: vit_b_16, vit_b_32, vit_l_16
- **Swin Transformer**: swin_t, swin_s, swin_b
- **ConvNeXt**: convnext_tiny, convnext_small, convnext_base
- **MobileNet**: mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
- **DenseNet**: densenet121, densenet169, densenet201
- And many more...

## Custom Models

Extend VisFlow with your own models:

```python
from visflow.resources.models import BaseClassifier, register_model


@register_model('my_custom_model')
class MyCustomModel(BaseClassifier):
    def __init__(self, num_classes: int):
        super().__init__(num_classes)
        # Define your model architecture

    def forward(self, x):
        # Implement forward pass
        pass

    def last_conv(self):
        # Required for Grad-CAM support
        pass
```

## CLI Commands

### Training
```bash
visflow train [OPTIONS]

Options:
  -c, --config PATH   Configuration file path
  -v, --verbose       Enable verbose output
```

### Grad-CAM Visualization
```bash
visflow gradcam [OPTIONS]

Options:
  -k, --ckpt-path PATH     Model checkpoint path
  -i, --image-path PATH    Input image path
  -o, --output-dir PATH    Output directory
  -l, --target-layer TEXT  Target layer name
  -t, --target-class TEXT  Target class
  -a, --alpha FLOAT        Overlay transparency (0-1)
  -c, --colormap TEXT      Colormap (jet/turbo/viridis/inferno/plasma)
  --heatmap-only           Save heatmap only
  --eigen-smooth           Apply eigen smoothing
  --aug-smooth             Apply augmented smoothing
  -d, --device TEXT        Device (cpu/cuda)
  -v, --verbose            Enable verbose output
```

## Configuration

VisFlow uses YAML configuration files for training. Here's a complete example:

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
See the [Configuration Example](.config.example.yml) for more details.

## License

MIT License - see the [LICENSE](LICENSE) file for details.
