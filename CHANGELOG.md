# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.2] - 2025-09-15

### Added

- GradCAM Batch Support: Added ability to input an image directory for GradCAM and process all images in batch
- Early Stopping in TrainPipeline: Support for early stopping during training

### Changed

- Various code optimizations and performance improvements

## [1.0.1] - 2025-09-11

### Added

- Enhanced documentation for better user experience

### Changed

- Improved plotting output quality with higher DPI (300) for better visualization

## [1.0.0] - 2025-09-11

### Added

- 🎯 **Easy Training**: Simple YAML configuration for model training
- 🔥 **GradCAM Visualization**: Built-in support for model interpretability with `visflow gradcam` command
- 🏗️ **Multiple Architectures**: Support for 50+ pre-trained models from torchvision
    - ResNet family (resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d)
    - EfficientNet family (efficientnet_b0 through efficientnet_b7, efficientnet_v2_s/m/l)
    - Vision Transformers (vit_b_16, vit_b_32, vit_l_16, swin_t, swin_s, swin_b)
    - MobileNet family (mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large)
    - DenseNet family (densenet121, densenet169, densenet201, densenet161)
    - And many more: VGG, ConvNeXt, RegNet, MaxViT, etc.
- 🎨 **Extensible Model System**: Easy custom model registration with `@register_model` decorator
- ⚡ **Dual Interface**: Both CLI and Python API support
- 📊 **Rich Logging**: Beautiful terminal output with progress tracking
- **CLI Commands**:
    - `visflow train`: Model training with configuration files
    - `visflow gradcam`: Generate GradCAM visualizations
- **Comprehensive Configuration**: YAML-based configuration system with support for:
    - Model architecture selection
    - Training parameters (optimizer, learning rate, batch size, etc.)
    - Data augmentation options
    - Logging and output settings
- **Data Augmentation**: Built-in support for various augmentation techniques
    - Horizontal flip
    - Rotation
    - Color jitter
    - Normalization with customizable mean and std
- **Advanced Training Features**:
    - Early stopping
    - Learning rate scheduling
    - Weighted sampling
    - Label smoothing
- **PyPI Package**: Available for installation via `pip install visflow`

### Documentation

- Comprehensive README with quick start guide
- CLI reference documentation
- Complete configuration examples
- Custom model implementation guide
- Installation and setup instructions

[unreleased]: https://github.com/6ixGODD/visflow/compare/v1.0.2...HEAD

[1.0.2]: https://github.com/6ixGODD/visflow/compare/v1.0.1...v1.0.2

[1.0.1]: https://github.com/6ixGODD/visflow/compare/v1.0.0...v1.0.1

[1.0.0]: https://github.com/6ixGODD/visflow/releases/tag/v1.0.0