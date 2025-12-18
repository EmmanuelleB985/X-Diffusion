# X-Diffusion: Generating Detailed 3D MRI Volumes From a Single Image Using Cross-Sectional Diffusion Models

[![ArXiv](https://img.shields.io/badge/ArXiv-Paper-%3CCOLOR%3E.svg)](https://arxiv.org/abs/2404.19604)
[![ProjectPage](https://img.shields.io/badge/Project_Page-blue)](https://emmanuelleb985.github.io/XDiffusion/)
[![CI/CD](https://github.com/EmmanuelleB985/X-Diffusion/actions/workflows/ci.yml/badge.svg)](https://github.com/EmmanuelleB985/X-Diffusion/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/EmmanuelleB985/X-Diffusion/branch/main/graph/badge.svg)](https://codecov.io/gh/EmmanuelleB985/X-Diffusion)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Checked](https://img.shields.io/badge/type_checked-mypy-blue.svg)](http://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## Overview

X-Diffusion is a state-of-the-art deep learning model for generating detailed 3D MRI volumes from single 2D images using cross-sectional diffusion models. This project implements the methods described in our [paper](https://arxiv.org/abs/2404.19604), providing a robust framework for medical image synthesis.

## Features

- Generate complete 3D MRI volumes from single 2D slices
- Optimized inference with mixed precision support
- Support for BRATS and UK Biobank datasets
- Modular architecture for easy customization
- Comprehensive evaluation metrics (PSNR, SSIM)
- Pre-trained models available
- Docker support for easy deployment
- Documentation 

## Installation

### Requirements

- Python 3.8 or higher
- CUDA 11.0+ (for GPU support)
- 32GB RAM recommended
- 30GB VRAM for training (GPU)

### Setup

1. **Clone the repository:**

```bash
git clone https://github.com/EmmanuelleB985/X-Diffusion.git
cd X-Diffusion
```

2. **Create a virtual environment:**

```bash
conda create -n XDiffusion python=3.9
conda activate XDiffusion
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Install development dependencies (optional):**

```bash
pip install -r requirements-dev.txt
```

5. **Install pre-commit hooks (for contributors):**

```bash
pre-commit install
```

6. **Download pre-trained models:**

```bash
# Zero-123 checkpoint
wget https://cv.cs.columbia.edu/zero123/assets/300000.ckpt -P Zero123/

# X-Diffusion checkpoint (if available)
tar -xzvf ckpt.tar.gz
```

## Quick Start

### Training

```bash
python scripts/main.py \
    -t \
    --base configs/sd-brats-finetune-c_concat-256.yaml \
    --gpus 0 \
    --scale_lr False \
    --num_nodes 1 \
    --seed 42 \
    --check_val_every_n_epoch 10 \
    --finetune_from Zero123/300000.ckpt
```

### Inference

```bash
# Single image inference
python scripts/inference.py \
    --model_path checkpoints/best_model.ckpt \
    --input_image path/to/image.png \
    --output_dir results/

# Batch inference
python scripts/inference.py \
    --model_path checkpoints/best_model.ckpt \
    --input_dir data/test_images/ \
    --output_dir results/ \
    --batch_size 4
```

Run all checks:

```bash
# Format code
black scripts/ tests/

# Sort imports
isort scripts/ tests/

# Run linting
flake8 scripts/ tests/

# Type checking
mypy scripts/

# Run pylint
pylint scripts/

# Check docstring coverage
interrogate scripts/ -v --fail-under=90
```

Or use pre-commit to run all checks automatically:

```bash
pre-commit run --all-files
```

### Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=scripts --cov-report=html

# Run specific test categories
pytest -m unit           # Unit tests only
pytest -m integration    # Integration tests only
pytest -m "not slow"     # Skip slow tests
```

### Documentation

Build the documentation:

```bash
cd docs
make html
# Open docs/_build/html/index.html in your browser
```

## Dataset

### BRATS2023

Download the BRATS2023 dataset:

1. Create an account on [Synapse](https://www.synapse.org/#!Synapse:syn27046444/wiki/616571)
2. Download the dataset
3. Extract to `data/BRATS2023/`

### UK Biobank

Access UK Biobank data:

1. Register on the [UK Biobank platform](https://www.ukbiobank.ac.uk/)
2. Follow instructions from [UKBiobankDXAMRIPreprocessing](https://github.com/rwindsor1/UKBiobankDXAMRIPreprocessing)

## Model Architecture

X-Diffusion uses a cross-sectional diffusion approach based on:

- **Backbone**: Modified Stable Diffusion architecture
- **Conditioning**: Cross-sectional image embeddings
- **Denoising**: Progressive refinement through diffusion steps
- **3D Reconstruction**: Volumetric assembly from 2D slices

## Results

Our model achieves state-of-the-art performance on:

| Dataset | PSNR ↑ | SSIM ↑ | FID ↓ |
|---------|--------|--------|-------|
| BRATS   | 28.3   | 0.912  | 12.4  |
| UK Biobank | 29.1 | 0.925 | 10.8  |

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@misc{bourigault2025xdiffusion,
    title={X-Diffusion: Generating Detailed 3D MRI Volumes From a Single Image Using Cross-Sectional Diffusion Models}, 
    author={Emmanuelle Bourigault and Abdullah Hamdi and Amir Jamaludin},
    year={2025},
    eprint={2404.19604},
    archivePrefix={arXiv},
    primaryClass={eess.IV}
}
```

## Acknowledgements

This project builds upon:

- [Zero-123](https://github.com/cvlab-columbia/zero123)
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [PyTorch Lightning](https://github.com/Lightning-AI/lightning)

We thank the authors for making their code publicly available.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! 

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
