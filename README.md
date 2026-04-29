# 🛰️ Aerospace Computer Vision Pipeline

> Deep Learning system for aerospace imagery classification
> Built for production with MLOps best practices

## 🏗️ Architecture
Input Image (512x512)
      ↓
EfficientNetV2 (ImageNet pretrained)
      ↓
Custom Head (LayerNorm → Dropout → Linear)
      ↓
5 Classes


## 🚀 Quick Start

```bash
# Install
pip install -r requirements.txt

# Train
make train

# API
make api
# → http://localhost:8000/docs

# Tests
make test

# MLflow dashboard
make mlflow
# → http://localhost:5000
