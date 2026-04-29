🛰️ Aerospace Computer Vision Pipeline

<img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT" />
<img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python 3.8+" />
<img src="https://img.shields.io/badge/MLflow-Tracking-orange" alt="MLflow" />
<img src="https://img.shields.io/badge/FastAPI-0.95.0-green" alt="FastAPI" />

Production-ready Deep Learning pipeline for aerospace imagery classification, built with MLOps best practices.

Production-ready Deep Learning pipeline for aerospace imagery classification, built with MLOps best practices.
📌 Overview

This project implements a computer vision system for classifying aerospace imagery (e.g., satellite, drone, or aerial photos) into 5 predefined classes using:

    EfficientNetV2 (pretrained on ImageNet) as the backbone.
    A custom classification head (LayerNorm → Dropout → Linear).
    MLflow for experiment tracking and model registry.
    FastAPI for serving predictions via a REST API.
    Docker and GitHub Actions for CI/CD.

🏗️ Architecture

graph TD
    A[Input Image (512x512)] --> B[EfficientNetV2 (Pretrained)]
    B --> C[Custom Head: LayerNorm → Dropout → Linear]
    C --> D[5-Class Prediction]

Key Features:

✅ Modular Design: Separation of concerns (data, model, API, MLOps).
✅ Reproducibility: Versioned datasets, models, and experiments.
✅ Scalability: Containerized deployment with Docker.
✅ Monitoring: MLflow for tracking metrics, parameters, and artifacts.
🚀 Quick Start
1️⃣ Prerequisites

    Python 3.8+
    Docker (optional, for containerized deployment)
    Git

2️⃣ Installation

git clone https://github.com/JoGrand/aerospace-pipeline.git
cd aerospace-pipeline
pip install -r requirements.txt

3️⃣ Run the Pipeline
Command 	Description 	URL
make train 	Train the model 	-
make api 	Launch the FastAPI server 	http://localhost:8000
make test 	Run unit/integration tests 	-
make mlflow 	Start MLflow tracking dashboard 	http://localhost:5000
📂 Project Structure

aerospace-pipeline/
├── api/               # FastAPI endpoints and models
├── configs/           # Configuration files (hyperparameters, paths)
├── mlops/             # CI/CD and MLOps scripts
├── notebooks/         # Jupyter notebooks for EDA/prototyping
├── src/               # Core ML code (data, model, training)
│   ├── data/          # Data loading and preprocessing
│   ├── models/        # Model architecture and training logic
│   └── utils/         # Helper functions
├── tests/             # Unit and integration tests
├── .github/workflows/ # GitHub Actions CI/CD
├── Dockerfile         # Containerization
├── Makefile           # Shortcuts for common tasks
└── README.md          # Project documentation

🔧 Configuration

Edit configs/train_config.yaml to customize:

    Model hyperparameters (learning rate, batch size, epochs).
    Data paths (train/val/test splits).
    MLflow tracking (experiment name, artifact storage).

🤖 API Usage

After running make api, interact with the model via:

curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/image.jpg"

Or visit the Swagger UI at http://localhost:8000/docs.
📊 Experiment Tracking

Launch MLflow to monitor:

    Training metrics (loss, accuracy).
    Hyperparameters.
    Model artifacts.

make mlflow

Access the dashboard at http://localhost:5000.
🧪 Testing

Run tests with:

make test

Includes:

    Unit tests for data preprocessing.
    Integration tests for the API.
    Model validation tests.

🐳 Docker Deployment

Build and run the container:

docker build -t aerospace-pipeline .
docker run -p 8000:8000 aerospace-pipeline

🤝 Contributing

Contributions are welcome! Follow these steps:

    Fork the repository.
    Create a feature branch (git checkout -b feature/your-feature).
    Commit your changes (git commit -m "Add your feature").
    Push to the branch (git push origin feature/your-feature).
    Open a Pull Request.

📜 License

This project is licensed under the MIT License - see LICENSE for details.
📬 Contact

For questions or feedback:

    GitHub Issues: https://github.com/JoGrand/aerospace-pipeline/issues
    Email: [joseph.grandchamp@gmail.com]

🌟 Acknowledgements

    EfficientNetV2 (Tan & Le, 2021)
    MLflow
    FastAPI

Star this repo if you find it useful! ⭐