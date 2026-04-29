# Makefile

.PHONY: install train evaluate api test docker

install:
    pip install -r requirements.txt

train:
    python -m src.train

evaluate:
    python -m src.evaluate

api:
    uvicorn api.main:app --reload --port 8000

mlflow:
    mlflow ui --port 5000

test:
    pytest tests/ -v

docker-build:
    docker build -t aerospace-cv-api .

docker-run:
    docker run -p 8000:8000 aerospace-cv-api

clean:
    find . -type f -name "*.pyc" -delete
    find . -type d -name "__pycache__" -delete
