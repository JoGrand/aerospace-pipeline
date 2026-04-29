# MLflow tracking configuration
import mlflow

def setup_mlflow():
    mlflow.set_tracking_uri("http://localhost:5000")