import os
import unittest
import mlflow
from mlflow.tracking import MlflowClient

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up DagsHub credentials for MLflow tracking
        dagshub_token = os.getenv("DAGSHUB_TOKEN")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        # Set up MLflow tracking URI for DagsHub
        dagshub_url = "https://dagshub.com"
        repo_owner = "bhattpriyang"
        repo_name = "ci_cd_test"
        mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

        # Model name in MLflow
        cls.new_model_name = "Best Model"
        # Get the latest model version
        cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)

        if not cls.new_model_version:
            raise ValueError(f"No model version found for {cls.new_model_name}")

        # Construct the model URI
        cls.new_model_uri = f"models:/{cls.new_model_name}/{cls.new_model_version}"

        try:
            # Load the model from MLflow
            cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {cls.new_model_uri}. Error: {e}")

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        # Get the latest version of the model from MLflow registry
        client = MlflowClient()
        latest_versions = client.get_latest_versions(model_name, stages=[stage])
        return latest_versions[0].version if latest_versions else None

    def test_model_loaded_properly(self):
        # Ensure the model is loaded successfully
        self.assertIsNotNone(self.new_model, "Model is not loaded properly.")

if __name__ == "__main__":
    unittest.main()
