import unittest
import mlflow
from mlflow.tracking import MlflowClient
import os

# Load DagsHub token from environment variables for secure access
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

# Set up environment variables for MLflow
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# Set the tracking URI to your DagsHub MLflow instance
mlflow.set_tracking_uri("https://dagshub.com/bhattpriyang/mlops_project.mlflow")

# Specify the model name
model_name = "Best Model"  # Replace with your registered model name

class TestModelLoading(unittest.TestCase):
    """Unit test class to verify MLflow model loading from the Staging stage."""

    def test_model_in_staging(self):
        """Test if the model exists in the 'Staging' stage."""
        client = MlflowClient()

        # Get the latest version of the model in the Staging stage
        versions = client.get_latest_versions(model_name, stages=["Staging"])

        # Assert that there is at least one version in the Staging stage
        self.assertGreater(len(versions), 0, "No model found in the 'Staging' stage.")

    def test_model_loading(self):
        """Test if the model can be loaded properly from the Staging stage."""
        client = MlflowClient()

        # Get the latest version of the model in the Staging stage
        versions = client.get_latest_versions(model_name, stages=["Staging"])

        if not versions:
            self.fail("No model found in the 'Staging' stage, skipping model loading test.")

        # Get details of the latest version
        latest_version = versions[0].version
        run_id = versions[0].run_id  # Fetch the run ID from the latest version

        # Construct the logged_model string
        logged_model = f"runs:/{run_id}/{model_name}"

        try:
            # Try loading the model
            loaded_model = mlflow.pyfunc.load_model(logged_model)
        except Exception as e:
            self.fail(f"Failed to load the model: {e}")

        # Assert that the loaded model is not None
        self.assertIsNotNone(loaded_model, "The loaded model is None.")
        print(f"Model successfully loaded from {logged_model}.")

if __name__ == "__main__":
    unittest.main()
