import unittest
import mlflow
from mlflow.tracking import MlflowClient
import os

# Load DagsHub token from environment variables for secure access
# The DagsHub token is required for authentication when interacting with the DagsHub MLflow server
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    # Raise an error if the DAGSHUB_TOKEN is not set in the environment variables
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

# Set the environment variables for MLflow using the DagsHub token
# These environment variables are used for authenticating with MLflow
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# Set the tracking URI for MLflow to point to your DagsHub MLflow instance
# The URI connects MLflow to the repository where your models are tracked
dagshub_url = "https://dagshub.com"
repo_owner = "bhattpriyang"
repo_name = "ci_cd_test"
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

# Specify the name of the model that we want to load and test
model_name = "Best Model"  # This is the model name registered in MLflow

# Unit test class to test the loading of models from the 'Staging' stage in MLflow
class TestModelLoading(unittest.TestCase):
    """Unit test class to verify MLflow model loading from the Staging stage."""

    def test_model_in_staging(self):
        """Test if the model exists in the 'Staging' stage."""
        
        # Initialize the MLflow client to interact with the MLflow server
        client = MlflowClient()

        # Retrieve the latest versions of the model in the 'Staging' stage
        versions = client.get_latest_versions(model_name, stages=["Staging"])

        # Assert that at least one version of the model exists in the 'Staging' stage
        # If no versions are found, it will raise an error
        self.assertGreater(len(versions), 0, "No model found in the 'Staging' stage.")

    def test_model_loading(self):
        """Test if the model can be loaded properly from the Staging stage."""
        
        # Initialize the MLflow client again to interact with the server
        client = MlflowClient()

        # Retrieve the latest versions of the model in the 'Staging' stage
        versions = client.get_latest_versions(model_name, stages=["Staging"])

        # If no versions are found, fail the test and skip the model loading part
        if not versions:
            self.fail("No model found in the 'Staging' stage, skipping model loading test.")

        # Get the version details of the latest model in the 'Staging' stage
        latest_version = versions[0].version
        run_id = versions[0].run_id  # Retrieve the run ID of the model version

        # Construct the string needed to load the model using its run ID
        logged_model = f"runs:/{run_id}/{model_name}"

        try:
            # Try to load the model from the specified path
            loaded_model = mlflow.pyfunc.load_model(logged_model)
        except Exception as e:
            # If loading the model fails, fail the test and output the error message
            self.fail(f"Failed to load the model: {e}")

        # Assert that the model is not None, meaning it was loaded successfully
        self.assertIsNotNone(loaded_model, "The loaded model is None.")
        print(f"Model successfully loaded from {logged_model}.")

# This ensures the tests run when executing the script directly
if __name__ == "__main__":
    unittest.main()
