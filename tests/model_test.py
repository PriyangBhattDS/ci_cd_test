import os
import unittest
import mlflow

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up DagsHub credentials for MLflow tracking
        dagshub_token = os.getenv("DAGSHUB_TOKEN")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

       
        dagshub_url = "https://dagshub.com"
        repo_owner = "bhattpriyang"
        repo_name='ci_cd_test'
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        # Load the new model
        cls.new_model_name = "Best Model"
        cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
        cls.new_model_uri = f'models:/{cls.new_model_name}/{cls.new_model_version}'
        cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=[stage])
        return latest_version[0].version if latest_version else None

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

if __name__ == "__main__":
    unittest.main()