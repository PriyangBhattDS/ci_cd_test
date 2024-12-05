import json
import os
from mlflow.tracking import MlflowClient
import mlflow

# Load DagsHub token from environment variables
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# DagsHub repository details
dagshub_url = "https://dagshub.com"
repo_owner = "bhattpriyang"
repo_name = "ci_cd_test"
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

# Load the run ID and model name from the saved JSON file
reports_path = "reports/run_info.json"
if not os.path.exists(reports_path):
    raise FileNotFoundError(f"{reports_path} not found. Ensure the JSON file exists.")

with open(reports_path, 'r') as file:
    run_info = json.load(file)

try:
    run_id = run_info['run_id']
    model_name = run_info['model_name']
    model_version = run_info.get('model_version')  # Optional, ensure it's in the JSON
except KeyError as e:
    raise KeyError(f"Missing required key in run_info.json: {e}")

# Create an MLflow client
client = MlflowClient()

# Transition the existing model version to Production
new_stage = "Production"

try:
    if not model_version:
        # If version is not provided, find the latest version
        versions = client.get_latest_versions(name=model_name)
        if not versions:
            raise ValueError(f"No versions found for model: {model_name}")
        model_version = versions[0].version  # Use the latest version by default

    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage=new_stage,
        archive_existing_versions=True
    )
    print(f"Model {model_name} version {model_version} transitioned to {new_stage} stage.")
except Exception as e:
    raise RuntimeError(f"Failed to transition model to {new_stage} stage: {e}")
