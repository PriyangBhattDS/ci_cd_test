import mlflow
import dagshub
dagshub.init(repo_owner='bhattpriyang', repo_name='ci_cd_test', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)