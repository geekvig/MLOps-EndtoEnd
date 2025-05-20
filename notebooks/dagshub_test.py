import mlflow
import dagshub

mlflow.set_tracking_uri("https://dagshub.com/geekvig/MLOps-EndtoEnd.mlflow")

dagshub.init(repo_owner='geekvig', repo_name='MLOps-EndtoEnd', mlflow=True)

with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)