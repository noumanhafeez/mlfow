from mlflow.tracking import MlflowClient

client = MlflowClient()
experiment_id = client.get_experiment_by_name("Default").experiment_id
runs = client.search_runs(experiment_id, order_by=["metrics.accuracy DESC"], max_results=1)
best_run = runs[0]
print("Best run:", best_run.data.metrics, best_run.data.params)