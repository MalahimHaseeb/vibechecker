import mlflow
from mlflow import MlflowClient

EXPERIMENT_NAME = "vibechecker"
MODEL_NAME = "vibechecker"

client = MlflowClient()
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="metrics.test_macro_f1 > 0",
    order_by=["metrics.test_macro_f1 DESC"],
    max_results=1,
)

if not runs:
    raise SystemExit("no run with test_macro_f1 found")

best = runs[0]
print("best run:", best.info.run_name, best.info.run_id)
print("test macro_f1:", round(best.data.metrics["test_macro_f1"], 4))

outputs = getattr(best, "outputs", None)
if outputs is not None and outputs.model_outputs:
    model_uri = "models:/" + outputs.model_outputs[0].model_id
else:
    model_uri = "runs:/" + best.info.run_id + "/model"

version = mlflow.register_model(model_uri, MODEL_NAME)
client.set_registered_model_alias(MODEL_NAME, "champion", version.version)
print("registered version", version.version, "as champion")