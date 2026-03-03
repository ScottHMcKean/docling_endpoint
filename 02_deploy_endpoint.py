# Databricks notebook source
# MAGIC %md
# MAGIC # Docling on Databricks -- Deploy Serving Endpoint
# MAGIC
# MAGIC Logs the Docling pyfunc to Unity Catalog and creates (or updates) a model
# MAGIC serving endpoint. Workload type and scale-to-zero are driven by `config.yaml`.

# COMMAND ----------

# MAGIC %pip install -r requirements.txt -q
# MAGIC %restart_python

# COMMAND ----------

import yaml, os
import mlflow

_nb = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
bundle_root = "/Workspace/" + "/".join(_nb.strip("/").split("/")[:-1])

with open(f"{bundle_root}/config.yaml") as f:
    cfg = yaml.safe_load(f)

with open(f"{bundle_root}/requirements.txt") as f:
    pip_reqs = [line.strip() for line in f if line.strip()]

catalog, schema = cfg["catalog"], cfg["schema"]
secrets = cfg["secrets"]
mlflow_cfg = cfg["mlflow"]
serving_cfg = cfg["serving"]

model_file = f"{bundle_root}/docling_endpoint.py"
output_volume_root = f"/Volumes/{catalog}/{schema}/{cfg['volumes']['exports']}"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log Model to Unity Catalog

# COMMAND ----------

mlflow.set_experiment(mlflow_cfg["experiment_path"])

registered_model_name = f"{catalog}.{schema}.{mlflow_cfg['registered_model_name']}"

with mlflow.start_run():
    logged = mlflow.pyfunc.log_model(
        artifact_path="docling_model",
        python_model=model_file,
        input_example=[f"/Volumes/{catalog}/{schema}/raw_docs/wind_turbine.pdf"],
        pip_requirements=pip_reqs,
        registered_model_name=registered_model_name,
    )

print(f"Registered: {registered_model_name} ({logged.model_uri})")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deploy Endpoint

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    ServingModelWorkloadType,
)
from databricks.sdk.errors import ResourceAlreadyExists

WORKLOAD_TYPE_MAP = {
    "CPU": ServingModelWorkloadType.CPU,
    "GPU_SMALL": ServingModelWorkloadType.GPU_SMALL,
    "GPU_MEDIUM": ServingModelWorkloadType.GPU_MEDIUM,
    "GPU_LARGE": ServingModelWorkloadType.GPU_LARGE,
}

w = WorkspaceClient()

versions = list(w.model_versions.list(full_name=registered_model_name))
latest_version = str(max(v.version for v in versions))

workload_type_str = serving_cfg.get("workload_type", "CPU")
workload_type = WORKLOAD_TYPE_MAP.get(workload_type_str)
if workload_type is None:
    raise ValueError(
        f"Unknown workload_type '{workload_type_str}'. "
        f"Valid: {list(WORKLOAD_TYPE_MAP.keys())}"
    )

env_vars = {
    "DATABRICKS_HOST": cfg["host"],
    "DATABRICKS_CLIENT_ID": f"{{{{secrets/{secrets['scope']}/{secrets['client_id_key']}}}}}",
    "DATABRICKS_CLIENT_SECRET": f"{{{{secrets/{secrets['scope']}/{secrets['client_secret_key']}}}}}",
    "OUTPUT_VOLUME_ROOT": output_volume_root,
}

entity = ServedEntityInput(
    name=serving_cfg["endpoint_name"],
    entity_name=registered_model_name,
    entity_version=latest_version,
    workload_size=serving_cfg["workload_size"],
    workload_type=workload_type,
    scale_to_zero_enabled=serving_cfg.get("scale_to_zero", True),
    environment_vars=env_vars,
)

endpoint_name = serving_cfg["endpoint_name"]
try:
    w.serving_endpoints.create(
        name=endpoint_name,
        config=EndpointCoreConfigInput(
            name=endpoint_name,
            served_entities=[entity],
        ),
    )
    print(f"Created endpoint: {endpoint_name}")
except ResourceAlreadyExists:
    w.serving_endpoints.update_config(
        name=endpoint_name,
        served_entities=[entity],
    )
    print(f"Updated endpoint: {endpoint_name}")
