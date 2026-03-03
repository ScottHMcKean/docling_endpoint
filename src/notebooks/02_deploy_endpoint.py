# Databricks notebook source
# MAGIC %md
# MAGIC # Docling on Databricks
# MAGIC ## 02 - Deploy Serving Endpoint
# MAGIC
# MAGIC Registers a new version of the Docling pyfunc model in Unity Catalog and creates
# MAGIC (or updates) a model serving endpoint. Workload type (CPU / GPU) and throughput
# MAGIC are driven by `config.yaml`.

# COMMAND ----------

# MAGIC %pip install mlflow docling onnxruntime databricks-sdk databricks-ai-bridge pymssql==2.3.8 hf_transfer --upgrade
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Configuration

# COMMAND ----------

import yaml, os
import mlflow

notebook_path = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .notebookPath()
    .get()
)
bundle_root = "/Workspace/" + "/".join(
    notebook_path.strip("/").split("/")[:-3]
)

with open(f"{bundle_root}/config.yaml") as f:
    config = yaml.safe_load(f)

catalog = config["catalog"]
schema = config["schema"]
host = config["host"]
secrets = config["secrets"]
mlflow_cfg = config["mlflow"]
serving_cfg = config["serving"]
pip_reqs = config["pip_requirements"]

model_file = f"{bundle_root}/src/docling_endpoint.py"
output_volume_root = f"/Volumes/{catalog}/{schema}/{config['volumes']['exports']}"

print(f"Model file: {model_file}")
print(f"Output volume root: {output_volume_root}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log Model to MLflow and Register in Unity Catalog

# COMMAND ----------

mlflow.set_experiment(mlflow_cfg["experiment_path"])

registered_model_name = f"{catalog}.{schema}.{mlflow_cfg['registered_model_name']}"

with mlflow.start_run() as run:
    logged_model = mlflow.pyfunc.log_model(
        artifact_path="docling_model",
        python_model=model_file,
        input_example=[f"/Volumes/{catalog}/{schema}/raw_docs/wind_turbine.pdf"],
        pip_requirements=pip_reqs,
        registered_model_name=registered_model_name,
    )

print(f"Model logged: {logged_model.model_uri}")
print(f"Registered as: {registered_model_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Resolve Latest Model Version

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    ServingModelWorkloadType,
)
from databricks.sdk.errors import ResourceAlreadyExists

w = WorkspaceClient()

versions = list(w.model_versions.list(full_name=registered_model_name))
latest_version = max(v.version for v in versions)
print(f"Latest model version: {latest_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configure and Deploy the Serving Endpoint
# MAGIC
# MAGIC Workload type is resolved from `config.yaml` (`CPU`, `GPU_SMALL`, `GPU_MEDIUM`,
# MAGIC or `GPU_LARGE`). Provisioned throughput and scale-to-zero behaviour are also
# MAGIC configurable.

# COMMAND ----------

WORKLOAD_TYPE_MAP = {
    "CPU": ServingModelWorkloadType.CPU,
    "GPU_SMALL": ServingModelWorkloadType.GPU_SMALL,
    "GPU_MEDIUM": ServingModelWorkloadType.GPU_MEDIUM,
    "GPU_LARGE": ServingModelWorkloadType.GPU_LARGE,
}

workload_type_str = serving_cfg.get("workload_type", "CPU")
workload_type = WORKLOAD_TYPE_MAP.get(workload_type_str)
if workload_type is None:
    raise ValueError(
        f"Unknown workload_type '{workload_type_str}'. "
        f"Valid options: {list(WORKLOAD_TYPE_MAP.keys())}"
    )

endpoint_name = serving_cfg["endpoint_name"]
scale_to_zero = serving_cfg.get("scale_to_zero", True)
max_throughput = serving_cfg.get("max_provisioned_throughput")

print(f"Endpoint: {endpoint_name}")
print(f"Workload: {workload_type_str} / {serving_cfg['workload_size']}")
print(f"Scale to zero: {scale_to_zero}")
if max_throughput:
    print(f"Max provisioned throughput: {max_throughput}")

# COMMAND ----------

env_vars = {
    "DATABRICKS_HOST": host,
    "DATABRICKS_CLIENT_ID": f"{{{{secrets/{secrets['scope']}/{secrets['client_id_key']}}}}}",
    "DATABRICKS_CLIENT_SECRET": f"{{{{secrets/{secrets['scope']}/{secrets['client_secret_key']}}}}}",
    "OUTPUT_VOLUME_ROOT": output_volume_root,
}

entity_kwargs = dict(
    name=endpoint_name,
    entity_name=registered_model_name,
    entity_version=str(latest_version),
    workload_size=serving_cfg["workload_size"],
    workload_type=workload_type,
    scale_to_zero_enabled=scale_to_zero,
    environment_vars=env_vars,
)
if max_throughput:
    entity_kwargs["max_provisioned_throughput"] = max_throughput

entity = ServedEntityInput(**entity_kwargs)

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
