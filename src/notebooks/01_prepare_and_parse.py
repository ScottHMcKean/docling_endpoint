# Databricks notebook source
# MAGIC %md
# MAGIC # Docling on Databricks
# MAGIC ## 01 - Prepare Data and Parse Documents
# MAGIC
# MAGIC This notebook provisions the required Unity Catalog infrastructure, uploads raw PDF
# MAGIC documents to a managed volume, and exercises the `DoclingModel` pyfunc to validate
# MAGIC the end-to-end parsing pipeline before it is deployed as a serving endpoint.

# COMMAND ----------

# MAGIC %pip install mlflow docling onnxruntime databricks-sdk databricks-ai-bridge pymssql==2.3.8 hf_transfer --upgrade
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Configuration

# COMMAND ----------

import yaml, os, sys, shutil

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
vol_raw = config["volumes"]["raw_docs"]
vol_exports = config["volumes"]["exports"]

print(f"Catalog: {catalog}, Schema: {schema}")
print(f"Bundle root: {bundle_root}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Provision Schema and Volumes

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{vol_raw}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{vol_exports}")

print("Schema and volumes ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Upload Raw Documents to Volume

# COMMAND ----------

raw_docs_ws = f"{bundle_root}/raw_docs"
volume_path = f"/Volumes/{catalog}/{schema}/{vol_raw}"

pdf_files = [f for f in os.listdir(raw_docs_ws) if f.endswith(".pdf")]
print(f"Found {len(pdf_files)} PDF files: {pdf_files}")

for fname in pdf_files:
    src = f"{raw_docs_ws}/{fname}"
    dst = f"{volume_path}/{fname}"
    shutil.copy2(src, dst)
    print(f"  Copied {fname} -> {dst}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Initialize the DoclingModel Pyfunc
# MAGIC
# MAGIC The model is loaded from `src/docling_endpoint.py` -- the same code that runs inside
# MAGIC the serving endpoint. Service principal credentials are sourced from the configured
# MAGIC secret scope so the model can read from and write to Unity Catalog volumes.

# COMMAND ----------

os.environ["DATABRICKS_HOST"] = host
os.environ["DATABRICKS_CLIENT_ID"] = dbutils.secrets.get(
    scope=secrets["scope"], key=secrets["client_id_key"]
)
os.environ["DATABRICKS_CLIENT_SECRET"] = dbutils.secrets.get(
    scope=secrets["scope"], key=secrets["client_secret_key"]
)

sys.path.insert(0, f"{bundle_root}/src")
from docling_endpoint import DoclingModel

model = DoclingModel()
model.load_context(None)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parse a Sample Document

# COMMAND ----------

test_file = pdf_files[0]
volume_input = [f"/Volumes/{catalog}/{schema}/{vol_raw}/{test_file}"]
print(f"Parsing with DoclingModel.predict(): {volume_input}")

output_paths = model.predict(volume_input)
print(f"Output paths: {output_paths}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save the Docling Document Model
# MAGIC
# MAGIC The pyfunc writes markdown to the exports volume. Here we also persist the full
# MAGIC Docling document model as JSON so notebook 03 can inspect the parsed structure.

# COMMAND ----------

import json

exports_path = f"/Volumes/{catalog}/{schema}/{vol_exports}"

result = model.converter.convert(f"{volume_path}/{test_file}")
stem = os.path.splitext(test_file)[0]
json_path = f"{exports_path}/{stem}.json"
with open(json_path, "w") as f:
    f.write(result.document.model_dump_json(indent=2))
print(f"Saved document model: {json_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Review Outputs

# COMMAND ----------

for f in sorted(os.listdir(exports_path)):
    size = os.path.getsize(f"{exports_path}/{f}")
    print(f"  {f} ({size:,} bytes)")

# COMMAND ----------

md_files = [f for f in os.listdir(exports_path) if f.endswith(".md")]
if md_files:
    with open(f"{exports_path}/{md_files[0]}") as f:
        content = f.read()
    print(f"--- {md_files[0]} Preview ---")
    print(content[:2000])
