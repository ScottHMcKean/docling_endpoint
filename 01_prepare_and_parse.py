# Databricks notebook source
# MAGIC %md
# MAGIC # Docling on Databricks -- Prepare Data and Parse Documents
# MAGIC
# MAGIC Provisions Unity Catalog volumes, uploads raw PDFs, and exercises the `DoclingModel`
# MAGIC pyfunc to validate parsing before endpoint deployment.

# COMMAND ----------

# MAGIC %pip install -r requirements.txt -q
# MAGIC %restart_python

# COMMAND ----------

import yaml, os, sys, shutil, json

_nb = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
bundle_root = "/Workspace/" + "/".join(_nb.strip("/").split("/")[:-1])

with open(f"{bundle_root}/config.yaml") as f:
    cfg = yaml.safe_load(f)

catalog, schema = cfg["catalog"], cfg["schema"]
secrets = cfg["secrets"]
vol_raw, vol_exports = cfg["volumes"]["raw_docs"], cfg["volumes"]["exports"]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Provision Schema and Volumes

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{vol_raw}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{vol_exports}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Upload Raw Documents

# COMMAND ----------

raw_docs_ws = f"{bundle_root}/raw_docs"
volume_path = f"/Volumes/{catalog}/{schema}/{vol_raw}"

pdf_files = [f for f in os.listdir(raw_docs_ws) if f.endswith(".pdf")]
for fname in pdf_files:
    shutil.copy2(f"{raw_docs_ws}/{fname}", f"{volume_path}/{fname}")
print(f"Copied {len(pdf_files)} PDFs to {volume_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parse with DoclingModel
# MAGIC
# MAGIC Uses the same pyfunc code that runs inside the serving endpoint.
# MAGIC Service principal credentials come from the configured secret scope.

# COMMAND ----------

os.environ["DATABRICKS_HOST"] = cfg["host"]
os.environ["DATABRICKS_CLIENT_ID"] = dbutils.secrets.get(
    scope=secrets["scope"], key=secrets["client_id_key"]
)
os.environ["DATABRICKS_CLIENT_SECRET"] = dbutils.secrets.get(
    scope=secrets["scope"], key=secrets["client_secret_key"]
)

sys.path.insert(0, bundle_root)
from docling_endpoint import DoclingModel

model = DoclingModel()
model.load_context(None)

test_file = pdf_files[0]
output_paths = model.predict([f"{volume_path}/{test_file}"])
print(f"Parsed: {output_paths}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Review Outputs

# COMMAND ----------

exports_path = f"/Volumes/{catalog}/{schema}/{vol_exports}"
for f in sorted(os.listdir(exports_path)):
    size = os.path.getsize(f"{exports_path}/{f}")
    print(f"  {f} ({size:,} bytes)")
