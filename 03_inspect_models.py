# Databricks notebook source
# MAGIC %md
# MAGIC # Docling on Databricks -- Inspect Models and Test Endpoint
# MAGIC
# MAGIC Reviews parsed document models from notebook 01 and sends a live request
# MAGIC to the serving endpoint.

# COMMAND ----------

# MAGIC %pip install -r requirements.txt -q
# MAGIC %restart_python

# COMMAND ----------

import yaml, os, json

_nb = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
bundle_root = "/Workspace/" + "/".join(_nb.strip("/").split("/")[:-1])

with open(f"{bundle_root}/config.yaml") as f:
    cfg = yaml.safe_load(f)

catalog, schema = cfg["catalog"], cfg["schema"]
vol_raw, vol_exports = cfg["volumes"]["raw_docs"], cfg["volumes"]["exports"]
exports_path = f"/Volumes/{catalog}/{schema}/{vol_exports}"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exported Artifacts

# COMMAND ----------

files = os.listdir(exports_path)
for f in sorted(files):
    size = os.path.getsize(f"{exports_path}/{f}")
    print(f"  {f} ({size:,} bytes)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Document Model Structure

# COMMAND ----------

json_files = [f for f in files if f.endswith(".json")]
if not json_files:
    dbutils.notebook.exit("No JSON document models found")

with open(f"{exports_path}/{json_files[0]}") as f:
    doc = json.load(f)

print(f"Document: {json_files[0]}")
print(f"Keys: {list(doc.keys())}")

if "pages" in doc:
    print(f"\nPages: {len(doc['pages'])}")
if "texts" in doc:
    print(f"Text items: {len(doc['texts'])}")
    for t in doc["texts"][:5]:
        print(f"  [{t.get('label', '?')}] {t.get('text', '')[:80]}")
if "tables" in doc:
    print(f"Tables: {len(doc['tables'])}")
if "pictures" in doc:
    print(f"Pictures: {len(doc['pictures'])}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Markdown Preview

# COMMAND ----------

md_files = [f for f in files if f.endswith(".md")]
if md_files:
    with open(f"{exports_path}/{md_files[0]}") as f:
        content = f.read()
    print(f"--- {md_files[0]} ({len(content)} chars) ---")
    print(content[:3000])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Deployed Endpoint

# COMMAND ----------

from mlflow.deployments import get_deploy_client

endpoint_name = cfg["serving"]["endpoint_name"]

pdf_files = [
    f for f in os.listdir(f"/Volumes/{catalog}/{schema}/{vol_raw}")
    if f.endswith(".pdf")
]
test_path = f"/Volumes/{catalog}/{schema}/{vol_raw}/{pdf_files[0]}"

print(f"Endpoint: {endpoint_name}")
print(f"Input:    {test_path}")

deploy_client = get_deploy_client("databricks")
response = deploy_client.predict(endpoint=endpoint_name, inputs={"inputs": [test_path]})
print(f"Response: {response}")
