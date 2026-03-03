# Databricks notebook source
# MAGIC %md
# MAGIC # Docling on Databricks
# MAGIC ## 03 - Inspect Document Models and Test Endpoint
# MAGIC
# MAGIC Reviews the Docling document models produced by notebook 01, then sends a live
# MAGIC request to the deployed serving endpoint to verify end-to-end functionality.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Configuration

# COMMAND ----------

import yaml, os, json

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
vol_raw = config["volumes"]["raw_docs"]
vol_exports = config["volumes"]["exports"]
serving_cfg = config["serving"]

exports_path = f"/Volumes/{catalog}/{schema}/{vol_exports}"
print(f"Exports path: {exports_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### List Exported Artifacts

# COMMAND ----------

files = os.listdir(exports_path)
print(f"Exported files ({len(files)}):")
for f in sorted(files):
    size = os.path.getsize(f"{exports_path}/{f}")
    print(f"  {f} ({size:,} bytes)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inspect Document Model Structure
# MAGIC
# MAGIC Docling produces a rich document model that captures pages, text blocks, tables,
# MAGIC pictures, and key-value items. The JSON representation below was persisted in
# MAGIC notebook 01 alongside the markdown export.

# COMMAND ----------

json_files = [f for f in files if f.endswith(".json")]
if not json_files:
    print("No JSON document models found.")
    dbutils.notebook.exit("No models to inspect")

doc_file = json_files[0]
with open(f"{exports_path}/{doc_file}") as f:
    doc = json.load(f)

print(f"Document: {doc_file}")
print(f"Top-level keys: {list(doc.keys())}")

# COMMAND ----------

if "pages" in doc:
    pages = doc["pages"]
    print(f"Pages: {len(pages)}")
    for page_id, page in pages.items():
        dims = page.get("size", {})
        print(f"  {page_id}: {dims.get('width', '?')} x {dims.get('height', '?')}")

if "texts" in doc:
    texts = doc["texts"]
    print(f"\nText items: {len(texts)}")
    for t in texts[:5]:
        label = t.get("label", "unknown")
        text = t.get("text", "")[:80]
        print(f"  [{label}] {text}")

if "tables" in doc:
    tables = doc["tables"]
    print(f"\nTables: {len(tables)}")
    for i, tbl in enumerate(tables):
        grid = tbl.get("data", {}).get("table_cells", [])
        print(f"  Table {i}: {len(grid)} cells")

if "pictures" in doc:
    print(f"\nPictures: {len(doc['pictures'])}")

if "key_value_items" in doc:
    print(f"\nKey-value items: {len(doc['key_value_items'])}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Markdown Preview

# COMMAND ----------

md_files = [f for f in files if f.endswith(".md")]
if md_files:
    with open(f"{exports_path}/{md_files[0]}") as f:
        md_content = f.read()
    print(f"--- {md_files[0]} ({len(md_content)} chars) ---")
    print(md_content[:3000])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Raw Document Model

# COMMAND ----------

if json_files:
    raw_json = spark.read.text(f"{exports_path}/{doc_file}", wholetext=True)
    display(raw_json)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test the Deployed Endpoint
# MAGIC
# MAGIC Sends a volume path to the live `docling-parser` endpoint and displays the
# MAGIC returned export path. The endpoint must be in READY state for this cell to succeed.

# COMMAND ----------

from mlflow.deployments import get_deploy_client

endpoint_name = serving_cfg["endpoint_name"]

pdf_files = [f for f in os.listdir(f"/Volumes/{catalog}/{schema}/{vol_raw}") if f.endswith(".pdf")]
test_path = f"/Volumes/{catalog}/{schema}/{vol_raw}/{pdf_files[0]}"

print(f"Endpoint: {endpoint_name}")
print(f"Input:    {test_path}")

deploy_client = get_deploy_client("databricks")
response = deploy_client.predict(
    endpoint=endpoint_name,
    inputs=[test_path],
)

print(f"Response: {response}")
