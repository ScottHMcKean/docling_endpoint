import io
import os
import tempfile
from pathlib import Path
from typing import List

import mlflow
from mlflow.pyfunc import PythonModel

from databricks.sdk import WorkspaceClient

class DoclingModel(PythonModel):
    def __init__(self):
        # fix for opencv bug
        import os

        os.environ.pop("OPENSSL_FORCE_FIPS_MODE", None)

        self.client = None
        self.converter = None
        self.output_volume_root = os.environ.get(
            "OUTPUT_VOLUME_ROOT", "/Volumes/shm/multimodal/exports"
        )

    def initialize_agent(self):
        # service principal authorization
        self.w = WorkspaceClient(
            host=os.environ["DATABRICKS_HOST"],
            client_id=os.environ["DATABRICKS_CLIENT_ID"],
            client_secret=os.environ["DATABRICKS_CLIENT_SECRET"],
        )

    def load_context(self, context):
        print("loading_context")
        # load this with context since it triggers library installs
        from docling.document_converter import DocumentConverter

        self.converter = DocumentConverter()

    def predict(self, model_input: List[str], params=None) -> List[str]:
        if self.converter is None:
            print("Load context to enable converter")
            return model_input

        print("initialize agent")
        self.initialize_agent()

        output_paths = []

        for input_path in model_input:
            file_stem = Path(input_path).stem

            with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
                resp = self.w.files.download(input_path)
                tmpfile.write(resp.contents.read())
                local_input_path = tmpfile.name

            result = self.converter.convert(local_input_path)

            md_path = f"{self.output_volume_root}/{file_stem}.md"
            self.w.files.upload(
                md_path,
                io.BytesIO(result.document.export_to_markdown().encode("utf-8")),
                overwrite=True,
            )

            json_path = f"{self.output_volume_root}/{file_stem}.json"
            self.w.files.upload(
                json_path,
                io.BytesIO(result.document.model_dump_json(indent=2).encode("utf-8")),
                overwrite=True,
            )

            output_paths.append(md_path)

        return output_paths


model = DoclingModel()
mlflow.models.set_model(model)
