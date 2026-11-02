"""Execute and save both IMP302 notebooks with the project kernel."""
from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
for name in ("01_dataset_eda.ipynb", "02_features_and_model.ipynb"):
    path = ROOT / "notebooks" / name
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=300,
        kernel_name="imp302",
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
    )
    print("Running", name, flush=True)
    client.execute()
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    print("Saved", name, flush=True)
