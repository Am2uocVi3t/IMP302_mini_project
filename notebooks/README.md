# IMP302 exploratory notebooks

`01_dataset_eda.ipynb` checks the dataset, image-level splits, quality, train-image properties, and synthetic noise. `02_features_and_model.ipynb` shows the denoising filters, the ten feature channels, the residual MLP, validation selection, and frozen final-test results. Image-content exploration uses only the train split. An integrity audit across splits checks metadata and duplicate candidates; it does not guide model choices.

From `mini-project/` on Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-notebooks.txt
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name imp302 --display-name "IMP302 (.venv)"
.\.venv\Scripts\python.exe notebooks\run_notebooks.py
```

The notebook builder `notebooks/populate.py` records the source cells and uses the imported Jupyter skill's notebook scaffolds. Running the notebooks saves plots to `notebooks/figures/` and an EDA JSON to `results/eda_summary.json`. Rerunning `populate.py` clears notebook outputs; run `run_notebooks.py` afterward.
