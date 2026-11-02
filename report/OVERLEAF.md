# Import the English report into Overleaf

The local source is `main.tex`, the measured tables are generated into `results.tex`, and the compiled PDF is `IMP302_Mini_Project.pdf`. Run `report/package_overleaf.py` after generating tables and EDA figures to refresh `IMP302_Overleaf.zip`.

To use Overleaf, select **New Project → Existing project (.zip)** and upload `IMP302_Overleaf.zip`. Set **Compiler → XeLaTeX** and select `main.tex` as the main document. The ZIP contains `main.tex` at its root, the tables, and all four referenced figures. It does not need the raw image dataset.

The ZIP layout and compiler setting follow [Overleaf's official project upload guide](https://docs.overleaf.com/managing-projects-and-files/uploading-a-project). It is ready for import; uploading it to a personal Overleaf account is an external action for the user.
