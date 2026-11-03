# DLBS Mini-Challenge – Code

Conda-Env: `dlbs` (Python 3.10, TensorFlow/Keras 2.11, JupyterLab, pandas, numpy, matplotlib).
Kernel in Jupyter: **dlbs**.

## Starten
```powershell
conda activate dlbs
cd 02_Code
jupyter lab
```

## Struktur
- `notebooks/` – Abgabe-Notebook(s), jedes läuft eigenständig (Run All)
- `src/` – wiederverwendbarer Code (`from src import ...`)
- `data/` – Datensätze (nicht im Git)
- `outputs/` – Modelle, Plots, Logs (nicht im Git)

## Abgabe
Notebook sauber durchlaufen lassen (Kernel → Restart & Run All), dann ggf. als HTML/PDF exportieren:
```powershell
jupyter nbconvert --to html notebooks/abgabe.ipynb
```
