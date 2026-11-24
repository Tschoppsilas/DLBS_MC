# DLBS Mini-Challenge – Eishockey Object Detection

Detektion von Spielern, Torhütern, Schiedsrichtern und Toren auf Eishockeybildern
(Roboflow-Datensatz `hockey_without_puck`) mit Ultralytics YOLO26.

## Umgebung
Conda-Env `dlbs-mc` (Python 3.12, PyTorch mit CUDA 13.2, Ultralytics, JupyterLab).
Kernel in Jupyter/VS Code: **dlbs-mc**.

```powershell
conda create -n dlbs-mc python=3.12 -y
conda activate dlbs-mc
pip install -r requirements.txt
```

## Struktur
- `notebooks/abgabe.ipynb` – Haupt-Notebook (Einstiegspunkt), gliedert sich in Setup, Dateninspektion, Gruppen-Split, Baseline, Overfit, Regularize, Tune, Ergebnisse
- `src/` – wiederverwendbarer Code
- `data/` – Datensatz (nicht im Git, Download siehe `src/data_download.py`)
- `models/` – vortrainierte Gewichte (nicht im Git)
- `outputs/` – Trainingsläufe, Plots, Logs (nicht im Git)

## Abgabe
Notebook sauber durchlaufen lassen (Kernel → Restart & Run All), dann als HTML exportieren:
```powershell
jupyter nbconvert --to html notebooks/abgabe.ipynb
```
