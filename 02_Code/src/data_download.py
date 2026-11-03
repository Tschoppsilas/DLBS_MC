import os
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow

CODE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = CODE_DIR / "data"

# API-Key aus 02_Code/.env (ROBOFLOW_API_KEY=...) oder aus der Umgebung
load_dotenv(CODE_DIR / ".env")


def download(version=1):
    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    project = rf.workspace("silass-workspace").project("hockey_without_puck")
    return project.version(version).download("yolov8", location=str(DATA_DIR / "hockey_without_puck"))


if __name__ == "__main__":
    dataset = download()
    print(dataset.location)
