"""Quellvideo-Gruppen und gruppierter Split für den Datensatz hockey_without_puck."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SPLITS = ["train", "valid", "test"]
BIG_GROUP = "VAN_NJD_HI_v2"  # ein Spiel mit 36 % der Bilder, zu gross für Valid/Test


def load_class_names(data_dir):
    """Klassennamen {index: name} aus data.yaml."""
    with open(Path(data_dir) / "data.yaml", encoding="utf-8") as f:
        names = yaml.safe_load(f)["names"]
    if isinstance(names, list):
        return dict(enumerate(names))
    return {int(k): v for k, v in names.items()}


def video_id(filename):
    """Quellvideo bzw. Bildserie aus dem Dateinamen (ohne Frame-Nummer und Roboflow-Hash)."""
    b = re.sub(r"\.rf\.[0-9a-f]+\.jpg$", "", filename)
    b = re.sub(r"_(jpg|png)$", "", b)
    if b.startswith("VAN_NJD_HI_v2"):                       # Clips desselben Spiels (Vancouver gegen New Jersey)
        return BIG_GROUP
    if re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}", b):       # Screenshots mit Zeitstempel -> nach Datum
        return b[:10]
    b = re.sub(r"[-_]?\d+-?$", "", b)                       # Frame-Nummer am Ende entfernen
    return b or "png_unbenannt"


def load_boxes(data_dir, names):
    """Eine Zeile pro Box mit Split, Bild, Gruppe, Klasse und normierter Box (x, y, w, h)."""
    data_dir = Path(data_dir)
    rows = []
    for split in SPLITS:
        for img in sorted((data_dir / split / "images").glob("*.jpg")):
            label_file = data_dir / split / "labels" / (img.stem + ".txt")
            for line in label_file.read_text().splitlines():
                c, x, y, w, h = line.split()[:5]
                rows.append(dict(split=split, image=img.name, group=video_id(img.name),
                                 cls=names[int(c)], x=float(x), y=float(y), w=float(w), h=float(h)))
    return pd.DataFrame(rows)


def make_group_split(boxes, images, seed, n_trials=20000, big_group=BIG_GROUP, target=15.0, max_group_share=0.5):
    """Gruppierter Split per Zufallssuche über Gruppenzuweisungen (jede Gruppe liegt in genau einem Split).

    Die grosse Gruppe kommt komplett ins Training. Die übrigen Gruppen werden n_trials-mal zufällig
    gemischt und der Reihe nach erst dem Test, dann der Validierung zugewiesen, bis jede Menge etwa
    target Prozent der Bilder enthält; der Rest ist Training. Bewertet wird nur die Qualität des
    Splits (ohne Modell): grösste Abweichung der Test-/Valid-Grösse von target (in %) plus grösste
    Abweichung der Klassenanteile (Boxen) in Test/Valid vom Gesamtanteil (in Prozentpunkten).
    Aufteilungen, in denen eine einzelne Gruppe mehr als max_group_share des Testsets ausmacht,
    werden verworfen. Rückgabe: Tabelle mit Spalte split_new.
    """
    img_cls = pd.crosstab(boxes["image"], boxes["cls"])
    imgs = images.set_index("image").loc[img_cls.index].reset_index()

    group_boxes = img_cls.groupby(imgs["group"].to_numpy()).sum()       # Boxen pro Gruppe und Klasse
    group_images = imgs["group"].value_counts().loc[group_boxes.index]
    overall = group_boxes.sum().to_numpy(float)
    overall = overall / overall.sum() * 100

    names = [g for g in group_boxes.index if g != big_group]
    n_img = group_images.loc[names].to_numpy()
    n_box = group_boxes.loc[names].to_numpy(float)
    goal = target / 100 * len(imgs)

    rng = np.random.default_rng(seed)
    best_score, best_assignment = np.inf, None
    for _ in range(n_trials):
        order = rng.permutation(len(names))
        csum = np.cumsum(n_img[order])
        n_test = int(np.searchsorted(csum, goal)) + 1
        rest = csum[n_test:] - csum[n_test - 1]
        n_valid = int(np.searchsorted(rest, goal)) + 1
        if n_test + n_valid > len(names):
            continue
        test, valid = order[:n_test], order[n_test:n_test + n_valid]
        if n_img[test].max() / n_img[test].sum() > max_group_share:
            continue
        size_dev = max(abs(n_img[test].sum() / len(imgs) * 100 - target),
                       abs(n_img[valid].sum() / len(imgs) * 100 - target))
        class_dev = max(np.abs(n_box[ids].sum(axis=0) / n_box[ids].sum() * 100 - overall).max()
                        for ids in (test, valid))
        if size_dev + class_dev < best_score:
            best_score, best_assignment = size_dev + class_dev, (test, valid)
    if best_assignment is None:
        raise ValueError("Kein gültiger Split gefunden, max_group_share erhöhen oder n_trials vergrössern.")

    split_of = {names[i]: "test" for i in best_assignment[0]}
    split_of.update({names[i]: "valid" for i in best_assignment[1]})
    imgs["split_new"] = imgs["group"].map(split_of).fillna("train")
    imgs.attrs["abweichung"] = float(best_score)
    return imgs


def write_split_files(imgs, data_dir, split_dir, names):
    """Schreibt train.txt, val.txt, test.txt (Bildpfade) und eine data.yaml für Ultralytics."""
    data_dir, split_dir = Path(data_dir), Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    list_names = {"train": "train.txt", "valid": "val.txt", "test": "test.txt"}
    for split, fname in list_names.items():
        paths = [
            (data_dir / row.split / "images" / row.image).resolve().as_posix()
            for row in imgs[imgs["split_new"] == split].itertuples()
        ]
        (split_dir / fname).write_text("\n".join(paths) + "\n", encoding="utf-8")

    data_yaml = {
        "train": (split_dir / "train.txt").resolve().as_posix(),
        "val": (split_dir / "val.txt").resolve().as_posix(),
        "test": (split_dir / "test.txt").resolve().as_posix(),
        "names": names,
    }
    with open(split_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)
