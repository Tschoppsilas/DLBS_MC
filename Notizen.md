# Notizen DLBS Mini-Challenge

Stand: 17.11.2026. Nächstes Mal zuerst die Abschnitte "Offene Punkte" besprechen.

## Termine
- M2 Lernaustausch: **1.12.2026**. 10 Min Demo + 5 Min Q&A. Inhalt: Problemstellung und Anwendungsfall, Dateninspektion, Baseline bis zum Overfitting-Stadium.
- Zusätzlich strukturiertes Feedback an ein anderes Team: Stärken, Verbesserungspotenzial, Key Takeaway.
- Proposal (M1) ist abgegeben. Die übrigen Termine (M3, M4) stehen im Spaces-Kalender, noch nachsehen.

## Stand
- Env `dlbs-mc` (Python 3.12, PyTorch 2.15.0+cu132, Ultralytics 8.4.208), GPU RTX 4050 (6 GB) wird erkannt.
- `requirements.txt` ist ein `pip freeze`. Prüfen, dass die erste Zeile `--extra-index-url https://download.pytorch.org/whl/cu132` enthält.
- Haupt-Notebook `02_Code/notebooks/abgabe.ipynb` mit Kapiteln 0 Setup, 1 Dateninspektion, 2 Gruppen-Split, 3 Baseline (COCO-Test erledigt), 4 Overfit, 5 Regularize, 6 Tune, 7 Ergebnisse. Kapitel 3.2 bis 7 sind Platzhalter.
- Split-Code ausgelagert nach `02_Code/src/splits.py` (`video_id`, `load_boxes`, `make_group_split`, `write_split_files`). Das Notebook ruft die Funktionen auf und zeigt Parameter und Kontrollen. Braucht nur `numpy`, `pandas` und `pyyaml`.
- Susannes Notebook (Split, Leakage, Duplikate, Embeddings) liegt in `02_Code/Susannes_Notebooks/` (nicht im Git).

## Entscheidungen zum Split (Kapitel 2)
- Mitgelieferter Roboflow-Split wird nicht verwendet: 14 von 23 Gruppen liegen in mehreren Splits, 68 % der Bilder betroffen.
- Beim Ansehen von Beispielbildern: Die sechs Clips `VAN_NJD_HI_v2-<Zeitstempel>` zeigen alle dasselbe Spiel (Vancouver gegen New Jersey, gleiche Halle, Trikots, Spielstandanzeige). Deshalb sind sie **eine Gruppe** (599 Bilder, 36 %) und kommen komplett ins Training.
- Übrige Gruppen: Zufallssuche (20'000 Mischungen, `numpy`, fester Seed) in `make_group_split`: Gruppen der Reihe nach zu Test, dann Valid, bis je etwa 15 % der Bilder. Bewertet nur nach Grösse und Klassenanteilen (Boxen), keine Gruppe über 50 % des Testsets. `StratifiedGroupKFold` wurde verworfen: Bei gleichem Seed kam auf Silas' Laptop (1290 / 179 / 183, Test zu 89 % ein Video) ein anderer Split als in meiner Umgebung (1158 / 249 / 245) heraus, die scikit-learn-Version ändert das Ergebnis. Ausserdem braucht es ein Label pro Bild.
- Ergebnis in meiner Umgebung (Seed 42): Train 1153, Valid 249, Test 250. Nach dem Neulauf auf dem Laptop prüfen (sollte identisch sein, da nur numpy).
- Test enthält viele kleine Gruppen, darunter `png_unbenannt` (120 Bilder, Herkunft unbekannt, knapp die Hälfte des Tests), Valid zwei Videos (138 und 111 Bilder). Bei den offenen Punkten besprechen.
- Klassenanteile (goal/goalie/player/referee) Valid: 6.3 / 6.9 / 76.5 / 10.4, Test: 5.8 / 6.0 / 78.7 / 9.4.
- Split wird als `train.txt`, `val.txt`, `test.txt` und `data.yaml` in `02_Code/outputs/splits/` gespeichert.

## Offene Punkte (nächstes Mal diskutieren)
1. **Gruppendefinition geprüft (24.11.).** Je ein Bild aus `-10-10-01_12_2024`, `-16-16-01_12_2024`, `-13-13-30_11_2024`, `InShot_20250113` und `InShot_20250114` angesehen: verschiedene Hallen und Mannschaften, also getrennte Spiele. Der Split bleibt unverändert. Gemeinsam sind Sendergrafiken und Wasserzeichen ("PRO VIDEO"), als Einschränkung im Bericht erwähnen. Es wurde nur ein Bild pro Gruppe angesehen.
2. **Zusammensetzung von Valid und Test (Stand 24.11., Seed 42, auf dem Laptop bestätigt: 1153 / 249 / 250).** Test = 11 kleine Gruppen aus vielen verschiedenen Spielen und Hallen (NHL-/NCAA-Sendungen, Hallenaufnahmen, `png_unbenannt` mit 120 Bildern, die nach SPBHL.ru-Sendergrafik aus mehreren Spielen stammen). Das ist breit, aber anders als die Videoclips im Training. Valid = nur zwei Videos (138 und 111 Bilder). Da Valid für Regularisierung und Tuning zur Modellwahl dient, ist es mit zwei Spielen eine schmale Basis. Diskutieren: reicht das, oder soll Valid breiter werden (z. B. Seed ändern oder Bedingung \"keine Gruppe über 50 %\" auch für Valid)?
3. **Das eine Spiel macht 53 % des Trainings aus.** Auswirkung auf die Generalisierung? Eventuell in der Diskussion erwähnen.
4. **Proposal vs. Umsetzung.** Im Proposal steht "gruppiert nach Quellvideo (Dateinamenpräfix)". Jetzt wird nach Spiel gruppiert. Im Bericht erwähnen, ob das Proposal angepasst oder nur erklärt werden muss.
5. **Seed-Auswahl** (300 Kandidaten, nur nach Splitqualität) im Bericht kurz begründen. Alternative zum Label "seltenste Klasse" besprechen.
6. **COCO-Referenz nach dem Split bauen:** Labelkopie mit Spieler/Torhüter/Schiedsrichter als Klasse `person`, Tore entfallen, eigene `data.yaml`, `model.val(..., classes=[0])` auf dem Test-Set. Der COCO-Test an einem Bild zeigte: Torhüter und Spieler nicht unterscheidbar, Stock/Schlittschuhe teils als `skis`, Tor nicht erkannt.
7. **Optional aus Susannes Notebook:** pHash-Duplikatsuche und DINOv2-Embeddings (UMAP) zur Dateninspektion. Nur bei genug Zeit.
8. **Aufwand abschätzen:** Zeit pro Epoche mit einem kurzen Probelauf messen, dann Läufe planen (3 Seeds pro Konfiguration, max. 100 Epochen).

## Nächste Schritte
1. In VS Code das Notebook neu laden, ab Kapitel 1 alle Zellen neu ausführen, dann das Speichern der Splits (Zelle "Speichern der Splits").
2. Kontrolle: Split-Zelle zeigt die besten Seeds. Zeilenzahlen von `train.txt`, `val.txt`, `test.txt` notieren.
3. Kapitel 3.2: Baseline yolo26n (Standardparameter, fester Seed, ohne zusätzliche Augmentation) mit Probelauf, dann 3 Seeds.
4. Kapitel 3.1 vervollständigen: COCO-Referenz auf dem Test-Set (siehe Punkt 6).
5. Diskussionstexte (2 bis 3 Sätze) pro Kapitel schreiben, mit Bezug zu [Kar19] und [Lon23].

## Aufräumen
- Leeren Ordner `02_Code/archive` von Hand löschen (wurde versehentlich angelegt).
- Vor der Abgabe: kein auskommentierter Code, Notebook mit "Restart & Run All" prüfen, HTML exportieren (`jupyter nbconvert --to html notebooks/abgabe.ipynb`), README aktuell halten, Zugriff für den Dozenten im Repository (GitHub `susuter` bzw. GitLab `susanne.suter@fhnw.ch`).

## Quellen
- [Kar19] Karpathy, A Recipe for Training Neural Networks, 2019.
- [Lon23] Lones, How to avoid machine learning pitfalls, arXiv:2108.02497.

## Probelauf (24.11., 3 Epochen mit altem Split)
- Gesamt etwa 2 Minuten für 3 Epochen (inkl. Validierung), also grob 35 bis 40 s pro Epoche. 100 Epochen wären etwa eine Stunde pro Lauf, 3 Seeds grob 3 Stunden. Genaue Zeit nochmals messen.
- Warnung "Slow image access": Daten liegen in OneDrive. Eventuell Daten lokal ablegen oder `cache="ram"` testen.
- Ultralytics speicherte in den alten Ordner `06_Semester\KIP\runs\...`, weil ein relativer `project`-Pfad an die Einstellung `runs_dir` angehängt wird. Fix: absoluter Pfad. Der falsch angelegte Ordner `KIP\runs\outputs\runs\probe` kann von Hand gelöscht werden.
