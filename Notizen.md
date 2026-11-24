# Notizen DLBS Mini-Challenge

Stand: 24.11.2026. Kapitel 0 bis 4 im Notebook sind fertig, als Nächstes Kapitel 5 (Regularize). Nebenbei die "Offene Punkte" besprechen.

## Termine
- M2 Lernaustausch: **1.12.2026**. 10 Min Demo + 5 Min Q&A. Inhalt: Problemstellung und Anwendungsfall, Dateninspektion, Baseline bis zum Overfitting-Stadium.
- Zusätzlich strukturiertes Feedback an ein anderes Team: Stärken, Verbesserungspotenzial, Key Takeaway.
- Proposal (M1) ist abgegeben. Die übrigen Termine (M3, M4) stehen im Spaces-Kalender, noch nachsehen.

## Stand
- Env `dlbs-mc` (Python 3.12, PyTorch 2.15.0+cu132, Ultralytics 8.4.208), GPU RTX 4050 (6 GB) wird erkannt.
- `requirements.txt` ist ein `pip freeze`. Prüfen, dass die erste Zeile `--extra-index-url https://download.pytorch.org/whl/cu132` enthält.
- Haupt-Notebook `02_Code/notebooks/abgabe.ipynb` mit Kapiteln 0 Setup, 1 Dateninspektion, 2 Gruppen-Split, 3 Baseline (inkl. COCO-Referenz auf Valid), 4 Overfit, 5 Regularize, 6 Tune, 7 Ergebnisse. Kapitel 0 bis 4 sind fertig und laufen ohne Fehler durch, 5 bis 7 sind Platzhalter.
- Baseline (yolo26n, ohne Augmentation, 100 Epochen, Seeds 42/43/44): mAP50-95 auf Valid 0.548 (Std 0.005), AP50 goal 0.929, goalie 0.788, player 0.913, referee 0.785. Bestes Modell (`best.pt`) wird nach Ultralytics-Fitness gewählt (0.1 mAP50 + 0.9 mAP50-95), nicht nach mAP50 allein. Valid-Loss steigt nach wenigen Epochen, Trainingsloss sinkt weiter (Overfitting).
- COCO-Referenz (Klasse `person`, Valid): AP50 0.482, mAP50-95 0.168. Test-Auswertung folgt in Kapitel 7.
- Ein-Batch-Overfit (16 Bilder, `weight_decay=0`, `nbs=16`, `warmup_epochs=0`, `patience=0`, 200 Epochen): Loss geht gegen null, mAP50 0.99 in allen 3 Seeds.
- Hilfsfunktionen `evaluiere` und `tabelle_ueber_seeds` im Setup, werden in den Kapiteln 5 bis 7 wiederverwendet.
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
6. **COCO-Referenz (erledigt auf Valid, 24.11.):** Auf Valid gebaut (Labelkopie mit `person`, ohne `goal`). Die Auswertung auf dem Test-Set folgt in Kapitel 7 mit derselben Funktion (`baue_person_split` für den Testsplit).
7. **Optional aus Susannes Notebook:** pHash-Duplikatsuche und DINOv2-Embeddings (UMAP) zur Dateninspektion. Nur bei genug Zeit.
8. **Aufwand abschätzen:** Zeit pro Epoche mit einem kurzen Probelauf messen, dann Läufe planen (3 Seeds pro Konfiguration, max. 100 Epochen).

## Nächste Schritte
1. Kapitel 5 Regularize: Augmentation, Early Stopping, Weight Decay je einzeln (Karpathy: eine Änderung auf einmal), 3 Seeds pro Konfiguration, Vergleich auf Valid. Nach Lon23 mehrfach auswerten, statistische Tests und mehrere Metriken angeben, Vorhersagen ansehen.
2. Kapitel 6 Tune: Random Search über die Lernrate mit kleinem Budget, ausgehend vom besten Modell aus Kapitel 5.
3. Kapitel 7: Endmodelle einmal auf dem Testset auswerten (inkl. COCO-Referenz auf Test), H1 und H2 prüfen.
4. Im Proposal nachsehen, ob für Kapitel 4 ein zweites Modell (Bonus, z. B. `yolo26s`) versprochen wurde.
5. Für M2 (1.12.): Folien/Demo mit Problemstellung, Dateninspektion, Baseline bis Overfitting. Feedback an das andere Team vorbereiten.
6. Report (am Schluss): Problem 150, Daten 150, Methoden 300 (mit Workflow-Skizze), Ergebnisse 450, Diskussion 450, Reflexion 150 inkl. Selbstbewertung. Daten und Metrikbegründung gehören in den Report, die Experimente ins Notebook.

## Aufräumen
- Leeren Ordner `02_Code/archive` von Hand löschen (wurde versehentlich angelegt).
- Vor der Abgabe: lange Trainingsausgabe (Baseline-Zelle) verstecken oder leeren, kein auskommentierter Code, Notebook mit "Restart & Run All" prüfen, HTML exportieren (`jupyter nbconvert --to html notebooks/abgabe.ipynb`), README aktuell halten, Zugriff für den Dozenten im Repository (GitHub `susuter` bzw. GitLab `susanne.suter@fhnw.ch`).

## Quellen
- [Kar19] Karpathy, A Recipe for Training Neural Networks, 2019.
- [Lon23] Lones, How to avoid machine learning pitfalls, arXiv:2108.02497.

## Laufzeiten und Technik (24.11.)
- Baseline: 3 Seeds mit je 100 Epochen liefen in rund 2.5 Stunden durch. Ein-Batch-Test: 3 Seeds à 200 Epochen in wenigen Minuten.
- Absturz beim Probelauf: `CUDA error: resource already mapped` im Pin-Memory-Thread (Speicherproblem). Behoben mit `workers=4` und `PIN_MEMORY=False` (vor dem Import von ultralytics setzen).
- Warnung "Slow image access": Daten liegen in OneDrive. Bei Bedarf Daten lokal ablegen oder `cache="ram"` testen.
- Ultralytics speicherte früher in den alten Ordner `06_Semester\KIP\runs\...`, weil ein relativer `project`-Pfad an die Einstellung `runs_dir` angehängt wird. Fix: absoluter Pfad. Den falsch angelegten Ordner `KIP\runs\outputs\runs\probe` kann man von Hand löschen.
