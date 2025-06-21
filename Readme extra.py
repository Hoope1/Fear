🧩 Gesamtüberblick

Das Projekt Despair ist eine komplette, produktionsreife Desktop-Anwendung für KI-gestützte Kantenerkennung. Es kombiniert eine moderne PyQt6-GUI, Torch-basierte Modelle (TEED & DexiNed), performante Batch-Verarbeitung, automatischen Download der Modellgewichte, Windows-Spezialskripte für PyQt-DLL-Probleme, eine Docker-Containerisierung, Pre-Commit-Automatisierung, GitHub Actions-CI und eine saubere Python-Paketstruktur.


---

🚀 Start- und Laufzeitmechanik

Schritt	Datei / Komponente	Aufgabe

1	main.py	✔ Initialisiert High-DPI-Scaling<br>✔ fügt unter Windows dynamisch Qt-DLL-Suchpfade hinzu<br>✔ konfiguriert logging<br>✔ startet die PyQt-Event-Loop und öffnet das Hauptfenster
2	gui/main_window.py	✔ Erzeugt das Haupt-GUI: Ordnerwahl, Fortschrittsbalken, Log-Konsole, Start-Button<br>✔ Verwaltet Benutzeraktionen und ruft die Hintergrundverarbeitung auf
3	processing/batch_processor.py (läuft in eigenem QThread)	✔ Sammelt alle Bilder im gewählten Ordner (PNG, JPG, JPEG, BMP, TIFF)<br>✔ Lädt TEED & DexiNed einmalig („Warm-up“)<br>✔ Verarbeitet jedes Bild GPU-beschleunigt (fällt bei Bedarf auf CPU zurück)<br>✔ Aktualisiert Fortschrittsbalken per Callback (0 – 100 %)<br>✔ schreibt Log-Meldungen an die GUI<br>✔ legt Kantenergebnis als PNG im Unterordner output/ ab
4	processing/image_utils.py	✔ Lädt Bilder mit OpenCV<br>✔ Normalisiert & skaliert Ausgaben, erhält Originalgröße bei Bedarf<br>✔ Speichert 8-Bit-Edge-Maps
5	Rückkehr	✔ GUI meldet „✅ Verarbeitung abgeschlossen“; Fortschrittsbalken steht auf 100 %



---

🖼️ GUI-Funktionsumfang

UI-Element	Funktion

Ordnerwähler	System-Dialog, setzt self.input_path, aktiviert Start-Button
Start-Button	löst BatchProcessor-Thread aus; deaktiviert sich bis Fertigstellung
Fortschrittsbalken	bekommt pro Bild einen Prozent-Tick
Log-Konsole (read-only)	Zeigt: Modell-Init, verarbeitete Dateien, Warnungen, Exceptions
Error-Handling	• Ungültiger Ordner → QMessageBox.critical<br>• Exceptions im Thread → Stacktrace in Log



---

🧠 Modelle & KI-Pipeline

Architektur-Highlights

Modell	Kern-Bausteine	Besonderheiten

TEEDModel	2 × DoubleFusion-Blöcke → 1×1 Conv	Depthwise-Separable-Faltungen, 3×3 + 5×5-Pfad, CONCAT
DexiNedModel	2 × DexiBlock (3 × Conv-BN-ReLU) → 1×1 Conv	klassische tiefere Feature-Extraktion
BaseEdgeDetector	–	gemeinsame Logik:<br>• CUDA/CPU-Auto-Device<br>• process_large_image() mit Tiling & Overlap (gegen OOM)<br>• abstrakte Hooks: load_model / preprocess / postprocess


Gewichte-Verwaltung

models/model_manager.py

legt Cache-Ordner checkpoints/ an

prüft, ob Gewicht schon vorhanden → sonst Download via gdown

robustes Error-Logging


Gewichts-URLs sind austauschbar (Platzhalter in README & AGENTS.md)



---

⚙️ Entwicklungs- & Betriebswerkzeuge

Tool / Datei	Nutzen

diagnose_pyqt6.py	CLI-Skript – listet OS, Python-Pfad, VEnv-Status, Visual C++ Runtime, PyQt6-Installation
fix_pyqt6_dll.py	Automatisiert Neu-Installation von PyQt6 + Cleanup des Pip-Caches + verweist auf VC-Redistributable
.pre-commit-config.yaml	Black Formatter, Flake8-Linting, Mypy-Typecheck – alles läuft automatisch vorm Commit
.github/workflows/python.yml	GitHub Actions: Checkout → Install Deps → Black Check → Flake8 Lint
Dockerfile (Python 3.10-slim)	Minimal-Image mit OpenCV-Systemlib (libgl1-mesa-glx) und Pip-Abhängigkeiten
launch_edge_detection_rye.bat	Windows-Batch: aktiviert lokale VEnv, startet App



---

📦 Projekt- und CI-Struktur

Despair/
├── main.py
├── gui/ …
├── processing/ …
├── models/ …
├── checkpoints/           # wird automatisch angelegt
├── output/                # Kantenergebnisse
├── tests/                 # (Platz für künftige Unittests)
├── .github/workflows/python.yml
├── requirements.txt       # Runtime + Dev-Tools
├── .pre-commit-config.yaml
├── .gitignore
├── Dockerfile
└── LICENSE (MIT)


---

🏎️ Performance-Features

1. CUDA-Fallback: automatische Wahl zwischen cuda und cpu.


2. torch.no_grad() in der Inferenz → verhindert Gradienten-Berechnung.


3. Tiling-Strategie: process_large_image(tile_size=1024, overlap=128); bricht 8k-Bilder in überlappende Patches und setzt Resultat per np.maximum wieder zusammen, sodass Speicherverbrauch kontrolliert bleibt.


4. Einmalige Modell-Initialisierung pro Batch: reduziert Overhead.




---

🔒 Robustheit & Wartbarkeit

Aspekt	Umsetzung

Typisierung	PEP 604-`
Logging	zentrale Konfiguration in main.py, Modul-Logger in jeder Datei
Exception-Handling	Try/Except in Batch-Thread; Stacktrace → GUI-Log
Code-Style	Black (formatiert), Flake8 (PEP8-Linting), Mypy-Checks



---

🔧 Erweiterbarkeit

Weitere Modelle können per Subklasse von BaseEdgeDetector mit minimalem Boilerplate ergänzt werden.

CLI-Skripte erlauben Diagnose/Fix ohne GUI – nützlich für Server-Deploys.

Docker-Image kann per docker run headless genutzt werden (z. B. in CI).

GitHub Actions kann einfach um Tests oder Upload artefacts erweitert werden.



---

📝 Fazit

Der Code liefert eine schlüsselfertige, plattformunabhängige Lösung zum Stapel-basierten Ausführen modernster Edge-Detection-Modelle mit einer Benutzeroberfläche, die Fortschritte und Logs in Echtzeit anzeigt. Dank automatischem Gewichts-Download, GPU-Support, Tiling-Handling, Pre-Commit-Hooks und CI-Workflow lässt sich das Projekt direkt in professionelle DevOps-Pipelines und Lehr-/Anwendungs-Szenarien integrieren – ohne manuelle Nacharbeiten.

