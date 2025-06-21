<!-- SPDX-FileCopyrightText: 2024 The Despair Authors -->
<!-- SPDX-License-Identifier: MIT -->
## Task
Lese die Dateien `README.md` und `AGENTS.md` im Repository-Stammverzeichnis. Erstelle daraus ein vollständig lauffähiges Projekt:

1. Analysiere **README.md**: Architektur, Datei-/Ordnerstruktur, Code-Blöcke.  
2. Analysiere **AGENTS.md**: Agent-Definitionen, Code-Blöcke, Ablaufbeschreibungen.  
3. Erstelle alle beschriebenen Ordner und Dateien exakt wie spezifiziert.  
4. Übernimm vorhandenen Code unverändert in die entsprechenden Dateien.  
5. Ergänze fehlende Implementierungen so, dass das Projekt sofort nutzbar ist (inkl. `requirements.txt`, Tests, Start-Skript).  
6. Optimiere Lesbarkeit, Fehlerbehandlung und Performance, ohne Stilvorgaben zu verletzen.  
7. Aktualisiere **README.md** (Usage-Anleitung) und **AGENTS.md** (Schritt-für-Schritt-How-To) nach den Änderungen.

## Specific requirements
- Nutze ausschließlich in `README.md` oder `AGENTS.md` dokumentierte Sprachen/Frameworks.  
- Schreibe sauberen, idiomatischen Code mit Docstrings (PEP 257) und Typannotationen (PEP 484) für Python.  
- Verwende relative Importe; keine absoluten Pfade zur lokalen Umgebung.  
- Erstelle für jede Benutzereingabe robuste Validierungen.  
- Keine Hard-Codierung von geheimen Schlüsseln; lies sie aus Umgebungsvariablen.

## Validation
- Führe `flake8`-Linting ohne Fehler durch.  
- Führe `pytest` gegen alle Tests aus (erstelle bei Bedarf Smoke-Tests).  
- Stelle sicher, dass `python -m main` ohne Fehler läuft.

## Citations and documentation
- Erstelle am Ende eine kurze Änderungsübersicht (Datei → Änderung).  
- Liste neue/aktualisierte Abhängigkeiten in `requirements.txt`.

## Du darfst die Readme.md und die AGENTS.md bei Bedarf anpassen, du darfst aber nur dinge Ergänzen, nicht streichen!

# AGENTS.md – Modern Edge Detection App

This **Agents.md** file equips OpenAI Codex (and compatible AI tools) with the precise context it needs to navigate, extend, and safely refactor the **Modern Edge Detection App** code‑base. Adhere to this guidance when generating, reviewing, or modifying code.

> **AI note:** Files or directories **not** listed in *Project Structure* are considered **read‑only** unless explicitly referenced in future updates.

---

## 1  Project Structure

| Path               | Purpose                                                                                                                                                            |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `/main.py`         | Application bootstrapper. Creates required folders, checks CUDA, and launches the Qt event loop.                                                                   |
| `/gui/`            | PyQt6 GUI widgets.<br>`main_window.py` houses the primary window.                                                                                                  |
| `/models/`         | Edge‑detector implementations.<br>`base_model.py` (abstract base), `teed_model.py`, `dexined_model.py`, and `model_manager.py` for checkpoint caching/downloading. |
| `/processing/`     | Non‑GUI business logic.<br>`batch_processor.py` (QThread that orchestrates inference) and `image_utils.py` I/O helpers.                                            |
| `/checkpoints/`    | Downloaded model weights (ignored by VCS; `.gitkeep` only).                                                                                                        |
| `/output/`         | Generated edge maps organised per‑model (also `.gitkeep`).                                                                                                         |
| `/tests/`          | **(to be created)** pytest suite: unit, integration, GUI (`pytest‑qt`).                                                                                            |
| `requirements.txt` | Runtime dependency lockfile.                                                                                                                                       |
| `/scripts/`        | Utility helpers like `download_checkpoints.py`. |
| `README.md`        | End‑user instructions.                                                                                                                                             |
| `AGENTS.md`        | **← this file**.                                                                                                                                                   |

---

## 2  Coding Conventions

### 2.1  Language & Runtime

* **Python ≥ 3.8 ≤ 3.10**. Avoid 3.11+ features until CI image is upgraded.
* Guard GPU calls with `torch.cuda.is_available()`.

### 2.2  Style & Formatting

* **PEP 8** via **flake8**.
* **black** (line length 100) ‑ run before each commit.
* **isort** (profile "black") for imports.
* Docstrings: **NumPy** style.
* Naming: `snake_case` (vars/funcs), `PascalCase` (classes), `UPPER_SNAKE` (constants).
* GUI strings must remain translatable (`tr()` wrappers) once localisation lands.

### 2.3  Typing & Static Analysis

* **mypy --strict** is mandatory – all public functions/classes require annotations.
* Prefer `pathlib.Path` over raw strings for file paths.

### 2.4  Dependencies

* Runtime deps pinned in `requirements.txt`.
* Dev‑only tools live in `requirements‑dev.txt` or `pyproject.toml` (black, mypy, pytest, pre‑commit, etc.).
* Heavy checkpoints remain out of VCS; fetched on‑demand by `ModelManager`.

---

### 2.5 Code Completeness & README Consistency

* **No placeholders or dummy code:** Every committed implementation must be production‑ready and runnable end to end; TODO stubs, mock implementations, and synthetic returns are prohibited.
* **README as source‑of‑truth:** When implementing functionality already documented in `README.md`, mirror the documented CLI commands, parameters, and expected outputs precisely so that users can copy‑paste commands without modification.
* **Verified external resources:** Each link to pretrained models, checkpoints, or datasets **must** be live, publicly accessible, and validated. Document where the weight can be obtained (PyPI package, CDN, GitHub release, Google Drive, etc.) along with its expected SHA‑256 checksum and file size.
* **Automated link audit:** A CI job (e.g., `scripts/verify_resources.py`) must run on every PR to ensure all URLs respond with HTTP 200 and that downloaded files match the documented hash.

## 3  Testing Requirements

### How to use this repository

1. Create a virtual environment and install requirements with `pip install -r requirements.txt`.
2. Download model weights via `python scripts/download_weights.py`.
3. Run the application using `python -m main`.
4. Execute `flake8` and `pytest -q` to validate before committing changes.

### Pretrained weights

| Model | File | Source URL |
|-------|------|------------|
| TEED | `checkpoints/teed_simplified.pth` | https://drive.google.com/uc?id=1V56vGTsu7GYiQouCIKvTWl5UKCZ6yCNu |
| TEED-Alt | `checkpoints/teed_checkpoint.pth` | https://drive.google.com/uc?id=1V56vGTsu7GYiQouCIKvTWl5UKCZ6yCNu |
| DexiNed | `checkpoints/dexined_checkpoint.pth` | https://drive.google.com/uc?id=1u3zrP5TQp3XkQ41RUOEZutnDZ9SdpyRk |
