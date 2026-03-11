# WakeWord Lab 🎙️

**WakeWord Lab** is a professional wake-word model trainer with a full browser UI.
Open it in your browser, type your wake phrase, pick a preset, and get a trained `.onnx` + `.tflite` model — no terminal commands required for everyday use.

> 🖥️ **`ui_app.py`** — the WakeWord Lab web app **← start here**
> 🖥️ **`train_voice.py`** — advanced CLI for terminal / scripting use only

## What's New In WakeWord Lab (2026 UI)

- Full web UI app with presets, progress bars, and downloads
- Health diagnostics tab for environment readiness checks
- One-click publish ZIP export from the Outputs tab
- Standalone CLI plus UI workflow in the same project

## Launch In 30 Seconds

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run ui_app.py
```

### Windows (PowerShell)

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run ui_app.py
```

### Windows (Command Prompt)

```bat
py -3 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
streamlit run ui_app.py
```

## Why This Project Is Public-Friendly 🌍

- ✅ Separate app folder and codebase
- ✅ Clear one-file CLI workflow
- ✅ Step-by-step onboarding for new users
- ✅ Open-source friendly project files
- ✅ Reproducible commands and output locations

## Using the WakeWord Lab UI 🖥️

Launch the app (`streamlit run ui_app.py`) and open it in your browser. You will see five tabs:

| Tab | What it does |
|-----|--------------|
| 🎤 Phrase Lab | Test how your wake phrase sounds before training |
| 📦 Data Setup | Download background audio and prepare the training dataset |
| 🏋️ Training Lab | Configure settings, pick a preset, and run training |
| 📁 Outputs | Download your trained `.onnx` / `.tflite` files and export a publish ZIP |
| 🩺 Health | Verify all required tools, folders, and packages are installed |

Training presets available as one-click buttons in the **Training** tab:

| Preset | Samples | Steps | Best For |
|--------|---------|-------|----------|
| ⚡ Quick Smoke | 1,000 | 500 | Fast test to verify the pipeline works |
| ⚖️ Balanced | 15,000 | 10,000 | Good personal wake-word model |
| 🔒 High Robustness | 50,000 | 25,000 | Production quality, minimal false positives |

## Project Structure 📁

```text
wakeword-lab/
├── ui_app.py            ← WakeWord Lab web app  (primary — run this)
├── train_voice.py       ← CLI training engine   (advanced / scripting only)
├── requirements.txt     ← Python dependencies
├── README.md
├── LICENSE
├── pyproject.toml       ← package metadata and CLI entrypoint
├── .gitignore
├── config/              ← generated training YAML configs
├── data/                ← downloaded training assets
├── output/              ← trained model files (.onnx, .tflite)
└── third_party/         ← cloned openWakeWord + piper-sample-generator
```

## Quick Start 🚀

## Platform Setup

### Windows

1. Install Python 3.11+ from python.org
2. Install Git for Windows
3. Install Git LFS:

```powershell
winget install GitHub.GitLFS
git lfs install
```

### macOS

1. Install Python 3.11+ (Homebrew or python.org)
2. Install Git LFS:

```bash
brew install git-lfs
git lfs install
```

### Linux (Ubuntu/Debian)

1. Install Python 3.11+, git, and pip
2. Install Git LFS:

```bash
sudo apt-get update
sudo apt-get install -y git-lfs
git lfs install
```

### 1) Create and activate a virtual environment

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Windows Command Prompt:

```bat
py -3 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
```

### 2) Install dependencies

```bash
python -m pip install -r requirements.txt
```

You can also auto-install while running the first command using `--install-deps`.

Optional install as a CLI tool:

```bash
python -m pip install .
wakeword-lab --help
```

### 2.5) 🚀 Launch WakeWord Lab — the Web UI (primary interface)

```bash
streamlit run ui_app.py
```

If `streamlit` is not found on Windows, use:

```powershell
python -m streamlit run ui_app.py
```

UI features:

- 🎛️ Safe sliders with min/max training limits
- 🧾 Add as many phrase variants as you want (one per line)
- ⚡ One-click training presets (Quick Smoke, Balanced, High Robustness)
- 🎧 Generate preview audio button
- ⏳ Stage-based loading/progress bars for long operations
- 🧬 Optional personal voice verifier using your own recorded clips
- 📦 Output browser with direct downloads for `.onnx`, `.tflite`, and `.wav`
- 🗂️ One-click publish ZIP export for sharing your app
- 🩺 Health tab with environment and dependency diagnostics

### 2.6) Optional: Personalize It With Your Own Voice

WakeWord Lab can also train a speaker-specific verifier so the wake word works better for one person.
This is useful if you want the model to respond mainly to your voice instead of general voices.

Record two folders of WAV files:

- `positive/`: 3 to 10 short clips of you saying the wake phrase
- `negative/`: 3 to 10 short clips of you saying anything else

Recommended recording format:

- mono WAV
- 16 kHz sample rate
- one phrase per clip
- quiet room, varied speaking speed, varied distance from mic

Then in the 🏋️ Training Lab:

1. Train the base model first
2. Scroll to `🧬 Personal Voice Verifier`
3. Paste the positive and negative clip folder paths
4. Click `Train Personal Voice Verifier`

Output:

- `output/<model_name>_verifier.pkl`

This verifier is meant to be used together with the base wake-word model, not instead of it.

---

## Advanced: Terminal / CLI Use 🖥️

> The commands below use `train_voice.py` directly from the terminal.
> These are for power users, automation, and CI pipelines.
> **Most users should use the WakeWord Lab UI (Step 2.5) instead.**

### Test your wake phrase sound

```bash
python train_voice.py test-phrase --target-word hey_seere
```

Output file:

- `output/test_generation.wav`

If pronunciation is not right, make it more phonetic (example: `hey_seeri`).

### 4) Prepare assets and data

Fast setup:

```bash
python train_voice.py prepare-data
```

More robust setup (downloads extra background audio, slower):

```bash
python train_voice.py prepare-data --include-streaming-sets
```

### 5) Train your model

```bash
python train_voice.py train \
  --model-name hey_seere \
  --target-phrases hey_seere hey_seeri hey_seeree hey_syree \
  --n-samples 5000 \
  --steps 3000 \
  --false-activation-penalty 600
```

Training argument bounds:

- `n_samples`: 100 to 200000
- `steps`: 100 to 100000
- `false_activation_penalty`: 100 to 5000

Output files:

- `output/hey_seere.onnx`
- `output/hey_seere.tflite`

## Suggested Presets 🧪

> These same presets are available as **one-click buttons** in the 🏋️ Training tab of the WakeWord Lab UI.
> Use the CLI commands below only if you prefer the terminal.

Quick smoke test:

```bash
python train_voice.py train --model-name hey_seere --target-phrases hey_seere --n-samples 1000 --steps 500 --false-activation-penalty 300
```

Balanced personal model:

```bash
python train_voice.py train --model-name hey_seere --target-phrases hey_seere hey_seeri hey_seeree hey_syree --n-samples 15000 --steps 10000 --false-activation-penalty 900
```

High robustness model:

```bash
python train_voice.py train --model-name hey_seere --target-phrases hey_seere hey_seeri hey_seeree hey_syree --n-samples 50000 --steps 25000 --false-activation-penalty 1600
```

## Requirements and Notes 🧩

- Python 3.11 recommended
- `git-lfs` required for MIT RIR assets
- Windows install example: `winget install GitHub.GitLFS`
- macOS install example: `brew install git-lfs`
- Linux (Debian/Ubuntu) install example: `sudo apt-get install -y git-lfs`
- First run may be long because it downloads large files

## Attribution and Data Licensing 📜

This app uses third-party projects and datasets, including openWakeWord and piper-sample-generator.

Some downloaded training datasets have usage restrictions and may be non-commercial only.
If you plan to publish models or use them commercially, verify each dataset license first.

## Publish Checklist ✅

Before making this repository public:

1. Remove generated data/model artifacts from git history
2. Confirm no API keys or private files are committed
3. Keep `requirements.txt` pinned and tested
4. Add example command logs/screenshots (optional)
5. Verify licenses for all datasets and model outputs

## Need Help? 🤝

If something fails, open an issue with:

1. Your OS and Python version
2. The command you ran
3. The full error output
