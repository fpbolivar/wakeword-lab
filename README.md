# WakeWord Lab 🎙️

Train your own wake-word model with a clean, standalone workflow.

This app is designed to be easy to publish and easy for others to run.

## Why This Project Is Public-Friendly 🌍

- ✅ Separate app folder and codebase
- ✅ Clear one-file CLI workflow
- ✅ Step-by-step onboarding for new users
- ✅ Open-source friendly project files
- ✅ Reproducible commands and output locations

## What This App Does ⚙️

`train_voice.py` provides three commands:

1. `test-phrase` to generate a sample wake phrase WAV
2. `prepare-data` to download training assets
3. `train` to train and export `.onnx` and `.tflite` models

## Project Structure 📁

```text
wakeword-lab/
├── train_voice.py
├── ui_app.py
├── requirements.txt
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── pyproject.toml
├── .gitignore
├── config/
├── data/
├── output/
└── third_party/
```

## Quick Start 🚀

### 1) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
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

### 2.5) Launch the Modern UI App

```bash
streamlit run ui_app.py
```

UI features:

- 🎛️ Safe sliders with min/max training limits
- 🧾 Add as many phrase variants as you want (one per line)
- ⚡ One-click training presets (Quick Smoke, Balanced, High Robustness)
- 🎧 Generate preview audio button
- ⏳ Stage-based loading/progress bars for long operations
- 📦 Output browser with direct downloads for `.onnx`, `.tflite`, and `.wav`
- 🗂️ One-click publish ZIP export for sharing your app
- 🩺 Health tab with environment and dependency diagnostics

### 3) Test your wake phrase sound

```bash
python train_voice.py test-phrase --target-word hey_att_la
```

Output file:

- `output/test_generation.wav`

If pronunciation is not right, make it more phonetic (example: `hey_at_luh`).

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
  --model-name hey_att_la \
  --target-phrases hey_att_la hey_at_luh hey_at_lah hey_aht_la \
  --n-samples 5000 \
  --steps 3000 \
  --false-activation-penalty 600
```

Training argument bounds:

- `n_samples`: 100 to 200000
- `steps`: 100 to 100000
- `false_activation_penalty`: 100 to 5000

Output files:

- `output/hey_att_la.onnx`
- `output/hey_att_la.tflite`

## Suggested Presets 🧪

Quick smoke test:

```bash
python train_voice.py train --model-name hey_att_la --target-phrases hey_att_la --n-samples 1000 --steps 500 --false-activation-penalty 300
```

Balanced personal model:

```bash
python train_voice.py train --model-name hey_att_la --target-phrases hey_att_la hey_at_luh hey_at_lah hey_aht_la --n-samples 15000 --steps 10000 --false-activation-penalty 900
```

High robustness model:

```bash
python train_voice.py train --model-name hey_att_la --target-phrases hey_att_la hey_at_luh hey_at_lah hey_aht_la --n-samples 50000 --steps 25000 --false-activation-penalty 1600
```

## Requirements and Notes 🧩

- Python 3.11 recommended
- `git-lfs` required for MIT RIR assets
- macOS install example: `brew install git-lfs`
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
