# Contributing Guide

Thanks for helping improve this project.

## Development Setup

1. Create environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

2. Run quick syntax check:

```bash
python -m py_compile train_voice.py
```

## Pull Request Guidelines

1. Keep changes focused and small.
2. Update README when behavior or CLI options change.
3. Include exact run commands used for testing.
4. Do not commit generated folders (`data/`, `output/`, `third_party/`).

## Reporting Issues

Please include:

1. Operating system
2. Python version
3. Command that failed
4. Full traceback/error output
