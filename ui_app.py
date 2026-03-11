#!/usr/bin/env python3
"""Modern Streamlit UI for the standalone wake-word trainer app."""

from __future__ import annotations

import io
import traceback
import zipfile
from datetime import datetime
from pathlib import Path

import streamlit as st

from train_voice import VoiceTrainerApp

APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "output"


def apply_custom_style() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Manrope:wght@500;700&display=swap');

        :root {
            --bg0: #f5f7fb;
            --bg1: #ffffff;
            --ink0: #0f172a;
            --ink1: #334155;
            --brand: #0ea5a8;
            --brand2: #0284c7;
            --good: #16a34a;
            --warn: #d97706;
            --line: #dbe4ef;
        }

        html, body, [class*="css"] {
            font-family: 'Space Grotesk', 'Manrope', sans-serif;
            color: var(--ink0);
            background: radial-gradient(1200px 500px at 10% -10%, #e7f6ff 0%, transparent 65%),
                        radial-gradient(1000px 450px at 100% 0%, #e8fff7 0%, transparent 60%),
                        var(--bg0);
        }

        .main .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }

        .hero {
            background: linear-gradient(120deg, rgba(14,165,168,0.09), rgba(2,132,199,0.10));
            border: 1px solid rgba(2,132,199,0.20);
            border-radius: 18px;
            padding: 1.1rem 1.25rem;
            margin-bottom: 1rem;
        }

        .hero h2 {
            margin: 0;
            color: #0b4a59;
            font-weight: 700;
            letter-spacing: 0.2px;
        }

        .hero p {
            margin: 0.4rem 0 0;
            color: #0f3d4d;
        }

        .card {
            background: var(--bg1);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 1rem 1rem 0.8rem;
            box-shadow: 0 8px 28px rgba(15, 23, 42, 0.04);
            margin-bottom: 1rem;
        }

        .pill {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            font-size: 0.8rem;
            color: #0c4a6e;
            background: #e0f2fe;
            border: 1px solid #bae6fd;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
        }

        .muted {
            color: var(--ink1);
            font-size: 0.92rem;
        }

        button[kind="primary"] {
            border-radius: 12px !important;
            border: 1px solid rgba(2,132,199,0.30) !important;
            background: linear-gradient(90deg, #0ea5a8, #0284c7) !important;
            color: white !important;
            font-weight: 700 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    if "phrases_text" not in st.session_state:
        st.session_state.phrases_text = "hey_att_la"
    if "model_name" not in st.session_state:
        st.session_state.model_name = "hey_att_la"
    if "n_samples" not in st.session_state:
        st.session_state.n_samples = 5000
    if "steps" not in st.session_state:
        st.session_state.steps = 3000
    if "false_penalty" not in st.session_state:
        st.session_state.false_penalty = 600


def preset_config(name: str) -> dict[str, int | str]:
    presets = {
        "Quick Smoke": {
            "n_samples": 1000,
            "steps": 500,
            "false_penalty": 300,
            "phrases_text": "hey_att_la",
        },
        "Balanced": {
            "n_samples": 15000,
            "steps": 10000,
            "false_penalty": 900,
            "phrases_text": "hey_att_la\nhey_at_luh\nhey_at_lah\nhey_aht_la",
        },
        "High Robustness": {
            "n_samples": 50000,
            "steps": 25000,
            "false_penalty": 1600,
            "phrases_text": "hey_att_la\nhey_at_luh\nhey_at_lah\nhey_aht_la",
        },
    }
    return presets[name]


def parse_phrases(text: str) -> list[str]:
    phrases: list[str] = []
    for raw_line in text.splitlines():
        phrase = raw_line.strip()
        if phrase and phrase not in phrases:
            phrases.append(phrase)
    return phrases


def sidebar(app: VoiceTrainerApp) -> None:
    st.sidebar.header("Workspace")
    st.sidebar.caption("Standalone training app")
    st.sidebar.write(f"App folder: {APP_DIR}")
    st.sidebar.write(f"Output folder: {app.output_dir}")
    st.sidebar.write(f"Data folder: {app.data_dir}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Safe Limits")
    st.sidebar.write(f"n_samples: {app.N_SAMPLES_MIN} to {app.N_SAMPLES_MAX}")
    st.sidebar.write(f"steps: {app.STEPS_MIN} to {app.STEPS_MAX}")
    st.sidebar.write(f"false penalty: {app.FALSE_PENALTY_MIN} to {app.FALSE_PENALTY_MAX}")


def health_panel(app: VoiceTrainerApp) -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("System Health")
    st.caption("Quick diagnostics to help users verify environment readiness.")

    checks = [
        ("App directory", APP_DIR.exists(), str(APP_DIR)),
        ("Output directory", app.output_dir.exists(), str(app.output_dir)),
        ("Data directory", app.data_dir.exists(), str(app.data_dir)),
        ("openwakeword repo", app.oww_repo.exists(), str(app.oww_repo)),
        ("piper repo", app.piper_repo.exists(), str(app.piper_repo)),
    ]

    for label, ok, path in checks:
        if ok:
            st.success(f"{label}: OK")
        else:
            st.warning(f"{label}: Missing ({path})")

    import_checks = ["streamlit", "numpy", "scipy", "yaml", "librosa", "soundfile", "datasets"]
    import_results: list[str] = []
    for pkg in import_checks:
        try:
            __import__(pkg)
            import_results.append(f"{pkg}: OK")
        except Exception:
            import_results.append(f"{pkg}: Missing")

    st.write("Package checks:")
    st.code("\n".join(import_results), language="text")
    st.markdown("</div>", unsafe_allow_html=True)


def phrase_lab(app: VoiceTrainerApp) -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Phrase Lab")
    st.caption("Generate and review a sample pronunciation before training.")

    target_word = st.text_input("Target phrase", value="hey_att_la", help="Use phonetic style with underscores.")
    generate = st.button("Generate Preview Audio", type="primary", use_container_width=True)

    if generate:
        progress = st.progress(0)
        status = st.empty()
        try:
            app.ensure_dirs()
            progress.progress(20)
            status.info("Loading TTS model")
            wav_path = app.generate_test_clip(target_word)
            progress.progress(100)
            status.success("Preview generated")
            st.audio(str(wav_path))
            st.success(f"Saved: {wav_path}")
        except Exception as exc:
            status.error("Preview generation failed")
            st.error(str(exc))
            st.code(traceback.format_exc(), language="text")

    st.markdown("</div>", unsafe_allow_html=True)


def data_setup(app: VoiceTrainerApp) -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Data Setup")
    st.caption("Prepare model assets and optional background datasets.")

    include_streaming_sets = st.checkbox(
        "Include sample AudioSet + FMA clips (slower, better robustness)",
        value=False,
    )

    prepare = st.button("Prepare Data", type="primary", use_container_width=True)
    if prepare:
        bar = st.progress(0)
        state = st.empty()
        try:
            state.info("Starting data preparation")
            bar.progress(10)
            app.prepare_data(include_streaming_sets=include_streaming_sets)
            bar.progress(100)
            state.success("Data preparation complete")
        except Exception as exc:
            state.error("Data preparation failed")
            st.error(str(exc))
            st.code(traceback.format_exc(), language="text")

    st.markdown("</div>", unsafe_allow_html=True)


def training_lab(app: VoiceTrainerApp) -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Training")
    st.caption("Configure training, run model creation, and monitor stage progress.")

    preset_col, apply_col = st.columns([3, 1])
    with preset_col:
        selected_preset = st.selectbox("Training preset", ["Quick Smoke", "Balanced", "High Robustness"])
    with apply_col:
        if st.button("Apply Preset", use_container_width=True):
            cfg = preset_config(selected_preset)
            st.session_state.n_samples = int(cfg["n_samples"])
            st.session_state.steps = int(cfg["steps"])
            st.session_state.false_penalty = int(cfg["false_penalty"])
            st.session_state.phrases_text = str(cfg["phrases_text"])
            st.rerun()

    c1, c2 = st.columns(2)
    with c1:
        model_name = st.text_input("Model name", key="model_name")
        n_samples = st.slider(
            "Number of synthetic samples",
            min_value=app.N_SAMPLES_MIN,
            max_value=app.N_SAMPLES_MAX,
            key="n_samples",
            step=100,
            help="More samples generally improve robustness but increase runtime.",
        )
    with c2:
        steps = st.slider(
            "Training steps",
            min_value=app.STEPS_MIN,
            max_value=app.STEPS_MAX,
            key="steps",
            step=100,
            help="More steps generally improve fit but increase runtime.",
        )
        false_penalty = st.slider(
            "False activation penalty",
            min_value=app.FALSE_PENALTY_MIN,
            max_value=app.FALSE_PENALTY_MAX,
            key="false_penalty",
            step=50,
            help="Higher values reduce false positives, but may reduce recall.",
        )

    st.text_area(
        "Target phrase variants (one phrase per line)",
        key="phrases_text",
        height=120,
        help="You can add as many phrase variants as you want.",
    )

    phrases = parse_phrases(st.session_state.phrases_text)
    st.write("Active variants:")
    if phrases:
        st.markdown("".join(f'<span class="pill">{p}</span>' for p in phrases), unsafe_allow_html=True)
    else:
        st.warning("Add at least one phrase variant.")

    run_training = st.button("Train Model", type="primary", use_container_width=True)
    if run_training:
        if not phrases:
            st.error("At least one target phrase is required.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        progress = st.progress(0)
        status = st.empty()

        def cb(value: float, label: str) -> None:
            progress.progress(int(max(0.0, min(1.0, value)) * 100))
            status.info(label)

        try:
            app.train(
                model_name=model_name.strip(),
                target_phrases=phrases,
                n_samples=n_samples,
                steps=steps,
                false_activation_penalty=false_penalty,
                progress_callback=cb,
            )
            status.success("Training complete")
            st.success("Model files are ready in the output folder.")
        except Exception as exc:
            status.error("Training failed")
            st.error(str(exc))
            st.code(traceback.format_exc(), language="text")

    st.markdown("</div>", unsafe_allow_html=True)


def build_publish_zip() -> bytes:
    include_files = [
        APP_DIR / "README.md",
        APP_DIR / "LICENSE",
        APP_DIR / "CONTRIBUTING.md",
        APP_DIR / "CODE_OF_CONDUCT.md",
        APP_DIR / ".gitignore",
        APP_DIR / "requirements.txt",
        APP_DIR / "pyproject.toml",
        APP_DIR / "train_voice.py",
        APP_DIR / "ui_app.py",
    ]

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in include_files:
            if file_path.exists() and file_path.is_file():
                zf.write(file_path, arcname=file_path.relative_to(APP_DIR))

        # Include generated model files if present.
        if OUTPUT_DIR.exists():
            for model_file in OUTPUT_DIR.glob("*.onnx"):
                zf.write(model_file, arcname=model_file.relative_to(APP_DIR))
            for model_file in OUTPUT_DIR.glob("*.tflite"):
                zf.write(model_file, arcname=model_file.relative_to(APP_DIR))

    buf.seek(0)
    return buf.getvalue()


def outputs_panel() -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Outputs")
    st.caption("Browse generated files and download artifacts.")

    package_data = build_publish_zip()
    package_name = f"wakeword_lab_publish_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    st.download_button(
        label="Create Publish ZIP",
        data=package_data,
        file_name=package_name,
        mime="application/zip",
        use_container_width=True,
    )

    if not OUTPUT_DIR.exists():
        st.info("No output folder yet. Run a step first.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    files = sorted([p for p in OUTPUT_DIR.rglob("*") if p.is_file()])
    if not files:
        st.info("No output files yet.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    for file_path in files:
        rel = file_path.relative_to(APP_DIR)
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(str(rel))
        with col2:
            if file_path.suffix.lower() in {".onnx", ".tflite", ".wav", ".yaml"}:
                with open(file_path, "rb") as f:
                    st.download_button(
                        label="Download",
                        data=f,
                        file_name=file_path.name,
                        key=f"download_{str(rel)}",
                        use_container_width=True,
                    )

    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(
        page_title="WakeWord Lab",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_custom_style()
    init_state()

    app = VoiceTrainerApp(APP_DIR)

    st.markdown(
        """
        <div class="hero">
            <h2>🎙️ WakeWord Lab</h2>
            <p class="muted">Professional wake-word model training UI with safe limits, phrase variants, guided stages, and downloadable outputs.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sidebar(app)

    t1, t2, t3, t4, t5 = st.tabs(["Phrase Lab", "Data Setup", "Training", "Outputs", "Health"])
    with t1:
        phrase_lab(app)
    with t2:
        data_setup(app)
    with t3:
        training_lab(app)
    with t4:
        outputs_panel()
    with t5:
        health_panel(app)


if __name__ == "__main__":
    main()
