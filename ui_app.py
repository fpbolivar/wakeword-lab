#!/usr/bin/env python3
"""Modern Streamlit UI for the standalone wake-word trainer app."""

from __future__ import annotations

import io
import shutil
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
        st.session_state.phrases_text = "hey_seere"
    if "model_name" not in st.session_state:
        st.session_state.model_name = "hey_seere"
    if "n_samples" not in st.session_state:
        st.session_state.n_samples = 5000
    if "steps" not in st.session_state:
        st.session_state.steps = 3000
    if "false_penalty" not in st.session_state:
        st.session_state.false_penalty = 600
    if "runtime_threshold" not in st.session_state:
        st.session_state.runtime_threshold = VoiceTrainerApp.RUNTIME_THRESHOLD_DEFAULT
    if "runtime_profile_model" not in st.session_state:
        st.session_state.runtime_profile_model = ""
    if "runtime_profile_threshold" not in st.session_state:
        st.session_state.runtime_profile_threshold = VoiceTrainerApp.RUNTIME_THRESHOLD_DEFAULT
    if "runtime_profile_loaded_model" not in st.session_state:
        st.session_state.runtime_profile_loaded_model = ""
    if "device_choice" not in st.session_state:
        st.session_state.device_choice = "auto"
    if "output_destinations_text" not in st.session_state:
        st.session_state.output_destinations_text = ""
    if "auto_copy_outputs" not in st.session_state:
        st.session_state.auto_copy_outputs = False
    if "positive_voice_dir" not in st.session_state:
        st.session_state.positive_voice_dir = ""
    if "negative_voice_dir" not in st.session_state:
        st.session_state.negative_voice_dir = ""
    if "verifier_output_name" not in st.session_state:
        st.session_state.verifier_output_name = ""


def preset_config(name: str) -> dict[str, int | str]:
    presets = {
        "⚡ Quick Smoke": {
            "n_samples": 1000,
            "steps": 500,
            "false_penalty": 300,
            "phrases_text": "hey_seere",
        },
        "⚖️ Balanced": {
            "n_samples": 15000,
            "steps": 10000,
            "false_penalty": 900,
            "phrases_text": "hey_seere\nhey_seeri\nhey_seeree\nhey_syree",
        },
        "🔒 High Robustness": {
            "n_samples": 50000,
            "steps": 25000,
            "false_penalty": 1600,
            "phrases_text": "hey_seere\nhey_seeri\nhey_seeree\nhey_syree",
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


def parse_destination_paths(text: str) -> list[Path]:
    destinations: list[Path] = []
    for raw in text.splitlines():
        entry = raw.strip()
        if not entry:
            continue
        path = Path(entry).expanduser()
        if not path.is_absolute():
            path = (APP_DIR / path).resolve()
        if path not in destinations:
            destinations.append(path)
    return destinations


def runtime_profile_path(model_name: str) -> Path:
    return OUTPUT_DIR / f"{Path(model_name).stem}_runtime.yaml"


def load_runtime_profile_threshold(model_name: str) -> float | None:
    profile_path = runtime_profile_path(model_name)
    if not profile_path.exists():
        return None

    try:
        import yaml

        data = yaml.safe_load(profile_path.read_text()) or {}
    except Exception:
        return None

    value = data.get("runtime_threshold")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def list_trained_model_names() -> list[str]:
    if not OUTPUT_DIR.exists():
        return []
    return sorted(path.stem for path in OUTPUT_DIR.glob("*.onnx"))


def resolve_user_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = (APP_DIR / path).resolve()
    return path


def count_wav_files(path_text: str) -> int:
    if not path_text.strip():
        return 0
    path = resolve_user_path(path_text)
    if not path.exists():
        return 0
    return len(list(path.rglob("*.wav")))


def save_audio_blob(audio_file, destination_dir: Path, prefix: str) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(getattr(audio_file, "name", "recording.wav")).suffix or ".wav"
    target = destination_dir / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}{suffix}"
    target.write_bytes(audio_file.getvalue())
    return target


def save_uploaded_files(uploaded_files: list, destination_dir: Path, prefix: str) -> int:
    destination_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for index, uploaded in enumerate(uploaded_files, start=1):
        suffix = Path(getattr(uploaded, "name", "clip.wav")).suffix or ".wav"
        target = destination_dir / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{index:02d}{suffix}"
        target.write_bytes(uploaded.getvalue())
        saved += 1
    return saved


def copy_files_to_destination(files: list[Path], destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in files:
        if src.exists() and src.is_file():
            shutil.copy2(src, destination / src.name)
            copied += 1
    return copied


def sidebar(app: VoiceTrainerApp) -> None:
    st.sidebar.header("🗂️ Workspace")
    st.sidebar.caption("WakeWord Lab — standalone training app")
    st.sidebar.write(f"📁 App: {APP_DIR}")
    st.sidebar.write(f"📤 Output: {app.output_dir}")
    st.sidebar.write(f"💾 Data: {app.data_dir}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🛡️ Safe Limits")
    st.sidebar.caption("Training inputs are clamped to these bounds:")
    st.sidebar.write(f"📊 n_samples: {app.N_SAMPLES_MIN:,} – {app.N_SAMPLES_MAX:,}")
    st.sidebar.write(f"⚙️ steps: {app.STEPS_MIN:,} – {app.STEPS_MAX:,}")
    st.sidebar.write(f"🔇 false penalty: {app.FALSE_PENALTY_MIN:,} – {app.FALSE_PENALTY_MAX:,}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ Compute Device")
    devices = app.detect_available_devices()
    device_ids = ["auto"] + [d["id"] for d in devices]
    device_labels = ["🤖 Auto (recommended)"] + [d["label"] for d in devices]

    # Show what "auto" will actually pick
    auto_pick = "mps" if any(d["id"] == "mps" for d in devices) else \
                "cuda:0" if any(d["id"] == "cuda:0" for d in devices) else "cpu"
    auto_hint = next((d["label"] for d in devices if d["id"] == auto_pick), "🖥️ CPU")
    st.sidebar.caption(f"Auto will use: {auto_hint}")

    selected_label = st.sidebar.radio(
        "Select training device",
        device_labels,
        key="device_label_radio",
        help="CPU works everywhere. MPS is fastest on Apple Silicon Macs. CUDA is fastest on NVIDIA GPUs.",
    )
    chosen_index = device_labels.index(selected_label)
    st.session_state.device_choice = device_ids[chosen_index]

    # Show a description for the selected device
    if chosen_index > 0:
        st.sidebar.caption(devices[chosen_index - 1]["desc"])

    st.sidebar.markdown("---")
    st.sidebar.subheader("📌 Final Output Destinations")
    st.sidebar.text_area(
        "Optional copy destinations (one path per line)",
        key="output_destinations_text",
        height=100,
        help="After training, model files can be copied to these folders. Relative paths are resolved from the app folder.",
    )
    st.sidebar.checkbox(
        "Auto-copy final model files after training",
        key="auto_copy_outputs",
        help="If enabled, .onnx/.tflite files are copied to destination folders automatically when training finishes.",
    )


def health_panel(app: VoiceTrainerApp) -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🩺 System Health")
    st.caption("Check that all required tools, folders, and Python packages are installed before training.")

    checks = [
        ("📁 App directory", APP_DIR.exists(), str(APP_DIR)),
        ("📤 Output directory", app.output_dir.exists(), str(app.output_dir)),
        ("💾 Data directory", app.data_dir.exists(), str(app.data_dir)),
        ("🤖 openWakeWord repo", app.oww_repo.exists(), str(app.oww_repo)),
        ("🗣️ Piper TTS repo", app.piper_repo.exists(), str(app.piper_repo)),
    ]

    for label, ok, path in checks:
        if ok:
            st.success(f"✅ {label}: found")
        else:
            st.warning(f"⚠️ {label}: missing — expected at {path}")

    st.markdown("---")
    st.write("⚡ Detected compute devices:")
    detected = app.detect_available_devices()
    for d in detected:
        st.info(f"{d['label']} — {d['desc']}")

    st.markdown("---")
    import_checks = [
        ("streamlit", "Streamlit web UI framework"),
        ("numpy", "Numerical computing"),
        ("scipy", "Signal processing"),
        ("sklearn", "Verifier model training"),
        ("yaml", "Config file parsing"),
        ("librosa", "Audio analysis"),
        ("soundfile", "WAV file I/O"),
        ("datasets", "HuggingFace datasets loader"),
    ]
    import_results: list[str] = []
    for pkg, desc in import_checks:
        try:
            __import__(pkg)
            import_results.append(f"✅  {pkg:<14} — {desc}")
        except Exception:
            import_results.append(f"❌  {pkg:<14} — {desc}  (pip install {pkg})")

    st.write("📦 Package checks:")
    st.code("\n".join(import_results), language="text")
    st.markdown("</div>", unsafe_allow_html=True)


def phrase_lab(app: VoiceTrainerApp) -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🎤 Phrase Lab")
    st.caption("Test how your wake phrase sounds before committing to a full training run. Tweak the spelling until it sounds right.")

    target_word = st.text_input(
        "🎤 Target phrase",
        value="hey_seere",
        help="Use phonetic spelling with underscores. e.g. 'hey_seeri' or 'hey_syree'. Adjust until the preview sounds natural.",
    )
    generate = st.button("🔊 Generate Preview Audio", type="primary", use_container_width=True)

    if generate:
        progress = st.progress(0)
        status = st.empty()
        try:
            app.ensure_dirs()
            progress.progress(20)
            status.info("⏳ Loading TTS model...")
            wav_path = app.generate_test_clip(target_word)
            progress.progress(100)
            status.success("✅ Preview ready — press play below")
            st.audio(str(wav_path))
            st.success(f"Saved: {wav_path}")
        except Exception as exc:
            status.error("Preview generation failed")
            st.error(str(exc))
            st.code(traceback.format_exc(), language="text")

    st.markdown("</div>", unsafe_allow_html=True)


def data_setup(app: VoiceTrainerApp) -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📦 Data Setup")
    st.caption("Download background audio and prepare the training dataset. Run this once before your first training session.")

    include_streaming_sets = st.checkbox(
        "📥 Include AudioSet + FMA background clips (slower download, better noise robustness)",
        value=False,
        help="Adds ~2–4 GB of background audio. Recommended for a production-quality model.",
    )

    prepare = st.button("⬇️ Prepare Data", type="primary", use_container_width=True)
    if prepare:
        bar = st.progress(0)
        state = st.empty()
        try:
            state.info("📥 Downloading and preparing datasets...")
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
    st.subheader("🏋️ Training Lab")
    st.caption("Pick a preset or fine-tune the sliders, then hit Train Model to generate your .onnx and .tflite files.")

    preset_col, apply_col = st.columns([3, 1])
    with preset_col:
        selected_preset = st.selectbox(
            "⚡ Training preset",
            ["⚡ Quick Smoke", "⚖️ Balanced", "🔒 High Robustness"],
            help="Quick Smoke: fast pipeline test (~2 min). Balanced: good personal model (~1 hr). High Robustness: production quality (~4+ hrs).",
        )
    with apply_col:
        if st.button("✨ Apply Preset", use_container_width=True):
            cfg = preset_config(selected_preset)
            st.session_state.n_samples = int(cfg["n_samples"])
            st.session_state.steps = int(cfg["steps"])
            st.session_state.false_penalty = int(cfg["false_penalty"])
            st.session_state.phrases_text = str(cfg["phrases_text"])
            st.rerun()

    c1, c2 = st.columns(2)
    with c1:
        model_name = st.text_input(
            "🏷️ Model name",
            key="model_name",
            help="Output files will be named after this. Use lowercase with underscores, e.g. hey_seere",
        )
        n_samples = st.slider(
            "📊 Synthetic sample count",
            min_value=app.N_SAMPLES_MIN,
            max_value=app.N_SAMPLES_MAX,
            key="n_samples",
            step=100,
            help="More samples generally improve robustness but increase runtime.",
        )
    with c2:
        steps = st.slider(
            "⚙️ Training steps",
            min_value=app.STEPS_MIN,
            max_value=app.STEPS_MAX,
            key="steps",
            step=100,
            help="More steps generally improve fit but increase runtime.",
        )
        false_penalty = st.slider(
            "🔇 False activation penalty",
            min_value=app.FALSE_PENALTY_MIN,
            max_value=app.FALSE_PENALTY_MAX,
            key="false_penalty",
            step=50,
            help="Higher values reduce false positives, but may reduce recall.",
        )

    runtime_threshold = st.number_input(
        "🎚️ Runtime activation threshold",
        key="runtime_threshold",
        step=0.05,
        format="%.2f",
        min_value=VoiceTrainerApp.RUNTIME_THRESHOLD_MIN,
        max_value=VoiceTrainerApp.RUNTIME_THRESHOLD_MAX,
        help=(
            "Detection score used when the trained model runs in a listener app or terminal script. "
            "This is saved with the model output and does not change training."
        ),
    )

    st.text_area(
        "📝 Phrase variants (one per line)",
        key="phrases_text",
        height=120,
        help="Add multiple phonetic spellings so the model learns different pronunciations. One phrase per line.",
    )

    phrases = parse_phrases(st.session_state.phrases_text)
    st.write("🏷️ Active variants:")
    if phrases:
        st.markdown("".join(f'<span class="pill">{p}</span>' for p in phrases), unsafe_allow_html=True)
    else:
        st.warning("Add at least one phrase variant.")

    # ── Live time estimate ──────────────────────────────────────────────────
    device_choice = st.session_state.get("device_choice", "auto")
    est = app.estimate_training_time(
        n_samples=n_samples,
        steps=steps,
        n_phrases=max(1, len(phrases)),
        device=device_choice,
    )
    with st.expander(f"⏱️ Estimated total time: **{est['label']}**", expanded=True):
        cols = st.columns(len(est["breakdown"]))
        for col, (stage, duration) in zip(cols, est["breakdown"]):
            col.metric(stage, duration)
        st.caption(
            "Estimates are based on average runtimes on an M2 MacBook Pro. "
            "Your actual time may vary depending on hardware and background load."
        )
    # ───────────────────────────────────────────────────────────────────────

    run_training = st.button("🚀 Train Model", type="primary", use_container_width=True)
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
                runtime_threshold=float(runtime_threshold),
                device=st.session_state.get("device_choice", "auto"),
                progress_callback=cb,
            )
            status.success("Training complete")
            st.success("Model files are ready in the output folder.")

            destinations = parse_destination_paths(st.session_state.get("output_destinations_text", ""))
            if st.session_state.get("auto_copy_outputs", False) and destinations:
                model_base = model_name.strip()
                export_files = [
                    OUTPUT_DIR / f"{model_base}.onnx",
                    OUTPUT_DIR / f"{model_base}.tflite",
                    OUTPUT_DIR / f"{model_base}_float16.tflite",
                    OUTPUT_DIR / f"{model_base}_runtime.yaml",
                ]
                export_files = [p for p in export_files if p.exists()]
                if export_files:
                    st.info("Copying final model files to destination folders...")
                    for dest in destinations:
                        copied_count = copy_files_to_destination(export_files, dest)
                        if copied_count:
                            st.success(f"Copied {copied_count} file(s) to: {dest}")
                        else:
                            st.warning(f"No model files found to copy for destination: {dest}")
        except Exception as exc:
            status.error("Training failed")
            st.error(str(exc))
            st.code(traceback.format_exc(), language="text")

    st.info("Need speaker-specific tuning? Use the `🧬 Personal Voice` tab to record clips, import WAVs, run readiness checks, and train a personal verifier.")

    st.markdown("</div>", unsafe_allow_html=True)


def personal_voice_lab(app: VoiceTrainerApp) -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧬 Personal Voice")
    st.caption(
        "Record or import your own voice clips, check readiness, and train a speaker-specific verifier for more personalized wake-word behavior."
    )

    example_root = APP_DIR / "personal_voice"
    example_positive = example_root / "positive"
    example_negative = example_root / "negative"

    top1, top2 = st.columns([1, 2])
    with top1:
        if st.button("✨ Autofill Example Paths", use_container_width=True):
            st.session_state.positive_voice_dir = str(example_positive)
            st.session_state.negative_voice_dir = str(example_negative)
            if not st.session_state.get("verifier_output_name"):
                st.session_state.verifier_output_name = f"{st.session_state.get('model_name', 'wakeword')}_verifier.pkl"
            st.rerun()
    with top2:
        st.caption("Creates a ready-to-use example layout under the app folder: `personal_voice/positive` and `personal_voice/negative`.")

    with st.expander("🎙️ Recording Guide", expanded=True):
        g1, g2 = st.columns(2)
        with g1:
            st.markdown(
                "\n".join(
                    [
                        "**Good recording habits**",
                        "- Record 3 to 10 clips for `positive/` and 3 to 10 clips for `negative/`",
                        "- Keep each clip short: about 1 to 2 seconds",
                        "- Use the same microphone you expect to use later if possible",
                        "- Record in a quiet room first, then add a few natural variations",
                        "- Change speed, tone, and distance slightly between takes",
                    ]
                )
            )
        with g2:
            st.markdown(
                "\n".join(
                    [
                        "**Avoid these issues**",
                        "- Do not put multiple phrases in one clip",
                        "- Do not trim so tightly that the phrase gets cut off",
                        "- Do not mix other speakers into the positive folder",
                        "- Do not use background music or TV noise if you can avoid it",
                        "- Do not use MP3 if you can save WAV directly",
                    ]
                )
            )

        st.caption("Suggested folder structure")
        st.code(
            "personal_voice/\n"
            "  positive/\n"
            "    take_01.wav\n"
            "    take_02.wav\n"
            "    take_03.wav\n"
            "  negative/\n"
            "    other_01.wav\n"
            "    other_02.wav\n"
            "    other_03.wav",
            language="text",
        )

    path_col1, path_col2 = st.columns(2)
    with path_col1:
        st.text_input(
            "🎤 Positive voice clips folder",
            key="positive_voice_dir",
            help="Folder containing WAV files of your voice saying the wake phrase.",
        )
    with path_col2:
        st.text_input(
            "🗣️ Negative voice clips folder",
            key="negative_voice_dir",
            help="Folder containing WAV files of your voice saying anything except the wake phrase.",
        )

    default_verifier_name = f"{st.session_state.get('model_name', 'wakeword').strip() or 'wakeword'}_verifier.pkl"
    if not st.session_state.get("verifier_output_name"):
        st.session_state.verifier_output_name = default_verifier_name
    st.text_input(
        "📄 Verifier output filename",
        key="verifier_output_name",
        help="Saved into the output folder unless you also configured final output destinations.",
    )

    positive_dir_text = st.session_state.get("positive_voice_dir", "")
    negative_dir_text = st.session_state.get("negative_voice_dir", "")
    positive_count = count_wav_files(positive_dir_text)
    negative_count = count_wav_files(negative_dir_text)
    base_model_path = OUTPUT_DIR / f"{(st.session_state.get('model_name', '').strip() or 'wakeword')}.onnx"

    st.write("📊 Readiness Check")
    r1, r2, r3 = st.columns(3)
    r1.metric("🎤 Positive clips", positive_count)
    r2.metric("🗣️ Negative clips", negative_count)
    r3.metric("🧠 Base model", "Ready" if base_model_path.exists() else "Missing")
    if positive_count < 3 or negative_count < 3 or not base_model_path.exists():
        st.warning("Recommended minimum: 3 positive clips, 3 negative clips, and a trained base ONNX model in the output folder.")
    else:
        st.success("Ready to train a personalized verifier.")

    st.markdown("---")
    st.subheader("🎙️ Record With Your Microphone")
    mic1, mic2 = st.columns(2)
    with mic1:
        positive_audio = st.audio_input("Record a positive clip", key="positive_audio_input")
        if st.button("💾 Save Positive Recording", use_container_width=True):
            if not positive_dir_text.strip():
                st.error("Set a positive clips folder first.")
            elif positive_audio is None:
                st.error("Record a positive clip first.")
            else:
                saved_path = save_audio_blob(positive_audio, resolve_user_path(positive_dir_text), "positive")
                st.success(f"Saved: {saved_path}")
    with mic2:
        negative_audio = st.audio_input("Record a negative clip", key="negative_audio_input")
        if st.button("💾 Save Negative Recording", use_container_width=True):
            if not negative_dir_text.strip():
                st.error("Set a negative clips folder first.")
            elif negative_audio is None:
                st.error("Record a negative clip first.")
            else:
                saved_path = save_audio_blob(negative_audio, resolve_user_path(negative_dir_text), "negative")
                st.success(f"Saved: {saved_path}")

    st.markdown("---")
    st.subheader("📥 Import Existing WAV Files")
    up1, up2 = st.columns(2)
    with up1:
        positive_uploads = st.file_uploader(
            "Upload positive WAV clips",
            type=["wav"],
            accept_multiple_files=True,
            key="positive_wav_uploads",
        )
        if st.button("📥 Import Positive WAVs", use_container_width=True):
            if not positive_dir_text.strip():
                st.error("Set a positive clips folder first.")
            elif not positive_uploads:
                st.error("Choose one or more positive WAV files first.")
            else:
                saved = save_uploaded_files(list(positive_uploads), resolve_user_path(positive_dir_text), "positive")
                st.success(f"Imported {saved} positive clip(s).")
    with up2:
        negative_uploads = st.file_uploader(
            "Upload negative WAV clips",
            type=["wav"],
            accept_multiple_files=True,
            key="negative_wav_uploads",
        )
        if st.button("📥 Import Negative WAVs", use_container_width=True):
            if not negative_dir_text.strip():
                st.error("Set a negative clips folder first.")
            elif not negative_uploads:
                st.error("Choose one or more negative WAV files first.")
            else:
                saved = save_uploaded_files(list(negative_uploads), resolve_user_path(negative_dir_text), "negative")
                st.success(f"Imported {saved} negative clip(s).")

    st.markdown("---")
    if st.button("🧬 Train Personal Voice Verifier", type="primary", use_container_width=True):
        try:
            verifier_output = app.train_personal_verifier(
                model_name=st.session_state.get("model_name", "").strip(),
                positive_reference_dir=positive_dir_text,
                negative_reference_dir=negative_dir_text,
                output_path=OUTPUT_DIR / st.session_state.get("verifier_output_name", default_verifier_name),
            )
            st.success(f"Personal verifier created: {verifier_output}")

            current_model_name = st.session_state.get("model_name", "").strip()
            if current_model_name:
                app.save_runtime_profile(
                    model_name=current_model_name,
                    runtime_threshold=float(
                        st.session_state.get("runtime_threshold", VoiceTrainerApp.RUNTIME_THRESHOLD_DEFAULT)
                    ),
                )

            destinations = parse_destination_paths(st.session_state.get("output_destinations_text", ""))
            if st.session_state.get("auto_copy_outputs", False) and destinations:
                verifier_exports = [verifier_output]
                current_model_name = st.session_state.get("model_name", "").strip()
                if current_model_name:
                    runtime_profile = runtime_profile_path(current_model_name)
                    if runtime_profile.exists():
                        verifier_exports.append(runtime_profile)
                for dest in destinations:
                    copied_count = copy_files_to_destination(verifier_exports, dest)
                    if copied_count:
                        st.success(f"Copied personalized verifier assets to: {dest}")
        except Exception as exc:
            st.error("Personal verifier training failed")
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
            for model_file in OUTPUT_DIR.glob("*.pkl"):
                zf.write(model_file, arcname=model_file.relative_to(APP_DIR))
            for model_file in OUTPUT_DIR.glob("*_runtime.yaml"):
                zf.write(model_file, arcname=model_file.relative_to(APP_DIR))

    buf.seek(0)
    return buf.getvalue()


def outputs_panel(app: VoiceTrainerApp) -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📁 Outputs")
    st.caption("Download your trained model files or export a shareable publish ZIP.")

    package_data = build_publish_zip()
    package_name = f"wakeword_lab_publish_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    st.download_button(
        label="📦 Create Publish ZIP",
        data=package_data,
        file_name=package_name,
        mime="application/zip",
        use_container_width=True,
        help="Bundles your source files and trained models into a single ZIP for sharing.",
    )

    destinations = parse_destination_paths(st.session_state.get("output_destinations_text", ""))
    if destinations:
        st.caption("Optional destinations configured:")
        for dest in destinations:
            st.write(f"📌 {dest}")

    if not OUTPUT_DIR.exists():
        st.info("No output folder yet. Run a step first.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    files = sorted([p for p in OUTPUT_DIR.rglob("*") if p.is_file()])
    if not files:
        st.info("No output files yet.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.markdown("---")
    st.subheader("🎚️ Runtime Threshold")
    st.caption("Save or update the detection threshold used when the trained model runs in your terminal or app.")

    model_names = list_trained_model_names()
    if model_names:
        current_model_name = st.session_state.get("model_name", "").strip()
        if current_model_name in model_names:
            default_index = model_names.index(current_model_name)
        else:
            default_index = 0

        if st.session_state.get("runtime_profile_model") not in model_names:
            st.session_state.runtime_profile_model = model_names[default_index]

        selected_runtime_model = st.selectbox(
            "Select trained model",
            model_names,
            index=default_index,
            key="runtime_profile_model",
        )

        if st.session_state.get("runtime_profile_loaded_model") != selected_runtime_model:
            existing_threshold = load_runtime_profile_threshold(selected_runtime_model)
            st.session_state.runtime_profile_threshold = (
                existing_threshold
                if existing_threshold is not None
                else VoiceTrainerApp.RUNTIME_THRESHOLD_DEFAULT
            )
            st.session_state.runtime_profile_loaded_model = selected_runtime_model

        runtime_profile_threshold = st.number_input(
            "Runtime activation threshold",
            key="runtime_profile_threshold",
            step=0.05,
            format="%.2f",
            min_value=VoiceTrainerApp.RUNTIME_THRESHOLD_MIN,
            max_value=VoiceTrainerApp.RUNTIME_THRESHOLD_MAX,
            help="Lower values trigger more easily. Higher values are stricter.",
        )

        if st.button("💾 Save Runtime Threshold", use_container_width=True):
            try:
                saved_profile = app.save_runtime_profile(
                    model_name=selected_runtime_model,
                    runtime_threshold=float(runtime_profile_threshold),
                )
                st.session_state.runtime_threshold = float(runtime_profile_threshold)
                st.success(f"Saved runtime profile: {saved_profile}")
            except Exception as exc:
                st.error(str(exc))

        selected_model_path = OUTPUT_DIR / f"{selected_runtime_model}.onnx"
        st.code(
            "\n".join(
                [
                    "from openwakeword.model import Model",
                    "",
                    f"model = Model(wakeword_models=[r\"{selected_model_path}\"])",
                    f"threshold = {float(runtime_profile_threshold):.2f}",
                    "",
                    "scores = model.predict(audio_chunk)",
                    f"if scores[\"{selected_runtime_model}\"] >= threshold:",
                    "    print(\"wake word detected\")",
                ]
            ),
            language="python",
        )
    else:
        st.info("Train a model first to save a runtime threshold profile.")

    exportable_suffixes = {".onnx", ".tflite", ".wav", ".yaml", ".pkl"}
    exportable_files = [f for f in files if f.suffix.lower() in exportable_suffixes]
    if destinations and exportable_files:
        if st.button("📤 Copy exportable outputs to destinations", use_container_width=True):
            for dest in destinations:
                copied_count = copy_files_to_destination(exportable_files, dest)
                st.success(f"Copied {copied_count} file(s) to: {dest}")

    _EXT_ICON = {
        ".onnx": "🧠",
        ".tflite": "📱",
        ".wav": "🔊",
        ".yaml": "⚙️",
        ".pkl": "🧬",
        ".npy": "📊",
        ".pb": "🔗",
    }
    for file_path in files:
        rel = file_path.relative_to(APP_DIR)
        icon = _EXT_ICON.get(file_path.suffix.lower(), "📄")
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"{icon} {str(rel)}")
        with col2:
            if file_path.suffix.lower() in exportable_suffixes:
                with open(file_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download",
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

    t1, t2, t3, t4, t5, t6 = st.tabs(["🎤 Phrase Lab", "📦 Data Setup", "🏋️ Training", "🧬 Personal Voice", "📁 Outputs", "🩺 Health"])
    with t1:
        phrase_lab(app)
    with t2:
        data_setup(app)
    with t3:
        training_lab(app)
    with t4:
        personal_voice_lab(app)
    with t5:
        outputs_panel(app)
    with t6:
        health_panel(app)


if __name__ == "__main__":
    main()
