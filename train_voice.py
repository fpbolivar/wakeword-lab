#!/usr/bin/env python3
"""Standalone wake-word training app.

This script provides a clean pipeline to:
1) Validate phrase generation (TTS sample)
2) Prepare datasets and openWakeWord training assets
3) Train and export your custom wake-word model
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from typing import Callable
from urllib.request import urlretrieve


def run_command(args: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    subprocess.run(args, cwd=str(cwd) if cwd else None, check=True, env=env)


class VoiceTrainerApp:
    N_SAMPLES_MIN = 100
    N_SAMPLES_MAX = 200000
    STEPS_MIN = 100
    STEPS_MAX = 100000
    FALSE_PENALTY_MIN = 100
    FALSE_PENALTY_MAX = 5000
    RUNTIME_THRESHOLD_DEFAULT = 0.5
    RUNTIME_THRESHOLD_MIN = 0.0
    RUNTIME_THRESHOLD_MAX = 1.0

    @staticmethod
    def estimate_training_time(
        n_samples: int,
        steps: int,
        n_phrases: int,
        device: str,
    ) -> dict:
        """Return a human-readable time estimate dict for the full training pipeline.

        Keys returned:
          clip_gen_sec  – TTS clip generation (CPU-bound, piper)
          feature_sec   – Feature extraction / augmentation
          train_sec     – Model training loop
          convert_sec   – ONNX → TFLite conversion (fixed ~30 s)
          total_sec     – Sum of all stages
          label         – Short human-readable string e.g. "~4 min 30 sec"
          breakdown     – List of (stage_name, human_label) for display
        """
        # -------------------------------------------------------------------
        # Empirical base rates measured on an M2 MacBook Pro (MPS/CPU similar
        # for these workloads because piper and feature extraction are CPU-only).
        # Device speed factors apply to the Torch training loop only.
        # -------------------------------------------------------------------
        SEC_PER_CLIP = 0.12          # piper TTS generation per clip (batch-assisted)
        SEC_PER_FEATURE = 0.0015     # feature extraction per sample (augmentation)
        SEC_PER_STEP_CPU = 0.022     # training loop per step on CPU
        SEC_PER_STEP_CUDA = 0.006    # training loop per step on CUDA GPU
        SEC_PER_STEP_MPS = 0.010     # training loop per step on Apple MPS
        CONVERT_SEC = 30.0           # onnx2tf conversion (roughly fixed)

        phrase_factor = 1.0 + (max(1, n_phrases) - 1) * 0.15
        total_clips = n_samples * phrase_factor

        clip_gen_sec = total_clips * SEC_PER_CLIP
        feature_sec = n_samples * SEC_PER_FEATURE * 8  # ~8 augmentations per sample

        dev = device.lower()
        if "cuda" in dev:
            sec_per_step = SEC_PER_STEP_CUDA
        elif dev == "mps":
            sec_per_step = SEC_PER_STEP_MPS
        else:
            sec_per_step = SEC_PER_STEP_CPU

        train_sec = steps * sec_per_step
        total_sec = clip_gen_sec + feature_sec + train_sec + CONVERT_SEC

        def _fmt(secs: float) -> str:
            secs = max(1, int(secs))
            if secs < 60:
                return f"~{secs} sec"
            minutes, s = divmod(secs, 60)
            if minutes < 60:
                return f"~{minutes} min {s} sec" if s else f"~{minutes} min"
            hours, m = divmod(minutes, 60)
            return f"~{hours} hr {m} min" if m else f"~{hours} hr"

        breakdown = [
            ("🎙️ Clip generation", _fmt(clip_gen_sec)),
            ("🔬 Feature extraction", _fmt(feature_sec)),
            ("🏋️ Model training", _fmt(train_sec)),
            ("📦 ONNX → TFLite", _fmt(CONVERT_SEC)),
        ]

        return {
            "clip_gen_sec": clip_gen_sec,
            "feature_sec": feature_sec,
            "train_sec": train_sec,
            "convert_sec": CONVERT_SEC,
            "total_sec": total_sec,
            "label": _fmt(total_sec),
            "breakdown": breakdown,
        }

    @classmethod
    def detect_available_devices(cls) -> list[dict]:
        """Return available compute devices with labels and descriptions.

        Each entry has keys: id (str passed to train()), label (display name),
        desc (short explanation shown to the user).
        """
        devices: list[dict] = [
            {
                "id": "cpu",
                "label": "🖥️ CPU",
                "desc": "Always available. Works everywhere. Slowest option.",
            }
        ]
        try:
            import torch

            if torch.cuda.is_available():
                try:
                    gpu_name = torch.cuda.get_device_name(0)
                except Exception:
                    gpu_name = "CUDA GPU"
                devices.append(
                    {
                        "id": "cuda:0",
                        "label": f"⚡ CUDA GPU — {gpu_name}",
                        "desc": "NVIDIA GPU via CUDA. Fastest for large training runs.",
                    }
                )

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                devices.append(
                    {
                        "id": "mps",
                        "label": "🍎 Mac GPU (MPS)",
                        "desc": "Apple Silicon unified GPU. Faster than CPU on M1/M2/M3/M4 Macs.",
                    }
                )
        except Exception:
            pass
        return devices

    def __init__(self, app_dir: Path) -> None:
        self.app_dir = app_dir
        self.third_party_dir = app_dir / "third_party"
        self.data_dir = app_dir / "data"
        self.output_dir = app_dir / "output"
        self.config_dir = app_dir / "config"

        self.piper_repo = self.third_party_dir / "piper-sample-generator"
        self.piper_model = self.piper_repo / "models" / "en_US-libritts_r-medium.pt"

        self.oww_repo = self.third_party_dir / "openwakeword"
        self.oww_models = self.oww_repo / "openwakeword" / "resources" / "models"
        self.oww_train_script = self.oww_repo / "openwakeword" / "train.py"
        self.oww_train_template = self.oww_repo / "examples" / "custom_model.yml"

        self.training_config = self.config_dir / "my_model.yaml"

    def ensure_dirs(self) -> None:
        self.third_party_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def install_packages(self, requirements: Path) -> None:
        run_command([sys.executable, "-m", "pip", "install", "-r", str(requirements)])

    def clone_if_missing(self, repo_url: str, target_dir: Path) -> None:
        if not target_dir.exists():
            run_command(["git", "clone", repo_url, str(target_dir)])

    def download_file(self, url: str, destination: Path, force: bool = False) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if force and destination.exists():
            destination.unlink()
        if not destination.exists():
            print(f"Downloading {destination.name}...")
            urlretrieve(url, destination)

    def ensure_tts_generator(self) -> None:
        self.clone_if_missing("https://github.com/rhasspy/piper-sample-generator", self.piper_repo)
        run_command(["git", "checkout", "213d4d5"], cwd=self.piper_repo)
        self.download_file(
            "https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt",
            self.piper_model,
        )

    def ensure_openwakeword(self) -> None:
        self.clone_if_missing("https://github.com/dscripka/openwakeword", self.oww_repo)
        run_command([sys.executable, "-m", "pip", "install", "-e", str(self.oww_repo), "--no-deps"])
        self._patch_openwakeword_train_script()

        model_downloads = {
            "embedding_model.onnx": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx",
            "embedding_model.tflite": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.tflite",
            "melspectrogram.onnx": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx",
            "melspectrogram.tflite": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.tflite",
        }

        self.oww_models.mkdir(parents=True, exist_ok=True)
        for name, url in model_downloads.items():
            self.download_file(url, self.oww_models / name)

    def _patch_openwakeword_train_script(self) -> None:
        train_py = self.oww_train_script
        if not train_py.exists():
            return

        source = train_py.read_text()
        updated = source

        # Fix argparse store_true defaults that should be boolean values.
        updated = updated.replace('default="False"', 'default=False')

        # Patch device selection to support MPS and the WAKEWORD_DEVICE env var.
        old_device_line = "        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')"
        new_device_block = (
            "        _ww_dev = os.environ.get(\"WAKEWORD_DEVICE\", \"auto\")\n"
            "        if _ww_dev == \"auto\":\n"
            "            if torch.cuda.is_available():\n"
            "                self.device = torch.device('cuda:0')\n"
            "            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():\n"
            "                self.device = torch.device('mps')\n"
            "            else:\n"
            "                self.device = torch.device('cpu')\n"
            "        else:\n"
            "            self.device = torch.device(_ww_dev)"
        )
        if old_device_line in updated:
            updated = updated.replace(old_device_line, new_device_block)

        # Avoid DataLoader multiprocessing pickling failures on macOS.
        old_block = """        n_cpus = os.cpu_count()\n        if n_cpus is None:\n            n_cpus = 1\n        else:\n            n_cpus = n_cpus//2\n        X_train = torch.utils.data.DataLoader(IterDataset(batch_generator),\n                                              batch_size=None, num_workers=n_cpus, prefetch_factor=16)\n"""
        new_block = """        n_cpus = 0\n        X_train = torch.utils.data.DataLoader(\n            IterDataset(batch_generator),\n            batch_size=None,\n            num_workers=n_cpus\n        )\n"""
        if old_block in updated:
            updated = updated.replace(old_block, new_block)

        if updated != source:
            train_py.write_text(updated)

    def _is_valid_npy(self, file_path: Path) -> bool:
        if not file_path.exists():
            return False
        try:
            import numpy as np

            np.load(file_path, mmap_mode="r")
            return True
        except Exception:
            return False

    def generate_test_clip(self, target_word: str) -> Path:
        import importlib.util
        import sys

        self.ensure_tts_generator()

        repo_path = str(self.piper_repo.resolve())
        if repo_path not in sys.path:
            sys.path.append(repo_path)

        module_path = self.piper_repo / "generate_samples.py"
        spec = importlib.util.spec_from_file_location("piper_generate_samples", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load {module_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        output_file = self.output_dir / "test_generation.wav"
        module.generate_samples(
            text=target_word,
            max_samples=1,
            length_scales=[1.1],
            noise_scales=[0.7],
            noise_scale_ws=[0.7],
            output_dir=str(self.output_dir),
            batch_size=1,
            auto_reduce_batch_size=True,
            file_names=[output_file.name],
        )
        return output_file

    def prepare_data(self, include_streaming_sets: bool) -> None:
        self.ensure_dirs()
        self.ensure_openwakeword()

        self.download_file(
            "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
            self.data_dir / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
            force=not self._is_valid_npy(self.data_dir / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"),
        )
        self.download_file(
            "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy",
            self.data_dir / "validation_set_features.npy",
            force=not self._is_valid_npy(self.data_dir / "validation_set_features.npy"),
        )

        self._prepare_mit_rirs()
        if include_streaming_sets:
            self._prepare_streaming_sets()

    def _prepare_mit_rirs(self) -> None:
        import numpy as np
        import scipy.io.wavfile
        import soundfile as sf
        from tqdm import tqdm

        rir_repo = self.third_party_dir / "MIT_environmental_impulse_responses"
        rir_output = self.data_dir / "mit_rirs"

        if rir_output.exists() and any(rir_output.glob("*.wav")):
            print("MIT RIR data already present.")
            return

        run_command(["git", "lfs", "install"])
        self.clone_if_missing(
            "https://huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses",
            rir_repo,
        )

        rir_output.mkdir(parents=True, exist_ok=True)
        rir_files = sorted((rir_repo / "16khz").glob("*.wav"))
        for wav_path in tqdm(rir_files):
            audio_array, sample_rate = sf.read(wav_path, dtype="float32")
            if getattr(audio_array, "ndim", 1) > 1:
                audio_array = np.mean(audio_array, axis=1)

            if sample_rate != 16000:
                raise ValueError(f"Unexpected sample rate {sample_rate} in {wav_path}")

            name = wav_path.name
            scipy.io.wavfile.write(
                rir_output / name,
                16000,
                (audio_array * 32767).astype(np.int16),
            )

    def _prepare_streaming_sets(self) -> None:
        import datasets
        import librosa
        import numpy as np
        import scipy.io.wavfile
        import soundfile as sf
        from tqdm import tqdm

        def load_audio_array(audio_data: dict, target_sr: int = 16000):
            audio_bytes = audio_data.get("bytes")
            audio_path = audio_data.get("path")

            if audio_bytes is not None:
                audio_array, sample_rate = sf.read(BytesIO(audio_bytes), dtype="float32")
            elif audio_path:
                audio_array, sample_rate = sf.read(audio_path, dtype="float32")
            else:
                raise ValueError("Audio data had neither bytes nor path")

            if getattr(audio_array, "ndim", 1) > 1:
                audio_array = np.mean(audio_array, axis=1)

            if sample_rate != target_sr:
                audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=target_sr)

            return audio_array.astype(np.float32)

        audioset_out = self.data_dir / "audioset_16k"
        if not (audioset_out.exists() and any(audioset_out.glob("*.wav"))):
            audioset_out.mkdir(parents=True, exist_ok=True)
            ds = datasets.load_dataset("agkphysics/AudioSet", "balanced", split="train", streaming=True)
            ds_iter = iter(ds.cast_column("audio", datasets.Audio(decode=False)))
            for _ in tqdm(range(500)):
                row = next(ds_iter)
                name = Path(row["audio"]["path"]).name.replace(".flac", ".wav")
                arr = load_audio_array(row["audio"])
                scipy.io.wavfile.write(audioset_out / name, 16000, (arr * 32767).astype(np.int16))
        else:
            print("AudioSet sample clips already present.")

        fma_out = self.data_dir / "fma"
        if not (fma_out.exists() and any(fma_out.glob("*.wav"))):
            fma_out.mkdir(parents=True, exist_ok=True)
            ds = datasets.load_dataset("rudraml/fma", name="small", split="train", streaming=True)
            ds_iter = iter(ds.cast_column("audio", datasets.Audio(decode=False)))
            for _ in tqdm(range(120)):
                row = next(ds_iter)
                name = Path(row["audio"]["path"]).name.replace(".mp3", ".wav")
                arr = load_audio_array(row["audio"])
                scipy.io.wavfile.write(fma_out / name, 16000, (arr * 32767).astype(np.int16))
        else:
            print("FMA sample clips already present.")

    def train(
        self,
        model_name: str,
        target_phrases: list[str],
        n_samples: int,
        steps: int,
        false_activation_penalty: int,
        runtime_threshold: float = RUNTIME_THRESHOLD_DEFAULT,
        device: str = "auto",
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> None:
        import yaml

        self._validate_training_inputs(target_phrases, n_samples, steps, false_activation_penalty)
        self._validate_runtime_threshold(runtime_threshold)

        if progress_callback:
            progress_callback(0.02, "Preparing training environment")

        self.ensure_dirs()
        self.ensure_openwakeword()

        if not self.oww_train_template.exists():
            raise FileNotFoundError("openWakeWord template config missing. Run prepare-data first.")

        config = yaml.load(self.oww_train_template.read_text(), yaml.Loader)

        feature_file = self.data_dir / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
        validation_file = self.data_dir / "validation_set_features.npy"

        config["target_phrase"] = target_phrases
        config["model_name"] = model_name
        config["n_samples"] = n_samples
        config["n_samples_val"] = max(500, n_samples // 10)
        config["steps"] = steps
        config["target_false_positives_per_hour"] = 0.2
        config["target_accuracy"] = 0.5
        config["target_recall"] = 0.25
        config["output_dir"] = str(self.output_dir)
        config["max_negative_weight"] = false_activation_penalty
        config["false_positive_validation_data_path"] = str(validation_file)
        config["feature_data_files"] = {}

        background_paths = []
        for folder in [self.data_dir / "audioset_16k", self.data_dir / "fma"]:
            if folder.exists() and any(folder.glob("*.wav")):
                background_paths.append(str(folder))
        config["background_paths"] = background_paths

        rir_dir = self.data_dir / "mit_rirs"
        if rir_dir.exists() and any(rir_dir.glob("*.wav")):
            config["rir_paths"] = [str(rir_dir)]

        config["batch_n_per_class"] = {"adversarial_negative": 50, "positive": 50}
        if self._is_valid_npy(feature_file):
            config["feature_data_files"]["ACAV100M_sample"] = str(feature_file)
            config["batch_n_per_class"]["ACAV100M_sample"] = 1024

        self.training_config.write_text(yaml.safe_dump(config, sort_keys=False))

        env = os.environ.copy()
        piper_path = str(self.piper_repo.resolve())
        env["PYTHONPATH"] = (
            piper_path if not env.get("PYTHONPATH") else f"{piper_path}{os.pathsep}{env['PYTHONPATH']}"
        )

        # Pass compute device choice to the openWakeWord training subprocess.
        env["WAKEWORD_DEVICE"] = device
        if device == "mps":
            # Required for ops that don't have native MPS kernels.
            env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        elif device == "cpu":
            # Prevent accidental CUDA use even if the driver is present.
            env["CUDA_VISIBLE_DEVICES"] = ""

        stage_items = [
            ("--generate_clips", "Generating synthetic clips"),
            ("--augment_clips", "Computing training features"),
            ("--train_model", "Training model weights"),
        ]

        for index, (stage_flag, stage_label) in enumerate(stage_items):
            if progress_callback:
                progress_callback(0.1 + index * 0.25, stage_label)
            run_command(
                [
                    sys.executable,
                    str(self.oww_train_script),
                    "--training_config",
                    str(self.training_config),
                    stage_flag,
                ],
                cwd=self.app_dir,
                env=env,
            )

        if progress_callback:
            progress_callback(0.88, "Converting ONNX model to TFLite")
        self._convert_to_tflite(model_name)

        onnx_file = self.output_dir / f"{model_name}.onnx"
        tflite_file = self.output_dir / f"{model_name}.tflite"
        runtime_profile = self.save_runtime_profile(
            model_name=model_name,
            runtime_threshold=runtime_threshold,
            target_phrases=target_phrases,
        )

        print(f"Created model: {onnx_file}")
        print(f"Created model: {tflite_file}")
        print(f"Created runtime profile: {runtime_profile}")
        if progress_callback:
            progress_callback(1.0, "Training complete")

    def train_personal_verifier(
        self,
        model_name: str,
        positive_reference_dir: str | Path,
        negative_reference_dir: str | Path,
        output_path: str | Path | None = None,
    ) -> Path:
        self.ensure_dirs()
        self.ensure_openwakeword()

        from openwakeword.custom_verifier_model import train_custom_verifier

        positive_dir = Path(positive_reference_dir).expanduser()
        negative_dir = Path(negative_reference_dir).expanduser()
        if not positive_dir.is_absolute():
            positive_dir = (self.app_dir / positive_dir).resolve()
        if not negative_dir.is_absolute():
            negative_dir = (self.app_dir / negative_dir).resolve()

        if not positive_dir.exists():
            raise FileNotFoundError(f"Positive reference folder not found: {positive_dir}")
        if not negative_dir.exists():
            raise FileNotFoundError(f"Negative reference folder not found: {negative_dir}")

        positive_clips = sorted(str(path) for path in positive_dir.rglob("*.wav"))
        negative_clips = sorted(str(path) for path in negative_dir.rglob("*.wav"))
        if not positive_clips:
            raise ValueError("No .wav files found in the positive reference folder.")
        if not negative_clips:
            raise ValueError("No .wav files found in the negative reference folder.")

        base_model = Path(model_name).expanduser()
        if not base_model.exists():
            candidate = self.output_dir / f"{model_name}.onnx"
            if candidate.exists():
                base_model = candidate
            else:
                raise FileNotFoundError(
                    f"Base model not found. Expected {candidate} or an explicit ONNX path."
                )

        if output_path is None:
            verifier_output = self.output_dir / f"{base_model.stem}_verifier.pkl"
        else:
            verifier_output = Path(output_path).expanduser()
            if not verifier_output.is_absolute():
                verifier_output = (self.app_dir / verifier_output).resolve()

        verifier_output.parent.mkdir(parents=True, exist_ok=True)
        train_custom_verifier(
            positive_reference_clips=positive_clips,
            negative_reference_clips=negative_clips,
            output_path=str(verifier_output),
            model_name=str(base_model),
        )
        return verifier_output

    def _validate_training_inputs(
        self,
        target_phrases: list[str],
        n_samples: int,
        steps: int,
        false_activation_penalty: int,
    ) -> None:
        cleaned_phrases = [p.strip() for p in target_phrases if p.strip()]
        if not cleaned_phrases:
            raise ValueError("At least one target phrase is required.")

        if not (self.N_SAMPLES_MIN <= n_samples <= self.N_SAMPLES_MAX):
            raise ValueError(
                f"n_samples must be between {self.N_SAMPLES_MIN} and {self.N_SAMPLES_MAX}."
            )

        if not (self.STEPS_MIN <= steps <= self.STEPS_MAX):
            raise ValueError(
                f"steps must be between {self.STEPS_MIN} and {self.STEPS_MAX}."
            )

        if not (self.FALSE_PENALTY_MIN <= false_activation_penalty <= self.FALSE_PENALTY_MAX):
            raise ValueError(
                f"false_activation_penalty must be between {self.FALSE_PENALTY_MIN} and {self.FALSE_PENALTY_MAX}."
            )

    def _validate_runtime_threshold(self, runtime_threshold: float) -> None:
        if not (self.RUNTIME_THRESHOLD_MIN <= runtime_threshold <= self.RUNTIME_THRESHOLD_MAX):
            raise ValueError(
                f"runtime_threshold must be between {self.RUNTIME_THRESHOLD_MIN} and {self.RUNTIME_THRESHOLD_MAX}."
            )

    def save_runtime_profile(
        self,
        model_name: str,
        runtime_threshold: float,
        target_phrases: list[str] | None = None,
    ) -> Path:
        import yaml

        self.ensure_dirs()
        self._validate_runtime_threshold(runtime_threshold)

        model_stem = Path(model_name).stem
        runtime_profile = self.output_dir / f"{model_stem}_runtime.yaml"
        verifier_file = self.output_dir / f"{model_stem}_verifier.pkl"

        profile = {
            "model_name": model_stem,
            "model_file": f"{model_stem}.onnx",
            "runtime_threshold": float(runtime_threshold),
        }
        if target_phrases:
            profile["target_phrases"] = [phrase.strip() for phrase in target_phrases if phrase.strip()]
        if verifier_file.exists():
            profile["verifier_file"] = verifier_file.name

        runtime_profile.write_text(yaml.safe_dump(profile, sort_keys=False))
        return runtime_profile

    def _convert_to_tflite(self, model_name: str) -> None:
        onnx_file = self.output_dir / f"{model_name}.onnx"
        float32_file = self.output_dir / f"{model_name}_float32.tflite"
        final_file = self.output_dir / f"{model_name}.tflite"

        run_command(
            [
                "onnx2tf",
                "-i",
                str(onnx_file),
                "-o",
                str(self.output_dir),
                "-nuo",
                "-kat",
                "onnx____Flatten_0",
            ],
            cwd=self.app_dir,
        )

        if float32_file.exists():
            float32_file.replace(final_file)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WakeWord Lab standalone wake-word trainer")
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install Python dependencies from requirements.txt before running the selected command.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_test = subparsers.add_parser("test-phrase", help="Generate one sample WAV for your wake phrase")
    p_test.add_argument("--target-word", required=True, help="Example: hey_att_la")

    p_data = subparsers.add_parser("prepare-data", help="Download training assets and optional sample background audio")
    p_data.add_argument(
        "--include-streaming-sets",
        action="store_true",
        help="Also download sample AudioSet and FMA clips (slower, recommended for better robustness).",
    )

    p_train = subparsers.add_parser("train", help="Train and export ONNX/TFLite wake-word model")
    p_train.add_argument("--model-name", required=True, help="Output model name (without extension)")
    p_train.add_argument(
        "--target-phrases",
        nargs="+",
        required=True,
        help="Space-separated phrase variants, e.g. hey_att_la hey_at_luh",
    )
    p_train.add_argument("--n-samples", type=int, default=5000)
    p_train.add_argument("--steps", type=int, default=3000)
    p_train.add_argument("--false-activation-penalty", type=int, default=600)
    p_train.add_argument(
        "--runtime-threshold",
        type=float,
        default=VoiceTrainerApp.RUNTIME_THRESHOLD_DEFAULT,
        help=(
            "Runtime activation threshold saved with the trained model for deployment use. "
            "Must be between 0.0 and 1.0 and does not change training itself."
        ),
    )

    return parser


def main() -> None:
    app_dir = Path(__file__).resolve().parent
    app = VoiceTrainerApp(app_dir)
    parser = build_parser()
    args = parser.parse_args()

    requirements = app_dir / "requirements.txt"
    if args.install_deps:
        app.install_packages(requirements)

    if args.command == "test-phrase":
        app.ensure_dirs()
        output_file = app.generate_test_clip(args.target_word)
        print(f"Generated test clip: {output_file}")
    elif args.command == "prepare-data":
        app.prepare_data(include_streaming_sets=args.include_streaming_sets)
        print("Data preparation complete.")
    elif args.command == "train":
        app.train(
            model_name=args.model_name,
            target_phrases=args.target_phrases,
            n_samples=args.n_samples,
            steps=args.steps,
            false_activation_penalty=args.false_activation_penalty,
            runtime_threshold=args.runtime_threshold,
        )


if __name__ == "__main__":
    main()
