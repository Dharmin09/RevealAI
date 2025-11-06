"""Continuous learning system for logging predictions and triggering auto-retraining."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class ContinuousLearningSystem:
    """Tracks model predictions, handles feedback, and launches retraining jobs."""

    def __init__(self, learning_dir: Optional[Path] = None, min_samples_for_retrain: int = 80) -> None:
        if learning_dir is None:
            learning_dir = Path(__file__).parent.parent / "learning_data"

        self.learning_dir = Path(learning_dir)
        self.learning_dir.mkdir(parents=True, exist_ok=True)

        self.predictions_log = self.learning_dir / "predictions_log.json"
        self.corrections_log = self.learning_dir / "corrections_log.json"
        self.accuracy_log = self.learning_dir / "accuracy_log.json"
        self.uploads_dir = self.learning_dir / "uploads"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

        self.lock = threading.Lock()
        self.min_samples_for_retrain = int(min_samples_for_retrain)
        self.retraining_state: Dict[str, Any] = {
            "status": "idle",
            "started_at": None,
            "completed_at": None,
            "dataset": None,
            "processes": [],
            "exit_codes": [],
        }
        self._retraining_processes: List[Dict[str, Any]] = []

        self.predictions: List[Dict[str, Any]] = []
        self.corrections: List[Dict[str, Any]] = []
        self.accuracy_history: List[Dict[str, Any]] = []

        self.load_logs()

    # ------------------------------------------------------------------
    # Persisted state helpers
    # ------------------------------------------------------------------

    def load_logs(self) -> None:
        with self.lock:
            self.predictions = self._load_json(self.predictions_log)
            self.corrections = self._load_json(self.corrections_log)
            self.accuracy_history = self._load_json(self.accuracy_log)

    def _load_json(self, filepath: Path) -> List[Dict[str, Any]]:
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as handle:
                return json.load(handle)
        return []

    def _save_json(self, filepath: Path, data: Any) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

    # ------------------------------------------------------------------
    # Logging predictions & corrections
    # ------------------------------------------------------------------

    def log_prediction(
        self,
        file_path: str,
        prediction: float,
        confidence: float,
        prediction_type: str = "video",
    ) -> Dict[str, Any]:
        """Persist a model prediction and return the stored record."""
        safe_type = prediction_type.lower() if prediction_type else "video"
        with self.lock:
            entry = {
                "id": len(self.predictions),
                "timestamp": datetime.now().isoformat(),
                "file": str(file_path),
                "type": safe_type,
                "prediction": float(prediction),
                "confidence": float(confidence),
                "ground_truth": None,
                "is_correct": None,
            }
            self.predictions.append(entry)
            self._save_json(self.predictions_log, self.predictions)
        return entry

    def log_correction(
        self,
        prediction_id: int,
        ground_truth: float,
        reason: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Attach human feedback to a prediction."""
        with self.lock:
            if prediction_id >= len(self.predictions) or prediction_id < 0:
                return None

            prediction = self.predictions[prediction_id]
            prediction["ground_truth"] = float(ground_truth)
            prediction["is_correct"] = (
                (prediction["prediction"] > 0.5 and ground_truth > 0.5)
                or (prediction["prediction"] <= 0.5 and ground_truth <= 0.5)
            )

            correction_entry = {
                "prediction_id": int(prediction_id),
                "timestamp": datetime.now().isoformat(),
                "prediction_value": float(prediction["prediction"]),
                "ground_truth": float(ground_truth),
                "was_correct": prediction["is_correct"],
                "reason": reason,
                "type": prediction.get("type", "unknown"),
            }

            self.corrections.append(correction_entry)
            self._save_json(self.predictions_log, self.predictions)
            self._save_json(self.corrections_log, self.corrections)
            return correction_entry

    def cache_media_file(self, source_path: str, desired_name: Optional[str] = None) -> str:
        """Persist a copy of analyzed media for future retraining."""
        if not source_path or not os.path.exists(source_path):
            return source_path

        try:
            original_name = Path(desired_name or source_path).name
            safe_name = original_name.replace(os.sep, "_").replace(" ", "_")
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            destination = self.uploads_dir / f"{timestamp}_{safe_name}"
            counter = 1
            while destination.exists():
                destination = self.uploads_dir / f"{timestamp}_{counter}_{safe_name}"
                counter += 1
            shutil.copy2(source_path, destination)
            return str(destination)
        except Exception as error:
            print(f"[LEARNING] Failed to cache media file {source_path}: {error}")
            return source_path

    # ------------------------------------------------------------------
    # Metrics & diagnostics
    # ------------------------------------------------------------------

    def get_current_accuracy(self, window_size: int = 100) -> Dict[str, Any]:
        with self.lock:
            recent_predictions = [
                p for p in self.predictions if p.get("ground_truth") is not None
            ][-window_size:]

        if not recent_predictions:
            return {
                "accuracy": 0,
                "total": 0,
                "correct": 0,
                "incorrect": 0,
                "false_positive_rate": 0,
                "false_negative_rate": 0,
                "window_size": window_size,
                "requires_retrain": False,
            }

        correct = sum(1 for p in recent_predictions if p.get("is_correct"))
        incorrect = len(recent_predictions) - correct

        false_positives = sum(
            1
            for p in recent_predictions
            if p["prediction"] > 0.5 and p["ground_truth"] <= 0.5
        )
        false_negatives = sum(
            1
            for p in recent_predictions
            if p["prediction"] <= 0.5 and p["ground_truth"] > 0.5
        )

        accuracy_value = round((correct / len(recent_predictions)) * 100, 2)
        return {
            "accuracy": accuracy_value,
            "total": len(recent_predictions),
            "correct": correct,
            "incorrect": incorrect,
            "false_positive_rate": round((false_positives / len(recent_predictions)) * 100, 2),
            "false_negative_rate": round((false_negatives / len(recent_predictions)) * 100, 2),
            "window_size": window_size,
            "requires_retrain": accuracy_value < 93.0,
        }

    def get_improvement_metrics(self) -> Dict[str, Any]:
        with self.lock:
            if len(self.predictions) < 20:
                return {
                    "insufficient_data": True,
                    "predictions_needed": max(0, 20 - len(self.predictions)),
                }

            mid = len(self.predictions) // 2
            first_half = [
                p for p in self.predictions[:mid] if p.get("ground_truth") is not None
            ]
            second_half = [
                p for p in self.predictions[mid:] if p.get("ground_truth") is not None
            ]

        def calc_accuracy(preds: List[Dict[str, Any]]) -> float:
            if not preds:
                return 0.0
            correct_local = sum(1 for p in preds if p.get("is_correct"))
            return (correct_local / len(preds)) * 100

        first_accuracy = calc_accuracy(first_half)
        second_accuracy = calc_accuracy(second_half)
        return {
            "first_period_accuracy": round(first_accuracy, 2),
            "second_period_accuracy": round(second_accuracy, 2),
            "improvement": round(second_accuracy - first_accuracy, 2),
            "is_improving": second_accuracy > first_accuracy,
        }

    def get_problem_areas(self, top_n: int = 5) -> List[Dict[str, Any]]:
        with self.lock:
            incorrect = [p for p in self.predictions if p.get("is_correct") is False]

        file_errors: Dict[str, int] = {}
        for pred in incorrect:
            file_path = pred.get("file", "unknown")
            file_errors[file_path] = file_errors.get(file_path, 0) + 1

        problem_areas = sorted(
            file_errors.items(), key=lambda item: item[1], reverse=True
        )[:top_n]

        return [
            {"file": path, "errors": count} for path, count in problem_areas
        ]

    # ------------------------------------------------------------------
    # Retraining workflow
    # ------------------------------------------------------------------

    def auto_retrain_if_needed(self, force: bool = False) -> Dict[str, Any]:
        """Evaluate accuracy and start retraining when performance drops or when forced."""
        self._refresh_retraining_state()
        accuracy = self.get_current_accuracy()
        if not force and not accuracy.get("requires_retrain"):
            return {
                "status": "idle",
                "reason": "Accuracy within threshold",
                "accuracy": accuracy,
            }

        dataset_info = self.prepare_retrain_dataset(self.learning_dir / "retrain_queue")
        if dataset_info.get("status") != "ready":
            return {
                "status": "pending",
                "required": dataset_info.get("required"),
                "missing": dataset_info.get("missing"),
                "samples": dataset_info.get("samples"),
                "accuracy": accuracy,
            }

        modalities = dataset_info.get("types", [])
        retrain_state = self._start_retraining_processes(
            dataset_file=dataset_info["file"], modalities=modalities
        )
        retrain_state["accuracy"] = accuracy
        return retrain_state

    def prepare_retrain_dataset(
        self,
        output_dir: Path,
        min_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        if min_samples is None:
            min_samples = self.min_samples_for_retrain

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with self.lock:
            corrected_files: List[Dict[str, Any]] = []
            for correction in self.corrections:
                pred_id = correction.get("prediction_id")
                if pred_id is None or pred_id >= len(self.predictions):
                    continue
                pred = self.predictions[pred_id]
                corrected_files.append(
                    {
                        "file": pred.get("file"),
                        "label": int(correction.get("ground_truth", 0)),
                        "original_prediction": float(pred.get("prediction", 0.5)),
                        "correction_reason": correction.get("reason", ""),
                        "type": pred.get("type", "unknown"),
                    }
                )

        modality_set = sorted({item.get("type", "unknown") for item in corrected_files})

        if len(corrected_files) < min_samples:
            return {
                "status": "insufficient_data",
                "samples": len(corrected_files),
                "required": min_samples,
                "missing": max(0, min_samples - len(corrected_files)),
                "types": modality_set,
            }

        output_file = output_dir / "retrain_data.json"
        with open(output_file, "w", encoding="utf-8") as handle:
            json.dump(corrected_files, handle, indent=2)

        return {
            "status": "ready",
            "samples": len(corrected_files),
            "file": str(output_file),
            "types": modality_set,
        }

    def _start_retraining_processes(
        self,
        dataset_file: str,
        modalities: List[str],
    ) -> Dict[str, Any]:
        if self.retraining_state.get("status") == "running":
            return {
                **self.retraining_state,
                "warning": "Retraining already running",
            }

        python_exe = sys.executable or "python"
        core_dir = Path(__file__).parent
        project_root = core_dir.parent
        commands: List[Dict[str, Any]] = []

        if "video" in modalities:
            commands.append(
                {
                    "type": "video",
                    "cmd": [python_exe, str(core_dir / "train_video.py"), "--corrections-json", dataset_file],
                }
            )
        if "audio" in modalities:
            commands.append(
                {
                    "type": "audio",
                    "cmd": [python_exe, str(core_dir / "train_audio.py"), "--corrections-json", dataset_file],
                }
            )

        if not commands:
            return {"status": "idle", "reason": "No matching modalities for retraining"}

        launched_processes: List[Dict[str, Any]] = []
        creation_flags = 0
        if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
            creation_flags = subprocess.CREATE_NO_WINDOW

        for cmd in commands:
            try:
                proc = subprocess.Popen(
                    cmd["cmd"],
                    cwd=str(project_root),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                    creationflags=creation_flags,
                )
                launched_processes.append({"type": cmd["type"], "proc": proc})
            except Exception as error:  # pragma: no cover - defensive
                print(f"[LEARNING] Failed to start {cmd['type']} retraining: {error}")

        if not launched_processes:
            return {"status": "failed", "reason": "Unable to start retraining jobs"}

        self._retraining_processes = launched_processes
        self.retraining_state = {
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
            "dataset": dataset_file,
            "processes": [
                {"type": info["type"], "pid": info["proc"].pid}
                for info in launched_processes
            ],
            "exit_codes": [],
        }
        return dict(self.retraining_state)

    def _refresh_retraining_state(self) -> None:
        if not self._retraining_processes:
            return

        still_running = False
        exit_codes: List[Dict[str, Any]] = []
        for info in list(self._retraining_processes):
            proc = info["proc"]
            ret = proc.poll()
            if ret is None:
                still_running = True
                continue
            exit_codes.append({"type": info["type"], "code": int(ret)})

        if still_running:
            return

        success = all(code["code"] == 0 for code in exit_codes) if exit_codes else False
        self.retraining_state.update(
            {
                "status": "completed" if success else "failed",
                "completed_at": datetime.now().isoformat(),
                "exit_codes": exit_codes,
            }
        )
        self._retraining_processes = []

    # ------------------------------------------------------------------
    # Status payload for API consumers
    # ------------------------------------------------------------------

    def get_status_summary(self) -> Dict[str, Any]:
        self._refresh_retraining_state()
        return {
            "current_performance": self.get_current_accuracy(),
            "improvement_metrics": self.get_improvement_metrics(),
            "problem_areas": self.get_problem_areas(),
            "total_predictions_logged": len(self.predictions),
            "total_corrections_made": len(self.corrections),
            "retraining": self.retraining_state,
        }


_LEARNING_SYSTEM: Optional[ContinuousLearningSystem] = None


def get_learning_system() -> ContinuousLearningSystem:
    global _LEARNING_SYSTEM
    if _LEARNING_SYSTEM is None:
        _LEARNING_SYSTEM = ContinuousLearningSystem()
    return _LEARNING_SYSTEM
