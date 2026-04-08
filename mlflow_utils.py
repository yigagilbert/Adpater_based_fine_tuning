"""
mlflow_utils.py — Optional MLflow integration for training and evaluation
=========================================================================

This module keeps MLflow concerns out of the core training logic.

Design goals
------------
1. MLflow should be opt-in via environment variables rather than hard-coded
   interactive prompts in the pipeline.
2. Training should continue to work when MLflow is not installed or not
   configured.
3. When MLflow is enabled, we want both:
   * step-wise Trainer logging via `report_to="mlflow"`
   * explicit run metadata, tags, artifacts, and evaluation metrics

Expected environment variables
------------------------------
The intended setup is:

    export MLFLOW_TRACKING_USERNAME="..."
    export MLFLOW_TRACKING_PASSWORD="..."
    export MLFLOW_TRACKING_URI="https://mlflow.sunbird.ai/"
    export MLFLOW_EXPERIMENT_NAME="med-gemma-adpater-exp"

Per MLflow's tracking server documentation, HTTP basic authentication requires
both `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` to be set.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

try:
    import mlflow
except ImportError:  # pragma: no cover - dependency is optional at runtime
    mlflow = None


def is_mlflow_requested() -> bool:
    """Return True when the environment indicates MLflow logging should be used."""

    return bool(os.environ.get("MLFLOW_TRACKING_URI"))


def configure_mlflow() -> bool:
    """
    Configure MLflow from environment variables.

    Returns `True` only when MLflow is both requested and importable.
    """

    if not is_mlflow_requested():
        return False

    if mlflow is None:
        logger.warning(
            "MLflow logging was requested via MLFLOW_TRACKING_URI, but the `mlflow` "
            "package is not installed. Install it to enable remote experiment logging."
        )
        return False

    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME")
    username = os.environ.get("MLFLOW_TRACKING_USERNAME")
    password = os.environ.get("MLFLOW_TRACKING_PASSWORD")

    if bool(username) ^ bool(password):
        logger.warning(
            "MLflow basic auth is only partially configured. Set both "
            "MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD."
        )

    mlflow.set_tracking_uri(tracking_uri)
    if experiment_name:
        mlflow.set_experiment(experiment_name)

    return True


@contextmanager
def start_mlflow_run(
    run_name: str,
    tags: Optional[dict[str, Any]] = None,
    run_id: Optional[str] = None,
) -> Iterator[Optional[Any]]:
    """
    Start or resume an MLflow run if MLflow is enabled.

    If MLflow is not enabled, this yields `None` and behaves as a no-op context.
    """

    if not configure_mlflow():
        yield None
        return

    kwargs: dict[str, Any] = {}
    if run_id:
        kwargs["run_id"] = run_id
    else:
        kwargs["run_name"] = run_name
        if tags:
            kwargs["tags"] = tags

    with mlflow.start_run(**kwargs) as run:
        if tags:
            mlflow.set_tags(tags)
        yield run


def active_run_id() -> Optional[str]:
    """Return the active MLflow run id if one exists."""

    if mlflow is None:
        return None
    run = mlflow.active_run()
    return run.info.run_id if run is not None else None


def log_params(params: dict[str, Any]) -> None:
    """Log parameters after converting complex values into strings."""

    if mlflow is None or mlflow.active_run() is None:
        return

    normalized = {}
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            normalized[key] = value
        else:
            normalized[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)

    if normalized:
        mlflow.log_params(normalized)


def log_metrics(metrics: dict[str, Any], prefix: str = "") -> None:
    """Log numeric metrics, optionally namespaced with a prefix."""

    if mlflow is None or mlflow.active_run() is None:
        return

    normalized = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            normalized[f"{prefix}{key}"] = value

    if normalized:
        mlflow.log_metrics(normalized)


def log_dict(payload: dict[str, Any], artifact_file: str) -> None:
    """Log a small JSON-serializable payload as an MLflow artifact."""

    if mlflow is None or mlflow.active_run() is None:
        return

    mlflow.log_dict(payload, artifact_file)


def log_artifact(path: str | Path, artifact_path: Optional[str] = None) -> None:
    """Log a local file artifact if MLflow is active."""

    if mlflow is None or mlflow.active_run() is None:
        return

    mlflow.log_artifact(str(path), artifact_path=artifact_path)
