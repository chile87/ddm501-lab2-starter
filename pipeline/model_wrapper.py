"""
MLflow pyfunc wrapper for Surprise recommendation models.

Surprise models are not natively supported by any MLflow flavour, so they
cannot be logged with ``mlflow.<flavour>.log_model()``. This module wraps a
trained Surprise estimator in a ``mlflow.pyfunc.PythonModel`` so that it can be:

- logged as a *real* MLflow model (not just a loose pickle artifact),
- registered in the MLflow Model Registry via ``runs:/<run_id>/model``,
- reloaded anywhere with ``mlflow.pyfunc.load_model(...)`` and served.

This file is an addition to the starter template (no TODOs) - it exists so that
the Model Registry stage operates on a loadable model rather than a bare file.
"""

import logging
import pickle
from typing import Any, List

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import ModelSignature, infer_signature

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Column names the wrapper expects at inference time.
USER_COLUMN = "user_id"
ITEM_COLUMN = "item_id"

# Key under which the pickled Surprise estimator is stored in the MLflow model.
SURPRISE_ARTIFACT_KEY = "surprise_model"


class SurpriseRecommender(mlflow.pyfunc.PythonModel):
    """
    pyfunc wrapper exposing a Surprise estimator as a rating predictor.

    Input : DataFrame with columns ``user_id`` and ``item_id`` (raw ids).
    Output: 1-D numpy array of estimated ratings, one per input row.
    """

    def load_context(self, context: Any) -> None:
        """Load the pickled Surprise estimator when the model is deserialised."""
        model_path = context.artifacts[SURPRISE_ARTIFACT_KEY]
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        logger.info(f"Loaded Surprise model of type {type(self.model).__name__}")

    def predict(
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: Any = None,
    ) -> np.ndarray:
        """
        Predict ratings for a batch of (user, item) pairs.

        Args:
            context: MLflow python model context (unused; artifacts loaded already)
            model_input: DataFrame with ``user_id`` and ``item_id`` columns
            params: Unused, present for MLflow >= 2.8 signature compatibility

        Returns:
            numpy array of estimated ratings

        Raises:
            ValueError: If required columns are missing from ``model_input``
        """
        if isinstance(model_input, dict):
            model_input = pd.DataFrame(model_input)

        missing = {USER_COLUMN, ITEM_COLUMN} - set(model_input.columns)
        if missing:
            raise ValueError(
                f"model_input is missing required column(s): {sorted(missing)}. "
                f"Expected columns: ['{USER_COLUMN}', '{ITEM_COLUMN}']"
            )

        # Surprise stores raw ids as strings; cast defensively so that callers
        # may pass ints without silently getting the global-mean fallback.
        estimates: List[float] = [
            self.model.predict(str(uid), str(iid)).est
            for uid, iid in zip(model_input[USER_COLUMN], model_input[ITEM_COLUMN])
        ]
        return np.array(estimates, dtype=float)


def build_input_example() -> pd.DataFrame:
    """
    Build a small input example stored alongside the logged model.

    Returns:
        DataFrame with two example (user, item) pairs
    """
    return pd.DataFrame(
        {
            USER_COLUMN: ["196", "186"],
            ITEM_COLUMN: ["242", "302"],
        }
    )


def build_signature() -> ModelSignature:
    """
    Build the MLflow model signature for the recommender.

    Returns:
        ModelSignature describing the (user_id, item_id) -> rating contract
    """
    return infer_signature(
        model_input=build_input_example(),
        model_output=np.array([4.0, 3.5], dtype=float),
    )
