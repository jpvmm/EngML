import numpy as np
import pandas as pd
from google.cloud.aiplatform.prediction.sklearn.predictor import SklearnPredictor


class DefaultRiskPredictor(SklearnPredictor):
    def __init__(self):
        return

    def preprocess(self, prediction_input: np.ndarray) -> np.ndarray:
        """Performs preprocessing by checking if clarity feature is in abbreviated form."""

        inputs = super().preprocess(prediction_input)

        inputs = pd.DataFrame.from_dict(inputs)
        return inputs

    def postprocess(self, prediction_results: np.ndarray) -> dict:
        """Performs postprocessing by rounding predictions and converting to str."""

        return {"predictions": [f"${value}" for value in np.round(prediction_results)]}
