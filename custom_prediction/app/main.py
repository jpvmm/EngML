import logging
import os
import pickle
from typing import Dict

import pandas as pd
from fastapi import FastAPI, Request
from google.cloud.aiplatform.utils import prediction_utils

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOGGER_LEVEL", logging.INFO))

app = FastAPI()


@app.get("/healthcheck")
def health_check():
    return 200


@app.post("/predict")
async def predict(request: Request):
    """
    Predict default risk loading a trained model
    """

    logger.info(f"Artifact URI: {os.environ['AIP_STORAGE_URI']}")
    logger.info(f"Request: {request}")
    try:
        artifact_uri = os.environ["AIP_STORAGE_URI"]
        prediction_utils.download_model_artifacts(artifact_uri)
        model = pickle.load(open("model.pkl", "rb"))
    except Exception as e:
        raise ValueError(f"Not possible to load model from {artifact_uri}, {e}")
    body = await request.json()
    instances = body["instances"]
    inputs = pd.DataFrame(instances)
    logger.info(f"Dataframes being used: {inputs}")
    preds = model.predict(inputs).tolist()
    return_response = {"predictions": preds}

    return return_response
