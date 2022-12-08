import json
from typing import Dict, List, Union

from google.cloud import aiplatform

PROJECT_ID = "qacomp"
REGION = "us-east1"
# Define variables
job_display_name = "default-credit-prediction-job"
MODEL_NAME = "defaultrisk-rf"
ENDPOINT_NAME = "default_risk_v2"
BUCKET_URI = "gs://preprocessed_infer"
input_file_name = "preds_test2.jsonl"
MODEL_ID = "4557695599457075200"
ENDPOINT_ID = "32176108275236864"

# Read data to send the request
with open("data/preds_test.json", "rb") as f:
    data = json.load(f)


def predict_custom_trained_model_sample(
    project: str,
    endpoint_id: str,
    data: Union[Dict, List[Dict]],
    location: str = "us-east1",
    api_endpoint: str = "us-east1-aiplatform.googleapis.com",
):
    """
    `instances` can be either single instance of type dict or a list
    of instances.
    """
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    # The format of each instance should conform to the deployed model's prediction input schema.
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=data["instances"], parameters=data["parameters"]
    )
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)
    # The predictions are a google.protobuf.Value representation of the model's predictions.
    predictions = response.predictions
    print(f"Predicted labels: {predictions}")
    return predictions


predict_custom_trained_model_sample(
    project="qacomp", endpoint_id=ENDPOINT_ID, data=data
)
