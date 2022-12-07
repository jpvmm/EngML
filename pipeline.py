from kfp.v2 import dsl

from components.bigquery_reader.component import compare_results_in_bigquery
from components.model_deployment.component import deploy_model
from components.read_data.component import read_and_process_data
from components.train_model.component import train_model

BUCKET_NAME = "gs://default_pipeline"
PIPELINE_ROOT = f"{BUCKET_NAME}/pipeline_root/"
PROJECT_ID = "qacomp"
REGION = "us-east1"


@dsl.pipeline(
    name="uci-testv1",
    description="Testing",
    pipeline_root=PIPELINE_ROOT,
)
def model_training(
    input_path: str = "/gcs/default_pipeline/credit_card_default.csv",
    project: str = PROJECT_ID,
    region: str = REGION,
    serving_container: str = "us-east1-docker.pkg.dev/qacomp/custom-predictor-repo/custom-predictor:latest",  # noqa E501
):
    dataset = read_and_process_data(input_path=input_path)
    train_task = train_model(
        train_set=dataset.outputs["train"], test_set=dataset.outputs["test"]
    )
    deploy = compare_results_in_bigquery(score=train_task.outputs["score"])

    with dsl.Condition(
        deploy.outputs["deploy"] == "true",
        name="model-comparison",
    ):

        deploy_model_op = deploy_model(
            model=train_task.outputs["model"],
            project=project,
            region=region,
            serving_container_image_uri=serving_container,
        )
