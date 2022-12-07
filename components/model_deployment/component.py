from kfp.v2.dsl import Artifact, Input, Model, Output, component


@component(
    packages_to_install=[
        "google-cloud-aiplatform",
        "scikit-learn==1.0.0",
        "kfp",
    ],  # noqa E501
    base_image="python:3.9",
)
def deploy_model(
    model: Input[Model],
    project: str,
    region: str,
    serving_container_image_uri: str,
    vertex_endpoint: Output[Artifact],
    vertex_model: Output[Model],
):
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=region)

    DISPLAY_NAME = "DefaultRisk"
    # MODEL_NAME = "defaultrisk-rf"
    ENDPOINT_NAME = "default_risk_v2"

    def create_endpoint(project: str = project, region: str = region) -> str:
        """Will list endpoints in the region and update it for the current
        model in case of deployment
        :param project: GCP Project for this operation
        :param regio: GCP Region for this operation"""
        endpoints = aiplatform.Endpoint.list(
            filter='display_name="{}"'.format(ENDPOINT_NAME),
            order_by="create_time desc",
            project=project,
            location=region,
        )
        if len(endpoints) > 0:
            endpoint = endpoints[0]  # will update the endpoint
        else:
            endpoint = aiplatform.Endpoint.create(
                display_name=ENDPOINT_NAME, project=project, location=region
            )
        return endpoint

    endpoint = create_endpoint()

    # Import a model programmatically
    model_upload = aiplatform.Model.upload(
        display_name=DISPLAY_NAME,
        artifact_uri=model.uri.replace("/model", "/"),
        serving_container_image_uri=serving_container_image_uri,
        serving_container_health_route="/healthcheck",
        serving_container_predict_route="/predict",
        serving_container_ports=[8050],
    )
    model_deploy = model_upload.deploy(
        machine_type="n1-standard-4",
        endpoint=endpoint,
        traffic_split={"0": 100},
        deployed_model_display_name=DISPLAY_NAME,
    )

    # Save data to the output params
    vertex_model.uri = model_deploy.resource_name
