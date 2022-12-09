import argparse
import json
from datetime import datetime

from google.cloud import aiplatform
from kfp.v2 import compiler

from pipeline.pipeline import model_training

PACKAGE_PATH = "pipeline_packages/uci.json"
TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")


def compile_pipeline(package_path: str):
    """Will compile the pipeline for job submitting
    Is used when something changes in the pipeline"""

    compiler.Compiler().compile(pipeline_func=model_training, package_path=package_path)
    print(f"Pipeline compiled to: {package_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--package_path",
        nargs="?",
        const=1,
        default="pipeline_packages/uci.json",
        type=str,
    )
    parser.add_argument(
        "--service_account",
        nargs="?",
        const=1,
        default="vertex@qacomp.iam.gserviceaccount.com",
        type=str,
    )
    parser.add_argument("--compile", nargs="?", const=1, default=False, type=bool)
    parser.add_argument(
        "--parameters", nargs="?", const=1, default="parameter_values.json"
    )

    args = parser.parse_args()

    with open(args.parameters, "r") as f:
        parameters = json.load(f)

    if args.compile:
        compile_pipeline(package_path=args.package_path)

    job = aiplatform.PipelineJob(
        display_name=f"default-risk-pipeline-{TIMESTAMP}",
        template_path=args.package_path,
        job_id="uci-{0}".format(TIMESTAMP),
        location="us-east1",
        enable_caching=True,
        parameter_values=parameters,
    )

    job.submit(service_account=args.service_account)
