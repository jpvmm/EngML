from datetime import datetime

from google.cloud import aiplatform
from kfp.v2 import compiler, dsl

from pipeline.pipeline import model_training

TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
PACKAGE_PATH = "pipeline_packages/uci.json"

compiler.Compiler().compile(pipeline_func=model_training, package_path=PACKAGE_PATH)

job = aiplatform.PipelineJob(
    display_name=f"default-risk-pipeline-{TIMESTAMP}",
    template_path=PACKAGE_PATH,
    job_id="uci-{0}".format(TIMESTAMP),
    location="us-east1",
    enable_caching=True,
)

job.submit(service_account="vertex@qacomp.iam.gserviceaccount.com")
