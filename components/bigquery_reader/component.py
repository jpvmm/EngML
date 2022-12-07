from kfp.v2.dsl import Dataset, Output, component, Input, Metrics, Model
from typing import NamedTuple, Tuple

@component(base_image='python:3.9', packages_to_install=['google-cloud-bigquery==2.34.3', 'pandas', 'pyarrow'])
def compare_results_in_bigquery(score: float)-> NamedTuple("Outputs", [("deploy", bool)]):
    """ Will query BigQuery table containing the scores of the historical models of this project 
        If the score is bigger than the biggest one in BQ this model will be set to deployed """
    from google.cloud import bigquery
    import logging
    logging.getLogger().setLevel(logging.INFO)

    table_id = 'qacomp.default_credit.default_credit_data'
    client = bigquery.Client(project='qacomp')
    def find_max_score_bq():
        """ Query BQ to get the max score of previous models"""
        try:
            sql = f"SELECT * FROM {table_id} LIMIT 1000"
            df = client.query(sql).to_dataframe()
            max_score = df.f1_score.max()
        except Exception as e:
            max_score = 0
            logging.error(f"Not able to query BQ {e}")
        return max_score
    
    def insert_row_to_bq(score):
        """Will insert the current trained model results to BQ"""
        #Insert new model values
        rows_to_insert = [
        {"clf_name": "RandomForest", "f1_score": score},
        ]
        errors = client.insert_rows_json(
            table_id, rows_to_insert, row_ids=[None] * len(rows_to_insert)
        )

        if errors == []:
            logging.info('No Errors on the update')
        else:
            logging.error(f"Errors on the update {errors}")

    max_score = find_max_score_bq()
    insert_row_to_bq(score)
    
    if score > max_score:
        deploy = True
    else:
        deploy = False
    logging.info(f"Deploy result: {deploy}")
    return (deploy,)