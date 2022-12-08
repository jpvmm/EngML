from typing import NamedTuple, Tuple

from kfp.v2.dsl import Dataset, Input, Metrics, Model, Output, component


@component(
    base_image="python:3.9", packages_to_install=["pandas", "scikit-learn", "xgboost"]
)
def train_model(
    train_set: Input[Dataset],
    test_set: Input[Dataset],
    metrics: Output[Metrics],
    model: Output[Model],
) -> NamedTuple("Outputs", [("score", float)]):
    """Will train a machine learning model for default credit prediction
    :param x_train: Train set for training
    :param x_test: dataset for testing"""

    import logging
    import pickle

    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    logging.getLogger().setLevel(logging.INFO)

    def read_and_drop_columns(set_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Read csv and drop unused columns
        :param set_path: this is a path beloning the Dataset type

        :return Tuple[x,y]: the dataframe for training/prediction and the labels
        for training/prediction"""
        try:
            df = pd.read_csv(set_path)
            logging.info(f"Data read from {set_path}")
        except Exception as e:
            logging.error(f"Data must be in CSV format {e} {set_path}")
            raise ValueError(f"Data must be in CSV format")

        y = df["default payment_next_month"].values
        x = df(["default payment_next_month", "ID"], axis=1)

        return x, y

    def create_columns_transformer() -> ColumnTransformer:
        """Will create the preprocessor used by the pipeline"""
        try:
            cat_features = ["SEX", "EDUCATION", "MARRIAGE"]
            cat_transformer = Pipeline(
                steps=[("imputer", OneHotEncoder(handle_unknown="ignore"))]
            )
        except Exception as e:
            logging.error(f"Not able to create categorical features pipeline {e}")
            raise ValueError(f"Not able to create categorical features pipeline {e}")

        try:
            numeric_features = [
                "LIMIT_BAL",
                "AGE",
                "PAY_0",
                "PAY_2",
                "PAY_3",
                "PAY_4",
                "PAY_5",
                "PAY_6",
                "BILL_AMT1",
                "BILL_AMT2",
                "BILL_AMT3",
                "BILL_AMT4",
                "BILL_AMT5",
                "BILL_AMT6",
                "PAY_AMT1",
                "PAY_AMT2",
                "PAY_AMT3",
                "PAY_AMT4",
                "PAY_AMT5",
                "PAY_AMT6",
                "diff_paid1",
                "diff_paid2",
                "diff_paid3",
                "diff_paid4",
                "diff_paid5",
                "diff_paid6",
                "sum_bill_amount",
                "sum_payment_delay",
                "sum_payment_amount",
                "%_sum_bill_in_limit",
            ]
            num_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
        except Exception as e:
            logging.error(f"Not able to create numerical features pipeline {e}")
            raise ValueError(f"Not able to create numerical features pipeline {e}")

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", cat_transformer, cat_features),
                ("num", num_transformer, numeric_features),
            ],
            remainder="passthrough",
        )
        return preprocessor

    def create_model_pipeline(preprocessor: ColumnTransformer):
        """Create a RandomForest Classifier and insert it in a pipeline
        :param preprocessor: ColumnTransformer for data preprocessing in pipeline

        :return GridSearchCV: a gridsearch model pipeline"""
        mdl = RandomForestClassifier(random_state=42)

        rf_pipe = Pipeline([("preprocessor", preprocessor), ("classifier", mdl)])

        param_grid_rf = {"classifier__max_depth": [500]}
        rf_grid = GridSearchCV(
            rf_pipe, param_grid_rf, cv=2, n_jobs=-1, scoring="f1_macro"
        )

        return rf_grid

    def log_metrics(y_test: pd.Series, y_pred: pd.Series) -> None:
        """Will log the metric to a Vertex Artifact
        :param y_test: Test labels out of the split
        :param y_pred: Predicted labels out of the model"""
        f1 = f1_score(y_test, y_pred, average="macro")
        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        metrics.log_metric("accuracy", acc)
        metrics.log_metric("recall", recall)
        metrics.log_metric("f1-score", f1)

        return f1

    x_train, y_train = read_and_drop_columns(train_set.path)
    x_test, y_test = read_and_drop_columns(test_set.path)

    preprocessor = create_columns_transformer()

    mdl = create_model_pipeline(preprocessor)

    mdl.fit(x_train, y_train)
    y_pred = mdl.predict(x_test)
    f1 = log_metrics(y_test, y_pred)

    # Saving model
    model.metadata["framework"] = "randomforest"
    file_name = model.path + ".pkl"
    with open(file_name, "wb") as file:
        pickle.dump(mdl, file)

    logging.info(f"Model saved at... {file_name}")
    logging.info(f"Model path at... {model.path}")

    return (f1,)
