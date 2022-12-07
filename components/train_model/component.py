from kfp.v2.dsl import Dataset, Output, component, Input, Metrics, Model
from typing import NamedTuple, Tuple

@component(base_image='python:3.9', packages_to_install=['pandas', 'scikit-learn', 'xgboost'])
def train_model(train_set: Input[Dataset], test_set: Input[Dataset], metrics: Output[Metrics], model: Output[Model])-> NamedTuple("Outputs", [("score", float)]):
    """ Will train a machine learning model for default credit prediction 
        :param x_train: Train set for training
        :param x_test: dataset for testing"""
    
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import f1_score, accuracy_score, recall_score
    from sklearn.ensemble import RandomForestClassifier

    import xgboost as xgb
    import pandas as pd
    import pickle
    import logging
    logging.getLogger().setLevel(logging.INFO)

    def read_and_drop_columns(set_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """ Read csv and drop unused columns """
        try:
            df = pd.read_csv(set_path)
            logging.info(f"Data read from {set_path}")
        except Exception as e:
            logging.error(f"Data must be in CSV format {e} {set_path}")
            raise ValueError(f"Data must be in CSV format")

        y = df['default payment_next_month'].values
        x = df(['default payment_next_month', 'ID'], axis=1)

        return x, y
    
    def create_columns_transformer() -> ColumnTransformer:
        try:
            cat_features = ["SEX", "EDUCATION", "MARRIAGE"]
            cat_transformer = Pipeline(
                steps=[("imputer", OneHotEncoder(handle_unknown = 'ignore'))]
                )
        except Exception as e:
            logging.error(f"Not able to create categorical features pipeline {e}")
            raise ValueError(f"Not able to create categorical features pipeline {e}")
        
        try:
            numeric_features = ['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5',
            'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4',
            'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
            'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'diff_paid1', 'diff_paid2',
            'diff_paid3', 'diff_paid4', 'diff_paid5', 'diff_paid6',
            'sum_bill_amount', 'sum_payment_delay', 'sum_payment_amount',
            '%_sum_bill_in_limit']
            num_transformer = Pipeline(
                steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
            )
        except Exception as e:
            logging.error(f"Not able to create numerical features pipeline {e}")
            raise ValueError(f"Not able to create numerical features pipeline {e}")           

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", cat_transformer, cat_features),
                ("num", num_transformer, numeric_features),
            ],remainder="passthrough"
        )
        return preprocessor
    
    def create_model_pipeline(preprocessor: ColumnTransformer):
        """ Create a RandomForest Classifier and insert it in a pipeline"""
        mdl = RandomForestClassifier(random_state=42)

        rf_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', mdl)
        ])

        param_grid_rf = {
            'classifier__max_depth': [500]
        }
        rf_grid = GridSearchCV(rf_pipe, param_grid_rf, cv=2, n_jobs=-1, scoring='f1_macro')

        return rf_grid

    def log_metrics(y_test, y_pred) -> None:
        """ Will log the metric to a Vertex Artifact """
        f1 = f1_score(y_test, y_pred, average='macro')
        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        metrics.log_metric('accuracy', acc)
        metrics.log_metric('recall', recall)
        metrics.log_metric('f1-score', f1)

        return f1

    x_train, y_train = read_and_drop_columns(train_set.path)
    x_test, y_test = read_and_drop_columns(test_set.path)

    preprocessor = create_columns_transformer()

    mdl = create_model_pipeline(preprocessor)

    mdl.fit(x_train, y_train)
    y_pred = mdl.predict(x_test)
    f1 = log_metrics(y_test, y_pred)

    #Saving model
    model.metadata["framework"] = "randomforest"
    file_name = model.path + f".pkl"
    with open(file_name, 'wb') as file:  
        pickle.dump(mdl, file)
    
    logging.info(f"Model saved at... {file_name}")
    logging.info(f"Model path at... {model.path}")

    return (f1, )