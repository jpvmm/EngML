from kfp.v2.dsl import Dataset, Output, component


@component(
    base_image="python:3.9", packages_to_install=["pandas", "scikit-learn"]
)  # noqa E501
def read_and_process_data(
    input_path: str, train: Output[Dataset], test: Output[Dataset]
):
    """Will read the dataset from GCP Bucket and validate with Evidently
    :param input_path: The FUSE FS path to a .csv file containing the data which
    the model will be trained
    The dataset will be splitted in train and test for the model training"""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    def read_and_make_features(input_path):
        """Read CSV and create new features based in modeling experimentation"""
        df = pd.read_csv(input_path)
        df["MARRIAGE"] = df["MARRIAGE"].astype("category")
        df["SEX"] = df["SEX"].astype("category")
        df["EDUCATION"] = df["EDUCATION"].astype("category")
        df["diff_paid1"] = df.apply(lambda x: x["BILL_AMT1"] - x["PAY_AMT1"], axis=1)
        df["diff_paid2"] = df.apply(lambda x: x["BILL_AMT2"] - x["PAY_AMT2"], axis=1)
        df["diff_paid3"] = df.apply(lambda x: x["BILL_AMT3"] - x["PAY_AMT3"], axis=1)
        df["diff_paid4"] = df.apply(lambda x: x["BILL_AMT4"] - x["PAY_AMT4"], axis=1)
        df["diff_paid5"] = df.apply(lambda x: x["BILL_AMT5"] - x["PAY_AMT5"], axis=1)
        df["diff_paid6"] = df.apply(lambda x: x["BILL_AMT6"] - x["PAY_AMT6"], axis=1)
        df["sum_bill_amount"] = df.apply(
            lambda x: x["BILL_AMT1"]
            + x["BILL_AMT2"]
            + x["BILL_AMT3"]
            + x["BILL_AMT4"]
            + x["BILL_AMT5"]
            + x["BILL_AMT6"],
            axis=1,
        )
        df["sum_payment_delay"] = df.apply(
            lambda x: x["PAY_0"]
            + x["PAY_2"]
            + x["PAY_3"]
            + x["PAY_4"]
            + x["PAY_5"]
            + x["PAY_6"],
            axis=1,
        )
        df["sum_payment_amount"] = df.apply(
            lambda x: x["PAY_AMT1"]
            + x["PAY_AMT2"]
            + x["PAY_AMT3"]
            + x["PAY_AMT4"]
            + x["PAY_AMT5"]
            + x["PAY_AMT6"],
            axis=1,
        )

        df["%_sum_bill_in_limit"] = df.apply(
            lambda x: (x["sum_bill_amount"] / x["LIMIT_BAL"]) * 100, axis=1
        )

    df = read_and_make_features(input_path)
    # Simple split data
    x_train, x_test = train_test_split(df, test_size=0.15, random_state=42)

    with open(train.path, "w") as f:
        x_train.to_csv(f, index=False)

    with open(test.path, "w") as f:
        x_test.to_csv(f, index=False)
