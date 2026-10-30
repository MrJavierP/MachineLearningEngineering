import argparse
import os
import pickle
import pprint
from datetime import datetime

import pandas as pd
import pyspark

from model_select import resolve_model_name


FEATURE_STORE_DIR = "datamart/gold/feature_store/"
MODEL_BANK_DIR = "model_bank/"
PREDICTIONS_DIR = "datamart/gold/model_predictions/"

# columns that identify rows rather than describe customers - never used as features
NON_FEATURE_COLS = ["loan_id", "Customer_ID", "label", "label_def",
                    "snapshot_date", "label_snapshot_date", "application_date"]


def prepare_features(df, feature_columns):
    # same encoding as training: one hot the categoricals,
    # then align to the training-time column list so the model always
    # sees the exact columns it was fitted on
    X = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns])
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    X = pd.get_dummies(X, columns=cat_cols, dummy_na=True)
    X = X.reindex(columns=feature_columns, fill_value=0)
    return X


def main(snapshotdate, modelname):
    print('\n\n---starting model inference job---\n\n')

    # --- set up config ---
    # "champion" means whichever model model_select.py chose; its predictions
    # go to their own gold folder so downstream users never have to know
    # which algorithm is live this month
    resolved = resolve_model_name(modelname)
    if resolved is None:
        print("skip: no champion selected yet")
        return
    config = {}
    config["snapshot_date_str"] = snapshotdate
    config["snapshot_date"] = datetime.strptime(snapshotdate, "%Y-%m-%d")
    config["model_name"] = resolved
    config["model_id"] = modelname[:-4] if modelname.endswith(".pkl") else modelname
    config["model_artefact_filepath"] = MODEL_BANK_DIR + resolved
    pprint.pprint(config)

    # the DAG calls this for every month of the backfill, but the model only exists
    # from its training date onwards - before that there is nothing to score with,
    # exactly like production before the first deployment
    if not os.path.exists(config["model_artefact_filepath"]):
        print("skip: " + config["model_artefact_filepath"] + " not in model bank yet")
        return

    spark = pyspark.sql.SparkSession.builder \
        .appName("model_inference") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # --- load model artefact from model bank ---
    with open(config["model_artefact_filepath"], 'rb') as file:
        model_artefact = pickle.load(file)
    print("model loaded:", config["model_artefact_filepath"])

    # --- load this month's partition of the gold feature store ---
    partition_name = "gold_feature_store_" + snapshotdate.replace('-', '_') + '.parquet'
    feature_filepath = FEATURE_STORE_DIR + partition_name
    features_sdf = spark.read.parquet(feature_filepath)
    features_pdf = features_sdf.toPandas()
    print("loaded from:", feature_filepath, "row count:", len(features_pdf))

    # --- preprocess with the transformers saved at training time ---
    X_raw = prepare_features(features_pdf, model_artefact["feature_columns"])
    imputer = model_artefact["preprocessing_transformers"]["imputer"]
    scaler = model_artefact["preprocessing_transformers"]["stdscaler"]
    X_inference = scaler.transform(imputer.transform(X_raw))
    print('X_inference', X_inference.shape[0])

    # --- model prediction inference ---
    model = model_artefact["model"]
    y_inference = model.predict_proba(X_inference)[:, 1]

    y_inference_pdf = features_pdf[["Customer_ID", "snapshot_date"]].copy()
    y_inference_pdf["model_name"] = config["model_name"]
    y_inference_pdf["model_predictions"] = y_inference

    # --- save predictions to datamart gold table, one folder per model ---
    model_id = config["model_id"]
    gold_directory = PREDICTIONS_DIR + model_id + "/"
    if not os.path.exists(gold_directory):
        os.makedirs(gold_directory)

    partition_name = model_id + "_predictions_" + snapshotdate.replace('-', '_') + '.parquet'
    filepath = gold_directory + partition_name
    spark.createDataFrame(y_inference_pdf).write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    spark.stop()
    print('\n\n---model inference job completed---\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="score one month of the feature store with a model from the bank")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True,
                        help="model pickle filename, or 'champion' for the selected model")
    args = parser.parse_args()
    main(args.snapshotdate, args.modelname)