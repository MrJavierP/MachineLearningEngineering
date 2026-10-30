import argparse
import os
import glob
import pickle
import pprint
from datetime import datetime

import numpy as np
import pandas as pd
import pyspark

from sklearn.metrics import roc_auc_score

from model_select import resolve_model_name


FEATURE_STORE_DIR = "datamart/gold/feature_store/"
LABEL_STORE_DIR = "datamart/gold/label_store/"
MODEL_BANK_DIR = "model_bank/"
PREDICTIONS_DIR = "datamart/gold/model_predictions/"
MONITORING_DIR = "datamart/gold/model_monitoring/"

# labels mature at mob 6, so the cohort we can evaluate this month applied 6 months ago
LABEL_MOB = 6

# governance thresholds (the SOP slide quotes these numbers)
#   matured-cohort AUC below 0.85: watch; two consecutive months below: retrain
#   psi vs the deployment month: > 0.10 investigate (watch), > 0.25 retrain
AUC_FLOOR = 0.85
PSI_WATCH, PSI_RETRAIN = 0.10, 0.25


def governance_status(auc, prev_auc, psi):
    low_now = auc is not None and auc < AUC_FLOOR
    low_prev = prev_auc is not None and prev_auc < AUC_FLOOR
    if (low_now and low_prev) or psi > PSI_RETRAIN:
        return "retrain"
    if low_now or psi > PSI_WATCH:
        return "watch"
    return "ok"


def previous_auc(model_id, snapshotdate):
    # last month's monitoring row for this model, if there is one
    files = sorted(glob.glob(MONITORING_DIR + model_id + "/*.parquet"))
    files = [f for f in files if f < MONITORING_DIR + model_id + "/" + model_id +
             "_monitoring_" + snapshotdate.replace('-', '_') + '.parquet']
    if not files:
        return None
    prev = pd.read_parquet(files[-1])
    val = prev["auc"].iloc[0]
    return None if pd.isna(val) else float(val)

# columns that identify rows rather than describe customers - never used as features
NON_FEATURE_COLS = ["loan_id", "Customer_ID", "label", "label_def",
                    "snapshot_date", "label_snapshot_date", "application_date"]


def prepare_features(df, feature_columns):
    # same encoding as training so the model sees the exact columns it was fitted on
    X = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns])
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    X = pd.get_dummies(X, columns=cat_cols, dummy_na=True)
    X = X.reindex(columns=feature_columns, fill_value=0)
    return X


def population_stability_index(baseline_scores, current_scores, n_bins=10):
    # bin edges come from the baseline month, then we compare how full
    # each bin is now vs then. rule of thumb: <0.1 stable, 0.1-0.25 watch, >0.25 shifted
    edges = np.quantile(baseline_scores, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    eps = 1e-6
    base_pct = np.histogram(baseline_scores, bins=edges)[0] / len(baseline_scores) + eps
    curr_pct = np.histogram(current_scores, bins=edges)[0] / len(current_scores) + eps
    return float(np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct)))


def main(snapshotdate, modelname):
    print('\n\n---starting model monitoring job---\n\n')

    resolved = resolve_model_name(modelname)
    if resolved is None:
        print("skip: no champion selected yet")
        return
    config = {}
    config["snapshot_date_str"] = snapshotdate
    config["snapshot_date"] = datetime.strptime(snapshotdate, "%Y-%m-%d")
    config["model_name"] = resolved
    config["model_artefact_filepath"] = MODEL_BANK_DIR + resolved
    pprint.pprint(config)

    model_id = modelname[:-4] if modelname.endswith(".pkl") else modelname

    # monitoring only makes sense once the model is deployed and scoring
    if not os.path.exists(config["model_artefact_filepath"]):
        print("skip: " + config["model_artefact_filepath"] + " not in model bank yet")
        return

    predictions_partition = (PREDICTIONS_DIR + model_id + "/" + model_id +
                             "_predictions_" + snapshotdate.replace('-', '_') + '.parquet')
    if not os.path.exists(predictions_partition):
        print("skip: no predictions for " + snapshotdate + " - model not live this month")
        return

    spark = pyspark.sql.SparkSession.builder \
        .appName("model_monitor") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    with open(config["model_artefact_filepath"], 'rb') as file:
        model_artefact = pickle.load(file)
    model = model_artefact["model"]
    imputer = model_artefact["preprocessing_transformers"]["imputer"]
    scaler = model_artefact["preprocessing_transformers"]["stdscaler"]
    feature_columns = model_artefact["feature_columns"]
    auc_oot_at_training = float(model_artefact["results"]["auc_oot"])

    # --- part 1: performance on the cohort that just matured ---
    # labels arriving this month belong to customers who applied 6 months ago.
    # we score that cohort and check how well the model separated good from bad.
    cohort_app_date = config["snapshot_date"] - pd.DateOffset(months=LABEL_MOB)
    cohort_app_str = cohort_app_date.strftime("%Y-%m-%d")

    label_partition = LABEL_STORE_DIR + "gold_label_store_" + snapshotdate.replace('-', '_') + '.parquet'
    feature_partition = FEATURE_STORE_DIR + "gold_feature_store_" + cohort_app_str.replace('-', '_') + '.parquet'

    auc = gini = actual_default_rate = avg_predicted = None
    n_labels = 0
    if os.path.exists(label_partition) and os.path.exists(feature_partition):
        labels_pdf = spark.read.parquet(label_partition).toPandas()
        features_pdf = spark.read.parquet(feature_partition).toPandas()
        cohort = labels_pdf[["Customer_ID", "label"]].merge(features_pdf, on="Customer_ID", how="inner")
        n_labels = len(cohort)
        if n_labels > 0:
            X = scaler.transform(imputer.transform(prepare_features(cohort, feature_columns)))
            scores = model.predict_proba(X)[:, 1]
            auc = float(roc_auc_score(cohort["label"], scores)) if cohort["label"].nunique() > 1 else None
            gini = round(2 * auc - 1, 4) if auc is not None else None
            actual_default_rate = float(cohort["label"].mean())
            avg_predicted = float(scores.mean())
    print("matured cohort:", cohort_app_str, "n =", n_labels, "| auc:", auc)

    # --- part 2: stability of this month's production scores ---
    # psi compares this month's score distribution against the first month
    # after deployment. drifting scores flag a changing population even
    # before any new labels arrive.
    preds_now = spark.read.parquet(predictions_partition).toPandas()

    baseline_files = sorted(glob.glob(PREDICTIONS_DIR + model_id + "/*.parquet"))
    baseline_partition = baseline_files[0]  # earliest month the model scored
    preds_baseline = spark.read.parquet(baseline_partition).toPandas()

    psi = population_stability_index(
        preds_baseline["model_predictions"].values,
        preds_now["model_predictions"].values)
    print("psi vs deployment month:", round(psi, 4), "| scored this month:", len(preds_now))

    # --- part 3: governance verdict for this month ---
    auc_drop = round(auc_oot_at_training - auc, 4) if auc is not None else None
    status = governance_status(auc, previous_auc(model_id, snapshotdate), psi)
    print("status:", status, "| auc drop vs training OOT:", auc_drop)

    # --- write one monitoring row to the gold monitoring table ---
    row = pd.DataFrame([{
        "snapshot_date": config["snapshot_date"].date(),
        "model_name": config["model_name"],
        "cohort_application_month": cohort_app_str,
        "n_labels_matured": n_labels,
        "auc": auc,
        "gini": gini,
        "actual_default_rate": actual_default_rate,
        "avg_predicted_prob": avg_predicted,
        "n_scored_this_month": len(preds_now),
        "avg_score_this_month": float(preds_now["model_predictions"].mean()),
        "psi_vs_deployment": round(psi, 4),
        "auc_oot_at_training": round(auc_oot_at_training, 4),
        "auc_drop": auc_drop,
        "status": status,
    }])

    gold_directory = MONITORING_DIR + model_id + "/"
    if not os.path.exists(gold_directory):
        os.makedirs(gold_directory)
    partition_name = model_id + "_monitoring_" + snapshotdate.replace('-', '_') + '.parquet'
    filepath = gold_directory + partition_name
    spark.createDataFrame(row).write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    spark.stop()
    print('\n\n---model monitoring job completed---\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="monitor one model for one month: matured-cohort performance plus score stability")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True,
                        help="model pickle filename, or 'champion' for the selected model")
    args = parser.parse_args()
    main(args.snapshotdate, args.modelname)