import argparse
import os
import glob
import pickle
import pprint
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

import pyspark

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
import lightgbm as lgb


# training only happens once enough label history has matured
# 12 months train/test + 2 months OOT counted back from this date (same windows as lab 5)
MODEL_TRAIN_DATE_STR = "2024-09-01"

LABEL_STORE_DIR = "datamart/gold/label_store/"
FEATURE_STORE_DIR = "datamart/gold/feature_store/"
MODEL_BANK_DIR = "model_bank/"

# columns that identify rows rather than describe customers - never used as features
NON_FEATURE_COLS = ["loan_id", "Customer_ID", "label", "label_def",
                    "snapshot_date", "label_snapshot_date", "application_date"]


def load_joined_data(spark):
    # read all label store partitions
    label_files = sorted(glob.glob(os.path.join(LABEL_STORE_DIR, '*')))
    labels_sdf = spark.read.parquet(*label_files)
    labels_pdf = labels_sdf.toPandas()
    print("label store row count:", len(labels_pdf))

    # read all feature store partitions
    feature_files = sorted(glob.glob(os.path.join(FEATURE_STORE_DIR, '*')))
    features_sdf = spark.read.parquet(*feature_files)
    features_pdf = features_sdf.toPandas()
    print("feature store row count:", len(features_pdf))

    # label sits at mob 6, so the application date is 6 months before the label snapshot
    # features are joined at application date - the model only ever sees information
    # that existed when the customer applied (no temporal leakage)
    labels_pdf["snapshot_date"] = pd.to_datetime(labels_pdf["snapshot_date"])
    labels_pdf["application_date"] = labels_pdf["snapshot_date"] - pd.DateOffset(months=6)
    labels_pdf = labels_pdf.rename(columns={"snapshot_date": "label_snapshot_date"})
    features_pdf["snapshot_date"] = pd.to_datetime(features_pdf["snapshot_date"])

    df = labels_pdf.merge(
        features_pdf,
        left_on=["Customer_ID", "application_date"],
        right_on=["Customer_ID", "snapshot_date"],
        how="left")
    print("joined row count:", len(df))
    return df


def prepare_features(df, feature_columns=None):
    # one hot encode the few categorical columns, everything else is already numeric
    X = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns])
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    X = pd.get_dummies(X, columns=cat_cols, dummy_na=True)

    if feature_columns is not None:
        # align to the training-time column list so train and inference always match
        X = X.reindex(columns=feature_columns, fill_value=0)
    return X


def main(snapshotdate, modeltype):
    print('\n\n---starting model train job---\n\n')

    # the DAG calls this script every month, but training only fires on the train date -
    # earlier months have not accumulated the 12+2 months of matured labels yet
    if snapshotdate != MODEL_TRAIN_DATE_STR:
        print(f"skip: {snapshotdate} is not the model train date ({MODEL_TRAIN_DATE_STR})")
        return

    spark = pyspark.sql.SparkSession.builder \
        .appName("model_train") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # set up config
    config = {}
    config["model_train_date_str"] = MODEL_TRAIN_DATE_STR
    config["train_test_period_months"] = 12
    config["oot_period_months"] = 2
    config["train_test_ratio"] = 0.8
    config["model_train_date"] = datetime.strptime(MODEL_TRAIN_DATE_STR, "%Y-%m-%d")
    config["oot_end_date"] = config["model_train_date"] - timedelta(days=1)
    config["oot_start_date"] = config["model_train_date"] - relativedelta(months=config["oot_period_months"])
    config["train_test_end_date"] = config["oot_start_date"] - timedelta(days=1)
    config["train_test_start_date"] = config["oot_start_date"] - relativedelta(months=config["train_test_period_months"])
    pprint.pprint(config)

    # load and join label store with feature store
    df = load_joined_data(spark)

    # split into train / test / oot by label snapshot date
    oot_pdf = df[(df["label_snapshot_date"] >= config["oot_start_date"]) &
                 (df["label_snapshot_date"] <= config["oot_end_date"])]
    train_test_pdf = df[(df["label_snapshot_date"] >= config["train_test_start_date"]) &
                        (df["label_snapshot_date"] <= config["train_test_end_date"])]

    X_all = prepare_features(pd.concat([train_test_pdf, oot_pdf]))
    feature_columns = X_all.columns.tolist()
    X_tt = X_all.iloc[:len(train_test_pdf)]
    X_oot_raw = X_all.iloc[len(train_test_pdf):]
    y_tt = train_test_pdf["label"].values
    y_oot = oot_pdf["label"].values

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_tt, y_tt,
        test_size=1 - config["train_test_ratio"],
        random_state=88,
        shuffle=True,
        stratify=y_tt)

    print('X_train', X_train_raw.shape[0])
    print('X_test', X_test_raw.shape[0])
    print('X_oot', X_oot_raw.shape[0])

    # preprocess: impute then scale, both fitted on train only
    # fitting on the full dataset would leak test/oot statistics into training
    imputer = SimpleImputer(strategy="median").fit(X_train_raw)
    scaler = StandardScaler().fit(imputer.transform(X_train_raw))

    X_train = scaler.transform(imputer.transform(X_train_raw))
    X_test = scaler.transform(imputer.transform(X_test_raw))
    X_oot = scaler.transform(imputer.transform(X_oot_raw))

    # train model with random search, hyperparameter grids from the bake-off
    if modeltype == "xgboost":
        estimator = xgb.XGBClassifier(random_state=88, eval_metric='logloss', n_jobs=-1)
        param_dist = {
            'n_estimators': [200, 400],
            'max_depth': [3, 4, 6],
            'learning_rate': [0.03, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
        }
    elif modeltype == "lightgbm":
        estimator = lgb.LGBMClassifier(random_state=88, n_jobs=1, verbose=-1)
        param_dist = {
            'n_estimators': [200, 400],
            'max_depth': [3, 4, 6],
            'learning_rate': [0.03, 0.1],
            'num_leaves': [15, 31],
            'subsample': [0.8, 1.0],
        }
    elif modeltype == "logreg":
        # linear baseline - kept in the candidate set so boosted models
        # always have to prove they earn their extra complexity
        estimator = LogisticRegression(max_iter=2000, random_state=88)
        param_dist = {
            'C': [0.01, 0.1, 1, 10],
        }
    elif modeltype == "randomforest":
        estimator = RandomForestClassifier(random_state=88, n_jobs=-1)
        param_dist = {
            'n_estimators': [200, 400],
            'max_depth': [6, 10, None],
            'min_samples_leaf': [1, 5, 10],
        }
    else:
        raise ValueError("modeltype must be xgboost, lightgbm, logreg or randomforest")

    random_search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        scoring='roc_auc',
        n_iter=10,
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1)
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    print("best params:", random_search.best_params_)

    # evaluate on all three windows
    train_auc = roc_auc_score(y_train, best_model.predict_proba(X_train)[:, 1])
    test_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
    oot_auc = roc_auc_score(y_oot, best_model.predict_proba(X_oot)[:, 1])
    print("train AUC:", round(train_auc, 4))
    print("test AUC:", round(test_auc, 4))
    print("OOT AUC:", round(oot_auc, 4))

    # package artefact
    model_artefact = {}
    model_artefact['model'] = best_model
    model_artefact['model_version'] = "credit_model_" + modeltype + "_" + MODEL_TRAIN_DATE_STR.replace('-', '_')
    model_artefact['preprocessing_transformers'] = {}
    model_artefact['preprocessing_transformers']['imputer'] = imputer
    model_artefact['preprocessing_transformers']['stdscaler'] = scaler
    model_artefact['feature_columns'] = feature_columns
    model_artefact['data_dates'] = config
    model_artefact['data_stats'] = {}
    model_artefact['data_stats']['X_train'] = X_train_raw.shape[0]
    model_artefact['data_stats']['X_test'] = X_test_raw.shape[0]
    model_artefact['data_stats']['X_oot'] = X_oot_raw.shape[0]
    model_artefact['data_stats']['y_train'] = round(float(y_train.mean()), 2)
    model_artefact['data_stats']['y_test'] = round(float(y_test.mean()), 2)
    model_artefact['data_stats']['y_oot'] = round(float(y_oot.mean()), 2)
    model_artefact['results'] = {}
    model_artefact['results']['auc_train'] = train_auc
    model_artefact['results']['auc_test'] = test_auc
    model_artefact['results']['auc_oot'] = oot_auc
    model_artefact['results']['gini_train'] = round(2 * train_auc - 1, 3)
    model_artefact['results']['gini_test'] = round(2 * test_auc - 1, 3)
    model_artefact['results']['gini_oot'] = round(2 * oot_auc - 1, 3)
    model_artefact['hp_params'] = random_search.best_params_

    # save artefact to model bank 
    if not os.path.exists(MODEL_BANK_DIR):
        os.makedirs(MODEL_BANK_DIR)

    file_path = os.path.join(MODEL_BANK_DIR, model_artefact['model_version'] + '.pkl')
    with open(file_path, 'wb') as file:
        pickle.dump(model_artefact, file)
    print("model saved to", file_path)

    spark.stop()
    print('\n\n---model train job completed---\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train a model and save the artefact to the model bank")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modeltype", type=str, required=True, help="xgboost, lightgbm, logreg or randomforest")
    args = parser.parse_args()
    main(args.snapshotdate, args.modeltype)