import argparse
import os
import glob
from datetime import datetime

import pyspark

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table


# datamart directory layout
BRONZE_LMS_DIR = "datamart/bronze/lms/"
BRONZE_CLICKSTREAM_DIR = "datamart/bronze/clickstream/"
BRONZE_ATTRIBUTES_DIR = "datamart/bronze/attributes/"
BRONZE_FINANCIALS_DIR = "datamart/bronze/financials/"

SILVER_LOAN_DAILY_DIR = "datamart/silver/loan_daily/"
SILVER_CLICKSTREAM_DIR = "datamart/silver/clickstream/"
SILVER_ATTRIBUTES_DIR = "datamart/silver/attributes/"
SILVER_FINANCIALS_DIR = "datamart/silver/financials/"

GOLD_LABEL_STORE_DIR = "datamart/gold/label_store/"
GOLD_FEATURE_STORE_DIR = "datamart/gold/feature_store/"

# label definition, default = 30dpd at mob 6
DPD = 30
MOB = 6


def main(snapshotdate):
    print('\n\n---starting data pipeline job---\n\n')

    # make sure all datamart folders exist before writing
    for directory in [BRONZE_LMS_DIR, BRONZE_CLICKSTREAM_DIR, BRONZE_ATTRIBUTES_DIR, BRONZE_FINANCIALS_DIR,
                      SILVER_LOAN_DAILY_DIR, SILVER_CLICKSTREAM_DIR, SILVER_ATTRIBUTES_DIR, SILVER_FINANCIALS_DIR,
                      GOLD_LABEL_STORE_DIR, GOLD_FEATURE_STORE_DIR]:
        os.makedirs(directory, exist_ok=True)

    # initialise spark session
    spark = pyspark.sql.SparkSession.builder \
        .appName("data_pipeline") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # bronze layer: ingest the raw monthly snapshots as-is
    utils.data_processing_bronze_table.process_bronze_table(snapshotdate, BRONZE_LMS_DIR, spark)
    utils.data_processing_bronze_table.process_bronze_clickstream(snapshotdate, BRONZE_CLICKSTREAM_DIR, spark)
    utils.data_processing_bronze_table.process_bronze_attributes(snapshotdate, BRONZE_ATTRIBUTES_DIR, spark)
    utils.data_processing_bronze_table.process_bronze_financials(snapshotdate, BRONZE_FINANCIALS_DIR, spark)

    # silver layer: clean, type and augment
    utils.data_processing_silver_table.process_silver_table(snapshotdate, BRONZE_LMS_DIR, SILVER_LOAN_DAILY_DIR, spark)
    utils.data_processing_silver_table.process_silver_clickstream(snapshotdate, BRONZE_CLICKSTREAM_DIR, SILVER_CLICKSTREAM_DIR, spark)
    utils.data_processing_silver_table.process_silver_attributes(snapshotdate, BRONZE_ATTRIBUTES_DIR, SILVER_ATTRIBUTES_DIR, spark)
    utils.data_processing_silver_table.process_silver_financials(snapshotdate, BRONZE_FINANCIALS_DIR, SILVER_FINANCIALS_DIR, spark)

    # gold layer: label store and feature store
    utils.data_processing_gold_table.process_labels_gold_table(snapshotdate, SILVER_LOAN_DAILY_DIR, GOLD_LABEL_STORE_DIR, spark, dpd=DPD, mob=MOB)
    utils.data_processing_gold_table.process_features_gold_table(snapshotdate, SILVER_ATTRIBUTES_DIR, SILVER_FINANCIALS_DIR, SILVER_CLICKSTREAM_DIR, GOLD_FEATURE_STORE_DIR, spark)

    spark.stop()
    print('\n\n---data pipeline job completed---\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run bronze, silver and gold processing for one snapshot date")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)