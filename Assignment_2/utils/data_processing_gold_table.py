import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_features_gold_table(snapshot_date_str, silver_attributes_directory, silver_financials_directory, silver_clickstream_directory, gold_feature_store_directory, spark):
 
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
 
    # connect to silver attributes table (snapshotted at loan application date)
    partition_name = "silver_attributes_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_attributes_directory + partition_name
    df_attributes = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df_attributes.count())
 
    # connect to silver financials table (snapshotted at loan application date)
    partition_name = "silver_financials_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_financials_directory + partition_name
    df_financials = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df_financials.count())
 
    # join attributes and financials on Customer_ID 
    df = df_attributes.join(df_financials.drop("snapshot_date"), on="Customer_ID", how="left")
 
    # point in time join for clickstream
    # only take clickstream partitions dated on or before the application date
    # otherwise future behaviour would leak into the features
    clickstream_files = sorted(glob.glob(os.path.join(silver_clickstream_directory, '*')))
    clickstream_files = [f for f in clickstream_files
                         if datetime.strptime(os.path.basename(f).replace('silver_clickstream_','').replace('.parquet',''), "%Y_%m_%d") <= snapshot_date]
 
    if clickstream_files:
        df_clickstream = spark.read.parquet(*clickstream_files)
 
        fe_columns = ["fe_" + str(i) for i in range(1, 21)]
 
        # aggregate full pre-application history per customer: mean and std of each fe column
        agg_exprs = []
        for fe_col in fe_columns:
            agg_exprs.append(F.avg(col(fe_col)).cast(FloatType()).alias(fe_col + "_mean"))
            agg_exprs.append(F.stddev(col(fe_col)).cast(FloatType()).alias(fe_col + "_std"))
        df_clickstream_agg = df_clickstream.groupBy("Customer_ID").agg(*agg_exprs)
 
        # take the latest pre-application month per customer
        from pyspark.sql.window import Window
        window_spec = Window.partitionBy("Customer_ID").orderBy(col("snapshot_date").desc())
        df_clickstream_latest = df_clickstream.withColumn("rn", F.row_number().over(window_spec)) \
                                              .filter(col("rn") == 1) \
                                              .select("Customer_ID", *[col(c).alias(c + "_latest") for c in fe_columns])
 
        # join clickstream aggregations onto the feature table
        df = df.join(df_clickstream_agg, on="Customer_ID", how="left")
        df = df.join(df_clickstream_latest, on="Customer_ID", how="left")
 
        # feature 5: has_clickstream flag - distinguishes "no history" from "zero activity"
        df = df.withColumn("has_clickstream", F.when(col("fe_1_mean").isNotNull(), 1).otherwise(0).cast(IntegerType()))
    else:
        df = df.withColumn("has_clickstream", F.lit(0).cast(IntegerType()))
 
    # feature 1: debt to income ratio, measures leverage
    df = df.withColumn("debt_to_income",
                       F.when(col("Annual_Income") > 0, col("Outstanding_Debt") / col("Annual_Income")).otherwise(None).cast(FloatType()))
 
    # feature 2: EMI to salary ratio, monthly repayment burden
    df = df.withColumn("emi_to_salary",
                       F.when(col("Monthly_Inhand_Salary") > 0, col("Total_EMI_per_month") / col("Monthly_Inhand_Salary")).otherwise(None).cast(FloatType()))
 
    # feature 3: loans per credit year, how fast customer takes on loans vs their credit experience
    # floor the denominator at 0.5 years so short histories dont blow up the ratio
    df = df.withColumn("loans_per_credit_year",
                       (col("Num_of_Loan") / F.greatest(col("Credit_History_Months") / 12, F.lit(0.5))).cast(FloatType()))
 
    # feature 4: total number of credit products held
    df = df.withColumn("num_credit_products",
                       (col("Num_Bank_Accounts") + col("Num_Credit_Card") + col("Num_of_Loan")).cast(IntegerType()))
 
    # save gold table - IRL connect to database to write
    partition_name = "gold_feature_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_feature_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
 
    return df
 