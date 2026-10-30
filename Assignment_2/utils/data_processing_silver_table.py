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


def process_silver_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # save silver table - IRL connect to database to write
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_silver_attributes(snapshot_date_str, bronze_attributes_directory, silver_attributes_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # connect to bronze table
    partition_name = "bronze_attributes_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_attributes_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: Age arrives as string with underscores (e.g. "34_") - strip and cast
    df = df.withColumn("Age", F.regexp_replace(col("Age").cast(StringType()), "_", "").cast(IntegerType()))

    # clean data: null out of age range 
    df = df.withColumn("Age", F.when((col("Age") >= 18) & (col("Age") <=100), col("Age")). otherwise(None))

     # clean data: replace placeholder occupation "_______" with null
    df = df.withColumn("Occupation", F.when(col("Occupation") == "_______", None).otherwise(col("Occupation")))

    # clean data: enforce schema / data type
    column_type_map = {
        "Customer_ID": StringType(),
        "Age": IntegerType(),
        "Occupation": StringType(),
        "snapshot_date": DateType(),
    }
 
    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))
 
    # select columns to save: drop Name and SSN (PII with no predictive value)
    df = df.select("Customer_ID", "Age", "Occupation", "snapshot_date")

    # save silver table 
    partition_name = "silver_attributes_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_attributes_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
 
    return df

def process_silver_financials(snapshot_date_str, bronze_financials_directory, silver_financials_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # connect to bronze table
    partition_name = "bronze_financials_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_financials_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data:strip wrapping underscores from numeric columns stored as string
     # (e.g. Amount_invested_monthly "__10000__")
    underscore_numeric_columns = [
        "Annual_Income",
        "Num_of_Loan",
        "Num_of_Delayed_Payment",
        "Changed_Credit_Limit",
        "Outstanding_Debt",
        "Amount_invested_monthly",
        "Monthly_Balance",
    ]
    for column in underscore_numeric_columns:
        df = df.withColumn(column, F.regexp_replace(col(column).cast(StringType()), "_", ""))
        df = df.withColumn(column, F.when(col(column) == "", None).otherwise(col(column)).cast(FloatType()))
 
    # clean data: replace placeholder categories with null
    df = df.withColumn("Credit_Mix", F.when(col("Credit_Mix") == "_", None).otherwise(col("Credit_Mix")))
    df = df.withColumn("Payment_Behaviour", F.when(col("Payment_Behaviour") == "!@9#%8", None).otherwise(col("Payment_Behaviour")))

    # clean data: null out impossible values (data entry errors / corrupted records)
    df = df.withColumn("Annual_Income", F.when((col("Annual_Income") > 0) & (col("Annual_Income") <= 300000), col("Annual_Income")).otherwise(None))
    df = df.withColumn("Num_Bank_Accounts", F.when((col("Num_Bank_Accounts") >= 0) & (col("Num_Bank_Accounts") <= 20), col("Num_Bank_Accounts")).otherwise(None))
    df = df.withColumn("Num_Credit_Card", F.when((col("Num_Credit_Card") >= 0) & (col("Num_Credit_Card") <= 20), col("Num_Credit_Card")).otherwise(None))
    df = df.withColumn("Interest_Rate", F.when((col("Interest_Rate") >= 1) & (col("Interest_Rate") <= 40), col("Interest_Rate")).otherwise(None))
    df = df.withColumn("Num_of_Loan", F.when((col("Num_of_Loan") >= 0) & (col("Num_of_Loan") <= 20), col("Num_of_Loan")).otherwise(None))
    df = df.withColumn("Num_of_Delayed_Payment", F.when((col("Num_of_Delayed_Payment") >= 0) & (col("Num_of_Delayed_Payment") <= 50), col("Num_of_Delayed_Payment")).otherwise(None))
    df = df.withColumn("Monthly_Balance", F.when((col("Monthly_Balance") >= -100000) & (col("Monthly_Balance") <= 100000), col("Monthly_Balance")).otherwise(None))
 
    # augment data: parse Credit_History_Age "10 Years and 9 Months" into total months
    df = df.withColumn("credit_history_years", F.regexp_extract(col("Credit_History_Age"), r"(\d+) Year", 1).cast(IntegerType()))
    df = df.withColumn("credit_history_months_part", F.regexp_extract(col("Credit_History_Age"), r"(\d+) Month", 1).cast(IntegerType()))
    df = df.withColumn("Credit_History_Months", (F.coalesce(col("credit_history_years"), F.lit(0)) * 12 + F.coalesce(col("credit_history_months_part"), F.lit(0))).cast(IntegerType()))
    df = df.withColumn("Credit_History_Months", F.when(col("Credit_History_Age").isNull(), None).otherwise(col("Credit_History_Months")))
    df = df.drop("credit_history_years", "credit_history_months_part", "Credit_History_Age")
 
    # augment data: count number of loan types listed in Type_of_Loan
    df = df.withColumn("Num_Loan_Types", F.when(col("Type_of_Loan").isNull(), 0).otherwise(F.size(F.split(col("Type_of_Loan"), ","))).cast(IntegerType()))
    df = df.drop("Type_of_Loan")
 
    # clean data: enforce schema / data type
    column_type_map = {
        "Customer_ID": StringType(),
        "Annual_Income": FloatType(),
        "Monthly_Inhand_Salary": FloatType(),
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": IntegerType(),
        "Num_of_Loan": IntegerType(),
        "Delay_from_due_date": IntegerType(),
        "Num_of_Delayed_Payment": IntegerType(),
        "Changed_Credit_Limit": FloatType(),
        "Num_Credit_Inquiries": IntegerType(),
        "Credit_Mix": StringType(),
        "Outstanding_Debt": FloatType(),
        "Credit_Utilization_Ratio": FloatType(),
        "Payment_of_Min_Amount": StringType(),
        "Total_EMI_per_month": FloatType(),
        "Amount_invested_monthly": FloatType(),
        "Payment_Behaviour": StringType(),
        "Monthly_Balance": FloatType(),
        "Credit_History_Months": IntegerType(),
        "Num_Loan_Types": IntegerType(),
        "snapshot_date": DateType(),
    }
 
    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))
 
    # save silver table - IRL connect to database to write
    partition_name = "silver_financials_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_financials_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
 
    return df
 
 
def process_silver_clickstream(snapshot_date_str, bronze_clickstream_directory, silver_clickstream_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
 
    # connect to bronze table
    partition_name = "bronze_clickstream_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_clickstream_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())
 
    # clean data: enforce schema / data type (clickstream is machine-generated and already clean)
    column_type_map = {"fe_" + str(i): IntegerType() for i in range(1, 21)}
    column_type_map["Customer_ID"] = StringType()
    column_type_map["snapshot_date"] = DateType()
 
    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))
 
    # save silver table - IRL connect to database to write
    partition_name = "silver_clickstream_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_clickstream_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
 
    return df
 