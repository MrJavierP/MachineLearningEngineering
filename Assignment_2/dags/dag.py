from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'credit_default_ml_pipeline',
    default_args=default_args,
    description='end to end ML pipeline run once a month',
    schedule_interval='0 0 1 * *',  # At 00:00 on day-of-month 1
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 1),
    catchup=True,
    max_active_runs=1,  # months must run in order: training in sep 2024 feeds inference from that month on
) as dag:

    # data pipeline: build this month's slice of the datamart 
    # one job runs the whole medallion for the month: bronze ingest, silver cleaning, then the gold label store and gold feature store

    dep_check_source_label_data = EmptyOperator(task_id="dep_check_source_label_data")

    dep_check_source_data_bronze_1 = EmptyOperator(task_id="dep_check_source_data_bronze_1")

    dep_check_source_data_bronze_2 = EmptyOperator(task_id="dep_check_source_data_bronze_2")

    dep_check_source_data_bronze_3 = EmptyOperator(task_id="dep_check_source_data_bronze_3")

    run_feature_pipeline = BashOperator(
        task_id='run_feature_pipeline',
        bash_command=(
            'cd /opt/airflow && '
            'python3 scripts/feature_pipeline.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    label_store_completed = EmptyOperator(task_id="label_store_completed")

    feature_store_completed = EmptyOperator(task_id="feature_store_completed")

    # all four sources feed the one monthly feature pipeline job
    dep_check_source_label_data >> run_feature_pipeline
    dep_check_source_data_bronze_1 >> run_feature_pipeline
    dep_check_source_data_bronze_2 >> run_feature_pipeline
    dep_check_source_data_bronze_3 >> run_feature_pipeline
    run_feature_pipeline >> label_store_completed
    run_feature_pipeline >> feature_store_completed

    # model auto training 
    # the scripts decide when to fire: training skips every month except the model train date (2024-09-01), when 12 months train/test + 2 months OOT of labels have matured;
    # model_select then promotes the best candidate by out-of-time AUC (first deployment automatic, later ones need a human --promote)

    model_automl_start = EmptyOperator(task_id="model_automl_start")

    model_1_automl = BashOperator(
        task_id='model_1_automl',
        bash_command=(
            'cd /opt/airflow && '
            'python3 scripts/model_train.py '
            '--snapshotdate "{{ ds }}" '
            '--modeltype xgboost'
        ),
    )

    model_2_automl = BashOperator(
        task_id='model_2_automl',
        bash_command=(
            'cd /opt/airflow && '
            'python3 scripts/model_train.py '
            '--snapshotdate "{{ ds }}" '
            '--modeltype lightgbm'
        ),
    )

    # the two finalists of the offline bake-off; model_train.py also supports
    # logreg and randomforest (both retired: lower out-of-time AUC)

    # model selection: compare the candidates trained this month on OOT AUC and
    # record the champion in model_bank/champion.json (skips on months with no training)
    model_select = BashOperator(
        task_id='model_select',
        bash_command=(
            'cd /opt/airflow && '
            'python3 scripts/model_select.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    model_automl_completed = EmptyOperator(task_id="model_automl_completed")

    feature_store_completed >> model_automl_start
    label_store_completed >> model_automl_start
    model_automl_start >> model_1_automl >> model_select
    model_automl_start >> model_2_automl >> model_select
    model_select >> model_automl_completed


    # model inference
    # training runs before inference so that in the month a new model ships, it scores that same month; the script skips any month where the model is not in the bank yet
    # the champion task writes the gold predictions table the business reads (model_predictions/champion/);
    # the two named tasks keep scoring both candidates in shadow so they can be compared in monitoring

    model_inference_start = EmptyOperator(task_id="model_inference_start")

    model_1_inference = BashOperator(
        task_id='model_1_inference',
        bash_command=(
            'cd /opt/airflow && '
            'python3 scripts/model_inference.py '
            '--snapshotdate "{{ ds }}" '
            '--modelname credit_model_xgboost_2024_09_01.pkl'
        ),
    )

    model_2_inference = BashOperator(
        task_id='model_2_inference',
        bash_command=(
            'cd /opt/airflow && '
            'python3 scripts/model_inference.py '
            '--snapshotdate "{{ ds }}" '
            '--modelname credit_model_lightgbm_2024_09_01.pkl'
        ),
    )

    champion_inference = BashOperator(
        task_id='champion_inference',
        bash_command=(
            'cd /opt/airflow && '
            'python3 scripts/model_inference.py '
            '--snapshotdate "{{ ds }}" '
            '--modelname champion'
        ),
    )

    model_inference_completed = EmptyOperator(task_id="model_inference_completed")

    model_automl_completed >> model_inference_start
    model_inference_start >> model_1_inference >> model_inference_completed
    model_inference_start >> model_2_inference >> model_inference_completed
    model_inference_start >> champion_inference >> model_inference_completed


    # model monitoring
    # evaluates the cohort whose labels matured this month and checks score stability vs the deployment month; skips until the model is live

    model_monitor_start = EmptyOperator(task_id="model_monitor_start")

    model_1_monitor = BashOperator(
        task_id='model_1_monitor',
        bash_command=(
            'cd /opt/airflow && '
            'python3 scripts/model_monitor.py '
            '--snapshotdate "{{ ds }}" '
            '--modelname credit_model_xgboost_2024_09_01.pkl'
        ),
    )

    model_2_monitor = BashOperator(
        task_id='model_2_monitor',
        bash_command=(
            'cd /opt/airflow && '
            'python3 scripts/model_monitor.py '
            '--snapshotdate "{{ ds }}" '
            '--modelname credit_model_lightgbm_2024_09_01.pkl'
        ),
    )

    champion_monitor = BashOperator(
        task_id='champion_monitor',
        bash_command=(
            'cd /opt/airflow && '
            'python3 scripts/model_monitor.py '
            '--snapshotdate "{{ ds }}" '
            '--modelname champion'
        ),
    )

    # refresh the performance / stability charts from every monitoring row so far
    model_visualise = BashOperator(
        task_id='model_visualise',
        bash_command=(
            'cd /opt/airflow && '
            'python3 scripts/model_visualise.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    model_monitor_completed = EmptyOperator(task_id="model_monitor_completed")

    model_inference_completed >> model_monitor_start
    model_monitor_start >> model_1_monitor >> model_visualise
    model_monitor_start >> model_2_monitor >> model_visualise
    model_monitor_start >> champion_monitor >> model_visualise
    model_visualise >> model_monitor_completed
