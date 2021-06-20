from datetime import timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago

default_args = {
    "owner": "Elena Soldatenko",
    "email": ["soldatenkoes@gmail.com"],
    'email_on_failure': True,
    'email_on_retry': True,
    "retries": 3,
    "retry_delay": timedelta(minutes=1),
}

DATA_DIR = Variable.get("DATA_DIR")

with DAG(
    "predict",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=days_ago(7),
) as dag:
    wait_data = FileSensor(
        task_id="wait_for_prediction_data",
        filepath="/opt/airflow/data/raw/{{ ds }}/data.csv",
        poke_interval=30,
    )

    wait_model = FileSensor(
        task_id="wait_for_prediction_model",
        filepath="/opt/airflow/data/models/{{ var.value.model }}/model.pkl",
        poke_interval=30,
    )

    preprocess = DockerOperator(
        image="airflow-preprocess",
        command="preprocess --input_dir /data/raw/{{ ds }} --output_dir /data/proceed/{{ ds }}",
        task_id="docker-airflow-preprocess-valid",
        do_xcom_push=False,
        volumes=[f"{DATA_DIR}:/data"],
    )

    predict = DockerOperator(
        image="airflow-predict",
        command="--data_path /data/proceed/{{ ds }}/data.csv "
                "--model_path /data/models/{{ var.value.model }}/model.pkl "
                "--pred_path /data/predictions/{{ ds }}/prediction.csv",
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        volumes=[f"{DATA_DIR}:/data"],
    )

    [wait_data, wait_model] >> preprocess >> predict
