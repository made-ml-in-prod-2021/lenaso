from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable

default_args = {
    "owner": "Elena Soldatenko",
    "email": ["soldatenkoes@gmail.com"],
    "email_on_failure": True,
    "email_on_retry": True,
    "retries": 3,
    "retry_delay": timedelta(minutes=1),
}

DATA_DIR = Variable.get("DATA_DIR")

with DAG(
    "dataload",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=days_ago(3),
) as dag:

    download = DockerOperator(
        image="airflow-generate",
        command="--data_path /data/raw/{{ ds }}/data.csv "
                "--target_path /data/raw/{{ ds }}/target.csv "
                "--n_samples 1000",
        network_mode="bridge",
        task_id="docker-airflow-dataload",
        do_xcom_push=False,
        volumes=[f"{DATA_DIR}:/data"],
    )
