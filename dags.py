import sys
import os

sys.path.append(os.path.dirname(__file__))

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from otomoto.preprocess_data import OtomotoPreprocessor
from otomoto.train_models import OtomotoModelTrainer
from otodom.preprocess_data import OtodomDomyPreprocessor, OtodomMieszkaniaPreprocessor
from otodom.train_models import OtodomModelTrainer

default_args = {
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

otomoto_pipeline = DAG(
    "otomoto_pipeline",
    default_args=default_args,
    description="Remove",
    schedule="0 19 * * *",
    catchup=False,
)

otomoto_pipeline_preprocess = PythonOperator(
    task_id="1",
    python_callable=OtomotoPreprocessor(),
    dag=otomoto_pipeline,
)

otomoto_pipeline_train = PythonOperator(
    task_id="2",
    python_callable=OtomotoModelTrainer(project_name="otomoto_car_price_predictor"),
    dag=otomoto_pipeline,
)

otomoto_pipeline_preprocess >> otomoto_pipeline_train


otodom_domy_pipeline = DAG(
    "otodom_domy_pipeline",
    default_args=default_args,
    description="Remove",
    schedule="15 19 * * *",
    catchup=False,
)

otodom_domy_pipeline_preprocess = PythonOperator(
    task_id="1",
    python_callable=OtodomDomyPreprocessor(),
    dag=otodom_domy_pipeline,
)

otodom_domy_pipeline_train = PythonOperator(
    task_id="2",
    python_callable=OtodomModelTrainer(project_name="otodom_domy_price_predictor"),
    dag=otodom_domy_pipeline,
)

otodom_domy_pipeline_preprocess >> otodom_domy_pipeline_train


otodom_mieszkania_pipeline = DAG(
    "otodom_mieszkania_pipeline",
    default_args=default_args,
    description="Remove",
    schedule="30 19 * * *",
    catchup=False,
)

otodom_mieszkania_pipeline_preprocess = PythonOperator(
    task_id="1",
    python_callable=OtodomMieszkaniaPreprocessor(),
    dag=otodom_mieszkania_pipeline,
)

otodom_mieszkania_pipeline_train = PythonOperator(
    task_id="2",
    python_callable=OtodomModelTrainer(
        project_name="otodom_mieszkania_price_predictor"
    ),
    dag=otodom_mieszkania_pipeline,
)

otodom_mieszkania_pipeline_preprocess >> otodom_mieszkania_pipeline_train
