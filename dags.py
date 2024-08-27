import sys
import os

sys.path.append(os.path.dirname(__file__))

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from otomoto.preprocess_data import OtomotoPreprocessor
from otomoto.train_models import ModelTrainer

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
    python_callable=ModelTrainer(project_name="otomoto_car_price_predictor"),
    dag=otomoto_pipeline,
)

# send_message = PythonOperator(
#     task_id='1',
#     python_callable=OtomotoPreprocessor(),
#     dag=dag,
# )

otomoto_pipeline_preprocess >> otomoto_pipeline_train
