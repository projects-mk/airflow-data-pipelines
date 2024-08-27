from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from otomoto.preprocess_data import OtomotoPreprocessor
import sys
import os

# sys.path.append(os.path.dirname(__file__))
default_args = {
    'depends_on_past': False,
    'start_date': datetime(2023, 10, 4),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'otomoto_pipeline',
    default_args=default_args,
    description='Remove',
    schedule="0 19 * * *",
)

preprocess = PythonOperator(
    task_id='1',
    python_callable=OtomotoPreprocessor(),
    dag=dag,
)

# train_models = PythonOperator(
#     task_id='1',
#     python_callable=OtomotoPreprocessor(),
#     dag=dag,
# )

# send_message = PythonOperator(
#     task_id='1',
#     python_callable=OtomotoPreprocessor(),
#     dag=dag,
# )

# preprocess >> train_models