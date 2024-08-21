from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# Make the Python script available for import
sys.path.append(os.path.dirname(__file__))  # Ensure the DAG is run from its own directory

# Import the function from your script
def my_function():
    print("Hello from my_function!")

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 10, 4),
    'email': ['your_email@example.com'],  # Add your email if you want notifications
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'my_python_script_execution',
    default_args=default_args,
    description='A simple DAG to execute a Python function',
    schedule_interval=timedelta(days=1),  # Run daily, modify as needed
)

# Create a PythonOperator to execute the function
run_my_function_task = PythonOperator(
    task_id='run_my_function',
    python_callable=my_function,
    dag=dag,
)

# If you have more tasks, you can define the dependencies as well
# For now our DAG has only one task