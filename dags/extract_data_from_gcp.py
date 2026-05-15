from airflow import DAG
from airflow.providers.google.cloud.transfers.gcs_to_local import GCSToLocalFilesystemOperator
from airflow.providers.google.cloud.operators.gcs import GCSListObjectsOperator
from airflow.operators.python import PythonOperator
from airflow.hooks.base import BaseHook
from datetime import datetime
import pandas as pd
import sqlalchemy

#### TRANSFORM STEP....
def load_to_sql(file_path):
    conn = BaseHook.get_connection('postgres_default')  
    engine = sqlalchemy.create_engine(f"postgresql+psycopg2://{conn.login}:{conn.password}@diabetes-disease-mlops_e1f712-postgres-1:{conn.port}/{conn.schema}")
    df = pd.read_csv(file_path)
    df.to_sql(name="diabetes", con=engine, if_exists="replace", index=False)

# Define the DAG
with DAG(
    dag_id="extract_diabetes_data",
    schedule=None,  # schedule_interval was changed to schedule.
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    # Extract STEP...
    list_files = GCSListObjectsOperator(
        task_id="list_files",
        bucket="diabetes_disease_bucket_neat_chain_464913", 
    )

    download_file = GCSToLocalFilesystemOperator(
        task_id="download_file",
        bucket="diabetes_disease_bucket_neat_chain_464913", 
        object_name="diabetes.csv", 
        filename="/tmp/diabetes.csv", 
    )
    
    ### TRANSFORM AND LOAD....
    load_data = PythonOperator(
        task_id="load_to_sql",
        python_callable=load_to_sql,
        op_kwargs={"file_path": "/tmp/diabetes.csv"}
    )

    list_files >> download_file >> load_data