import kfp
from kfp import dsl

@dsl.container_component
def data_ingestion_op():
    return dsl.ContainerSpec(
        image='htai99/diabetes-disease-mlops:latest',
        command=['python', 'src/data_ingestion.py']
    )

@dsl.container_component
def data_preprocessing_op():
    return dsl.ContainerSpec(
        image='htai99/diabetes-disease-mlops:latest',
        command=['python', 'src/data_preprocessing.py']
    )

@dsl.container_component
def model_training_op():
    return dsl.ContainerSpec(
        image='htai99/diabetes-disease-mlops:latest',
        command=['python', 'src/model_training.py']
    )

@dsl.pipeline(
    name='mlops-pipeline',
    description='Diabetes MLOps Pipeline'
)
def mlops_pipeline():
    # Step 1: Define tasks in separate lines
    task_ingestion = data_ingestion_op()
    task_preprocessing = data_preprocessing_op()
    task_training = model_training_op()

    # Step 2: Define dependencies
    task_preprocessing.after(task_ingestion)
    task_training.after(task_preprocessing)

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(
        pipeline_func=mlops_pipeline, 
        package_path="mlops_pipeline.yaml"
    )