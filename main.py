from cnnClassificationProject import logger
from cnnClassificationProject.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassificationProject.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassificationProject.pipeline.stage_03_model_training import ModelTrainingPipeline
from cnnClassificationProject.pipeline.stage_04_model_evaluation_mlflow import EvaluationPipeline

import dagshub
dagshub.init(repo_owner='b-1129',
             repo_name='chest-cancer-classification-using-MLFlow-and-DVC',
             mlflow=True)

STAGE_NAME = "Data Ingestion stage"

try:
    logger.info(f">>>> stage {STAGE_NAME} started <<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>> stage {STAGE_NAME} compeleted <<<<\n\nx==========x")
except Exception as e:
    raise e


STAGE_NAME = "Prepare base model"

try:
    logger.info(f"***************")
    logger.info(f">>>> stage {STAGE_NAME} started <<<<")
    prepare_base_model = PrepareBaseModelTrainingPipeline()
    prepare_base_model.main()
    logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Training"

try:
    logger.info(f"***************")
    logger.info(f">>>> stage {STAGE_NAME} started <<<<")
    model_trainer = ModelTrainingPipeline()
    model_trainer.main()
    logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Evaluation stage"

try:
    logger.info(f"***************")
    logger.info(f">>>> stage {STAGE_NAME} started <<<<")
    model_evaluation = EvaluationPipeline()
    model_evaluation.main()
    logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e