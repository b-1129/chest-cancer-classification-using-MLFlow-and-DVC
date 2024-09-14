from cnnClassificationProject.config.configuration import ConfigurationManager
from cnnClassificationProject.components.model_evaluation_mlflow import Evaluation
from cnnClassificationProject import logger

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)

STAGE_NAME = "Evaluation stage"

class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        evaluation.save_score()
        # evaluation.log_into_mlflow()


if __name__ == "__main__":
    try:
        logger.info(f"*************")
        logger.info(f">>>> stage {STAGE_NAME} started <<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n\nx===============x")
    except Exception as e:
        logger.exception(e)
        raise e
