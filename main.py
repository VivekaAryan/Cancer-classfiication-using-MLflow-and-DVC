from src.ChestCancerClassfication.utils import logger
from src.ChestCancerClassfication.pipeline.step_01_data_ingestion import DataIngestionTrainingPipeline
from src.ChestCancerClassfication.pipeline.step_02_prepare_base_model import PrepareBaseModelTrainingPipeline

##################################### Stage 1: Data Ingestion #####################################

STAGE_NAME = "Data Ingestion stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx=============x")

except Exception as e:
    logger.exception(e)
    raise e

##################################### Stage 1: Preparing Base Model #####################################

STAGE_NAME = "Prepare base model"

try:
    logger.info(f"****************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    prepare_base_model = PrepareBaseModelTrainingPipeline()
    prepare_base_model.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx=========x")

except Exception as e:
    logger.exception(e)
    raise e
