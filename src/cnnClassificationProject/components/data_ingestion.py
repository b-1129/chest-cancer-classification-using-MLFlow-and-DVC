import os
import zipfile
import gdown
from cnnClassificationProject import logger
from cnnClassificationProject.utils.common import get_size
from cnnClassificationProject.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self,config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> str:

        """ Fetch Data from the URL
        """
        try:
            dataset_url= self.config.source_URL
            zip_download_dir= self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(f"downloading data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = "https://drive.google.com/uc?/export=download&id="
            gdown.download(prefix+file_id,zip_download_dir)

            logger.info(f"downloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            raise e
        
    def extract_zip_file(self):
        """
        zip file path : str
        Extract the zip file into Data Directory
        Function returns None
        """

        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file,'r') as zip_ref:
            zip_ref.extractall(unzip_path)