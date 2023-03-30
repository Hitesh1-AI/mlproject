import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join('artifacts', 'train_csv')
    test_data_path : str = os.path.join('artifacts', 'test_csv')
    raw_data_path : str = os.path.join('artifacts', 'data_csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion Method and Component")
        try:
            df = pd.read_csv('src/notebook/data/CarPrice_data.csv')
            # num_col = [col for col in df.columns if df[col].dtype != 'object' ]
            # cat_col = [col for col in df.columns if df[col].dtype == 'object' ]
            # print(num_col, cat_col)
            logging.info("Read the dataset to Dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, header=True, index=False)
            logging.info("Train test split the data")
            train_set, test_set = train_test_split(df, test_size = 0.2, random_state = 42)
            train_set.to_csv(self.ingestion_config.train_data_path, index= False, header= True)
            test_set.to_csv(self.ingestion_config.test_data_path, index= False, header= True)
            logging.info("Data ingestion is complited")
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_set, test_set = obj.initiate_data_ingestion()
    
    obj2 = DataTransformation()
    train_arr, test_arr,_ = obj2.initiate_data_transformation(train_set, test_set)

    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_arr, test_arr)