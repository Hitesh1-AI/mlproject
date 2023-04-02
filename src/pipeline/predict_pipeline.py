import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self):
        '''
        here is the all data that we recieve from the user or our web app
        '''    
    def get_data_as_data_frame(self):
        '''
        Here is the code for converting the user data to dataframe
        '''
        
        try:
            pass
        except Exception as e:
            raise CustomException(e, sys)