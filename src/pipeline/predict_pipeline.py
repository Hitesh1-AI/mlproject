import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, CarName: str, symboling:float, str, fueltype: str, aspiration: str, doornumber: str, carbody: str,
                 drivewheel: str, enginelocation: str, wheelbase: float, carlength: float, carwidth: float,
                 carheight: float, curbweight: float, enginetype: str, cylindernumber: str, enginesize: float,
                 fuelsystem: str, boreratio: float, stroke: float, compressionratio: float, horsepower: float,
                 peakrpm: float, citympgL: float, highwaympg: float):
        self.CarName = CarName
        self.symboling = symboling
        self.fueltype = fueltype
        self.aspiration = aspiration
        self.doornumber = doornumber
        self.carbody = carbody
        self.drivewheel = drivewheel
        self.enginelocation = enginelocation
        self.wheelbase = wheelbase
        self.carlength = carlength
        self.carwidth = carwidth
        self.carheight = carheight
        self.curbweight = curbweight
        self.enginetype = enginetype
        self.cylindernumber = cylindernumber
        self.enginesize = enginesize
        self.fuelsystem = fuelsystem
        self.boreratio = boreratio
        self.stroke = stroke
        self.compressionratio = compressionratio
        self.horsepower = horsepower
        self.peakrpm = peakrpm
        self.citympgL = citympgL
        self.highwaympg = highwaympg

        '''
        here is the all data that we recieve from the user or our web app
        '''    

    def get_data_as_data_frame(self):
        '''
        Here is the code for converting the user data to dataframe
        '''
        try:
            custom_data_input_dict = {
                
            'CarName': [self.CarName],
            'symboling':[self.symboling],
            'fueltype': [self.fueltype],
            'aspiration': [self.aspiration],
            'doornumber': [self.doornumber],
            'carbody': [self.carbody],
            'drivewheel': [self.drivewheel],
            'enginelocation': [self.enginelocation],
            'wheelbase': [self.wheelbase],
            'carlength': [self.carlength],
            'carwidth': [self.carwidth],
            'carheight': [self.carheight],
            'curbweight': [self.curbweight],
            'enginetype': [self.enginetype],
            'cylindernumber': [self.cylindernumber],
            'enginesize': [self.enginesize],
            'fuelsystem': [self.fuelsystem],
            'boreratio': [self.boreratio],
            'stroke': [self.stroke],
            'compressionratio': [self.compressionratio],
            'horsepower': [self.horsepower],
            'peakrpm': [self.peakrpm],
            'citympg': [self.citympgL],
            'highwaympg': [self.highwaympg]
        }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)