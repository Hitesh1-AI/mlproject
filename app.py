from fastapi import FastAPI
from pydantic import BaseModel
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
import numpy as np


app = FastAPI()

class Data(BaseModel):
    CarName : str
    symboling: float
    fueltype :str 
    aspiration :str
    doornumber: str 
    carbody: str
    drivewheel :str 
    enginelocation: str 
    wheelbase: float
    carlength: float 
    carwidth: float 
    carheight:float 
    curbweight:float 
    enginetype:str
    cylindernumber: str 
    enginesize: float
    fuelsystem: str
    boreratio:float 
    stroke: float
    compressionratio: float 
    horsepower: float
    peakrpm: float
    citympgL: float 
    highwaympg: float

@app.get('/')
async def get_root():
    return 'hi!!'

@app.post('/')
async def post_method(data: Data):
    df = CustomData(
        data.CarName, data.symboling,
                    data.fueltype, data.aspiration,
                    data.doornumber, data.carbody, data.drivewheel, data.enginelocation, data.wheelbase,
                    data.carlength, data.carwidth, data.carheight, data.curbweight, data.enginetype,
                    data.cylindernumber, data.enginesize, data.fuelsystem, data.boreratio, data.stroke,
                    data.compressionratio, data.horsepower, data.peakrpm, data.citympgL, data.highwaympg,27
    )

    pred_df = df.get_data_as_data_frame()

    print(pred_df)

    predict_pipe = PredictPipeline()
    result = predict_pipe.predict(pred_df)
    return result
