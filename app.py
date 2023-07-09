from fastapi import FastAPI, Request
from pydantic import BaseModel
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
import numpy as np
from typing import List
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")

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
async def get_root(request: Request):
    return templates.TemplateResponse('index.html', {"request": request})

@app.post('/prediction')
async def post_method(data: Data):
    df = CustomData(
        data.CarName, data.symboling, data.fueltype,data.aspiration,
                    data.doornumber, data.carbody, data.drivewheel, data.enginelocation, data.wheelbase,
                    data.carlength, data.carwidth, data.carheight, data.curbweight,
                    data.cylindernumber, data.enginesize, data.fuelsystem, data.boreratio, data.stroke,
                    data.compressionratio, data.horsepower, data.peakrpm, data.citympgL, data.highwaympg
    )

    pred_df = df.get_data_as_data_frame()

    predict_pipe = PredictPipeline()
    result = predict_pipe.predict(pred_df).tolist()

    return result
