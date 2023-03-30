import sys, os
import pandas as pd
import numpy as np
import dill
from dataclasses import dataclass

# from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model
from sklearn.metrics import r2_score

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Split training and test input data")
            
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Linear Regression": LinearRegression(),
                "Ada Boost Regressor": AdaBoostRegressor(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Gradient Boosting": GradientBoostingRegressor()
            }

            model_report:dict= evaluate_model(X_train = X_train, y_train= y_train, X_test = X_test, y_test= y_test, models = models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score< 0.6:
                raise CustomException("No best model found")
            
            logging.info("Best model found on both Test and Training data")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )

            predicted = best_model.predict(X_test)

            score = r2_score(predicted, y_test)
            print(score)
            return score
          
        
        except Exception as e:
            raise CustomException(e, sys)