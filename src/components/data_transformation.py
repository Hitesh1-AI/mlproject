import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is used to transform the data
        '''
        try:

            numerical_columns = ['symboling', 'wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg']
            categorical_columns = ['CarName', 'fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem']


            num_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehotencode', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
                ]
            )
            logging.info("Numerical column scaling completed")
            logging.info("Categorical column encoding completed")

            preprocessor = ColumnTransformer(
                [
                
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
                
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info('Read train and test data completed')
            logging.info('Obtaining preprocessor object')
            
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'price'
            numerical_columns = ['symboling', 'wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg']
         
            input_feature_train_df =train_df.drop(columns=[target_column_name, 'car_ID'], axis=1)
            target_feature_train_df = train_df[target_column_name]
            print(input_feature_train_df.shape)
            input_feature_test_df =test_df.drop(columns=[target_column_name, 'car_ID'], axis = 1)
            target_feature_test_df = test_df[target_column_name]
            
            input_feature_train_arr =preprocessing_obj.fit_transform(input_feature_train_df).toarray()
            input_feature_test_arr =preprocessing_obj.fit_transform(input_feature_test_df).toarray()

            print(input_feature_train_arr.shape)
            # input_feature_train_arr = np.array(input_feature_train_arr)
            print(input_feature_train_arr.shape)
            

            train_arr= np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr= np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path 
        )
        except Exception as e:
            raise CustomException(e, sys)
