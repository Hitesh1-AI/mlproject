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
            categorical_columns = ['CarName', 'fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 'enginelocation', 'cylindernumber', 'fuelsystem']


            num_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehotencode', OneHotEncoder(categories = 'auto',handle_unknown= 'ignore')),
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
            print(train_df.index)
            print(test_df.index)


          

            logging.info('Read train and test data completed')
            logging.info('Obtaining preprocessor object')
            
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'price'
            drop_columns = ['price', 'car_ID','enginetype']
            numerical_columns = ['symboling', 'wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg']
         
            input_feature_train_df =train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]
            print('train columns', input_feature_train_df.columns)
          
            input_feature_test_df =test_df.drop(columns=drop_columns, axis = 1)
            target_feature_test_df = test_df[target_column_name]
            print('test columns', input_feature_test_df.columns)
            #Here i have an error that my test data heve different shape then I use handle_unknown parameter in OneHotEncoding
            # input_feature_train_ar =preprocessing_obj.fit(input_feature_train_df)
            # input_feature_train_arr = input_feature_train_ar.transform(input_feature_train_df).toarray()
            # input_feature_test_arr = input_feature_train_ar.transform(input_feature_test_df).toarray()
            input_feature_train_arr =preprocessing_obj.fit_transform(input_feature_train_df).toarray()
            input_feature_test_arr =preprocessing_obj.transform(input_feature_test_df).toarray()
            print('train size', input_feature_train_arr.shape)
            print('test size', input_feature_test_arr.shape)

            train_arr= np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr= np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
#             testing = preprocessing_obj.transform(pd.DataFrame([3,'alfa-romero giulia','gas','std','two','convertible','rwd','front',88.6,168.8,64.1,48.8,2548,'four',130,'mpfi',3.47,2.68,9,111,5000,21,27],
#                                                                index=['symboling', 'CarName', 'fueltype', 'aspiration', 'doornumber',
#        'carbody', 'drivewheel', 'enginelocation', 'wheelbase', 'carlength',
#        'carwidth', 'carheight', 'curbweight', 'cylindernumber', 'enginesize',
#        'fuelsystem', 'boreratio', 'stroke', 'compressionratio', 'horsepower',
#        'peakrpm', 'citympg', 'highwaympg']
# ).T)
#             print('testing.......')
#             print(testing)
#             print(testing.shape)
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
