import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_Transformer_obj(self):
        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
               "gender",
               "race_ethnicity",
               "parental_level_of_education",
               "lunch",
               "test_preparation_course"
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False))

                ]

            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                    ]
             ) 
            logging.info("Numerical columns Standard Scaling completed.")

            logging.info("Categorical columns encoding completed.")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("Cat_pipelines", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            return CustomException(e, sys)
        

    def initiated_data_transformation(self,train_path,test_path):

        try:
            train_df = pd.read_csv(train_path)

            test_pf = pd.read_csv(test_path)

            logging.info("NRead train and test Data completed.")

            logging.info("Obtaining preproessing Objrct")

            preprocessing_obj = self.get_data_Transformer_obj()

            target_column_name = "math_score"
            numerical_columns = ['writing_score', 'reading_score']

            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_featute_train_df = train_df[target_column_name]

            input_feature_test_df = test_pf.drop(columns=[target_column_name])
            target_featute_test_df = test_pf[target_column_name]

            logging.info("applying Preprocessing obj on training Dataframe and testing dataframe")

            input_feature_train_arr =  preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr =  preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_featute_train_df)]

            test_arr = np.c_[input_feature_test_arr, np.array(target_featute_test_df)]

            logging.info("Saved Preprocessing Object")

            save_object(

                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )




        except Exception as e:
            raise CustomException(e, sys)

