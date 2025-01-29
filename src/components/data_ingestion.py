import os
import sys

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.exception import CustomException
from src.logger import logging
from src.utils import data_prep, group_sleep_hours, group_categories
import pandas as pd
import numpy as np
import re


from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation, DataTransformationConfig


from src.components.model_trainer import ModelTrainerConfig, ModelTrainer





@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")
    dict_sleep_data_path: str=os.path.join('artifacts',"mapping_sleep.csv")
    dict_diet_data_path: str=os.path.join('artifacts',"mapping_diet.csv")
    
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv("notebook/data/Mental_Health.csv")
            logging.info("Read the dataset as dataframe")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            
            
            
            df['Sleep_Duration_Category'] = df['Sleep Duration'].apply(group_sleep_hours)
            # Create a mapping dictionary for future use when we will have new unseen data.
            mapping_sleep = dict(zip(df['Sleep Duration'], df['Sleep_Duration_Category']))
                        
            # Apply the grouping function 
            df['Dietary_Habits_Category'] = group_categories(df['Dietary Habits'], ['Moderate', 'Unhealthy', 'Healthy'])

            # Create a mapping dictionary for future use when we will have new unseen data.
            mapping_diet = dict(zip(df['Dietary Habits'], df['Dietary_Habits_Category']))
            
            df = data_prep(df = df,
                           mapping_sleep = mapping_sleep,
                           mapping_diet = mapping_diet)
            
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)
            logging.info("Raw data saved successfully")
            
            # Convert mapping dictionnaries to DataFrame
            mapping_sleep_df = pd.DataFrame(list(mapping_sleep.items()), columns=['Sleep Duration', 'Sleep_Duration_Category'])
            mapping_diet_df = pd.DataFrame(list(mapping_diet.items()), columns=['Dietary Habits', 'MDietary_Habits_Category'])
            
            # Save the DataFrame as a CSV file
            mapping_sleep_df.to_csv(self.ingestion_config.dict_sleep_data_path, index = False, header = True)
            mapping_diet_df.to_csv(self.ingestion_config.dict_diet_data_path, index = False, header = True)
            
            logging.info(f"Sleep Mapping dictionary saved to {self.ingestion_config.dict_sleep_data_path}")
            logging.info(f"Diet Mapping dictionary saved to {self.ingestion_config.dict_diet_data_path}")
            
            # Train-test split
            logging.info("Train test split initiated")      
            train_set, test_set = train_test_split(df, 
                                                   stratify = df['depression'],
                                                   test_size = 0.3, 
                                                   random_state = 42)
            
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)
            
            logging.info("Ingestion of the data is completed")
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path
                #self.ingestion_config.dict_sleep_data_path,
                #self.ingestion_config.dict_diet_data_path
            )
            
        except Exception as e:
           raise CustomException(e, sys)
       
       
if __name__ == "__main__":
    obj = DataIngestion()
    # Run data_ingestion
    train_path, test_path, raw_path = obj.initiate_data_ingestion()
    
    # Run data_transformation
    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_path, test_path)
    
    # Run model_trainer
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))

        
