import sys
import os
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging

from src.utils import load_object



class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        
        try:
            logging.info("Create artifacts paths")
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            
            logging.info("Create artifacts objects")
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            
            
            logging.info("Start data processing")
            data_scaled = preprocessor.transform(features)
            
            logging.info("Making Preds")
            preds = model.predict(data_scaled)

            
            return preds
        

        except Exception as e:
            raise CustomException(e, sys)
       
    
class CustomData:
    def __init__(self,
                 age: int,
                 work_study_hours: float,
                 gender: str, 
                 prof_or_student: str, 
                 suicidal: str,  
                 financial_stress: float,
                 family_hist: str,
                 academic_work_pressure: float,
                 sleep_dur_cat: str,
                 diet_cat: str
                  ):
       
        self.age = age
        self.work_study_hours = work_study_hours
        self.gender = gender
        self.prof_or_student = prof_or_student
        self.suicidal = suicidal
        self.financial_stress = financial_stress
        self.family_hist = family_hist
        self.academic_work_pressure = academic_work_pressure
        self.sleep_dur_cat = sleep_dur_cat
        self.diet_cat = diet_cat
        
        
        
    def get_data_as_data_frame(self):
        
        try:
            custom_data_input_dict = {
             
                "age": [self.age], 
                "work_study_hours": [self.work_study_hours], 
                "gender": [self.gender],
                "prof_or_student": [self.prof_or_student], 
                "suicidal": [self.suicidal],  
                "financial_stress": [self.financial_stress], 
                "family_hist": [self.family_hist], 
                "academic_work_pressure": [self.academic_work_pressure], 
                "sleep_dur_cat": [self.sleep_dur_cat], 
                "diet_cat": [self.diet_cat] 
                
            }
            
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)
        
        