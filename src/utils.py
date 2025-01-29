import os
import sys

from src.exception import CustomException
from src.logger import logging

import numpy as np 
import pandas as pd
import re
import dill  
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
from sklearn.model_selection import GridSearchCV



def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    


def group_sleep_hours(value):
    '''
    This function groups the original categories into more natural categorical grouping
    '''
    try:
        
        if value in ['Less than 5 hours', '2-3 hours', '1-3 hours', '1-2 hours']:
            return 'Low Sleep'
        elif value in ['5-6 hours', '6-7 hours', '4-5 hours', '6-8 hours', '7-8 hours', '8 hours']:
            return 'Moderate Sleep'
        elif value in ['More than 8 hours', '9-11 hours', '10-11 hours', '8-9 hours']:
            return 'High Sleep'
        else:
            return 'Other'
    
    except Exception as e:
        raise CustomException(e, sys) 



def group_categories(series, main_categories):
    '''
    This function groups the incorrect caegories into 'Oher' category
    '''
    try:
        
        return series.apply(lambda x: x if x in main_categories else 'Other')
    
    except Exception as e:
        raise CustomException(e, sys) 




def data_prep(df,
              col1 = 'Academic Pressure',
              col2 = 'Work Pressure',
              mapping_sleep = None,
              mapping_diet = None):
    '''
    Step 1: Combines Study and Work Pressure into one column
    Step 2: Creates sleep_dur_cat by grouping incorrect leep Duration levels
    Step 3: Creates Diet_cat by grouping incorrect leep Duration levels
    Step 4: Renaming columns
    Step 5: Drop irrelevant columns
    
    '''
    try:
        
        col_drop = ['id', 'Name', 'City', 'Profession', 'CGPA', 'Degree', 'Work Pressure', 'Academic Pressure', 
                        'Study Satisfaction', 'Job Satisfaction', 'Sleep Duration', 'Dietary Habits',
                        'Sleep_Duration_Category', 'Dietary_Habits_Category']
        
        
        df['academic_work_pressure'] = np.maximum(df[col1].fillna(-np.inf), df[col2].fillna(-np.inf)).replace(-np.inf, np.nan)
        #df['academic_work_satisfaction'] = np.maximum(df[col3].fillna(-np.inf), df[col4].fillna(-np.inf)).replace(-np.inf, np.nan)
            
        # Step 2
        df['sleep_dur_cat'] = df['Sleep Duration'].map(mapping_sleep)
        
        # Step 3
        df['diet_cat'] = df['Dietary Habits'].map(mapping_diet)
        
        # Step 4
        df = df.rename(columns = {'Working Professional or Student': 'prof_or_student',
                                'Have you ever had suicidal thoughts ?': 'suicidal',
                                'Work/Study Hours': 'work_study_hours',
                                'Family History of Mental Illness': 'family_hist',
                                'Gender': 'gender',
                                'Age': 'age',
                                'Depression': 'depression',
                                'Financial Stress': 'financial_stress'})
        
        # Step 5
        df = df.drop(columns = col_drop, axis = 1)
        
        return df
    
    except Exception as e:
        raise CustomException(e, sys) 
    
    




def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    
    '''
    
    This function calculates the Accuracy for both Train and Test sets and for each model, 
    then return the results in Pandas DataFrame
    
    '''
    
    try:
        
        data = {}

        # Loop through models and their corresponding hyperparameters
        for model_name, model in models.items():
            # Get the hyperparameter grid for the current model
            para = param[model_name]

            # Perform GridSearchCV
            gs = GridSearchCV(model, para, cv = 5)
            gs.fit(X_train, y_train)

            # Make predictions using the best model from GridSearchCV
            y_train_pred = gs.predict(X_train)
            y_test_pred = gs.predict(X_test)
            
            
            # Calculate the Accuracy
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            
            
            # Add the results to the report dictionary for this model
            data[model_name] = {
                                'train_accuracy': train_accuracy,
                                'test_accuracy': test_accuracy
                                    }
            
            # Now we take the dictionnary and create a nice data frame 
        
            report =  pd.DataFrame(data).T  # Transpose to make model names as rows

            report.rename(columns={'index': 'model_name'}, inplace = True)
            

        return report
    
    except Exception as e:
        raise CustomException(e, sys) 
















    
    
def metric_evaluation(actuals,
                      preds,
                      act_test = None,
                      pred_test = None):

    '''
    
    This function calculates the Accuracy, Precision, Recall and F1_scure
    
    '''
    try:
        
        accuracy = accuracy_score(actuals, preds)
        precision = precision_score(actuals, preds)
        recall = recall_score(actuals, preds)
        f1 = f1_score(actuals, preds)
        
        df = pd.DataFrame({'Train Data': [accuracy, precision, recall, f1]}, 
                        index=['Accuracy', 'Precision', 'Recall', 'F1 Scores'])

        if act_test is not None:
            accuracy = accuracy_score(act_test, pred_test)
            precision = precision_score(act_test, pred_test)
            recall = recall_score(act_test, pred_test)
            f1 = f1_score(act_test, pred_test)
            
            df["Test"] = [accuracy, precision, recall, f1]
        
        return df    
    
    except Exception as e:
        raise CustomException(e, sys) 