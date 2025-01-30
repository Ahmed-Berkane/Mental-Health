from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.exception import CustomException
   

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

from src.exception import CustomException



application = Flask(__name__)


## Route for a home page

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/predictdata', methods = ['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            age = int(request.form.get("age")),
            work_study_hours = float(request.form.get("work_study_hours")),
            gender = request.form.get("gender"),
            prof_or_student = request.form.get("prof_or_student"),
            suicidal = request.form.get("suicidal"),
            financial_stress = float(request.form.get("financial_stress")),
            family_hist = request.form.get("family_hist"),
            academic_work_pressure = float(request.form.get("academic_work_pressure")),
            sleep_dur_cat = request.form.get("sleep_dur_cat"),
            diet_cat = request.form.get("diet_cat"),
        )
        

        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        if results == 1:
            results = "Depressed"
        else:
            results = "Not Depressed"
        
        return render_template('home.html', results = results)
    
    

if __name__ == "__main__":
    application.run()
    
    
    
    