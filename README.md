# Mental Health Depression Classification


This project focuses on predicting the presence of depression based on various personal and lifestyle attributes. Using machine learning, we explored multiple models to identify the most accurate predictor of mental health outcomes. The analysis incorporates a diverse set of features, including behavioral patterns and health indicators. Additionally, we developed a Flask web application that enables users to input relevant data and receive real-time predictions through an intuitive interface.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data Source](#data-source)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Modeling](#modeling)
    - [Model Selection](#model-selection)
5. [Flask App](#flask-app)
6. [Setup](#setup)

---

### 1. Project Overview
   - Objective: Predict car prices based on various features.
   - Key Features: Year of manufacture, mileage, engine size, and more.
   - Outcome: Selected Linear Regression model for its high Adjusted R-squared score of 0.92.

### 2. Data Source
   - Dataset: [USA Mercedes Benz Prices Dataset](https://www.kaggle.com/competitions/playground-series-s4e11/data)
   - Target: Depression
   - Features: Contains various car attributes including Age, Work/Study Hours, Gender, Professional/Student, Suicidal Thoughts, Financial Stress, Family History, Acadimic/Work Pressure, Sleep Duration, and Diet.


### 3. Exploratory Data Analysis (EDA)
   - Analysis included:
      - Data cleaning and preprocessing.
      - Visualizations (bar plots, scatter plots) to identify the relationship with the target variable.
      - Chi-Square and ANOVA tests to determine whether there is a significant association with Depression.
      - Feature importance analysis to assess the impact of attributes on depression.

### 4. Modeling
   - Seven different models were tested:
      - Logistic Regression
      - Linear Discriminant Analysis
      - K Neighbors Classifier
      - Random Forest Classifier
      - XGBoost Classifier
      - CatBoost Classifier
      - GDClassifier
   - Model Comparison:
      - Evaluated each model based on RAccuracy, Precision, Recall, F1 Scores and AUC.
      - Catboost model achieved the highest Accuracy of 93.6%.

#### Model Selection
   - Final Model: **CatBoost Classifier**
      -  CatBoost was chosen as the final model due to its ability to handle categorical variables efficiently, robustness to class imbalances, and superior performance across all key metrics. Its ensemble-based approach provided a well-balanced trade-off between accuracy and generalizability.

### 5. Flask App
   - A Flask web application is included for local predictions.
   - Features:
      - Input car attributes via a form.
      - Provides predicted price based on user inputs.
   - Note: The app currently runs locally, with deployment planned for future iterations.

### 6. Setup
   - **Clone the repository**:
     ```bash
     github.com/Ahmed-Berkane/Mental-Health.git
     ```
   - **Create a virtual environment**:
     ```bash
     python3 -m venv venv

     # On Windows:
     .\venv\Scripts\activate 

     # On macOS:  
     source venv/bin/activate
     ```
   - **Install dependencies**:
     ```bash
     cd 02-ClassificationMentalHealth
     pip install -r requirements.txt
     ```
   - **Run the Flask app locally**:
     ```bash
     python application.py
     ```
   - Once the app is running, open your browser and go to http://127.0.0.1:5000/predictdata to see the app.

   - **Additional Notes**:
      - Ensure all required libraries are installed as per `requirements.txt`.
      - Data files should be placed in the specified directories if not included.



