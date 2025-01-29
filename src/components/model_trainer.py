import sys
import os
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import  save_object, evaluate_models, metric_evaluation

#from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis




@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Split training and test data')
            X_train, y_train, X_test, y_test = (
                                                train_array[:,:-1],
                                                train_array[:,-1],
                                                test_array[:,:-1],
                                                test_array[:,-1]
                                                  )
            # Creating dictionary of the models to run and their hyperparameters 
            models = {
            "Logistic Regression": LogisticRegression(max_iter= 1000),
            "LDA": LinearDiscriminantAnalysis(),
            "KNN": KNeighborsClassifier(),
            "Random Forest": RandomForestClassifier(),
            "XG Boost": XGBClassifier(),
            "CatBoost": CatBoostClassifier(verbose = 0),
            "SGDC": SGDClassifier(random_state=42)
            }

            params = {
                        "Logistic Regression": {},  # Add parameters for Logistic Regression if needed

                        "LDA": {
                            'solver': ['svd', 'lsqr', 'eigen'],
                            'shrinkage': [None, 'auto', 0.1, 0.5, 1.0],  # Only applicable for 'lsqr' and 'eigen'
                        },

                        "KNN": {
                            'n_neighbors': [3, 5, 7, 9],
                            'weights': ['uniform', 'distance']
                        },

                        "Random Forest": {
                            'n_estimators': [10, 50],
                            'min_samples_split': [2, 5, 10]
                        },

                        "XG Boost": {
                            'n_estimators': [10, 50],
                            'max_depth': [3, 5, 7, 10],
                            'learning_rate': [0.01, 0.1, 0.2]
                        },

                        "CatBoost": {
                            'iterations': [500, 1000],  # Number of boosting iterations
                            'learning_rate': [0.01, 0.1, 0.2]  # Learning rate
                        },

                        "SGDC": {}  # Add parameters for SGDC if needed
                    }

            model_report = evaluate_models(X_train = X_train, 
                                           y_train = y_train, 
                                           X_test = X_test, 
                                           y_test = y_test, 
                                           models = models, 
                                           param = params)
            
            best_model_name = model_report['test_accuracy'].idxmax()
            
            best_model = models[best_model_name]
            
            logging.info("Best Model Found")
            best_model.fit(X_train, y_train)
            
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
                    )

            pred_train = best_model.predict(X_train)
            pred_test = best_model.predict(X_test)
            Metrics = metric_evaluation(actuals = y_train,
                                        preds = pred_train,
                                        act_test = y_test,
                                        pred_test = pred_test)
            
            return Metrics
            
            
        except Exception as e:
            raise CustomException(e, sys)