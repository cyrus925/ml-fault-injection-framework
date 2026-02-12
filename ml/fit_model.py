import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from ingestion.utilites import FunctionsUtilites
from pathlib import Path
from ml.log_ml import MLLogger







class Fit():
    '''
    Process to train and save the machine learning model
    '''
    def __init__(self,name:str)-> None:
        '''
        Initialize parameters
        name(str): name of the dataset
        '''
        self.step='fit_model'
        self.name = name
        #Path to feature schema
        self.schema_path = Path(f'ml/schema/{name}_features.yaml') 
        #Path to features dataset
        self.features_path = Path(f'data/ml/features/{name}_features.parquet') 
        #Path where the trained model will be saved
        self.model_path = Path(f'data/ml/models/{name}_model.joblib') 
        pass

    def train(self):
        '''
        Train a RandomForest model using the features dataset.

        Steps:
        - Load feature schema
        - Load features dataset
        - Split data into train and test sets
        - Build preprocessing + model pipeline
        - Train model
        - Evaluate model (RMSE, MAE)
        - Save trained pipeline
        '''

        #Initialize logger for the training 
        logger = MLLogger(self.step,self.name)

        try:
            #Load feature schema
            schema = FunctionsUtilites.load_yaml(self.schema_path)
            NUM_FEATURES, CAT_FEATURES, TARGET = FunctionsUtilites.open_feature(schema)

            #Load features dataset
            df = pd.read_parquet(self.features_path)
            #Log input row in dataset
            logger.log_input(df)

            #Initialize features and target
            X = df[NUM_FEATURES + CAT_FEATURES]
            y = df[TARGET]

            #Split dataset into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            #Passes through numerical features and one-hot encodes categorical ones
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", "passthrough", NUM_FEATURES),
                    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
                ]
            )

            #Define the model
            model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                verbose=2
            )

            #Create the pipeline with the preprocessor and the model
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", model)
                ]
            )

            #Train the model
            pipeline.fit(X_train, y_train)

            #Predict on the test set
            preds = pipeline.predict(X_test)

            #Compute evaluation metrics
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)

            #Log metrics and row out dataset 
            logger.log_metrics(rmse=rmse,mae=mae)
            logger.log_output(df)
            joblib.dump(pipeline, self.model_path)

        except Exception as e:
            logger.log_critical(e)

        finally:
            logger.write()

        