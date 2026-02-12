import pandas as pd
import joblib
from datetime import datetime
from ingestion.utilites import FunctionsUtilites
from pathlib import Path
from ml.log_ml import MLLogger





class Predict:
    '''
    Process to run predictions using a trained model
    '''
    def __init__(self,name:str)-> None:
        '''
        Initialize parameters
        name(str): name of the dataset
        '''
        self.step='predict'
        self.name = name
        #Path to feature schema
        self.schema_path = Path(f'ml/schema/{name}_features.yaml') 
        #Path to the trained model
        self.model_path = Path(f'data/ml/models/{name}_model.joblib') 
        #Path to input data 
        self.input_path =  Path(f'data/ml/fault/fault_{name}.csv')
        #Path where predictions will be saved
        self.output_path = Path(f'data/ml/predict/predict_{name}.csv') 
        pass



    def predict(self):
        '''
        Run predictions 

        Steps:
        - Load feature schema
        - Load trained pipeline
        - Load input dataset
        - Extract features
        - Predict target
        - Log predictions
        - Save predictions
        '''
        # Initialize logger for the prediction
        logger = MLLogger(self.step,self.name)

        try:
            #Load feature schema
            schema = FunctionsUtilites.load_yaml(self.schema_path)
            NUM_FEATURES, CAT_FEATURES, TARGET = FunctionsUtilites.open_feature(schema)

            #Load trained pipeline
            pipeline = joblib.load(self.model_path)

            #Load input data
            df = pd.read_csv(self.input_path)
            #Log input row in dataset
            logger.log_input(df)

            #Select only feature columns
            X = df[NUM_FEATURES + CAT_FEATURES]

            #Run predictions
            preds = pipeline.predict(X)
            #Put the predictions stats in the logs
            logger.log_predictions(preds)

            df["pred"] = preds

            #Log row out dataset 
            logger.log_output(df)
            df.to_csv(self.output_path, index=False)

        except Exception as e:
            logger.log_critical(e)

        finally:
            logger.write() 

    

