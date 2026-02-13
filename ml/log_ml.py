import csv
import os
from datetime import datetime
import pandas as pd



class MLLogger:
    '''
    Logger for ML pipeline steps (features, training, prediction).
    '''
    def __init__(self, step:str, dataset:str):
        '''
        Initialize parameters

        step(str): name of the pipeline step 
        dataset(str): name of the dataset
        '''
        #Path to the CSV log file
        self.LOG_PATH = "logs/ml_log.csv"

        #Dictionary to store all log information
        self.event = {
            "pipeline_stage":"ml", # Stage of the pipeline
            "step":step, # ML Step
            "dataset": dataset, # Dataset name
            "status": "SUCCESS", # Current status: SUCCESS, FAILED, CRITICAL
            "rows_in": None, # Number of input rows
            "rows_out": None, # Number of output rows
            "error_count": 0, # Number of errors
            "errors": [], # List of errors

            # Feature layer
            "missing_rate": None, # Average missing rate of input features

            # Training layer
            "rmse": None, # Mean Squared Error
            "mae": None, # Mean Absolute Error

            # Predict layer
            "prediction_mean": None, # Mean of predicted values
            "prediction_std": None, # Standard deviation of predicted values
            "prediction_min": None, # Minimum predicted value
            "prediction_max": None, # Maximum predicted value

            "timestamp": None # Timestamp of logging
        }


    def log_input(self, df:pd.DataFrame) -> None:
        '''
        Log number of input rows
        df(DataFrame): input dataframe before the ML step
        '''
        self.event["rows_in"] = len(df)


    def log_output(self, df:pd.DataFrame) -> None:
        '''
        Log number of output rows

        df(DataFrame): output dataframe after the ML step
        '''
        self.event["rows_out"] = len(df)

    def log_dataframe_stats(self, df:pd.DataFrame) -> None:
        '''
        Log missing values about the dataframe

        df(DataFrame): dataframe to analyze
        '''
        self.event["missing_rate"] = float(df.isnull().mean().mean())

    def log_metrics(self, rmse=None, mae=None) -> None:
        '''
        Log training metrics

        rmse(float): Root Mean Squared Error
        mae(float): Mean Absolute Error
        '''
        self.event["rmse"] = rmse
        self.event["mae"] = mae

    def log_predictions(self, preds) -> None:
        '''
        Log prediction statistics

        preds(list): predictions 
        '''
        self.event["prediction_mean"] = float(preds.mean())
        self.event["prediction_std"] = float(preds.std())
        self.event["prediction_min"] = float(preds.min())
        self.event["prediction_max"] = float(preds.max())

    def log_errors(self, errors:list) -> None:
        '''
        Log any errors that occurred during the step

        errors(list): list of error messages
        '''
        if errors:
            self.event["status"] = "FAILED"
            self.event["error_count"] = len(errors)
            self.event["errors"] = errors

    def log_critical(self, exception:Exception) -> None:
        '''
        Log a critical error (exception) that stops the pipeline

        exception(Exception): exception object
        '''
        self.event["status"] = "CRITICAL"
        self.event["errors"] = [str(exception)]
    
    def write(self): 
            '''
            Write the current event log to a CSV file
            '''
            #Collect the timestamp
            self.event["timestamp"] = datetime.utcnow().isoformat()

            # Check if the file exists
            file_exists = os.path.isfile(self.LOG_PATH)

            # Open the file 
            with open(self.LOG_PATH, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.event.keys())

                # Write the CSV header only if the file is new
                if not file_exists:
                    writer.writeheader()
                    
                # Write the event data as a new row in the CSV file
                writer.writerow(self.event)