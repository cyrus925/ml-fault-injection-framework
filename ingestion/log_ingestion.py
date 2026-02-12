import csv
import os
from datetime import datetime
import pandas as pd



class IngestionLogger:
    '''
    Create a log file for the ingestion step
    '''
    def __init__(self, layer:str,dataset:str) -> None:
        '''
        Initialize parameters
        layer(str): layer of the transformation (bronze, silver, gold)
        dataset(str): name of the processed dataset 
        '''
        self.event = {
            "pipeline_stage":"ingestion",
            "layer":layer,
            "dataset": dataset,
            "status": "SUCCESS",
            "row_count": None,
            "missing_rate": None,
            "error_count": 0,
            "errors": [],
            "timestamp": None
        }
        self.log_path="logs/ingestion_log.csv"

    def log_dataframe_stats(self, df:pd.DataFrame) -> None:
        '''
        Save stats of the df
        
        df(DataFrame): dataframe processed
        '''
        # Count the number of rows
        self.event["row_count"] = len(df)

        # Average rate of missing values
        self.event["missing_rate"] = float(df.isnull().mean().mean())

    def log_errors(self, errors:list) -> None:
        '''
        Save error 
        
        errors(list) : list of the errors
        '''

        if errors:
            self.event["status"] = "FAILED"
            self.event["error_count"] = len(errors)
            self.event["errors"] = errors

    def log_critical(self, exception: Exception) -> None:
        '''
        Log a critical error from a caught exception
        
        exception(Exception)
        '''
        
        self.event["status"] = "CRITICAL"
        self.event["errors"] = [str(exception)]
    
    def write(self) -> None:  
            '''
            Write the current event log to a CSV file
            '''

            #Collect the timestamp
            self.event["timestamp"] = datetime.utcnow().isoformat()

            # Check if the file exists
            file_exists = os.path.isfile(self.log_path)

            # Open the file 
            with open(self.log_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.event.keys())

                # Write the CSV header only if the file is new
                if not file_exists:
                    writer.writeheader()

                # Write the event data as a new row in the CSV file
                writer.writerow(self.event)