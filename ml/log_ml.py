import csv
import os
from datetime import datetime

LOG_PATH = "logs/ml_log.csv"

class MLLogger:
    def __init__(self, step,dataset):
        self.event = {
            "pipeline_stage":"ml",
            "step":step,
            "dataset": dataset,
            "status": "SUCCESS",
            "rows_in": None,
            "rows_out": None,
            "error_count": 0,
            "errors": [],

            # Feature layer
            "missing_rate": None,

            # Training layer
            "rmse": None,
            "mae": None,

            # Predict layer
            "prediction_mean": None,
            "prediction_std": None,
            "prediction_min": None,
            "prediction_max": None,

            "timestamp": None
        }
    def log_input(self, df):
        self.event["rows_in"] = len(df)

    def log_output(self, df):
        self.event["rows_out"] = len(df)

    def log_dataframe_stats(self, df):
        self.event["missing_rate"] = float(df.isnull().mean().mean())

    def log_metrics(self, rmse=None, mae=None):
        self.event["rmse"] = rmse
        self.event["mae"] = mae

    def log_predictions(self, preds):
        self.event["prediction_mean"] = float(preds.mean())
        self.event["prediction_std"] = float(preds.std())
        self.event["prediction_min"] = float(preds.min())
        self.event["prediction_max"] = float(preds.max())

    def log_errors(self, errors):
        if errors:
            self.event["status"] = "FAILED"
            self.event["error_count"] = len(errors)
            self.event["errors"] = errors

    def log_critical(self, exception):
        self.event["status"] = "CRITICAL"
        self.event["errors"] = [str(exception)]
    
    def write(self):  
            self.event["timestamp"] = datetime.utcnow().isoformat()

            file_exists = os.path.isfile(LOG_PATH)

            with open(LOG_PATH, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.event.keys())

                if not file_exists:
                    writer.writeheader()

                writer.writerow(self.event)