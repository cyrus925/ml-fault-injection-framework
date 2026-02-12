import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path




class FaultPlayerInjector:

    def __init__(self, fault_rate=0.1, random_state=42):
        self.fault_rate = fault_rate
        np.random.seed(random_state)
        self.input_path = Path(f'data/gold/players.parquet') 
        self.output_path = Path("data/ml/fault/fault_players.csv")
        self.log_path= Path("logs/fault_log.csv")
        self.event = {
            "timestamp": None,
            "input_path": str(self.input_path),
            "rows_in": None,
            "rows_out": None,
            "fault_rate": fault_rate,

            # Fault metrics
            "age_corrupted": 0,
            "height_corrupted": 0,
            "unknown_positions": 0,
            "column_dropped": 0,

            # Status
            "status": "SUCCESS",
            "error_count": 0,
            "errors": []
        }




    def _write_log(self):
        self.event["timestamp"] = datetime.utcnow().isoformat()

        log_df = pd.DataFrame([self.event])

        try:
            if self.log_path.exists():
                existing = pd.read_csv(self.log_path)
                log_df = pd.concat([existing, log_df])
        except Exception:
            pass

        log_df.to_csv(self.log_path, index=False)


    def corrupt_age(self, df):
        mask = np.random.rand(len(df)) < self.fault_rate
        df.loc[mask, "age"] = np.random.choice(
            [-10, 0, 150, 300], size=mask.sum()
        )
        return mask.sum()

    def corrupt_height(self, df):
        mask = np.random.rand(len(df)) < self.fault_rate
        df.loc[mask, "height_in_cm"] = np.random.choice(
            [50, 300, None], size=mask.sum()
        )
        return mask.sum()

    def inject_unknown_category(self, df):
        mask = np.random.rand(len(df)) < self.fault_rate
        df.loc[mask, "position"] = "UNKNOWN_POSITION"
        return mask.sum()

    def drop_column(self, df, col):
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            return 1
        return 0


    def run(self):
        try:
            df = pd.read_parquet(self.input_path)
            self.event["rows_in"] = len(df)

        except Exception as e:
            self.event["status"] = "CRITICAL"
            self.event["error_count"] += 1
            self.event["errors"].append(f"READ_ERROR: {str(e)}")
            self._write_log()
            return

        # ---- Fault injections (isolées) ----
        for fault_name, fault_func in [
            ("age_corrupted", self.corrupt_age),
            ("height_corrupted", self.corrupt_height),
            ("unknown_positions", self.inject_unknown_category),
        ]:
            try:
                self.event[fault_name] = fault_func(df)
            except Exception as e:
                self.event["status"] = "DEGRADED"
                self.event["error_count"] += 1
                self.event["errors"].append(
                    f"{fault_name.upper()}_ERROR: {str(e)}"
                )

        # Column drop (volontairement dangereuse)
        try:
            self.event["column_dropped"] = self.drop_column(
                df, "country_of_citizenship"
            )
        except Exception as e:
            self.event["status"] = "DEGRADED"
            self.event["error_count"] += 1
            self.event["errors"].append(
                f"COLUMN_DROP_ERROR: {str(e)}"
            )

        # ---- Write output ----
        try:
            df.to_csv(self.output_path, index=False)
            self.event["rows_out"] = len(df)
        except Exception as e:
            self.event["status"] = "FAILED"
            self.event["error_count"] += 1
            self.event["errors"].append(f"WRITE_ERROR: {str(e)}")

        self._write_log()


