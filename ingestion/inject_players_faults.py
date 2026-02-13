import pandas as pd
import numpy as np
from ingestion.log_ingestion import IngestionLogger
from pathlib import Path




class FaultPlayerInjector:
    """
    Class to inject faults into a players dataset for testing ML robustness.
    """
    def __init__(self, fault_rate=0.1, random_state=42):
        """
        Initialize the injector.

        fault_rate (float): Fraction of rows to be affected by faults
        random_state (int): Seed 
        """
        self.fault_rate = fault_rate
        np.random.seed(random_state)
        self.layer='fault'
        self.name='player'
        #Path of the gold path
        self.input_path = Path(f'data/gold/players.parquet') 
        #Path for the table with faults
        self.output_path = Path("data/ml/fault/fault_players.csv")

    def corrupt_age(self, df:pd.DataFrame)->int:
        """
        Corrupt the age column with unrealistic values.

        df (DataFrame): Players dataframe.

        Returns:
        int: number of row infected
        """
        mask = np.random.rand(len(df)) < self.fault_rate
        df.loc[mask, "age"] = np.random.choice(
            [-10, 0, 150, 300], size=mask.sum()
        )
        return mask.sum()

    def corrupt_height(self, df:pd.DataFrame)->int:
        """
        Corrupt the height column with unrealistic values.

        df (DataFrame): Players dataframe.

        Returns:
        int: number of row infected
        """
        mask = np.random.rand(len(df)) < self.fault_rate
        df.loc[mask, "height_in_cm"] = np.random.choice(
            [50, 300, None], size=mask.sum()
        )
        return mask.sum()

    def corrupt_position(self, df):
        """
        Corrupt the position column with unrealistic values.

        df (DataFrame): Players dataframe.

        Returns:
        int: number of row infected
        """
        mask = np.random.rand(len(df)) < self.fault_rate
        df.loc[mask, "position"] = "UNKNOWN_POSITION"
        return mask.sum()

    def drop_column(self, df:pd.DataFrame, col:str)->None:
        '''
        Delete a column

        df(DataFrame): player dataframe
        col(str): column to delete
        '''
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
        pass


    def run(self):
        # Initialize logger for this ingestion process
        logger = IngestionLogger(self.layer,self.name)

        try:
            #Ingest the gold data
            df = pd.read_parquet(self.input_path)
            # Add row_count and missing_rate to the log
            logger.log_dataframe_stats(df)
                

            # Fault injections
            errors_age = self.corrupt_age(df)
            logger.log_corrupted_rows(errors_age)
            errors_height = self.corrupt_height(df)
            logger.log_corrupted_rows(errors_height)
            errors_unknown_category = self.corrupt_position(df)
            logger.log_corrupted_rows(errors_unknown_category)


            # Keep the infected columns
            cols_to_delete = [col for col in df.columns if col not in ["age", "height_in_cm", "position"]]
            #Drop a random column
            if cols_to_delete:
                col_to_drop = np.random.choice(cols_to_delete)
                df.drop(columns=[col_to_drop], inplace=True)

            #Save the df to the target path
            df.to_csv(self.output_path, index=False)

        except Exception as e:
            logger.log_critical(e)

        finally:
            logger.write() 

