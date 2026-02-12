import pandas as pd
from datetime import datetime
from ingestion.utilites import FunctionsUtilites
from ingestion.log_ingestion import IngestionLogger
from pathlib import Path



class Silver : 
    '''
    Process to table silver to gold
    '''
    def __init__(self,name:str)-> None:
        '''
        Initialize parameters
        name(str): name of the table
        '''
        self.layer='silver'
        self.name = name
        self.source_path=Path(f'data/silver/{name}.parquet')
        self.target_path=Path(f'data/gold/{name}.parquet')
        self.schema_path=Path(f'schema/{name}.yaml')

        # Load the schema path for the silver layer and the dataset
        self.schema_path=FunctionsUtilites.find_schema_ingestion(self.layer,name)
        pass
    
    def compute_age(self,birth_date:datetime) -> int:
        '''
        Returns the age with the birth date
        birth_date(date): player's date of birth

        Returns:
        age(int): player's age
        '''

        # Do the transformation only for the players table
        if pd.isnull(birth_date):
            return None
        
        # Collect today's date
        today = datetime.today()
        # Calculate age
        age = today.year - birth_date.year - (
            (today.month, today.day) < (birth_date.month, birth_date.day)
        )
        return age
    

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Create column age for the players table
        df(pd.DataFrame): player's df

        Returns:
        df(pd.DataFrame): player's df with the age column
        '''
        #Do the transformation for the players table
        if(self.name != "players"):
            return df
        
        # Create age column
        df["age"] = df["date_of_birth"].apply(self.compute_age)


        return df

    def load(self) -> None:
        '''
        Load the silver data with the transformation and columns for the gold data.
        '''

        # Initialize logger for this ingestion process
        logger = IngestionLogger(self.layer,self.name)

        try:
            #Ingest the silver data
            df = pd.read_parquet(self.source_path)

            # Add row_count and missing_rate to the log
            logger.log_dataframe_stats(df)
            
            #Add the column age
            df_transformed=self.transform(df)

            # Load the target schema 
            SCHEMA = FunctionsUtilites.load_yaml(self.schema_path)

            # Edit the column's type like the schema 
            df = FunctionsUtilites.edit_type_columns_schema(df, SCHEMA)

            # Keep the columns like the schema 
            df_gold = FunctionsUtilites.keep_columns_schema(df, SCHEMA)

            # Validate the schema (ex: columns, is_nullable etc...)
            errors = FunctionsUtilites.validate_schema(df_gold, SCHEMA)

            # Log any schema validation errors
            logger.log_errors(errors)

            #Save the df to the target path
            if not errors:
                df_gold.to_parquet(self.target_path, index=False)


        except Exception as e:
            logger.log_critical(e)

        finally:
            logger.write()        
        