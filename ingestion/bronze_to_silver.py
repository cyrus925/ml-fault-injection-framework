import pandas as pd
from ingestion.log_ingestion import IngestionLogger
from ingestion.utilites import FunctionsUtilites
from pathlib import Path




class Bronze :
    '''
    Process to table bronze to silver

    '''
    def __init__(self,name:str)-> None:
        '''
        Initialize parameters
        name(str): name of the table
        '''
        self.layer='bronze'
        self.name = name
        self.source_path=Path(f'data/raw/{name}.csv')
        self.target_path=Path(f'data/silver/{name}.parquet')

        # Load the schema path for the bronze layer and the dataset
        self.schema_path=FunctionsUtilites.find_schema_ingestion(self.layer,name)
        pass

    def ingest(self) -> None:
        '''
        Load the raw data with the type/columns defined in the schema.
        '''

        # Initialize logger for this ingestion process
        logger = IngestionLogger(self.layer,self.name)

        try:
            #Ingest the raw data
            df = pd.read_csv(self.source_path)

            # Add row_count and missing_rate to the log
            logger.log_dataframe_stats(df)

            # Load the target schema 
            SCHEMA = FunctionsUtilites.load_yaml(self.schema_path)

            # Edit the column's type like the schema 
            df = FunctionsUtilites.edit_type_columns_schema(df, SCHEMA)

            # Keep the columns like the schema 
            df = FunctionsUtilites.keep_columns_schema(df, SCHEMA)
    
            # Validate the schema (ex: columns, is_nullable etc...)
            errors = FunctionsUtilites.validate_schema(df, SCHEMA)

            # Log any schema validation errors
            logger.log_errors(errors)

            #Save the df to the target path
            if not errors:
                df.to_parquet(self.target_path, engine="pyarrow", index=False)

            try:
                df.to_parquet(self.target_path, index=False)
            except Exception as e:
                logger.log_critical(e)


        except Exception as e:
            logger.log_critical(e)

        finally:
            logger.write()