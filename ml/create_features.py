import pandas as pd
from ingestion.utilites import FunctionsUtilites
from pathlib import Path
from ml.log_ml import MLLogger






class Features:
    '''
    Process to create ML features dataset from gold layer
    '''
    def __init__(self,name:str)-> None:
        '''
        Initialize parameters
        name(str): name of the dataset
        '''
        self.step='create_feature'
        self.name = name

        #Path to the gold dataset
        self.gold_path = Path(f'data/gold/{name}.parquet') 
        #Path to the ML feature schema
        self.schema_path = Path(f'ml/schema/{name}_features.yaml') 
        #Path where the features dataset will be saved
        self.features_path = Path(f'data/ml/features/{name}_features.parquet') 
        pass

    def build_features(self):
        '''
        Create the ML features dataset from the gold data.

        Steps:
        - Load gold dataset
        - Load feature schema (numerical, categorical, target)
        - Remove missing values
        - Keep only selected columns
        - Save features dataset
        '''

        #Initialize logger for the feature creation 
        logger = MLLogger(self.step,self.name)

        try:
            #Load the gold dataset
            df = pd.read_parquet(self.gold_path)

            #Log input row in and missing_rate dataset
            logger.log_input(df)
            logger.log_dataframe_stats(df)

            #Load feature schema
            schema = FunctionsUtilites.load_yaml(self.schema_path)
            # Extract informations from the schema
            NUM_FEATURES, CAT_FEATURES, TARGET = FunctionsUtilites.open_feature(schema)

            #Remove rows with missing values 
            df = df.dropna(subset=NUM_FEATURES + CAT_FEATURES + [TARGET])

            #Select only features and target columns
            selected_columns = NUM_FEATURES + CAT_FEATURES + [TARGET]
            df = df[selected_columns]

            #Log row out dataset 
            logger.log_output(df)
            df.to_parquet(self.features_path, index=False)

        except Exception as e:
            logger.log_critical(e)

        finally:
            logger.write()