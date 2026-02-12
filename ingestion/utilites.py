import yaml 
from typing import Tuple
import pandas as pd

class FunctionsUtilites : 
    '''
    Functions used in all the scripts
    '''
    def find_schema_ingestion(layer:str,name:str) -> str:
        '''
        Find the path of a table schema for the ingestion
        
        layer(str): layer of the ingestion (bronze/silver)
        name(str): name of the table

        Returns:
        path of the table schema 
        '''
        return f'schema/{layer}_{name}.yaml'
    
    def load_yaml(path:str)-> dict:
        '''
        Load yaml file
        
        path(str): yaml file path
        Returns:
        dict
        '''
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def validate_schema(df: pd.DataFrame, SCHEMA: dict)-> dict:
        '''
        Validation of the df by the rules of the schema
        
        df(pd.DataFrame): table to check
        schema(dict): rules for the table

        Returns:
        errors(dict): schema errors in the table
        '''
        errors = []

        expected_columns = {col["name"]: col for col in SCHEMA["columns"]}

        # Missing columns
        for col in expected_columns:
            if col not in df.columns:
                errors.append(f"Missing column: {col}")

        # Columns unexpected
        for col in df.columns:
            if col not in expected_columns:
                errors.append(f"Unexpected column: {col}")

        # Nullable check
        for col_name, spec in expected_columns.items():
            if col_name not in df.columns:
                continue

            if not spec["nullable"] and df[col_name].isnull().any():
                errors.append(f"Null values in non-nullable column: {col_name}")

        return errors
    
    def edit_type_columns_schema(df: pd.DataFrame, SCHEMA: dict) -> pd.DataFrame:
        '''
        Cast DataFrame columns to the types defined in the schema
        
        df(pd.DataFrame): Input DataFrame whose columns must be cast
        SCHEMA(dict): Dictionary containing a columns key (type,name...)

        Returns
        df(pd.DataFrame): DataFrame with updated column types
        '''
        for col_def in SCHEMA["columns"]:
                col_name = col_def["name"]
                 # default to string if type not specified
                col_type = col_def.get("type", "string") 
                # Transform the type in int
                if col_type == "int":
                    df[col_name] = df[col_name].astype("Int64")  
                #Transform the type in float
                elif col_type == "float":
                    df[col_name] = df[col_name].astype("Float64")  
                # Transform the type in string
                elif col_type == "string":
                    df[col_name] = df[col_name].astype("string")
                # Transform the type in boolean
                elif col_type == "bool":
                    df[col_name] = df[col_name].astype("boolean") 
                # Transform the type in date
                elif col_type == "date":
                    df[col_name] = pd.to_datetime(df[col_name], errors="coerce")

        return df
    
    def keep_columns_schema(df: pd.DataFrame, SCHEMA: dict) -> pd.DataFrame:
        '''
        Filter the dataframe according to the schema definition  
        
        df(pd.DataFrame):input DataFrame containing all available columns
        SCHEMA(dict): dict with the columns needed

        Returns
        df(pd.DataFrame): df with filtered columns
        '''
        #Collect the columns in the schema
        schema_columns = [col["name"] for col in SCHEMA["columns"]]

        for col in schema_columns:
            # Add a nullable column if missisng in the df
            if col not in df.columns:
                df[col] = pd.NA
        df = df[schema_columns]

        return df
    

    def open_feature(config:dict)-> Tuple[list[str], list[str], str]:
        '''
        Extract feature definitions and target column from the configuration

        config(dict): Configuration dictionary
        Returns
        tuple
            - numerical_features (List[str])
            - categorical_features (List[str])
            - target (str)
        '''
        NUM_FEATURES = config["features"]["numerical"]
        CAT_FEATURES = config["features"]["categorical"]
        TARGET = config["dataset"]["target"]
        return NUM_FEATURES, CAT_FEATURES, TARGET
