from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np 


def scaler(sheets):
    
    scaler = MinMaxScaler()

    # Normalize numeric values in each sheet
    for sheet_name, df in sheets.items():
        df = df.drop("User-Id", axis=1)
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        sheets[sheet_name] = df
    
    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()

    # Label encode object type columns in each sheet
    for sheet_name, df in sheets.items():
        object_columns = df.select_dtypes(include=[object]).columns
        for column in object_columns:
            df[column] = label_encoder.fit_transform(df[column])
        sheets[sheet_name] = df

    return sheets
        