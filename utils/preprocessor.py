import pandas as pd
import numpy as np

def null_remover(sheets):
  # Numeric Null Remover
  for sheet_name, df in sheets.items():
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        mean_value = df[column].mean()
        df[column].fillna(mean_value, inplace=True)
    sheets[sheet_name] = df

  # Object Null Remover
  for sheet_name, df in sheets.items():
    # Select only object columns
    object_columns = df.select_dtypes(include=[object]).columns
    # Fill null values in object columns with an empty string ('')
    df[object_columns] = df[object_columns].fillna('neutral')
    sheets[sheet_name] = df

  return sheets
  

def sheet_loader(sheet_path, y):
  excel_data = pd.read_excel(sheet_path, sheet_name=None)
  sheet_names = excel_data.keys()
  
  sheets = {}
  for sheet_name in sheet_names:
    sheets[sheet_name] = excel_data[sheet_name]
    
  # Null removed
  sheets = null_remover(sheets)
  
  y_class = []
  for i in sheets[list(sheets.keys())[0]]['User-Id']:
    y_class.append(int(y[y['id'] == i].score))
  y = pd.DataFrame(y_class, columns=['score'])

  return sheets, y