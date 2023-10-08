import sys
sys.path.append('/root/code_Bao/Vietnamese_stocks_forecasting') 
import pandas as pd
import numpy as np

class TypeConverter:
    def __init__(self, df, cols_purpose: dict):
        self.df = df
        self.cols_purpose = cols_purpose

    def convert_types(self):
        self.df = self.convert_date_columns()
        self.df = self.reduce_numeric_memory_usage()
        return self.df

    def convert_date_columns(self):
        date_cols = self.cols_purpose['date_cols']
        for col in date_cols:
            self.df[col] = pd.to_datetime(self.df[col], format='mixed', dayfirst=False, errors="coerce")
        return self.df

    def reduce_numeric_memory_usage(self):
        numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
        num_cols = self.cols_purpose['num_cols']

        for col in num_cols:
            col_type = self.df[col].dtypes
            if col_type in numerics:
                c_min = self.df[col].min()
                c_max = self.df[col].max()
                if str(col_type)[:3] == "int":
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        self.df[col] = self.df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        self.df[col] = self.df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        self.df[col] = self.df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        self.df[col] = self.df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        self.df[col] = self.df[col].astype(np.float32) # We would use float32 for all, float 16 contains so many errors
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        self.df[col] = self.df[col].astype(np.float32)
                    else:
                        self.df[col] = self.df[col].astype(np.float64)
        return self.df