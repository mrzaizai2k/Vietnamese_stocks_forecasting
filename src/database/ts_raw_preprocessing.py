import sys
sys.path.append("")

import pandas as pd
import numpy as np
from typing import Literal

from src.database.ts_fill_data import *
from src.database.feature_engineer import *
from src.utils.convert_data_type import TypeConverter
from src.utils.config_parse import DataConfigReader
from src.utils.decorators import timeit, tqdm_decor


class TimeSeriesPreprocessingBase:
    def __init__(self, dataframe, 
                 data_config: DataConfigReader,
                group_id_col:str='ticker', 
                date_col:str='time',
                ):
        
        self.dataframe = dataframe
        self.data_config = data_config
        self.group_id_col = group_id_col
        self.date_col = date_col
        self.cols_purpose = self.read_cols_purpose()

    def read_cols_purpose(self):
        cols_purpose = self.data_config.read_data_purpose()
        return cols_purpose

    def convert_data_type(self):
        type_converter = TypeConverter(self.dataframe, cols_purpose=self.cols_purpose)
        self.dataframe = type_converter.convert_types()
        return self.dataframe
    
    def check_duplicate(self):
        # Find duplicate rows
        duplicate_checker = DuplicateCheck(self.dataframe)
        self.dataframe = duplicate_checker.remove_duplicate_rows()
    
        return self.dataframe


class HistoryStockPreprocess(TimeSeriesPreprocessingBase):
    def __init__(self, dataframe, 
                 data_config: DataConfigReader,
                group_id_col:str='ticker', 
                date_col:str='time',
                 special_days:dict=None,
                 window_width_list: list = [10], 
                 lag_dict=None,
                 ma_fill_method = None,
                 pct_fill_method = None,
                 lag_fill_method = None,
                 ):
        super().__init__(dataframe, data_config, group_id_col, date_col)

        self.special_days = special_days
        self.window_width_list = window_width_list
        self.lag_dict = lag_dict
        
        self.ma_fill_method = ma_fill_method
        self.pct_fill_method = pct_fill_method
        self.lag_fill_method = lag_fill_method


        self.cols_purpose = self.read_cols_purpose()
        self.pct_cols = self.cols_purpose['pct_cols']
        self.ma_cols = self.cols_purpose['ma_cols']
        self.lag_cols = self.cols_purpose['lag_cols']
 
    
    def create_MA(self):
        ma_generator = RollingMACalculator(self.dataframe,MA_cols=self.ma_cols, window_widths=self.window_width_list,
                                        fill_method = BackwardFillStrategy(limit = max(self.window_width_list)))
        ma_df = ma_generator.calculate_MA()
        return ma_df
    
    def create_pct_change(self):
        pct_generator = PercentageChangeCalculator(self.dataframe, pct_cols=self.pct_cols, 
                                                fill_method=ConstantFillStrategy(constant_value=0))
        pct_df = pct_generator.calculate_pct_change()
        return pct_df
    
    def transform_time(self):
        time_engine = TimeEngine(self.dataframe, date_col=self.date_col, 
                                 special_days=self.special_days)
        time_df = time_engine.generate_time_features()  
        return time_df
    
    def create_lag(self):
        lag_generator = LagEngine(self.dataframe, lag_dict=self.lag_dict,
                                fill_method=ForwardFillStrategy())
        lag_df = lag_generator.create_lag_features()
        return lag_df

    @timeit
    def get_preprocessing(self):
        self.convert_data_type()
        self.check_duplicate()
        self.dataframe = self.create_MA()
        self.dataframe = self.create_pct_change()
        self.dataframe = self.transform_time()
        self.dataframe = self.create_lag()
        return self.dataframe
