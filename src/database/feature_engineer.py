import sys
sys.path.append('/root/code_Bao/Vietnamese_stocks_forecasting') 

import pandas as pd
import numpy as np
from typing import Literal

from src.database.ts_fill_data import TSDataFill
from src.utils.decorators import timeit, tqdm_decor


class TimeEngine:
    """
    A class for engineering time-based features in a DataFrame.

    Args:
        dataframe (pandas.DataFrame): The DataFrame to perform time feature engineering on.
        date_col (str): The name of the date column in the DataFrame.
        special_days (Union[None, List[str]]): A list of special days to mark as holidays. Defaults to None.

    Attributes:
        dataframe (pandas.DataFrame): The DataFrame to perform time feature engineering on.
        date_col (str): The name of the date column in the DataFrame.
        special_days (Union[None, List[str]]): A list of special days to mark as holidays.

    Methods:
        engineer_time_features():
            Engineer time-based features such as holidays, day of the week, day of the month, etc., in the DataFrame.

    """
    def __init__(self, dataframe, date_col:str='time', special_days=None):
        self.dataframe = dataframe
        self.date_col = date_col
        self.special_days = special_days
        self.new_cols = []
    
    @timeit
    def generate_time_features(self):
        """
        Engineer time-based features in the DataFrame.

        Returns:
            pandas.DataFrame: The DataFrame with added time-based features.
        """
        self.dataframe[self.date_col] = pd.to_datetime(self.dataframe[self.date_col])
        # Sort the DataFrame by date_col in ascending order
        new_df = self.dataframe.sort_values(by=self.date_col, ascending=True)
        
        if self.special_days is not None:
            new_df['holidays'] = new_df[self.date_col].apply(lambda x: 1 if x in self.special_days else 0).astype(bool)

        new_df['day_in_week'] = new_df[self.date_col].dt.day_name().astype("category")
        new_df['day_in_month'] = new_df[self.date_col].dt.day.astype(np.uint8)
        new_df['day_in_year'] = new_df[self.date_col].dt.dayofyear.astype(np.uint16)
        new_df['month_in_year'] = new_df[self.date_col].dt.month.astype(np.uint8)
        new_df['weekend'] = (new_df['day_in_week'] == 'Saturday') | (new_df['day_in_week'] == 'Sunday')
        new_df["time_idx"] = (new_df[self.date_col] - new_df[self.date_col].min()).dt.days.astype(np.int32) # for pytorch forecasting

        # Compare columns between new_df and self.dataframe to find new columns
        self.new_cols = [col for col in new_df.columns if col not in self.dataframe.columns]

        return new_df


class PercentageChangeCalculator:
    def __init__(self, dataframe, pct_cols:list, 
                 date_col:str = 'time', group_id:str = "ticker",
                 fill_method=None):
        '''
        Initialize the PercentageChangeCalculator with a DataFrame, columns to calculate percentage change for, date column, and group_id.
        '''
        self.dataframe = dataframe
        self.pct_cols = pct_cols
        self.date_col = date_col
        self.group_id = group_id
        self.fill_method = fill_method
        self.new_cols = self.get_new_cols()

    @timeit
    def calculate_pct_change(self):
        '''
        Calculate percent changes for lagged columns.
        
        Args:
            fill_method (str, optional): Method for filling missing values in percent change columns ('bfill','constant', 'ffill', or None). Default is None.
        
        Returns:
            pd.DataFrame: The dataframe including percent change columns.
        '''
        new_df = self.dataframe.sort_values(self.date_col)
        new_df[self.pct_cols] = new_df[self.pct_cols].astype(np.float32)  # pct_change has error with float16
        for col in self.pct_cols:
            pct_col_name = f"{col}_pct"
            new_df[pct_col_name] = new_df.groupby([self.group_id])[col].pct_change()
            new_df[pct_col_name].replace([np.inf, -np.inf], np.nan, inplace=True)
        
            if self.fill_method is not None:
                fill_generator = TSDataFill(new_df.groupby([self.group_id])[pct_col_name], fill_method=self.fill_method)
                new_df[pct_col_name] = fill_generator.fill()
        return new_df

    def get_new_cols(self)-> list:
        new_cols =[]
        for col in self.pct_cols:
            pct_col_name = f"{col}_pct"
            new_cols.append(pct_col_name)
        return new_cols


class LagEngine:
    def __init__(self, dataframe, lag_dict=None, 
                 group_id='ticker', time_column='time',
                 fill_method=None):
        '''
        Initialize LagEngine with a dataframe and optional lag parameters.
        
        Args:
            dataframe (pd.DataFrame): The dataframe to create lagged features.
            lag_dict (dict, optional): Dictionary of variable names mapped to a list of time steps by which the variable should be lagged. Default is None.
            group_id (str, optional): The column to group by. Default is 'CUST_NO'.
            time_column (str, optional): The column representing time in 'YYYY-MM-DD' format. Default is 'time_idx'.
        '''
        self.dataframe = dataframe
        self.lag_dict = lag_dict
        self.group_id = group_id
        self.time_column = time_column
        self.fill_method = fill_method
        self.new_cols = self.get_new_cols()

    @timeit
    def create_lag_features(self):
        '''
        Create lagged columns in the dataframe if lag_dict is provided.
        
        Returns:
            pd.DataFrame: The dataframe including lagged features.
        '''
        result_df = self.dataframe.copy()
        if self.lag_dict is not None:
            for col, lags in self.lag_dict.items():
                for lag in lags:
                    new_col_name = f"{col}_lagged_by_{lag}"
                    result_df[new_col_name] = result_df.sort_values(self.time_column).groupby(self.group_id, observed=True)[col].shift(periods=lag, fill_value=None)
        
                    if self.fill_method is not None:
                        fill_generator = TSDataFill(dataframe=result_df[new_col_name], fill_method=self.fill_method)
                        result_df[new_col_name] = fill_generator.fill()

        return result_df
    
    def get_new_cols(self)-> list:
        new_cols =[]
        for col, lags in self.lag_dict.items():
            for lag in lags:
                new_col_name = f"{col}_lagged_by_{lag}"
                new_cols.append(new_col_name)
        return new_cols



class RollingMACalculator:
    def __init__(self, dataframe, MA_cols:list, 
                 window_widths:list = [10], group_id='ticker',
                 fill_method=None):
        self.dataframe = dataframe
        self.MA_cols = MA_cols
        self.window_widths = window_widths
        self.group_id = group_id
        self.fill_method = fill_method
        self.new_cols = self.get_new_cols()
        
    @timeit
    def calculate_MA(self) ->pd.DataFrame:
        """
        Calculate Moving Average for specified columns and window widths.

        Returns:
        pd.DataFrame: A DataFrame containing Moving Average columns.
        """
        # Create a copy of the original DataFrame to avoid modifying it
        result_df = self.dataframe.copy()

        result_df.set_index(self.group_id, inplace=True)

        # Loop through each window width
        for window_width in self.window_widths:
            # Loop through the columns to calculate MA
            for col in self.MA_cols:
                col_name_ma = f"{col}_MA_{window_width}"
                # Calculate the Moving Average within each group defined by 'CUST_NO' and store it in the result DataFrame
                result_df[col_name_ma] = result_df[col].rolling(window=window_width).mean().astype(np.float32)

                if self.fill_method is not None:
                    fill_generator = TSDataFill(dataframe=result_df[col_name_ma], fill_method=self.fill_method)
                    result_df[col_name_ma] = fill_generator.fill()

        # Reset index before returning the result
        result_df.reset_index(inplace=True)
        return result_df
    
    def get_new_cols(self)-> list:
        new_cols =[]
        for window_width in self.window_widths:
            # Loop through the columns to calculate MA
            for col in self.MA_cols:
                col_name_ma = f"{col}_MA_{window_width}"
                new_cols.append(col_name_ma)
        return new_cols

class DuplicateCheck:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def check_duplicate_rows(self):
        # Find duplicate rows
        duplicate_rows = self.dataframe[self.dataframe.duplicated()]
        # Count the number of duplicate rows
        number_of_duplicate_rows = len(duplicate_rows)
        print('There are {} duplicate rows.'.format(number_of_duplicate_rows))

    def remove_duplicate_rows(self):
        # First, check and print the number of duplicate rows
        self.check_duplicate_rows()
        new_df = self.dataframe.drop_duplicates()

        print("Duplicated rows has been removed!")
        return new_df
