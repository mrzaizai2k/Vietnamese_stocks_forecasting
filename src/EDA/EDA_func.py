import sys
sys.path.append("")

import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

# Draw
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
from matplotlib.offsetbox import AnchoredText
from statsmodels.graphics.tsaplots import plot_pacf
import math

from vnstock import * #Load vietnamese data
import holidays # Load Vietnamese holidays

from tqdm import tqdm
import yaml

from datetime import datetime
from datetime import date
from datetime import timedelta

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

from src.utils.decorators import timeit, tqdm_decor


def plot_columns_with_time(df, columns_to_plot:list, plot_title:str ='cols over time', idx_col:str = 'index'):
    '''Draw columns over time with plotly'''
    # Sort the DataFrame by the 'Time' column
    df_sorted = df.sort_values(by=idx_col)
    
    fig = go.Figure()
    
    for column in columns_to_plot:
        fig.add_trace(go.Scatter(x=df_sorted[idx_col], y=df_sorted[column], mode='lines', name=column))

    fig.update_layout(
        title=plot_title,
        xaxis_title='time',
        yaxis_title='Values',
        legend_title='Columns'
    )
    
    fig.show()

def summarize_categoricals(df, show_levels=False):
    """
        Display uniqueness in each column
    """
    data = [[df[c].unique(), len(df[c].unique()), df[c].isnull().sum(), (df[c].isnull().sum()/len(df[c]))*100] for c in df.columns]
    df_temp = pd.DataFrame(data, index=df.columns,
                           columns=['Levels', 'No. of Levels', 'No. of Missing Values', 'Missing Values (%)'])
    return df_temp.iloc[:, 0 if show_levels else 1:]

class financial_flow_process():
    def __init__(self, symbol='MWG', report_type='incomestatement', report_range='quarterly', 
                 start_date=None, end_date=None, idx_col = 'index', null_threshold=90):
        self.symbol = symbol
        # Initialize start_date and end_date here
        self.start_date = start_date
        self.end_date = end_date
        self.idx_col = idx_col
        self.report_type = report_type
        self.report_range = report_range
        self.null_threshold = null_threshold
        self.exclude_columns = ['ticker', 'index']
        self.quarter_date_dict = {
                                    'Q1': '04-26',
                                    'Q2': '07-27',
                                    'Q3': '10-24',
                                    'Q4': '01-30'
                                }
        self.df = self._load_data()

    def _load_data(self):
        # Create a method to load the financial data based on symbol, report_type, and report_range
        # Replace this with your data loading logic
        return financial_flow(symbol=self.symbol, report_type=self.report_type, report_range=self.report_range).reset_index()
    
    @timeit
    def preprocess(self):
        self._convert_quar_idx_2_date()
        self._time_filter()
        self.null_columns = self.get_null_columns()
        self._drop_null_columns()
        self.columns_to_plot = list(set(self.df.columns) - set(self.null_columns) -set(self.exclude_columns))
        return self.df
    
    def _convert_quar_idx_2_date(self):
        # Create a new DataFrame with the desired changes
        df_new = self.df.copy()
        df_new['year'] = df_new[self.idx_col].str[:4]
        df_new['quarter'] = df_new[self.idx_col].str[-2:]
        df_new['date'] = df_new['quarter'].map(self.quarter_date_dict)
        df_new[self.idx_col] = df_new['year'] + '-' + df_new['date']
        
        # Drop the temporary columns (year, quarter, date) if needed
        df_new.drop(columns=['year', 'quarter', 'date'], inplace=True)
        
        # Assign the new DataFrame to self.df
        self.df = df_new
        return self.df

    
    def _time_filter(self):
        """
        Filter the DataFrame based on start_date and end_date.
        """
        if (self.start_date and self.end_date):
            mask = (self.df[self.idx_col] >= self.start_date) & (self.df[self.idx_col] <= self.end_date)
            self.df = self.df[mask]
        else:
            pass
        return self.df

    def get_null_columns(self):
        """
        Return a list of column names with null values exceeding the given threshold.

        Args:
            threshold (int): The threshold percentage for null values (default is 90).

        Returns:
            list: A list of column names with null values exceeding the threshold.
        """
        null_columns = []

        for column in self.df.columns:
            null_percentage = (self.df[column].isnull().sum() / len(self.df[column])) * 100
            if null_percentage > self.null_threshold:
                null_columns.append(column)

        return null_columns
   
    def _drop_null_columns(self):
        """
        Drop columns with null values exceeding the given threshold.

        Args:
            threshold (int): The threshold percentage for null values (default is 90).
        """
        self.df.drop(columns=self.null_columns, inplace=True)

    def summarize_categoricals(self, show_levels=False):
        """
            Display uniqueness in each column
        """
        df = self.df
        data = [[df[c].unique(), len(df[c].unique()), df[c].isnull().sum(), (df[c].isnull().sum()/len(df[c]))*100] for c in df.columns]
        df_temp = pd.DataFrame(data, index=df.columns,
                            columns=['Levels', 'No. of Levels', 'No. of Missing Values', 'Missing Values (%)'])
        return df_temp.iloc[:, 0 if show_levels else 1:]        

    def plot_columns_with_time(self, plot_title:str ='cols over time'):
        '''Draw columns over time with plotly'''
        # Sort the DataFrame by the 'Time' column
        df_sorted = self.df.sort_values(by=self.idx_col)
        fig = go.Figure()
        
        for column in self.columns_to_plot:
            fig.add_trace(go.Scatter(x=df_sorted[self.idx_col], y=df_sorted[column], mode='lines', name=column))

        fig.update_layout(
            title=plot_title,
            xaxis_title='time',
            yaxis_title='Values',
            legend_title='Columns'
        )
        
        fig.show()
    
    def plot_heat_map(self):
        f, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(self.df[self.columns_to_plot].corr(), vmax=.8, square=True);        


def convert_types(dataframe):
    dataframe["time"] = pd.to_datetime(dataframe['time'],format='mixed', dayfirst=False,errors="coerce") 
    dataframe['ticker'] = dataframe['ticker'].astype('category')
    dataframe['volume'] = dataframe['volume'].astype(int)
    for col in ['open','high','low','close']:
        dataframe[col] = dataframe[col].astype(np.uint16)
    return dataframe

def time_engineer(dataframe, holidays, date_col='time', holiday_offset:int = 3):
    dataframe[date_col] = pd.to_datetime(dataframe[date_col])
    dataframe['holidays'] = dataframe[date_col].apply(lambda x: 1 if (x + pd.DateOffset(holiday_offset)) in holidays else 0).astype(bool) # create lagged holidays
    dataframe['day_in_week'] = dataframe[date_col].dt.day_name()
    dataframe['day_in_month'] = dataframe[date_col].dt.day
    dataframe['day_in_year'] = dataframe[date_col].dt.dayofyear
    dataframe['month_in_year'] = dataframe[date_col].dt.month
    dataframe['weekend'] = (dataframe['day_in_week'] == 'Saturday') | (dataframe['day_in_week'] == 'Sunday')
    return dataframe

def plot_lag(x, lag=1, ax=None, **kwargs):
    x_ = x.shift(lag)
    y_ = x
    if ax is None:
        fig, ax = plt.subplots()
    scatter_kws = dict(
        alpha=0.75,
        s=3,
    )
    line = dict(color='C3', )

    ax = sns.regplot(x=x_,
                     y=y_,
                     scatter_kws=scatter_kws,
                     line_kws=line,
                     lowess=True,
                     ax=ax,
                     **kwargs)
    
    # Adding correlation on plot
    at = AnchoredText(
        f"{y_.corr(x_):.2f}",
        prop=dict(size="large"),
        frameon=True,
        loc="upper left",
    )
    at.patch.set_boxstyle("square, pad=0.0")
    ax.add_artist(at)
    ax.set(title=f"Lag {lag}", xlabel=x_.name, ylabel=y_.name)
    return ax

def plot_autocorrelation(x, lags=6, lagplot_kwargs={}, **kwargs):
    kwargs.setdefault("nrows", 2)
    kwargs.setdefault("ncols", math.ceil(lags / 2))
    kwargs.setdefault("figsize", (kwargs["ncols"] *2, 2  * 2 + 0.5))
    fig, axs = plt.subplots(sharex=True, sharey=True, squeeze=False, **kwargs)
    for ax, k in zip(fig.get_axes(), range(2 * kwargs["ncols"])):
        if k + 1 <= lags:
            ax = plot_lag(x, lag=k + 1, ax=ax, **lagplot_kwargs)
            ax.set_title(f"Lag #{k + 1}", fontdict=dict(fontsize=14))
            ax.set(xlabel="", ylabel="")
        else:
            ax.axis("off")
    plt.setp(axs[-1, :]
, xlabel=x.name)
    fig.tight_layout(w_pad=0.1, h_pad=0.1)
    return fig

@timeit
def create_lag_features(dataframe, lag_dict:dict = {'close': [1,2,30]}, group_id = 'ticker'):
    '''Create lagged columns
    Args:
        dataframe: the dataframe we want to create lagged features
        lag_dict: (Dict[str, List[int]]) dictionary of variable names mapped to list of time steps by which the variable should be lagged
    return:
        dataframe: the dataframe include lagged features
    '''
    for col, lags in lag_dict.items():
        for lag in lags:
            new_col_name = f"{col}_lagged_by_{lag}"
            dataframe[new_col_name] = dataframe.sort_values('time').groupby(group_id, observed=True)[col].shift(periods=lag, fill_value=None)
    
    return dataframe

class Plot:
    def __init__(self, dataframe, cols_2_plot:list, holiday:dict = None, ticker:list = [], time_mode:str ='day'):
        self.dataframe = dataframe
        self.cols_2_plot = cols_2_plot
        self.time_mode = time_mode
        self.ticker = ticker
        self.holiday = holiday

    def get_filtered_data(self):
        '''
        If provide ticker then filter the dataframe of that customer 
        '''
        if self.ticker:
            return self.dataframe.loc[self.dataframe['ticker'].isin(self.ticker)].sort_values(by='time', ascending=True)
        else:
            return self.dataframe
    
    @timeit
    def sum_transaction(self):
        '''
        Sum transaction as time_mode = ['day', 'month']
        '''
        filtered_df = self.get_filtered_data()
        sum_df = filtered_df.groupby('time')[self.cols_2_plot].sum()
        if self.time_mode == 'month':
            sum_df = sum_df.resample('M').sum().reset_index()
        elif self.time_mode == 'day':
            sum_df = sum_df.reset_index()
        
        if self.holiday:
            sum_df['holidays'] = sum_df['time'].apply(lambda x: 1 if x in self.holiday else 0)
        return sum_df
    
    
    @timeit
    def calculate_MA(self, window_width_list:list = [10], plot = False):
        '''
        Calculate Moving Average
        window_width_list:list = [10]
        plot: True/False: plot MA or not
        '''
        sum_df = self.sum_transaction()
        for window_width in window_width_list:
            for col in self.cols_2_plot:
                col_name_ma = f"{col}_MA_{window_width}"
                sum_df[col_name_ma] = sum_df[col].rolling(window=window_width).mean().astype(np.float32)
        if plot == True:
            self.plot_MA(sum_df, window_width_list)
        return sum_df
    
    def plot_MA(self, sum_df, window_width_list):
        '''Plot the moving averages'''
        fig = go.Figure()
        for col in self.cols_2_plot:
            for window_width in window_width_list:
                col_name_ma = f"{col}_MA_{window_width}"
                fig.add_trace(go.Scatter(x=sum_df['time'], y=sum_df[col_name_ma], mode='lines', name=f'{col}_MA_{window_width}'))

        # Mark holidays with markers on the plot
        if self.holiday is not None:
            self.add_holiday_2_plot(fig)
        fig.update_layout(title='Moving Averages',
                            yaxis=dict(title='Moving Average', rangemode='tozero'),
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                            xaxis_tickangle=45)
        fig.show()
    
    @timeit
    def calculate_pct_change(self):
        '''
        Calculate the percentage change of the value next row
        '''
        sum_df = self.sum_transaction()
        pct_change_df = sum_df[self.cols_2_plot].pct_change()
        # Replace "inf" with NaN
        pct_change_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        pct_change_df['time'] = sum_df['time']
        return pct_change_df
    
    def plot_seasonal_decomposition(self, model='additive', period=30):
        '''
        model : {"additive", "multiplicative"}
        Period: the T period
        '''
        sum_df = self.sum_transaction()
        for col in self.cols_2_plot:
            decomposed_result = sm.tsa.seasonal_decompose(sum_df[col], period=period, model=model)
            figure = decomposed_result.plot()
        plt.tight_layout()
        plt.show()
        
    def check_stationary(self):
        ''''Check if the time series is stationary with adfuller'''
        sum_df = self.sum_transaction()
        for col in self.cols_2_plot:
            result = adfuller(sum_df[col], autolag='AIC')
            print(f'{col} ADF Statistic: {result[0]}')
            print(f'{col} n_lags: {result[1]}')
            print(f'{col} p-value: {result[1]} \n\n')
            
        for key, value in result[4].items():
            print('Critial Values:')
            print(f'   {key}, {value}')

    def add_holiday_2_plot(self,fig):
        sum_df = self.sum_transaction()
        holiday_dates = sum_df[sum_df['holidays'] == 1]['time']
        for date in holiday_dates:
            fig.add_shape(dict(type="line",x0=date,x1=date,y0=0,y1=1,xref="x",yref="paper",line=dict(color="red", width=1),))
    
    @timeit
    def sum_plot(self):
        '''Plot the sum_transaction and calculate_pct_change'''
        sum_df = self.sum_transaction()
        pct_change_df = self.calculate_pct_change()
        
        fig = go.Figure()
        
        for col in self.cols_2_plot:
            fig.add_trace(go.Scatter(x=sum_df['time'], y=sum_df[col], mode='lines', name=f'Sum_{col}'))
            fig.add_trace(go.Scatter(x=pct_change_df['time'], y=pct_change_df[col], mode='lines', name=f'Pct_Change_{col}', yaxis="y2"))

        # Mark holidays with markers on the plot
        if self.holiday is not None:
            # holiday_dates = sum_df[sum_df['holidays'] == 1]['time']
            # for date in holiday_dates:
            #     fig.add_shape(dict(type="line",x0=date,x1=date,y0=0,y1=1,xref="x",yref="paper",line=dict(color="red", width=1),))
            self.add_holiday_2_plot(fig)
            
        # Update the layout to display the second y-axis
        fig.update_layout(title='Sum and Percentage Change',
                          yaxis=dict(title='Sum', rangemode='tozero'),
                          yaxis2=dict(title='Percentage Change %', overlaying='y', side='right'),
                          legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                          xaxis_tickangle=45)
        
        fig.show()
    def mean_by_day_in_week(self):
        '''Average transaction by week'''
        sum_df = self.sum_transaction()
        sum_df['day_in_week'] = sum_df['time'].dt.day_name()
        mean_by_day_in_week = sum_df.groupby('day_in_week')[self.cols_2_plot].mean()
        return mean_by_day_in_week
    def mean_by_day_in_month(self):
        '''Average transaction by month'''
        sum_df = self.sum_transaction()
        sum_df['day_in_month'] = sum_df['time'].dt.day
        mean_by_day_in_month = sum_df.groupby('day_in_month')[self.cols_2_plot].mean()
        return mean_by_day_in_month
    
    def mean_by_month_in_year(self):
        '''Average transaction by year'''
        sum_df = self.sum_transaction()
        sum_df['month_in_year'] = sum_df['time'].dt.month
        mean_by_month_in_year = sum_df.groupby('month_in_year')[self.cols_2_plot].mean()
        return mean_by_month_in_year

    def plot_mean_day_month_week(self, data=None):
        '''
        Plot sum data by week or month
        data: [mean_by_day_in_week(), mean_by_day_in_month(), mean_by_month_in_year()]
        If data is None, draw all 3 plots for mean_by_day_in_week, mean_by_day_in_month, and mean_by_month_in_year
        '''
        if data is None:
            data = [self.mean_by_day_in_week(), self.mean_by_day_in_month(), self.mean_by_month_in_year()]
        else:
            data = [data]
        plot_titles = ['Day in Week', 'Day in Month', 'Month in Year']
        for idx, data_item in enumerate(data):
            x = data_item.index  # x-axis: days in week or days in month
            y = np.array(data_item.values.tolist())  # y-axis: sum of the columns
            fig = go.Figure()
            for i, col in enumerate(self.cols_2_plot):
                fig.add_trace(go.Scatter(x=x, y=y[:, i], mode='lines+markers', name=col))
            title = plot_titles[idx]
            fig.update_layout(
                title=f"Mean of {', '.join(self.cols_2_plot)} by {title}",
                xaxis_title=title,
                yaxis_title='Mean',
            )
            fig.show()

def count_plot(df, col="col1", hue="col2", figsize=(1500, 500), xrotation=0):
    '''Vẽ mối quan hệ giữa 2 biến'''
    fig = px.histogram(df, x=col, color=hue, barmode='group', nbins=len(df[col].unique()))

    # Calculate percentage values and add them as text annotations
    total_counts = df.groupby([col, hue]).size().unstack().fillna(0)
    total_counts_percentage = total_counts.div(total_counts.sum(axis=1), axis=0) * 100
    annotations = []
    for category in total_counts_percentage.index:
        for group in total_counts_percentage.columns:
            percentage = total_counts_percentage.loc[category, group]
            annotations.append(
                {
                    "x": category,
                    "y": total_counts.loc[category, group],
                    "text": f"{percentage:.2f}%",
                    "showarrow": False,
                }
            )
    fig.update_layout(annotations=annotations)

    # Customize the layout
    fig.update_layout(
        barmode="group",
        legend=dict(x=1.02, y=0.5),
        width=figsize[0],
        height=figsize[1],
        xaxis_title=col,
        yaxis_title="Count",
    )

    # Rotate x-axis labels
    fig.update_xaxes(tickangle=xrotation)

    # Show the plot
    fig.show()

def top_k_count_plot(dataframe, col, k=10):
    
    top_branch_codes = dataframe[col].value_counts().nlargest(k)
    sorted_top_codes = top_branch_codes.sort_values(ascending=False).index
    # Create the subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    sns.countplot(data=dataframe, y=col, palette='magma', ax=axes[0], order=sorted_top_codes).set_title(f'Count of Top {k} {col}', fontsize=12)
    sns.countplot(data=dataframe, y=col, palette='mako', hue='RECORD_STATUS', ax=axes[1], order=sorted_top_codes).set_title(f'Count of {col} per RECORD_STATUS', fontsize=12)
    # Display the plots
    plt.tight_layout()
    plt.show()

def missing_values_visualize(dataframe, label_col = 'ticker'):
    # Hiển thị có bao nhiêu mising values ở mới feature
    plt.figure(figsize=(19,3))
    sns.heatmap((dataframe.drop(columns=label_col).isna().sum()).to_frame(name='train_na').T,cmap='Spectral', 
                annot=True, fmt='0.0f').set_title('count missing values', fontsize=14)
    plt.figure(figsize=(19,3))
    sns.heatmap(((dataframe.drop(columns=label_col).isna().sum()/dataframe.shape[0])*100).to_frame(name='train_na').T,cmap='Spectral', 
                annot=True, fmt='0.0f').set_title('count missing values in percentage', fontsize=14)
