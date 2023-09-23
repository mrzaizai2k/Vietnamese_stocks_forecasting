import sys
sys.path.append('/root/code_Bao/Vietnamese_stocks_forecasting') 

import pandas as pd
import argparse
from vnstock import * #Load vietnamese data

from src.utils.decorators import timeit, tqdm_decor



@tqdm_decor
@timeit
def fetch_and_save_stock_data(stocks:list = ['MWG', 'DGC', 'FPT'], start_date:str="2021-01-01", end_date:str="2021-07-18"):
    for stock in stocks:
        # Fetch historical data for the given stock
        df = stock_historical_data(symbol=stock,
                                    start_date=start_date,
                                    end_date=end_date,
                                    resolution='1D',
                                    type='stock')

        # Define the root directory for storing data
        root_DB = 'data/raw/historical_price'
        path = f"{root_DB}/{stock}.csv"

        # Save the DataFrame to a CSV file
        df.to_csv(path, index=False)
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch and save stock data.')
    parser.add_argument('--stocks', nargs='+', default=['MWG', 'DGC', 'FPT'], help="List of stock symbols")
    parser.add_argument('--start_date', type=str, default='2021-01-01', help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', type=str, default='2021-07-18', help='End date in YYYY-MM-DD format')
    
    args = parser.parse_args()
    
    print(f"Fetching and saving data for {args.stocks}...")
    fetched_data = fetch_and_save_stock_data(args.stocks, args.start_date, args.end_date)
    print(f"Data for {args.stocks} saved to {fetched_data}")