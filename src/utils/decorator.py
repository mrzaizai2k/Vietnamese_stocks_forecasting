import time
import pandas as pd
from tqdm import tqdm
import psutil  # for memory usage

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} took {execution_time:.2f} seconds to execute.")
        return result

    return wrapper

def tqdm_decor(func):
    def wrapper(*args, **kwargs):
        iterable = func(*args, **kwargs)
        if hasattr(iterable, '__iter__'):
            iterable = tqdm(iterable)
        return iterable

    return wrapper

def mem_decor(func):
    def wrapper(*args, **kwargs):
        before_mem = psutil.virtual_memory().used
        result = func(*args, **kwargs)
        if isinstance(result, pd.DataFrame):
            after_mem = psutil.virtual_memory().used
            reduction = before_mem - after_mem
            print(f"Memory reduced by {reduction / (1024 * 1024):.2f} MB")
        return result

    return wrapper
