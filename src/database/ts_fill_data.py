from typing import Literal
import pandas as pd

class FillStrategy:
    def __init__(self, limit=None):
        self.limit = limit

    def fill(self, dataframe):
        pass

class BackwardFillStrategy(FillStrategy):
    def fill(self, dataframe):
        return dataframe.bfill(limit=self.limit)

class ForwardFillStrategy(FillStrategy):
    def fill(self, dataframe):
        return dataframe.ffill(limit=self.limit)

class ConstantFillStrategy(FillStrategy):
    def __init__(self, constant_value, limit=None):
        super().__init__(limit)
        self.constant_value = constant_value

    def fill(self, dataframe):
        return dataframe.fillna(value=self.constant_value, limit=self.limit)

class TSDataFill:
    def __init__(self, dataframe, fill_method: FillStrategy):
        self.dataframe = dataframe
        self.fill_method = fill_method

    def fill(self):
        return self.fill_method.fill(self.dataframe)

