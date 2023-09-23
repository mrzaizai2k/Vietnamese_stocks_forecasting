import sys
sys.path.append('/root/code_Bao/Vietnamese_stocks_forecasting') 

import yaml

class ConfigReader:
    def __init__(self, path):
        self.path = path

    def read_config(self, section=None):
        with open(self.path, mode='r', encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if section is not None:
            config = config[section]

        return config

class DataConfigReader:
    def __init__(self, data_config_path:str='config/raw_data_config.yaml', 
                 dtype_config_path:str='config/data_type_dict.yaml', section:str='Historical_price'):
        self.section = section
        self.data_config_path = data_config_path
        self.dtype_config_path = dtype_config_path
        self.data_config = ConfigReader(path=data_config_path)
        self.dtype_config = ConfigReader(path=dtype_config_path)

    def read_column_names(self):
        column_names = self.data_config.read_config('LOAD_DATA')[self.section]
        return column_names

    def read_data_type(self):
        data_type = self.dtype_config.read_config(self.section)
        return data_type

if __name__ == "__main__":
    data_config_reader = DataConfigReader()
    column_names = data_config_reader.read_column_names()
    data_type = data_config_reader.read_data_type()

    print("Column Names:", column_names)
    print("Data Types:", data_type)
