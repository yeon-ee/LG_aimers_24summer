import pandas as pd
from typing import Union

def split_data(data : pd.DataFrame)-> tuple[pd.DataFrame, pd.DataFrame]:
        df1 = data[data['Equipment_Dam'] == 'Dam dispenser #1']
        df2 = data[data['Equipment_Dam'] == 'Dam dispenser #2']
        return df1, df2