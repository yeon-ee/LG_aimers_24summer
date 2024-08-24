import pandas as pd
from typing import List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler

class Normalization:
    def __init__(self, df: pd.DataFrame, columns: Optional[List[str]] = None, method: str = 'min_max'):
        """
        Initialize the Normalization with a DataFrame and the columns to be normalized.
        
        Parameters:
        df (pandas.DataFrame): The input DataFrame.
        columns (list): The list of column names to normalize. Default is None.
        method (str): The normalization method to use. Default is 'min_max'.
        """
        self.df: pd.DataFrame = df
        self.columns: Optional[List[str]] = columns
        self.method: str = method
        self.normalized_df: Optional[pd.DataFrame] = self.normalize_data()
        
    def normalize_data(self) -> pd.DataFrame:
        """
        Normalize the data for the specified columns.
        
        Returns:
        pd.DataFrame: The normalized data.
        """
        if self.method == 'min_max':
            self.normalized_df = MinMaxScaler().fit_transform(self.df[self.columns])
        elif self.method == 'z_score':
            self.normalized_df = StandardScaler().fit_transform(self.df[self.columns])
        elif self.method == 'robust':
            self.normalized_df = RobustScaler().fit_transform(self.df[self.columns])
        elif self.method == 'max_abs':
            self.normalized_df = MaxAbsScaler().fit_transform(self.df[self.columns])
        else:
            raise ValueError("Invalid normalization method. Please choose either 'min_max' or 'z_score' or 'robust' or 'max_abs'.")
        return self.normalized_df
    
    def get_normalized_data(self) -> pd.DataFrame:
        """
        Get the normalized data.
        
        Returns:
        pd.DataFrame: The normalized data.
        """
        if self.normalized_df is None:
            raise ValueError("Data has not been normalized yet. Please run normalize_data() first.")
        return self.normalized_df
    
    def set_columns(self, columns: Optional[List[str]]) -> None:
        """_
        Set the columns to be used for normalization.
        """
        if columns is None:
            self.columns = [col for col in self.df.columns if self.df[col].dtype == 'int64' or self.df[col].dtype == 'float64']
        else:
            self.columns = columns