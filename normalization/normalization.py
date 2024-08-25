import pandas as pd
from typing import List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
import numpy as np

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
        self.df = self.manage_missing_values()
        if columns is None:
            self.columns: Optional[List[str]] = self.set_columns()
        else:
            self.columns: Optional[List[str]] = columns
        self.method: str = method
        self.normalized_df: Optional[pd.DataFrame] = self.normalize_data()
        
    def normalize_data(self) -> pd.DataFrame:
        """
        Normalize the data for the specified columns.
        
        Returns:
        pd.DataFrame: The normalized data.
        """
        df = self.df.copy()
        if self.method == 'min_max':
            self.normalized_df = MinMaxScaler().fit_transform(df[self.columns])
        elif self.method == 'z_score':
            self.normalized_df = StandardScaler().fit_transform(df[self.columns])
        elif self.method == 'robust':
            self.normalized_df = RobustScaler().fit_transform(df[self.columns])
        elif self.method == 'max_abs':
            self.normalized_df = MaxAbsScaler().fit_transform(df[self.columns])
        else:
            raise ValueError("Invalid normalization method. Please choose either 'min_max' or 'z_score' or 'robust' or 'max_abs'.")
        return self.normalized_df
    
    def get_normalized_data(self) -> pd.DataFrame:
        """
        Get the normalized data.
        
        Returns:
        pd.DataFrame: The normalized data.
        """
        df = self.df.copy()
        df[self.columns] = self.normalized_df
        return df

    def set_columns(self, columns: Optional[List[str]] = None)-> List[str]:
        """_
        Set the columns to be used for normalization.
        """
        if columns is None:
            self.columns = [col for col in self.df.columns if self.df[col].dtype == 'int64' or self.df[col].dtype == 'float64']
        else:
            self.columns = columns
        return self.columns
    
    def manage_missing_values(self) -> pd.DataFrame:
        """
        Manage the missing values in the DataFrame.
        """
        df_copy = self.df.copy()
        df_copy.replace('OK', np.nan, inplace=True)  # Replace 'OK' with NaN
        
        # Drop columns with any missing values
        df_copy.dropna(axis=1, how='any', inplace=True)
        
        # Drop columns with only one unique value
        columns_to_drop = [col for col in df_copy.columns if df_copy[col].nunique() == 1]
        df_copy.drop(columns=columns_to_drop, inplace=True)
        
        self.df = df_copy
        return self.df