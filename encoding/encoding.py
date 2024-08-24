import pandas as pd
from typing import List, Optional
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, TargetEncoder
import category_encoders as ce

class Encoding:
    def __init__(self, df: pd.DataFrame, columns: Optional[List[str]] = None, method: str = 'one_hot', target_column: Optional[str] = None):
        """
        Initialize the Encoding with a DataFrame and the columns to be encoded.
        
        Parameters:
        df (pandas.DataFrame): The input DataFrame.
        columns (list): The list of column names to encode. Default is None.
        method (str): The encoding method to use. Default is 'one_hot'.
        target_column (str): The name of the target column, if applicable. Default is None.
        """
        self.df: pd.DataFrame = df
        self.target_column: Optional[str] = target_column
        self.target: Optional[pd.Series] = None
        
        if self.target_column:
            self.target = self.df[self.target_column]  # Store the target column
            self.df = self.df.drop([self.target_column], axis=1)  # Drop the target column from the DataFrame
        
        if columns is None:
            self.columns: Optional[List[str]] = self.set_columns()
        else:
            self.columns: Optional[List[str]] = columns
        
        self.method: str = method
        self.encoded_df: pd.DataFrame = self.encode_data()
    
    def set_columns(self) -> List[str]:
        """
        Set the columns to be used for encoding.
        """
        # If no columns are provided, use all categorical (object) columns
        self.columns = [col for col in self.df.columns if self.df[col].dtype == 'object']
        return self.columns
    
    def encode_data(self) -> pd.DataFrame:
        """
        Encode the columns based on the method.
        """
        if self.method == 'one_hot':
            return self.one_hot_encoding()
        elif self.method == 'label':
            return self.label_encoding()
        elif self.method == 'binary':
            return self.binary_encoding()
        elif self.method == 'hash':
            return self.hash_encoding()
        elif self.method == 'target':
            return self.target_encoding()
        else:
            raise ValueError("Invalid encoding method. Please choose one of the following: 'one_hot', 'label', 'binary', 'hash', 'target'.")
        
    def one_hot_encoding(self) -> pd.DataFrame:
        """
        Perform One-Hot Encoding on the specified columns.
        """
        encoder = OneHotEncoder(sparse_output=False)
        one_hot_encoded = encoder.fit_transform(self.df[self.columns])
        one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(self.columns))
        return pd.concat([self.df.drop(self.columns, axis=1), one_hot_df], axis=1)

    def label_encoding(self) -> pd.DataFrame:
        """
        Perform Label Encoding on the specified columns.
        """
        encoder = LabelEncoder()
        df_encoded = self.df.copy() 
        for col in self.columns:
            df_encoded[col] = encoder.fit_transform(df_encoded[col])
        return pd.concat([self.df.drop(self.columns, axis=1), df_encoded[self.columns]], axis=1)
    
    def binary_encoding(self) -> pd.DataFrame:
        """
        Perform Binary Encoding on the specified columns.
        """
        encoder = ce.BinaryEncoder(cols=self.columns)
        binary_encoded = encoder.fit_transform(self.df[self.columns])
        return pd.concat([self.df.drop(self.columns, axis=1), binary_encoded], axis=1)
    
    def hash_encoding(self) -> pd.DataFrame:
        """
        Perform Hash Encoding on the specified columns.
        """
        encoder = ce.HashingEncoder(cols=self.columns)
        hash_encoded = encoder.fit_transform(self.df[self.columns])
        return pd.concat([self.df.drop(self.columns, axis=1), hash_encoded], axis=1)
    
    
    def target_encoding(self) -> pd.DataFrame:
        """
        Perform Target Encoding on the specified columns.
        """
        encoder = TargetEncoder(smooth="auto")
        target_encoded = encoder.fit_transform(self.df[self.columns], self.target)
        target_df = pd.DataFrame(target_encoded, columns=encoder.get_feature_names_out(self.columns))
        return pd.concat([self.df.drop(self.columns, axis=1), target_df], axis=1)

    
    def get_encoded_data(self) -> pd.DataFrame:
        """
        Get the encoded DataFrame, with the target column restored if applicable.
        """
        if self.target_column:
            self.encoded_df[self.target_column] = self.target  # Restore the target column
        return self.encoded_df