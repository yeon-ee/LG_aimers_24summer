import pandas as pd
from typing import List, Optional, Tuple
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import category_encoders as ce

class Encoding:
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                 columns: Optional[List[str]] = None, method: str = 'one_hot', 
                 target_column: Optional[str] = None):
        """
        Initialize the Encoding with a train and test DataFrame, the columns to be encoded, and encoding method.
        
        Parameters:
        train_df (pd.DataFrame): The training DataFrame.
        test_df (pd.DataFrame): The testing DataFrame.
        columns (list, optional): The list of column names to encode. If None, infer categorical columns.
        method (str): The encoding method to use ('one_hot', 'label', 'binary', 'target').
        target_column (str, optional): The name of the target column, if applicable.
        """
        self.train_df = train_df.copy()
        self.test_df = test_df.copy()
        self.target_column = target_column
        self.target = None
        
        if self.target_column:
            self.target = self.train_df[self.target_column]  # Store the target column
            self.train_df = self.train_df.drop([self.target_column], axis=1)  # Drop the target column from the train DataFrame
        
        if columns is None:
            self.columns = self.set_columns()  # Set columns to be all non-numeric (categorical) columns
        else:
            self.columns = columns
        
        self.method = method
        self.encoder = None
    
    def set_columns(self) -> List[str]:
        """
        Set the columns to be used for encoding. Uses all non-numeric columns if none provided.
        
        Returns:
        List[str]: The list of column names to encode.
        """
        self.columns = [col for col in self.train_df.columns if self.train_df[col].dtype == 'object']
        return self.columns
    
    def handle_missing_values(self) -> None:
        """
        Handle missing values by filling them with a placeholder value before encoding.
        """
        self.train_df[self.columns] = self.train_df[self.columns].fillna('MISSING')
        self.test_df[self.columns] = self.test_df[self.columns].fillna('MISSING')
    
    def align_categories(self) -> None:
        """
        Align the categories between train and test data to ensure they have the same categories.
        """
        for col in self.columns:
            combined = pd.concat([self.train_df[col], self.test_df[col]], axis=0)
            combined = pd.Categorical(combined).categories
            
            self.train_df[col] = pd.Categorical(self.train_df[col], categories=combined)
            self.test_df[col] = pd.Categorical(self.test_df[col], categories=combined)
    
    def one_hot_encoding(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform One-Hot Encoding on the specified columns.
        
        Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Encoded training and testing DataFrames.
        """
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        train_encoded = self.encoder.fit_transform(self.train_df[self.columns])
        test_encoded = self.encoder.transform(self.test_df[self.columns])
        train_encoded_df = pd.DataFrame(train_encoded, columns=self.encoder.get_feature_names_out(self.columns))
        test_encoded_df = pd.DataFrame(test_encoded, columns=self.encoder.get_feature_names_out(self.columns))
        # Reset index to ensure proper concatenation
        train_final = pd.concat([self.train_df.drop(self.columns, axis=1).reset_index(drop=True), 
                                train_encoded_df.reset_index(drop=True)], axis=1)
        test_final = pd.concat([self.test_df.drop(self.columns, axis=1).reset_index(drop=True), 
                                test_encoded_df.reset_index(drop=True)], axis=1)
        return train_final, test_final
    
    def label_encoding(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform Label Encoding on the specified columns.
        
        Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Encoded training and testing DataFrames.
        """
        self.encoder = LabelEncoder()
        train_encoded_df = self.train_df.copy()
        test_encoded_df = self.test_df.copy()
        
        for col in self.columns:
            self.encoder.fit(pd.concat([self.train_df[col], self.test_df[col]], axis=0))
            train_encoded_df[col] = self.encoder.transform(self.train_df[col])
            test_encoded_df[col] = self.encoder.transform(self.test_df[col])
        
        return train_encoded_df, test_encoded_df
    
    def binary_encoding(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform Binary Encoding on the specified columns.
        
        Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Encoded training and testing DataFrames.
        """
        self.encoder = ce.BinaryEncoder(cols=self.columns)
        train_encoded = self.encoder.fit_transform(self.train_df)
        test_encoded = self.encoder.transform(self.test_df)
        return train_encoded, test_encoded
    
    def target_encoding(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform Target Encoding on the specified columns.
        
        Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Encoded training and testing DataFrames.
        """
        if not self.target_column:
            raise ValueError("Target encoding requires a target column.")
        
        self.encoder = ce.TargetEncoder(cols=self.columns)
        train_encoded = self.encoder.fit_transform(self.train_df, self.target)
        test_encoded = self.encoder.transform(self.test_df)
        return train_encoded, test_encoded
    
    def get_encoded_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform encoding on the data and return the encoded train and test DataFrames.
        
        Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Encoded training and testing DataFrames.
        """
        self.handle_missing_values()
        self.align_categories()

        if self.method == 'one_hot':
            train_encoded_df, test_encoded_df = self.one_hot_encoding()
        elif self.method == 'label':
            train_encoded_df, test_encoded_df = self.label_encoding()
        elif self.method == 'binary':
            train_encoded_df, test_encoded_df = self.binary_encoding()
        elif self.method == 'target':
            train_encoded_df, test_encoded_df = self.target_encoding()
        else:
            raise ValueError("Invalid encoding method. Please choose one of 'one_hot', 'label', 'binary', 'target'.")

        # If target column exists, add it back to the encoded dataframes
        if self.target_column:
            train_encoded_df[self.target_column] = self.target

        return train_encoded_df, test_encoded_df