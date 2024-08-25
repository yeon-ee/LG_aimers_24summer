from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Union

class PCAProcessor:
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, options: Union[int, List[int]]):
        """
        Initialize the PCAProcessor with train and test DataFrames and the processing options.
        
        Parameters:
        train_df (pd.DataFrame): The training DataFrame.
        test_df (pd.DataFrame): The testing DataFrame.
        options (int or list of int): The processing option(s). 
                                      1: Perform PCA on HEAD NORMAL COORDINATE columns.
                                      2: Perform PCA on Stage columns.
                                      3: Perform PCA on Head columns.
        """
        self.train_df: pd.DataFrame = train_df
        self.test_df: pd.DataFrame = test_df
        self.options: List[int] = [options] if isinstance(options, int) else options
        
        # Define the column sets based on the options
        self.columns_to_pca: List[Tuple[List[str], int]] = []
        self._initialize_column_sets()

    def _initialize_column_sets(self) -> None:
        """
        Initialize the column sets based on the options provided.
        """
        # Define the column sets based on options
        head_normal_coordinate_columns = [col for col in self.train_df.columns if col.startswith("HEAD NORMAL COORDINATE")]
        stage_columns = [col for col in self.train_df.columns if col.startswith("Stage")]
        head_columns = [col for col in self.train_df.columns if col.startswith("Head")]

        if 1 in self.options:
            self.columns_to_pca.append((head_normal_coordinate_columns, 5))
        if 2 in self.options:
            self.columns_to_pca.append((stage_columns, 4))
        if 3 in self.options:
            self.columns_to_pca.append((head_columns, 2))

    def standardize_data(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Standardize the data for the specified columns.
        
        Parameters:
        df (pd.DataFrame): The DataFrame to standardize.
        columns (list): The list of columns to standardize.
        
        Returns:
        pd.DataFrame: The standardized data.
        """
        scaler = StandardScaler()
        return pd.DataFrame(scaler.fit_transform(df[columns]), columns=columns)
    
    def perform_pca(self, columns: List[str], n_components: int) -> Tuple[pd.DataFrame, PCA]:
        """
        Perform PCA on the standardized data for the train dataset and return the PCA object.
        
        Parameters:
        columns (list): The list of columns to apply PCA on.
        n_components (int): The number of principal components to keep.
        
        Returns:
        Tuple[pd.DataFrame, PCA]: DataFrame with the principal components for the train dataset and the fitted PCA object.
        """
        standardized_train_data = self.standardize_data(self.train_df, columns)
        pca = PCA(n_components=n_components)
        pca_train_components = pca.fit_transform(standardized_train_data)
        pca_df = pd.DataFrame(
            data=pca_train_components, 
            columns=[f'PC{i+1}_{columns[0].split()[0]}' for i in range(pca_train_components.shape[1])]
        )
        return pca_df, pca
    
    def transform_test_data(self, pca: PCA, columns: List[str]) -> pd.DataFrame:
        """
        Apply the same PCA transformation to the test data using the PCA object from the train data.
        
        Parameters:
        pca (PCA): The fitted PCA object from the train data.
        columns (list): The list of columns to apply PCA on.
        
        Returns:
        pd.DataFrame: DataFrame with the principal components for the test dataset.
        """
        standardized_test_data = self.standardize_data(self.test_df, columns)
        pca_test_components = pca.transform(standardized_test_data)
        pca_test_df = pd.DataFrame(
            data=pca_test_components, 
            columns=[f'PC{i+1}_{columns[0].split()[0]}' for i in range(pca_test_components.shape[1])]
        )
        return pca_test_df
    
    def replace_with_pcs(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Replace the original columns with the selected principal components in both train and test DataFrames.
        
        Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The train and test DataFrames with the original columns replaced by the selected principal components.
        """
        train_final = self.train_df.copy()
        test_final = self.test_df.copy()
        
        for columns, n_pcs in self.columns_to_pca:
            if columns:
                pca_train_df, pca = self.perform_pca(columns, n_pcs)
                train_final = train_final.drop(columns=columns, axis=1)
                train_final = pd.concat([train_final.reset_index(drop=True), pca_train_df], axis=1)
                
                pca_test_df = self.transform_test_data(pca, columns)
                test_final = test_final.drop(columns=columns, axis=1)
                test_final = pd.concat([test_final.reset_index(drop=True), pca_test_df], axis=1)
        
        return train_final, test_final
