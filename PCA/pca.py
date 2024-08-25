from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Union

class PCAProcessor:
    def __init__(self, df: pd.DataFrame, options: Union[int, List[int]]):
        """
        Initialize the PCAProcessor with a DataFrame and the processing options.
        
        Parameters:
        df (pandas.DataFrame): The input DataFrame.
        options (int or list of int): The processing option(s). 
                                      1: Perform PCA on HEAD NORMAL COORDINATE columns.
                                      2: Perform PCA on Stage columns.
                                      3: Perform PCA on Head columns.
        """
        self.df: pd.DataFrame = df
        self.options: List[int] = [options] if isinstance(options, int) else options
        
        # Define the column sets based on the options
        self.columns_to_pca: List[Tuple[List[str], int]] = []
        self._initialize_column_sets()

    def _initialize_column_sets(self) -> None:
        """
        Initialize the column sets based on the options provided.
        """
        # Define the column sets based on options
        head_normal_coordinate_columns = [col for col in self.df.columns if col.startswith("HEAD NORMAL COORDINATE")]
        stage_columns = [col for col in self.df.columns if col.startswith("Stage")]
        head_columns = [col for col in self.df.columns if col.startswith("Head")]

        if 1 in self.options:
            self.columns_to_pca.append((head_normal_coordinate_columns, 5))
        if 2 in self.options:
            self.columns_to_pca.append((stage_columns, 4))
        if 3 in self.options:
            self.columns_to_pca.append((head_columns, 2))

    def standardize_data(self, columns: List[str]) -> pd.DataFrame:
        """
        Standardize the data for the specified columns.
        
        Parameters:
        columns (list): The list of columns to standardize.
        
        Returns:
        pd.DataFrame: The standardized data.
        """
        scaler = StandardScaler()
        return pd.DataFrame(scaler.fit_transform(self.df[columns]), columns=columns)
    
    def perform_pca(self, columns: List[str], n_components: int) -> Tuple[pd.DataFrame, PCA]:
        """
        Perform PCA on the standardized data.
        
        Parameters:
        columns (list): The list of columns to apply PCA on.
        n_components (int): The number of principal components to keep.
        
        Returns:
        Tuple[pd.DataFrame, PCA]: DataFrame with the principal components and the fitted PCA object.
        """
        standardized_data = self.standardize_data(columns)
        pca = PCA(n_components=n_components)
        pca_components = pca.fit_transform(standardized_data)
        pca_df = pd.DataFrame(
            data=pca_components, 
            columns=[f'PC{i+1}_{columns[0].split()[0]}' for i in range(pca_components.shape[1])]
        )
        return pca_df, pca
    
    def plot_elbow(self, pca: PCA) -> None:
        """
        Plots an elbow plot to help determine the number of principal components to select.
        
        Parameters:
        pca: The fitted PCA object.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
                 pca.explained_variance_ratio_.cumsum(), marker='o')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Elbow Plot')
        plt.grid(True)
        plt.show()
    
    def replace_with_pcs(self) -> pd.DataFrame:
        """
        Replace the original columns with the selected principal components in the DataFrame.
        
        Returns:
        pd.DataFrame: The DataFrame with the original columns replaced by the selected principal components.
        """
        df_final = self.df.copy()
        
        for columns, n_pcs in self.columns_to_pca:
            if columns:
                pca_df, pca = self.perform_pca(columns, n_pcs)
                df_final = df_final.drop(columns=columns, axis=1)
                df_final = pd.concat([df_final.reset_index(drop=True), pca_df], axis=1)
        
        return df_final