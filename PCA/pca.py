from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

class PCAProcessor:
    def __init__(self, df: pd.DataFrame, columns: List[str]):
        """
        Initialize the PCAProcessor with a DataFrame and the columns to be used for PCA.
        
        Parameters:
        df (pandas.DataFrame): The input DataFrame.
        columns (list): The list of column names to apply PCA on.
        """
        self.df: pd.DataFrame = df
        self.columns: List[str] = columns
        self.scaler: StandardScaler = StandardScaler()
        self.pca: Optional[PCA] = None
        self.pca_df: Optional[pd.DataFrame] = None
    
    def standardize_data(self) -> pd.DataFrame:
        """
        Standardize the data for the specified columns.
        
        Returns:
        pd.DataFrame: The standardized data.
        """
        return pd.DataFrame(self.scaler.fit_transform(self.df[self.columns]), columns=self.columns)
    
    def perform_pca(self, n_components: Optional[int] = None) -> Tuple[pd.DataFrame, PCA]:
        """
        Perform PCA on the standardized data.
        
        Parameters:
        n_components (int, optional): The number of principal components to keep. If None, all components are kept.
        
        Returns:
        Tuple[pd.DataFrame, PCA]: DataFrame with the principal components and the fitted PCA object.
        """
        standardized_data = self.standardize_data()
        self.pca = PCA(n_components=n_components)
        pca_components = self.pca.fit_transform(standardized_data)
        self.pca_df = pd.DataFrame(
            data=pca_components, 
            columns=[f'PC{i+1}_{self.columns[0].split()[0]}' for i in range(pca_components.shape[1])]
        )
        return self.pca_df, self.pca
    
    def plot_elbow(self) -> None:
        """
        Plots an elbow plot to help determine the number of principal components to select.

        Parameters:
        pca: The fitted PCA object.
        """
        if self.pca is None:
            raise ValueError("PCA has not been performed yet. Please run perform_pca() first.")
        
        # Plot the explained variance ratio (cumulative)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.pca.explained_variance_ratio_) + 1), 
                 self.pca.explained_variance_ratio_.cumsum(), marker='o')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Elbow Plot')
        plt.grid(True)
        plt.show()
    
    def replace_with_pcs(self, n_pcs: int) -> pd.DataFrame:
        """
        Replace the original columns with the selected principal components in the DataFrame.
        
        Parameters:
        n_pcs (int): The number of principal components to keep.
        
        Returns:
        pd.DataFrame: The DataFrame with the original columns replaced by the selected principal components.
        """
        if self.pca_df is None:
            raise ValueError("PCA has not been performed yet. Please run perform_pca() first.")
        
        pca_selected_df = self.pca_df.iloc[:, :n_pcs]
        df_dropped = self.df.drop(columns=self.columns, axis=1)
        df_final = pd.concat([df_dropped.reset_index(drop=True), pca_selected_df], axis=1)
        
        return df_final