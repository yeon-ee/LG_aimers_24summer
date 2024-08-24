import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

class Undersample:
    def __init__(self, data: pd.DataFrame, target: str, undersample_size: int):
        """
        Initializes the Undersample class with the given data and parameters.
        
        Parameters:
        data (pd.DataFrame): The input DataFrame.
        target (str): The name of the target column containing class labels.
        undersample_size (int, optional): The number of samples to undersample the majority class to.
        """
        self.data = data
        self.target = target
        self.undersample_size = undersample_size
        self.random_state = 42  # Fixed random_state

    def perform(self) -> pd.DataFrame:
        """
        Performs the undersampling process based on the initialized parameters.
        
        Returns:
        pd.DataFrame: The undersampled DataFrame.
        """
        X, y = self._separate_features_and_target()
        
        # Calculate the sampling strategy
        # Assuming you want to undersample the majority class to a fixed size
        class_counts = y.value_counts()
        majority_class = class_counts.idxmax()
        minority_class = class_counts.idxmin()
        sampling_strategy = {majority_class: self.undersample_size, minority_class: len(y[y == minority_class])}
        
        # Apply undersampling
        undersampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=self.random_state)
        X_resampled, y_resampled = undersampler.fit_resample(X, y)
        
        # Combine the resampled data
        df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=[self.target])], axis=1)
        
        return df_resampled

    def _separate_features_and_target(self):
        """
        Separates the features and the target column from the DataFrame.

        """
        X = self.data.drop(columns=[self.target])
        y = self.data[self.target]
        return X, y
