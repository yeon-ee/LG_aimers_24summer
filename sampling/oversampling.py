import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek


class Oversample:
    def __init__(self, data:pd.DataFrame, target:str, numeric_method:str, ratio:float):
        self.data: pd.DataFrame = data
        self.target: str = target
        self.numeric_method: str = numeric_method
        self.ratio: float = ratio
        self.random_state = 42  # Fixed random_state
        self.n_neighbors = 5    # Fixed n_neighbors
        self.label_encoders = {}


    def perform(self) -> pd.DataFrame:
        """
        Performs the oversampling process based on the initialized parameters.

        Returns:
        pd.DataFrame: The oversampled DataFrame.
        """
        X, y = self._separate_features_and_target()

        # Apply oversampling method to numeric data
        if self.numeric_method == 'SMOTE':
            oversampler = SMOTE(sampling_strategy=self.ratio, random_state=self.random_state, k_neighbors=self.n_neighbors)
        elif self.numeric_method == 'SMOTE-Tomek':
            oversampler = SMOTETomek(sampling_strategy=self.ratio, random_state=self.random_state, smote=SMOTE(k_neighbors=self.n_neighbors))
        else:
            raise NotImplementedError(f"Oversampling method '{self.numeric_method}' is not implemented. Please use 'SMOTE' or 'SMOTE-Tomek'.")

        X_smote_numeric, y_smote = oversampler.fit_resample(X.select_dtypes(include=['number']), y)

        # Handle non-numeric columns with KNN Imputer
        X_resampled = self._handle_non_numeric_columns(X, X_smote_numeric)

        # Combine with the target column
        df_resampled = pd.concat([X_resampled, pd.DataFrame(y_smote, columns=[self.target]).reset_index(drop=True)], axis=1)

        return df_resampled
    
    def _separate_features_and_target(self):
        """
        Separates the features and the target column from the DataFrame.
        
        """
        X = self.data.drop(columns=[self.target])
        y = self.data[self.target]
        return X, y
    
    def _handle_non_numeric_columns(self, X: pd.DataFrame, X_smote_numeric: pd.DataFrame) -> pd.DataFrame:
        """
        Handles the non-numeric columns by applying label encoding and imputation, then aligning with resampled numeric data.
        
        Returns:
        pd.DataFrame: The combined DataFrame with both numeric and non-numeric data.
        """
        numeric_columns = X.select_dtypes(include=['number']).columns
        non_numeric_columns = X.select_dtypes(exclude=['number']).columns

        if len(non_numeric_columns) > 0:
            non_numeric_data = X[non_numeric_columns].copy()
            for col in non_numeric_columns:
                if non_numeric_data[col].dtype == 'object':
                    le = LabelEncoder()
                    non_numeric_data[col] = le.fit_transform(non_numeric_data[col].astype(str))
                    self.label_encoders[col] = le

            # Use SimpleImputer for missing values in non-numeric data
            simple_imputer = SimpleImputer(strategy='most_frequent')
            X_imputed_non_numeric = simple_imputer.fit_transform(non_numeric_data)
            X_imputed_non_numeric = pd.DataFrame(X_imputed_non_numeric, columns=non_numeric_columns)

            # Duplicate non-numeric data to match the number of rows in SMOTE output
            X_imputed_non_numeric_resampled = pd.DataFrame(
                [X_imputed_non_numeric.iloc[i % len(X_imputed_non_numeric)] for i in range(len(X_smote_numeric))],
                columns=non_numeric_columns
            )
        else:
            X_imputed_non_numeric_resampled = pd.DataFrame(index=X_smote_numeric.index)  # Empty DataFrame if no non-numeric columns

        # Combine numeric and non-numeric data
        X_smote_numeric = pd.DataFrame(X_smote_numeric, columns=numeric_columns).reset_index(drop=True)
        X_imputed_non_numeric_resampled = X_imputed_non_numeric_resampled.reset_index(drop=True)
        X_resampled = pd.concat([X_smote_numeric, X_imputed_non_numeric_resampled], axis=1)
        
        return X_resampled