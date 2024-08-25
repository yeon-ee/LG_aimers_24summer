import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

class Oversample:
    def __init__(self, data: pd.DataFrame, target: str, numeric_method: str, ratio: float):
        self.data: pd.DataFrame = data
        self.target: str = target
        self.numeric_method: str = numeric_method
        self.ratio: float = ratio
        self.random_state = 42
        self.n_neighbors = 5
        self.label_encoders = {}

        # Ensure the target column exists
        if self.target not in self.data.columns:
            raise ValueError(f"'{self.target}' column not found in the dataframe.")

    def perform(self) -> pd.DataFrame:
        # Convert 'None' and string 'nan' values to actual np.nan
        self.data[self.target] = self.data[self.target].replace('nan', np.nan)
        self.data[self.target] = self.data[self.target].fillna(np.nan)

        # Remove rows where the target is NaN
        self.data = self.data.dropna(subset=[self.target])

        X, y = self._separate_features_and_target()

        # Encode y labels if they are strings
        # if y.dtype == 'O':  # dtype 'O' means object, which is typically a string
        #     le = LabelEncoder()
        #     y = le.fit_transform(y)

        # Apply oversampling method to numeric data
        if self.numeric_method == 'SMOTE':
            oversampler = SMOTE(sampling_strategy=self.ratio, random_state=self.random_state, k_neighbors=self.n_neighbors)
        elif self.numeric_method == 'SMOTE-Tomek':
            oversampler = SMOTETomek(sampling_strategy=self.ratio, random_state=self.random_state, smote=SMOTE(k_neighbors=self.n_neighbors))
        else:
            raise NotImplementedError(f"Oversampling method '{self.numeric_method}' is not implemented. Please use 'SMOTE' or 'SMOTE-Tomek'.")

        X_smote_numeric, y_smote = oversampler.fit_resample(X.select_dtypes(include=['number']), y)

        # Handle non-numeric columns with Label Encoding and Simple Imputation
        X_resampled = self._handle_non_numeric_columns(X, X_smote_numeric)

        # Combine with the target column
        df_resampled = pd.concat([X_resampled, pd.DataFrame(y_smote, columns=[self.target]).reset_index(drop=True)], axis=1)

        return df_resampled
    
    def _separate_features_and_target(self):
        X = self.data.drop(columns=[self.target])
        y = self.data[self.target]
        return X, y
    
    def _handle_non_numeric_columns(self, X: pd.DataFrame, X_smote_numeric: pd.DataFrame) -> pd.DataFrame:
        numeric_columns = X.select_dtypes(include=['number']).columns
        non_numeric_columns = X.select_dtypes(exclude=['number']).columns

        if len(non_numeric_columns) > 0:
            non_numeric_data = X[non_numeric_columns].copy()
            for col in non_numeric_columns:
                if non_numeric_data[col].dtype == 'object':
                    le = LabelEncoder()
                    non_numeric_data[col] = le.fit_transform(non_numeric_data[col].astype(str))
                    self.label_encoders[col] = le

            simple_imputer = SimpleImputer(strategy='most_frequent')
            X_imputed_non_numeric = simple_imputer.fit_transform(non_numeric_data)
            X_imputed_non_numeric = pd.DataFrame(X_imputed_non_numeric, columns=non_numeric_columns)

            X_imputed_non_numeric_resampled = pd.DataFrame(
                [X_imputed_non_numeric.iloc[i % len(X_imputed_non_numeric)] for i in range(len(X_smote_numeric))],
                columns=non_numeric_columns
            )
        else:
            X_imputed_non_numeric_resampled = pd.DataFrame(index=X_smote_numeric.index)

        X_smote_numeric = pd.DataFrame(X_smote_numeric, columns=numeric_columns).reset_index(drop=True)
        X_imputed_non_numeric_resampled = X_imputed_non_numeric_resampled.reset_index(drop=True)
        X_resampled = pd.concat([X_smote_numeric, X_imputed_non_numeric_resampled], axis=1)
        
        return X_resampled

