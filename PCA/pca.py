import pandas as pd
from sklearn.decomposition import PCA
from typing import Tuple, Optional

def perform_pca(data : pd.Dataframe, times: Optional[int] = None) -> Tuple[pd.DataFrame, PCA]:
    pass