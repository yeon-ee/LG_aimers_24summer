# python main.py -c configs/my.yaml
import matplotlib.pyplot as plt
import pycaret
from pycaret.classification import *    
import os
import logging
import yaml  #pip install pyyaml
import argparse
import pandas as pd
from typing import Tuple
from PCA.pca import PCAProcessor
from encoding.encoding import Encoding
from normalization.normalization import Normalization
from sampling.oversampling import Oversample
from sampling.undersampling import Undersample
from split.split_data import split_data


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(module)s:%(levelname)s:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load the YAML configuration file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def apply_normalization(train_data: pd.DataFrame, test_data: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply normalization to the train and test datasets."""
    col_to_normalize = config["normalization_config"].get("col_list")
    
    train_normalizer = Normalization(train_data, columns=col_to_normalize, method=config["normalization_config"]["method"])
    train_data = train_normalizer.manage_missing_values()
    train_data = train_normalizer.get_normalized_data()

    test_normalizer = Normalization(test_data, columns=col_to_normalize, method=config["normalization_config"]["method"])
    test_data = test_normalizer.manage_missing_values()
    test_data = test_normalizer.get_normalized_data()

    return train_data, test_data

def apply_encoding(train_data: pd.DataFrame, test_data: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply encoding to the train and test datasets."""
    col_to_encode = config["encoding_config"].get("col_list")
    
    train_encoder = Encoding(train_data, columns=col_to_encode, method=config["encoding_config"]["method"], target_column="target")
    train_data = train_encoder.get_encoded_data()
    test_encoder = Encoding(test_data, columns=col_to_encode, method=config["encoding_config"]["method"])
    test_data = test_encoder.get_encoded_data()

    return train_data, test_data

def apply_pca(train_data: pd.DataFrame, test_data: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply PCA to the train and test datasets."""
    col_to_pca = config["pca_config"]["col_list"]
    if col_to_pca is None or len(col_to_pca) == 0:
        raise ValueError("No columns provided for PCA. Please specify columns in the 'pca_config'.")

    train_pca_processor = PCAProcessor(train_data, columns=col_to_pca)
    train_data, _ = train_pca_processor.perform_pca()
    test_pca_processor = PCAProcessor(test_data, columns=col_to_pca)
    test_data, _ = test_pca_processor.perform_pca()

    return train_data, test_data

def apply_sampling(train_data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply sampling to the train dataset."""
    logger.info(f"Train_target distribution: {train_data.value_counts('target')}")
    
    if config["sampling_config"].get("undersampling_apply", False):
        undersample_size = config["sampling_config"]["undersampling_size"]
        undersampler = Undersample(data=train_data, 
                                   target="target", 
                                   undersample_size=undersample_size
                                   )
        train_data = undersampler.perform()
        logger.info(f"Undersampling complete. Train_target distribution: {train_data.value_counts('target')}")

    if config["sampling_config"].get("oversampling_apply", False):
        sampling_method = config["sampling_config"]["oversampling_method"]
        sampling_ratio = config["sampling_config"]["oversampling_ratio"]    
        if sampling_method in ["SMOTE", "SMOTE_Tomek"]:
            oversampler = Oversample(data=train_data, 
                                     target="target", 
                                     numeric_method=sampling_method, 
                                     ratio=sampling_ratio
                                     )
            train_data = oversampler.perform()
        else:
            raise NotImplementedError(f"Oversampling method '{sampling_method}' is not implemented. Please use 'SMOTE' or 'SMOTE-Tomek'.")  
        logger.info(f"Oversampling complete. Train_target distribution: {train_data.value_counts('target')}")

    return train_data

def preprocessing(config_path: str):
    
    config = load_config(config_path)
    train_data = pd.read_csv("data/train.csv")
    test_data = pd.read_csv("data/test.csv")

    # Apply normalization 
    if config["normalization_config"]["apply"]:
        train_data, test_data = apply_normalization(train_data, test_data, config)
        logger.info("Normalization complete.")


    # Apply encoding 
    if config["encoding_config"]["apply"]:
        train_data, test_data = apply_encoding(train_data, test_data, config)
        logger.info("Encoding complete.")
        logger.info(f"categorical data columns: {train_data.select_dtypes(include=['object']).columns}")    

    # Apply PCA 
    if config["pca_config"]["apply"]:
        train_data, test_data = apply_pca(train_data, test_data, config)
        logger.info("PCA complete.")    

    if config["sampling_config"]["apply"]:
        train_data = apply_sampling(train_data, config)
        logger.info(f"Sampling complete. Train_target distribution: {train_data.value_counts('target')}")

    return train_data, test_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to the config file", required=True)
    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    train, test = preprocessing(args.config)

    common_columns = list(set(train.columns).intersection(test.columns)) 
    test = test[common_columns]
    for col in train.columns:
        if col not in test.columns:
            test[col] = "0"
    
    # if target column is in test
    if "target" in test.columns:
        test = test.drop(columns=["target"], axis=1)
    
    
    clf_setup = setup(data=train,
                      target="target",
                      session_id=45,
                      train_size=0.8)
    top_models = compare_models(n_select=int(config["ensemble_num"]))
    result_df = pull()

    if isinstance(top_models, list):
        final_models = [finalize_model(model) for model in top_models]
    else:
        final_models = [finalize_model(top_models)]

    predictions = []
    for model in final_models:
        pred_df = predict_model(model, data=test)
        predictions.append(pred_df['prediction_label'].values)

    final_predictions = pd.DataFrame(predictions).mode().iloc[0].values

    # save results
    df_sub = pd.read_csv('submission.csv')
    df_sub['target'] = final_predictions
    save_root = os.path.join(args.config["folder_prefix"], "submission.csv")
    if not os.path.exists(args.config["folder_prefix"]):
        os.makedirs(args.config["folder_prefix"])
    df_sub.to_csv(save_root, index=False)

    fig, ax = plt.subplots(figsize=(12, 8))  # Set the figure size as needed
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=result_df.values, colLabels=result_df.columns, cellLoc = 'center', loc='center')
    save_image_root = os.path.join(config["folder_prefix"], "df.png")
    plt.savefig(save_image_root, bbox_inches='tight', dpi=300)
    plt.close()