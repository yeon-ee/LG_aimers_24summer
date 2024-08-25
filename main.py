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
    train_data = train_normalizer.normalize_data()
    train_data = train_normalizer.get_normalized_data()

    test_normalizer = Normalization(test_data, columns=col_to_normalize, method=config["normalization_config"]["method"])
    test_data = test_normalizer.manage_missing_values()
    test_data = test_normalizer.normalize_data()
    test_data = test_normalizer.get_normalized_data()

    return train_data, test_data

def apply_encoding(train_data: pd.DataFrame, test_data: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply encoding to the train and test datasets using the specified configuration.
    
    Parameters:
    train_data (pd.DataFrame): The training data.
    test_data (pd.DataFrame): The testing data.
    config (dict): The configuration dictionary containing encoding settings.
    
    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: The encoded training and testing data.
    """
    encoder = Encoding(
        train_df=train_data,
        test_df=test_data,
        columns=config.get("col_list"),
        method=config.get("method", "one_hot"),
        target_column='target'
    )
    return encoder.get_encoded_data()

def apply_pca(train_data: pd.DataFrame, test_data: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply PCA to the train and test datasets."""
    col_to_pca = config["pca_config"]["col_list"]
    if col_to_pca is None or len(col_to_pca) == 0:
        raise ValueError("No columns provided for PCA. Please specify columns in the 'pca_config'.")

    pca_processor = PCAProcessor(train_data, test_data, options=col_to_pca)
    train_data, test_data = pca_processor.replace_with_pcs()

    return train_data, test_data

def apply_sampling(train_data: pd.DataFrame, config: dict) -> pd.DataFrame:
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
        if sampling_method in ["SMOTE", "SMOTE-Tomek"]:
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

    # Apply split if configured
    if config.get("split_apply", False):
        train_data_1, train_data_2 = split_data(train_data)
        test_data_1, test_data_2 = split_data(test_data)
    else:
        train_data_1, train_data_2 = train_data, None
        test_data_1, test_data_2 = test_data, None

    train_list = [(train_data_1, test_data_1)]
    if train_data_2 is not None and test_data_2 is not None:
        train_list.append((train_data_2, test_data_2))

    processed_train_list = []
    processed_test_list = []

    for train_data, test_data in train_list:
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

        processed_train_list.append(train_data)
        processed_test_list.append(test_data)

    return processed_train_list, processed_test_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to the config file", required=True)
    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    processed_train_list, processed_test_list = preprocessing(args.config)

    # Initialize an empty DataFrame to store all results
    all_submissions_df = pd.DataFrame()

    # Loop over the train and test datasets
    for i, (train, test) in enumerate(zip(processed_train_list, processed_test_list), start=1):
        set_id = test["Set ID"]
        common_columns = list(set(train.columns).intersection(test.columns)) 
        test = test[common_columns]
        
        for col in train.columns:
            if col not in test.columns:
                test[col] = "0"
        
        if "target" in test.columns:
            test = test.drop(columns=["target"], axis=1)
            
        train.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)
        
        # PyCaret setup for the experiment
        clf_setup = setup(data=train,
                        target="target",
                        session_id=45,
                        train_size=0.8)

        top_models = compare_models(n_select=int(config["ensemble_num"]))
        result_df = pull()

        # Finalize the models
        if isinstance(top_models, list):
            final_models = [finalize_model(model) for model in top_models]
        else:
            final_models = [finalize_model(top_models)]

        predictions = []
        
        for model in final_models:
            pred_df = predict_model(model, data=test)
            predictions.append(pred_df["prediction_label"]) 
        
        all_predictions_df = pd.DataFrame(predictions).T
        all_predictions_df.columns = [f"model_{i}" for i in range(len(final_models))]

        final_predictions_df = all_predictions_df.mode(axis=1)[0]
        final_predictions_df.index = test.index

        submission_df = pd.DataFrame({
            "Set ID": set_id,
            "target": final_predictions_df
        })
        
        all_submissions_df = pd.concat([all_submissions_df, submission_df], axis=0)

        # Create the directory if it does not exist
        if not os.path.exists(config["folder_prefix"]):
            os.makedirs(config["folder_prefix"])

        # Save the result_df as an image using matplotlib
        fig, ax = plt.subplots(figsize=(12, 8))  # Set the figure size as needed
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=result_df.values, colLabels=result_df.columns, cellLoc='center', loc='center')
        plt.savefig(os.path.join(config["folder_prefix"], f"df_{i}.png"), bbox_inches='tight', dpi=300)
        plt.close()

    # Save the final submission to a CSV file
    save_root = os.path.join(config["folder_prefix"], "submission.csv")
    if not os.path.exists(config["folder_prefix"]):
        os.makedirs(config["folder_prefix"])
    all_submissions_df.to_csv(save_root, index=False)
