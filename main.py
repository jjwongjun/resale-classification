"""
Main script for the HDB resale price classification pipeline.

Loads the configuration and data, cleans the data, trains and tunes the models,
selects the best model on the validation set (ROC AUC) and evaluates it on the
test set.

Usage:
    python main.py [--config ./src/config.yaml]
"""

# Standard library imports
import argparse
import logging
import warnings

# Third-party imports
import pandas as pd
import yaml
from sklearn.exceptions import ConvergenceWarning

# Local application/library specific imports
from src.data_preparation import DataPreparation
from src.model_training import ModelTraining

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Only silence the known convergence warning, not all warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

REQUIRED_CONFIG_KEYS = {
    "file_path": str,
    "target_column": str,
    "val_test_size": float,
    "val_size": float,
    "logreg_param_grid": dict,
    "knn_param_grid": dict,
    "dt_param_grid": dict,
    "hgb_param_grid": dict,
    "cv": int,
    "scoring": str,
    "numerical_features": list,
    "nominal_features": list,
    "ordinal_features": list,
    "passthrough_features": list,
    "flat_type_categories": list,
}


def validate_config(config: dict) -> None:
    """
    Checks that all required config keys exist, have the right type and
    sensible values.

    Args:
        config (dict): The configuration dictionary loaded from the YAML file.

    Raises:
        ValueError: If a key is missing, has the wrong type or an invalid value.
    """
    for key, expected_type in REQUIRED_CONFIG_KEYS.items():
        if key not in config:
            raise ValueError(f"Missing required config key: '{key}'")
        if not isinstance(config[key], expected_type):
            raise ValueError(
                f"Config key '{key}' should be of type {expected_type.__name__}"
            )
    for key in ["val_test_size", "val_size"]:
        if not 0 < config[key] < 1:
            raise ValueError(f"Config key '{key}' must be between 0 and 1")
    if config["cv"] < 2:
        raise ValueError("Config key 'cv' must be at least 2")


def load_config(config_path: str) -> dict:
    """
    Loads and validates the YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: The validated configuration dictionary.
    """
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        logging.error(f"Config file not found: {config_path}")
        raise
    except yaml.YAMLError:
        logging.exception(f"Config file is not valid YAML: {config_path}")
        raise
    validate_config(config)
    return config


def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads the resale transactions CSV file into a DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: The raw data.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        logging.error(f"Data file not found: {file_path}")
        raise
    except pd.errors.ParserError:
        logging.exception(f"Data file could not be parsed: {file_path}")
        raise
    logging.info(f"Loaded {len(df)} rows from {file_path}")
    return df


def main():
    """Run the full pipeline: load, clean, train, tune, select and evaluate."""

    parser = argparse.ArgumentParser(description="HDB resale classification pipeline")
    parser.add_argument("--config", default="./src/config.yaml", help="Config path")
    args = parser.parse_args()

    # Load and validate the configuration file
    config = load_config(args.config)

    # Load CSV file into a DataFrame
    df = load_data(config["file_path"])

    # Initialize and run data preparation
    data_prep = DataPreparation(config)
    cleaned_df = data_prep.clean_data(df)

    # Initialize model training with the created preprocessor
    model_training = ModelTraining(config, data_prep.preprocessor)

    # Split the data
    X_train, X_val, X_test, y_train, y_val, y_test = model_training.split_data(
        cleaned_df
    )

    # Train and evaluate baseline models with default hyperparameters
    baseline_models, baseline_metrics = (
        model_training.train_and_evaluate_baseline_models(
            X_train, y_train, X_val, y_val
        )
    )

    # Train and evaluate tuned models with hyperparameter tuning
    tuned_models, tuned_metrics = model_training.train_and_evaluate_tuned_models(
        X_train, y_train, X_val, y_val
    )

    # Combine all models and their metrics into dictionaries
    all_models = {**baseline_models, **tuned_models}
    all_metrics = {**baseline_metrics, **tuned_metrics}

    # Side-by-side comparison of all models on the validation set
    comparison = pd.DataFrame(all_metrics).T.sort_values("ROC AUC", ascending=False)
    logging.info(f"Validation metrics for all models:\n{comparison.round(4)}")

    # Find the best model based on ROC AUC score
    best_model_name = max(all_metrics, key=lambda k: all_metrics[k]["ROC AUC"])
    best_model = all_models[best_model_name]
    logging.info(f"Best Model Found: {best_model_name}")

    # Retrain the best model on train + validation data, then evaluate on test set
    final_model = model_training.retrain_on_train_val(
        best_model, X_train, X_val, y_train, y_val
    )
    final_metrics = model_training.evaluate_final_model(
        final_model, X_test, y_test, best_model_name
    )

    # Explain which features matter most for the final model
    model_training.log_feature_importance(final_model, X_test, y_test)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Pipeline failed.")
        raise SystemExit(1)