# Standard library imports
import logging
import re
from typing import Any, Dict

# Related third-party imports
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


class DataPreparation:
    """
    A class used to clean and preprocess HDB resale prices data.

    Attributes:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing parameters for data cleaning and preprocessing.
    preprocessor : sklearn.compose.ColumnTransformer
        A preprocessor pipeline for transforming numerical, nominal, and ordinal features.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the DataPreparation class with a configuration dictionary.

        Args:
        -----
        config (Dict[str, Any]): Configuration dictionary containing parameters for data cleaning and preprocessing.
        """
        self.config = config
        self.preprocessor = self._create_preprocessor()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the input DataFrame by performing several preprocessing steps.

        Args:
        -----
        df (pd.DataFrame): The input DataFrame containing the raw data.

        Returns:
        --------
        pd.DataFrame: The cleaned DataFrame.
        """
        logging.info("Starting data cleaning.")
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        target = self.config["target_column"]
        missing_columns = {target, "flat_type", "storey_range"} - set(df.columns)
        if missing_columns:
            raise ValueError(f"Input data is missing columns: {missing_columns}")
        df = df.copy()
        self._log_data_summary(df)
        rows_before = len(df)
        df = df.drop_duplicates()
        logging.info(f"Removed {rows_before - len(df)} duplicate rows.")
        df["flat_type"] = df["flat_type"].replace("FOUR ROOM", "4 ROOM")
        # Some lease_commence_date values are stored as negative years (entry error)
        df["lease_commence_date"] = df["lease_commence_date"].abs()
        df["storey_range"] = df["storey_range"].apply(self._convert_storey_range)
        df = self._fill_missing_names(df, "town_id", "town_name")
        df = self._fill_missing_names(df, "flatm_id", "flatm_name")
        df.drop(columns=["id", "town_id", "flatm_id"], inplace=True)
        df["year_month"] = pd.to_datetime(df["month"], format="%Y-%m")
        df["year"] = df["year_month"].dt.year
        df["month"] = df["year_month"].dt.month
        df.drop(columns=["year_month"], inplace=True)
        df["remaining_lease_months"] = df["remaining_lease"].apply(
            self._extract_lease_info
        )
        df.drop(columns=["remaining_lease", "block", "street_name"], inplace=True)
        # Encode the target; unexpected labels raise an error instead of silently becoming NaN
        target_labels = {"Above Median": 1, "Below Median": 0}
        unexpected = set(df[target].unique()) - set(target_labels)
        if unexpected:
            raise ValueError(f"Unexpected labels in '{target}': {unexpected}")
        df[target] = df[target].map(target_labels)
        logging.info(
            f"Target distribution:\n{df[target].value_counts(normalize=True).round(3)}"
        )
        logging.info("Data cleaning completed.")
        return df

    @staticmethod
    def _log_data_summary(df: pd.DataFrame) -> None:
        """
        Logs a short data validation report: duplicates, missing values and
        the number of unique values in each categorical column.

        Args:
        -----
        df (pd.DataFrame): The raw DataFrame to summarise.
        """
        logging.info(f"Raw data shape: {df.shape}")
        logging.info(f"Duplicate rows found: {df.duplicated().sum()}")
        missing = df.isna().sum()
        missing = missing[missing > 0]
        if missing.empty:
            logging.info("No missing values found.")
        else:
            logging.info(f"Missing values per column: {missing.to_dict()}")
        for column in df.select_dtypes(include=["object", "string"]).columns:
            logging.info(f"Unique values in '{column}': {df[column].nunique()}")

    def _create_preprocessor(self) -> ColumnTransformer:
        """
        Creates a preprocessor pipeline for transforming numerical, nominal, and ordinal features.

        Returns:
        --------
        sklearn.compose.ColumnTransformer: A ColumnTransformer object for preprocessing the data.
        """
        numerical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        nominal_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        ordinal_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "ordinal",
                    OrdinalEncoder(
                        categories=[self.config["flat_type_categories"]],
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                )
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, self.config["numerical_features"]),
                ("nom", nominal_transformer, self.config["nominal_features"]),
                ("ord", ordinal_transformer, self.config["ordinal_features"]),
                ("pass", "passthrough", self.config["passthrough_features"]),
            ],
            remainder="passthrough",
            sparse_threshold=0,  # dense output so tree-based models can be used
            n_jobs=-1,
        )
        return preprocessor

    @staticmethod
    def _convert_storey_range(storey_range: str) -> float:
        """
        Converts a storey range string into its average numerical value.

        Args:
        -----
        storey_range (str): A string representing a range of storeys, in the format 'XX TO YY'.

        Returns:
        --------
        float: The average value of the two storeys in the range.
        """
        range_values = storey_range.split(" TO ")
        return (int(range_values[0]) + int(range_values[1])) / 2

    @staticmethod
    def _fill_missing_names(
        df: pd.DataFrame, id_column: str, name_column: str
    ) -> pd.DataFrame:
        """
        Fills missing values in the 'name_column' using the 'id_column'.

        Args:
        -----
        df (pd.DataFrame): The DataFrame containing the columns to be filled.
        id_column (str): The name of the column containing the IDs.
        name_column (str): The name of the column containing the names to be filled.

        Returns:
        --------
        pd.DataFrame: The DataFrame with missing values in 'name_column' filled.
        """
        missing_names = df[name_column].isna()
        name_mapping = (
            df[[id_column, name_column]]
            .dropna()
            .drop_duplicates()
            .set_index(id_column)[name_column]
            .to_dict()
        )
        df.loc[missing_names, name_column] = df.loc[missing_names, id_column].map(
            name_mapping
        )
        return df

    @staticmethod
    def _extract_lease_info(lease_str: str) -> int:
        """
        Converts lease information from a string format to total months.

        Args:
        -----
        lease_str (str): The remaining lease period as a string.

        Returns:
        --------
        int: The total number of months, or None if the input is NaN.
        """
        if pd.isna(lease_str):
            return None
        years_match = re.search(r"(\d+)\s*years?", lease_str)
        months_match = re.search(r"(\d+)\s*months?", lease_str)
        number_match = re.match(r"^\d+$", lease_str.strip())
        if years_match:
            years = int(years_match.group(1))
        elif number_match:
            years = int(number_match.group(0))
        else:
            years = 0
        months = int(months_match.group(1)) if months_match else 0
        total_months = years * 12 + months
        return total_months