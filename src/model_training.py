# Standard library imports
import logging
from typing import Any, Dict, Tuple

# Related third-party imports
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


class ModelTraining:
    """
    A class used to train and evaluate machine learning models on HDB resale prices data.

    Attributes:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing parameters for model training and evaluation.
    preprocessor : sklearn.compose.ColumnTransformer
        A preprocessor pipeline for transforming numerical, nominal, and ordinal features.
    """

    def __init__(self, config: Dict[str, Any], preprocessor: ColumnTransformer):
        """
        Initialize the ModelTraining class with configuration and preprocessor.

        Args:
        -----
        config (Dict[str, Any]): Configuration dictionary containing parameters for model training and evaluation.
        preprocessor (sklearn.compose.ColumnTransformer): A preprocessor pipeline for transforming numerical, nominal, and ordinal features.
        """
        self.config = config
        self.preprocessor = preprocessor

    def split_data(
        self, df: pd.DataFrame
    ) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
    ]:
        """
        Split the data into training, validation, and test sets.

        Args:
        -----
        df (pd.DataFrame): The input DataFrame containing the cleaned data.

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]: A tuple containing the training, validation, and test features and target variables.
        """
        logging.info("Starting data splitting.")
        if self.config["target_column"] not in df.columns:
            raise ValueError(f"Target column '{self.config['target_column']}' not found.")
        X = df.drop(columns=self.config["target_column"])
        y = df[self.config["target_column"]]
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=self.config["val_test_size"],
            random_state=self.config.get("random_state", 42),
            stratify=y,
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=self.config["val_size"],
            random_state=self.config.get("random_state", 42),
            stratify=y_temp,
        )
        logging.info("Data split into train, validation, test sets.")
        for split_name, split_y in [("train", y_train), ("val", y_val), ("test", y_test)]:
            logging.info(
                f"Class distribution ({split_name}): "
                f"{split_y.value_counts(normalize=True).round(3).to_dict()}"
            )
        # Classes are (almost) balanced, so no resampling or class weights are used
        logging.info("Classes are balanced; no class imbalance strategy is applied.")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_and_evaluate_baseline_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
        """
        Create, train, and evaluate baseline models.

        Args:
        -----
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target variable.
        X_val (pd.DataFrame): The validation features.
        y_val (pd.Series): The validation target variable.

        Returns:
        --------
        Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]: A tuple containing the trained pipelines and their evaluation metrics.
        """
        logging.info("Training and evaluating baseline models.")
        seed = self.config.get("random_state", 42)
        models = {
            # Minimum benchmark: always predicts the class prior
            "dummy": DummyClassifier(strategy="prior"),
            "logreg": LogisticRegression(max_iter=1000),
            "knn": KNeighborsClassifier(),
            "dt": DecisionTreeClassifier(criterion='gini', min_samples_split=2, min_samples_leaf=1, random_state=seed),
            # Tree-based ensembles
            "rf": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed),
            "hgb": HistGradientBoostingClassifier(random_state=seed),
        }
        pipelines = {}
        metrics = {}

        for model_name, model in models.items():
            pipeline = Pipeline(
                steps=[("preprocessor", self.preprocessor), ("classifier", model)]
            )
            pipeline.fit(X_train, y_train)
            pipelines[model_name] = pipeline
            metrics[model_name] = self._evaluate_model(
                pipeline, X_val, y_val, model_name
            )
            self._log_cv_summary(pipeline, X_train, y_train, model_name)

        return pipelines, metrics

    def train_and_evaluate_tuned_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
        """
        Perform hyperparameter tuning for the classification models and evaluate them.

        Args:
        -----
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target variable.
        X_val (pd.DataFrame): The validation features.
        y_val (pd.Series): The validation target variable.

        Returns:
        --------
        Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]: A tuple containing the tuned pipelines and their evaluation metrics.
        """
        logging.info("Starting hyperparameter tuning.")
        tuned_models = {}
        tuned_metrics = {}
        logreg_param_grid = self.config["logreg_param_grid"]
        knn_param_grid = self.config["knn_param_grid"]
        dt_param_grid = self.config["dt_param_grid"]
        hgb_param_grid = self.config["hgb_param_grid"]
        param_grids = {
            "logreg_tuned": logreg_param_grid,
            "knn_tuned": knn_param_grid,
            "dt_tuned": dt_param_grid,
            "hgb_tuned": hgb_param_grid,
        }
        seed = self.config.get("random_state", 42)
        cv = self.config["cv"]
        scoring = self.config["scoring"]

        models = {
            "logreg_tuned": LogisticRegression(max_iter=1000),
            "knn_tuned": KNeighborsClassifier(),
            "dt_tuned": DecisionTreeClassifier(criterion='gini', min_samples_split=2, min_samples_leaf=1, random_state=seed),
            "hgb_tuned": HistGradientBoostingClassifier(random_state=seed),
        }

        for model_name, model in models.items():
            pipeline = Pipeline(
                steps=[("preprocessor", self.preprocessor), ("classifier", model)]
            ) 
            grid_search = GridSearchCV(
                pipeline, param_grids[model_name], cv=cv, scoring=scoring, n_jobs=1
            )
            grid_search.fit(X_train, y_train)
            logging.info(f"{model_name} best parameters: {grid_search.best_params_}")
            tuned_models[model_name] = grid_search.best_estimator_
            self._log_cv_summary(tuned_models[model_name], X_train, y_train, model_name)
            tuned_metrics[model_name] = self._evaluate_model(
                tuned_models[model_name], X_val, y_val, model_name + " (tuned)"
            )

        logging.info("Hyperparameter tuning completed.")
        return tuned_models, tuned_metrics

    def evaluate_final_model(
        self, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, model_name: str
    ) -> Dict[str, float]:
        """
        Evaluate the final model on the test set and log the metrics.

        Args:
        -----
        model (Pipeline): The trained model pipeline.
        X_test (pd.DataFrame): The test features.
        y_test (pd.Series): The test target variable.
        model_name (str): The name of the model being evaluated.

        Returns:
        --------
        Dict[str, float]: A dictionary containing the evaluation metrics.
        """
        y_test_pred = model.predict(X_test)
        metrics = self._compute_metrics(model, X_test, y_test)
        logging.info(f"Final Test Metrics for {model_name}:")
        for metric_name, metric_value in metrics.items():
            logging.info(f"{metric_name}: {metric_value}")
        logging.info(f"Confusion matrix:\n{confusion_matrix(y_test, y_test_pred)}")
        logging.info(
            f"Classification report:\n{classification_report(y_test, y_test_pred)}"
        )
        return metrics

    @staticmethod
    def _compute_metrics(
        model: Pipeline, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, float]:
        """
        Compute classification metrics. ROC AUC and PR AUC use predicted
        probabilities rather than hard class predictions.

        Args:
        -----
        model (Pipeline): The trained model pipeline.
        X (pd.DataFrame): The features.
        y (pd.Series): The true labels.

        Returns:
        --------
        Dict[str, float]: A dictionary containing the evaluation metrics.
        """
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        return {
            "Accuracy": accuracy_score(y, y_pred),
            "Precision": precision_score(y, y_pred, zero_division=0),
            "Recall": recall_score(y, y_pred),
            "F1 Score": f1_score(y, y_pred),
            "ROC AUC": roc_auc_score(y, y_pred_proba),
            "PR AUC": average_precision_score(y, y_pred_proba),
        }

    def retrain_on_train_val(
        self,
        model: Pipeline,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
    ) -> Pipeline:
        """
        Retrain the selected model (same hyperparameters) on train + validation data.

        Args:
        -----
        model (Pipeline): The selected model pipeline.
        X_train, X_val (pd.DataFrame): The training and validation features.
        y_train, y_val (pd.Series): The training and validation target variables.

        Returns:
        --------
        Pipeline: A new pipeline fitted on the combined data.
        """
        logging.info("Retraining best model on training + validation data.")
        final_model = clone(model)
        final_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
        return final_model

    def log_feature_importance(
        self, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
    ) -> pd.Series:
        """
        Log permutation importance (drop in ROC AUC when a feature is shuffled)
        on the test set. Works for any model.

        Args:
        -----
        model (Pipeline): The trained model pipeline.
        X_test (pd.DataFrame): The test features.
        y_test (pd.Series): The test target variable.

        Returns:
        --------
        pd.Series: Feature importances sorted from most to least important.
        """
        result = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=5,
            scoring="roc_auc",
            random_state=self.config.get("random_state", 42),
            n_jobs=-1,
        )
        importances = pd.Series(result.importances_mean, index=X_test.columns)
        importances = importances.sort_values(ascending=False)
        logging.info(f"Permutation feature importance:\n{importances.round(4)}")
        return importances

    def _log_cv_summary(
        self, model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, model_name: str
    ) -> None:
        """
        Log the mean and standard deviation of accuracy, F1 and ROC AUC across CV folds.

        Args:
        -----
        model (Pipeline): The model pipeline.
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target variable.
        model_name (str): The name of the model being evaluated.
        """
        scoring = {"Accuracy": "accuracy", "F1 Score": "f1", "ROC AUC": "roc_auc"}
        scores = cross_validate(
            model, X_train, y_train, cv=self.config["cv"], scoring=scoring
        )
        logging.info(f"{model_name} {self.config['cv']}-fold CV results:")
        for metric in scoring:
            values = scores[f"test_{metric}"]
            logging.info(f"{metric}: {values.mean():.4f} (+/- {values.std():.4f})")

    def _evaluate_model(
        self, model: Pipeline, X_val: pd.DataFrame, y_val: pd.Series, model_name: str
    ) -> Dict[str, float]:
        """
        Evaluate a model on the validation set and log the metrics.

        Args:
        -----
        model (Pipeline): The trained model pipeline.
        X_val (pd.DataFrame): The validation features.
        y_val (pd.Series): The validation target variable.
        model_name (str): The name of the model being evaluated.

        Returns:
        --------
        Dict[str, float]: A dictionary containing the evaluation metrics.
        """
        metrics = self._compute_metrics(model, X_val, y_val)
        logging.info(f"{model_name} Validation Metrics:")
        for metric_name, metric_value in metrics.items():
            logging.info(f"{metric_name}: {metric_value}")
        return metrics