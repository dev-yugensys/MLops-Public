"""
Iris model training module for YugenAI project
"""

import logging
import os
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Import from project modules
from src.config.paths import IRIS_EXPERIMENTS, DefaultPaths, EnvVars, get_artifact_path
from src.data.preprocessing import preprocess_iris_data, validate_dataframe
from src.utils.logger import setup_logger

# Suppress warnings
warnings.filterwarnings("ignore")

# Set matplotlib backend to 'Agg' to avoid Tkinter threading issues
import matplotlib

matplotlib.use("Agg")  # Must be set before importing pyplot

# Set matplotlib style
plt.style.use("seaborn-v0_8")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150

logger = setup_logger("iris_training")


class IrisModelTrainer:
    """Iris model trainer with MLflow integration"""

    def __init__(self, experiment_name: str = None):
        """
        Initialize the iris model trainer

        Args:
            experiment_name: MLflow experiment name
        """
        # Set up experiment name and directories
        self.experiment_name = experiment_name or os.getenv(
            EnvVars.MLFLOW_EXPERIMENT_NAME,
            f"iris_classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        self.artifacts_dir = IRIS_EXPERIMENTS
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Create console handler and set level to INFO
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)

        # Add handler to logger
        if not self.logger.handlers:
            self.logger.addHandler(ch)

        # Setup MLflow
        self._setup_mlflow()

        # Model configurations
        self.models = {
            "logistic_regression": {
                "model": LogisticRegression(random_state=42, max_iter=1000),
                "params": {"C": 1.0, "penalty": "l2", "solver": "lbfgs"},
            },
            "decision_tree": {
                "model": DecisionTreeClassifier(random_state=42),
                "params": {
                    "max_depth": 5,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                },
            },
            "random_forest": {
                "model": RandomForestClassifier(random_state=42),
                "params": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                },
            },
            "gradient_boosting": {
                "model": GradientBoostingClassifier(random_state=42),
                "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
            },
            "svm": {
                "model": SVC(random_state=42, probability=True),
                "params": {"C": 1.0, "kernel": "rbf", "gamma": "scale"},
            },
            "knn": {
                "model": KNeighborsClassifier(),
                "params": {"n_neighbors": 5, "weights": "uniform", "algorithm": "auto"},
            },
        }

        # Class mapping for iris species
        self.class_mapping = {
            0: "Iris-setosa",
            1: "Iris-versicolor",
            2: "Iris-virginica",
        }

        self.results = {}

    def _setup_mlflow(self) -> bool:
        """
        Set up MLflow tracking for the experiment.

        Returns:
            bool: True if MLflow setup was successful, False otherwise
        """
        try:
            # Disable MLflow autologging for more control
            mlflow.sklearn.autolog(disable=True)

            # Create mlruns directory if it doesn't exist
            mlruns_dir = Path("mlruns").absolute()
            mlruns_dir.mkdir(parents=True, exist_ok=True)

            # Convert Windows path to proper file URI format
            # On Windows, we need to add an extra slash after file://
            path_str = str(mlruns_dir).replace("\\", "/")
            tracking_uri = f"file:///{path_str}"

            self.logger.info(f"Setting MLflow tracking URI to: {tracking_uri}")
            mlflow.set_tracking_uri(tracking_uri)

            # Enable autologging with more detailed options
            mlflow.sklearn.autolog(
                log_input_examples=True,
                log_model_signatures=True,
                log_models=True,
                log_datasets=True,
            )

            # Get or create experiment
            try:
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                if experiment is None:
                    experiment_id = mlflow.create_experiment(
                        self.experiment_name,
                        artifact_location=str(
                            mlruns_dir / self.experiment_name.replace(" ", "_")
                        ),
                    )
                    self.logger.info(
                        f"Created new MLflow experiment: {self.experiment_name} (ID: {experiment_id})"
                    )
                else:
                    logger.info(
                        f"Using existing MLflow experiment: {self.experiment_name}"
                    )

                # Set the experiment
                mlflow.set_experiment(self.experiment_name)
                return True
            except Exception as exp_err:
                self.logger.error(
                    f"Error setting up MLflow experiment: {str(exp_err)}", exc_info=True
                )
                return False
        except Exception as e:
            self.logger.error(f"Error in MLflow setup: {str(e)}", exc_info=True)
            self.logger.warning("Continuing without MLflow tracking")
            return False

    def load_and_preprocess_data(
        self, data_path: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Load and preprocess iris data

        Args:
            data_path: Path to iris data file

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Loading and preprocessing data from: {data_path}")

        # Preprocess data
        df = preprocess_iris_data(data_path)

        # Validate data
        if not validate_dataframe(df):
            raise ValueError("Data validation failed")

        # Split features and target
        target_col = "Species"
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info(
            f"Data split: {X_train.shape[0]} train, {X_test.shape[0]} test samples"
        )
        logger.info(
            f"Class distribution in training set: {y_train.value_counts().to_dict()}"
        )

        return X_train, X_test, y_train, y_test

    def train_model(
        self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Any:
        """
        Train a single model

        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training target

        Returns:
            Trained model
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")

        model_config = self.models[model_name]
        model = model_config["model"]
        params = model_config["params"]

        # Set parameters
        model.set_params(**params)

        logger.info(f"Training {model_name} with parameters: {params}")

        # Train model
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        logger.info(f"{model_name} training completed in {training_time:.2f} seconds")

        return model, training_time

    def evaluate_model(
        self, model: Any, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate a trained model

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target

        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = (
            model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        )

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)

        # Cross-validation score
        cv_scores = cross_val_score(model, X_test, y_test, cv=5)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        metrics = {
            "accuracy": accuracy,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "cv_min": cv_scores.min(),
            "cv_max": cv_scores.max(),
        }

        logger.info(f"Model evaluation metrics: {metrics}")

        return metrics

    def plot_confusion_matrix(
        self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str
    ) -> str:
        """
        Plot and save confusion matrix

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model

        Returns:
            Path to saved plot
        """
        # Make predictions
        y_pred = model.predict(X_test)

        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Create plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=list(self.class_mapping.values()),
            yticklabels=list(self.class_mapping.values()),
        )
        plt.title(f"{model_name} - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()

        # Save plot
        plot_path = self.artifacts_dir / f"{model_name}_confusion_matrix.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Confusion matrix plot saved to: {plot_path}")

        return str(plot_path)

    def plot_feature_importance(
        self, model: Any, feature_names: list, model_name: str
    ) -> str:
        """
        Plot and save feature importance

        Args:
            model: Trained model
            feature_names: List of feature names
            model_name: Name of the model

        Returns:
            Path to saved plot
        """
        if not hasattr(model, "feature_importances_"):
            logger.warning(f"{model_name} does not support feature importance")
            return None

        # Get feature importance
        importance = model.feature_importances_

        # Create DataFrame for plotting
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": importance}
        ).sort_values("importance", ascending=False)

        # Create plot
        plt.figure(figsize=(8, 6))
        sns.barplot(data=importance_df, x="importance", y="feature")
        plt.title(f"{model_name} - Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()

        # Save plot
        plot_path = self.artifacts_dir / f"{model_name}_feature_importance.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Feature importance plot saved to: {plot_path}")

        return str(plot_path)

    def train_all_models(self, data_path: str) -> Dict[str, Any]:
        """
        Train all models and log results with manual MLflow logging

        Args:
            data_path: Path to iris data file

        Returns:
            Dictionary of training results
        """
        logger.info("Starting training of all models")

        # Setup MLflow
        mlflow_enabled = self._setup_mlflow()
        logger.info(f"MLflow enabled: {mlflow_enabled}")

        try:
            # Load and preprocess data
            logger.info(f"Loading data from: {data_path}")
            X_train, X_test, y_train, y_test = self.load_and_preprocess_data(data_path)
            logger.info(
                f"Data loaded - Train shape: {X_train.shape}, Test shape: {X_test.shape}"
            )

            if X_train is None or len(X_train) == 0:
                raise ValueError("Training data is empty after preprocessing")

            # Initialize results dictionary
            self.results = {}

            # Train each model
            for model_name in self.models.keys():
                run_id = None
                try:
                    logger.info(f"\n{'='*50}")
                    logger.info(f"Starting training for model: {model_name}")

                    # Create run with manual logging
                    if mlflow_enabled:
                        try:
                            run = mlflow.start_run(run_name=f"{model_name}_training")
                            run_id = run.info.run_id
                            logger.info(f"MLflow run started with ID: {run_id}")
                        except Exception as e:
                            logger.error(f"Failed to start MLflow run: {str(e)}")
                            mlflow_enabled = False

                    # Train model
                    logger.info(f"Training {model_name} model...")
                    model, training_time = self.train_model(
                        model_name, X_train, y_train
                    )
                    logger.info(f"{model_name} trained in {training_time:.2f} seconds")

                    if model is None:
                        raise ValueError(f"Model {model_name} training returned None")

                    # Evaluate model
                    logger.info(f"Evaluating {model_name}...")
                    metrics = self.evaluate_model(model, X_test, y_test)
                    logger.info(f"Evaluation metrics: {metrics}")

                    # Manual logging
                    if mlflow_enabled and run_id:
                        try:
                            # Log parameters
                            mlflow.log_params(self.models[model_name].get("params", {}))

                            # Log metrics
                            mlflow.log_metrics(metrics)
                            mlflow.log_metric("training_time_seconds", training_time)

                            # Log model (without signature to avoid permission issues)
                            mlflow.sklearn.log_model(
                                sk_model=model,
                                artifact_path=model_name,
                                registered_model_name=None,
                                signature=None,
                            )
                            logger.info(f"Logged {model_name} model to MLflow")

                        except Exception as e:
                            logger.error(f"MLflow logging failed: {str(e)}")

                    # Create and save plots
                    logger.info(f"Generating plots for {model_name}...")
                    try:
                        confusion_plot = self.plot_confusion_matrix(
                            model, X_test, y_test, model_name
                        )
                        feature_plot = self.plot_feature_importance(
                            model, X_train.columns, model_name
                        )

                        if mlflow_enabled and run_id:
                            if confusion_plot and os.path.exists(confusion_plot):
                                mlflow.log_artifact(confusion_plot)
                            if feature_plot and os.path.exists(feature_plot):
                                mlflow.log_artifact(feature_plot)
                    except Exception as e:
                        logger.error(f"Error generating plots: {str(e)}")
                        confusion_plot = None
                        feature_plot = None

                    # Store results
                    self.results[model_name] = {
                        "model": model,
                        "metrics": metrics,
                        "training_time": training_time,
                        "confusion_plot": confusion_plot,
                        "feature_plot": feature_plot,
                    }

                    logger.info(f"{model_name} training completed successfully")

                except Exception as e:
                    logger.error(
                        f"Error training {model_name}: {str(e)}", exc_info=True
                    )
                    if model_name in self.results:
                        del self.results[model_name]
                    continue

                finally:
                    # Always end the MLflow run if it was started
                    if mlflow_enabled and run_id:
                        try:
                            mlflow.end_run()
                        except Exception as e:
                            logger.error(f"Error ending MLflow run: {str(e)}")

            if not self.results:
                raise ValueError("No models were successfully trained")

            return self.results

        except Exception as e:
            logger.error(f"Fatal error in train_all_models: {str(e)}", exc_info=True)
            raise

        return self.results

    def save_best_model(self, metric: str = "accuracy") -> str:
        """
        Save the best performing model

        Args:
            metric: Metric to use for selecting best model

        Returns:
            Path to saved model
        """
        if not self.results:
            raise ValueError("No models have been trained yet")

        # Find best model
        best_model_name = max(
            self.results.keys(), key=lambda x: self.results[x]["metrics"][metric]
        )
        best_model = self.results[best_model_name]["model"]

        # Create models/iris directory if it doesn't exist
        model_dir = Path("models/iris")
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model to DVC-tracked location
        model_path = model_dir / "iris_model.pkl"
        joblib.dump(best_model, model_path)

        # Also save to src/models/saved for backward compatibility
        saved_dir = Path("src/models/saved")
        saved_dir.mkdir(parents=True, exist_ok=True)
        saved_path = saved_dir / "iris_model.pkl"
        joblib.dump(best_model, saved_path)

        logger.info(f"Best model ({best_model_name}) saved to: {model_path}")
        logger.info(f"Also saved to: {saved_path} for backward compatibility")

        return str(model_path)

    def save_results_summary(self) -> str:
        """
        Save training results summary

        Returns:
            Path to saved summary
        """
        if not self.results:
            raise ValueError("No models have been trained yet")

        # Create summary DataFrame
        summary_data = []
        for model_name, result in self.results.items():
            summary_data.append(
                {
                    "model": model_name,
                    **result["metrics"],
                    "training_time": result["training_time"],
                }
            )

        summary_df = pd.DataFrame(summary_data)

        # Save summary to CSV in the artifacts directory
        summary_path = self.artifacts_dir / "iris_model_comparison.csv"
        summary_df.to_csv(summary_path, index=False)

        # Also save to metrics directory for DVC tracking
        metrics_dir = Path("artifacts/metrics")
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = metrics_dir / "iris_model_comparison.csv"
        summary_df.to_csv(metrics_path, index=False)

        logger.info(f"Results summary saved to: {summary_path}")

        return str(summary_path)


def main():
    """Main training function"""
    # Setup paths
    data_path = "data/raw/iris.csv"

    # Initialize trainer
    trainer = IrisModelTrainer()

    # Train all models
    results = trainer.train_all_models(data_path)

    # Save best model
    best_model_path = trainer.save_best_model()

    # Save results summary
    summary_path = trainer.save_results_summary()

    logger.info("Training completed successfully!")
    logger.info(f"Best model saved to: {best_model_path}")
    logger.info(f"Results summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
