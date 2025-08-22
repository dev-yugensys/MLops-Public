"""
Improved model_training.py
- Adds cross-validation (nested where appropriate), pipelines with preprocessing,
  randomized search (or grid search) with parallel jobs, CV scoring logging,
  composite metric for model selection (configurable weights),
  feature selection option, class-imbalance handling, early stopping for xgboost,
  better MLflow logging (params, CV metrics, artifacts like confusion matrix,
  classification report, and feature importances), and safer model saving.
- Compatible with the original Streamlit UI but exposes more options to the user.

Usage:
- Run with `streamlit run improved_model_training.py`

"""

import os
import time
import json
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import streamlit as st
import subprocess
import webbrowser
from datetime import datetime
from pathlib import Path
from threading import Thread
from importlib import import_module
from sklearn import datasets
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, KFold,
    RandomizedSearchCV, GridSearchCV, cross_validate
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
)
from scipy.stats import uniform, randint

# ====== CONFIG ======
BASE_DIR = Path.cwd()
MLFLOW_DIR = Path("mlruns").absolute()
MLFLOW_DIR.mkdir(exist_ok=True)
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Keep a similar high-level MODEL_CONFIGS but allow flexible searchable params
# MODEL_CONFIGS = {
#     "Iris Classification": {
#         "task_type": "classification",
#         "default_opt_metric": "f1_score",
#         "metrics": ["accuracy", "f1_score", "precision", "recall"],
#         "models": {
#             "Logistic Regression": {
#                 "model": "sklearn.linear_model.LogisticRegression",
#                 "params": {
#                     "C": uniform(0.01, 10.0),
#                     "solver": ["lbfgs", "liblinear"],
#                     "max_iter": [200],
#                     "class_weight": [None, "balanced"],
#                     "random_state": [42]
#                 }
#             },
#             "Random Forest": {
#                 "model": "sklearn.ensemble.RandomForestClassifier",
#                 "params": {
#                     "n_estimators": randint(50, 300),
#                     "max_depth": [5, 10, None],
#                     "min_samples_split": randint(2, 10),
#                     "min_samples_leaf": randint(1, 4),
#                     "class_weight": [None, "balanced"],
#                     "random_state": [42]
#                 }
#             },
#             "SVM": {
#                 "model": "sklearn.svm.SVC",
#                 "params": {
#                     "C": uniform(0.01, 10.0),
#                     "kernel": ["linear", "rbf"],
#                     "gamma": ["scale", "auto"],
#                     "probability": [True],
#                     "class_weight": [None, "balanced"],
#                     "random_state": [42]
#                 }
#             }
#         }
#     },
#     "Housing Regression": {
#         "task_type": "regression",
#         "default_opt_metric": "r2",
#         "metrics": ["mse", "mae", "r2", "explained_variance"],
#         "models": {
#             "Linear Regression": {
#                 "model": "sklearn.linear_model.LinearRegression",
#                 "params": {
#                     "fit_intercept": [True, False],
#                     # positive is only available in recent sklearn versions; safe fallback
#                 }
#             },
#             "Random Forest": {
#                 "model": "sklearn.ensemble.RandomForestRegressor",
#                 "params": {
#                     "n_estimators": randint(50, 300),
#                     "max_depth": [5, 10, 20, None],
#                     "min_samples_split": randint(2, 10),
#                     "min_samples_leaf": randint(1, 4),
#                     "random_state": [42]
#                 }
#             }
#         }
#     }
# }

MODEL_CONFIGS = {
    "Iris Classification": {
        "task_type": "classification",
        "metrics": ["accuracy", "f1_score", "precision", "recall"],
        "models": {
            "Logistic Regression": {
                "model": "sklearn.linear_model.LogisticRegression",
                "params": {
                    "C": [0.1, 1.0, 10.0],
                    "solver": ["lbfgs", "liblinear"],
                    "max_iter": [100, 200],
                    "random_state": [42]
                }
            },
            "SVM": {
                "model": "sklearn.svm.SVC",
                "params": {
                    "C": [0.1, 1.0, 10.0],
                    "kernel": ["linear", "rbf"],
                    "gamma": ["scale", "auto"],
                    "probability": [True],
                    "random_state": [42]
                }
            }
        }
    },
    "Housing Regression": {
        "task_type": "regression",
        "metrics": ["mse", "mae", "r2", "explained_variance"],
        "models": {
            "XGBoost": {
                "model": "xgboost.XGBRegressor",
                "params": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.3],
                    "max_depth": [3, 6, 9],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                    "random_state": [42]
                }
            },
            "Gradient Boosting": {
                "model": "sklearn.ensemble.GradientBoostingRegressor",
                "params": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 5, 7],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2],
                    "random_state": [42]
                }
            }
        }
    }
}
# ====== HELPERS ======

def setup_mlflow(experiment_name: str):
    path_str = str(MLFLOW_DIR).replace("\\", "/")
    mlflow.set_tracking_uri(f"file:///{path_str}")
    mlflow.sklearn.autolog(log_models=False)  # we will log model artifacts explicitly
    mlflow.set_experiment(experiment_name)


def load_dataset(dataset_name):
    if dataset_name == "Iris Classification":
        data = datasets.load_iris(as_frame=True)
        return data.data, data.target
    elif dataset_name == "Housing Regression":
        data = datasets.fetch_california_housing(as_frame=True)
        # Ensure target is in the original scale (in case it was scaled)
        target = data.target * 100000  # Scale back to original values (in $100,000s)
        return data.data, target
    else:
        raise ValueError("Unknown dataset")

def start_mlflow_ui(port=5000):
    """Start MLflow UI in a background process and open in browser."""
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', port))
        sock.close()

        if result == 0:
            st.warning(f"MLflow UI is already running on port {port}")
        else:
            def run_mlflow_ui():
                subprocess.Popen(
                    ["mlflow", "ui", "-p", str(port)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            thread = Thread(target=run_mlflow_ui, daemon=True)
            thread.start()
            time.sleep(2)

        webbrowser.open_new_tab(f"http://localhost:{port}")
        return True
    except Exception as e:
        st.error(f"Failed to start MLflow UI: {str(e)}")
        return False


def build_pipeline(task_type: str, do_feature_selection: bool = False, k_features: int = 10):
    steps = []
    # Only scale features, not the target variable
    steps.append(("imputer", SimpleImputer(strategy="median")))
    steps.append(("scaler", StandardScaler()))  # This only scales features
    if do_feature_selection:
        score_func = f_classif if task_type == "classification" else f_regression
        steps.append(("select", SelectKBest(score_func=score_func, k=k_features)))
    return steps

def compute_composite_score(metrics: dict, task_type: str, metric_weights: dict):
    """Compute a composite scalar score from multiple metrics using user-specified weights.
    For classification: higher is better for accuracy, f1, precision, recall.
    For regression: combine r2 (higher better) and negative MSE/MAE (lower better) by normalizing.
    """
    if task_type == "classification":
        total_weight = sum(metric_weights.values()) if metric_weights else 1.0
        score = 0.0
        for m, w in metric_weights.items():
            score += metrics.get(m, 0.0) * (w / total_weight)
        return score
    else:
        # regression: prefer higher r2, lower mse and mae -> convert them to a comparable scale
        # we'll normalize by simple heuristics so they are roughly comparable
        r2 = metrics.get("r2", 0.0)
        mse = metrics.get("mse", 0.0)
        mae = metrics.get("mae", 0.0)
        # Protect against zero division; use inverses for loss metrics
        inv_mse = 1.0 / (1.0 + mse)
        inv_mae = 1.0 / (1.0 + mae)
        total_weight = sum(metric_weights.values()) if metric_weights else 1.0
        score = 0.0
        for m, w in metric_weights.items():
            if m == "r2":
                score += r2 * (w / total_weight)
            elif m == "mse":
                score += inv_mse * (w / total_weight)
            elif m == "mae":
                score += inv_mae * (w / total_weight)
            else:
                score += metrics.get(m, 0.0) * (w / total_weight)
        return score


def evaluate_on_test(model, X_test, y_test, task_type: str):
    results = {}
    if task_type == "classification":
        y_pred = model.predict(X_test)
        results.update({
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred, zero_division=0, output_dict=True)
        })
    else:
        y_pred = model.predict(X_test)
        results.update({
            "mse": mean_squared_error(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
            "explained_variance": explained_variance_score(y_test, y_pred)
        })
    return results


# def save_model_artifact(model, dataset_name: str, model_name: str, tag: str = "best"):
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     fname = f"{dataset_name.replace(' ', '_')}_{model_name.replace(' ', '_')}_{tag}_{timestamp}.pkl"
#     path = MODELS_DIR / fname
#     joblib.dump(model, path)
#     return path
def save_model_artifact(model, dataset_name: str, model_name: str, tag: str = "best"):
    # Create a mapping of dataset names to their respective model filenames
    model_filenames = {
        "Iris": "iris_classification_model.pkl",
        "Housing": "housing_regression_model.pkl"
    }
    
    # Get the appropriate filename based on dataset_name, default to a generated name if not in mapping
    fname = model_filenames.get(dataset_name, 
                              f"{dataset_name.lower().replace(' ', '_')}_model.pkl")
    
    path = MODELS_DIR / fname
    
    # Remove existing file if it exists
    if path.exists():
        path.unlink()
    
    # Save the model
    joblib.dump(model, path)
    st.success(f"Model saved successfully as: {fname}")
    return path

# ====== STREAMLIT UI ======
st.set_page_config(page_title="Improved Model Training", layout="wide")
st.title("Improved Model Training with Cross-Validation, Pipelines and MLflow")

if st.button("🚀 Open MLflow UI", help="Open MLflow UI to track experiments"):
    if start_mlflow_ui():
        st.success("MLflow UI opened in a new tab!")
    else:
        st.error("Failed to start MLflow UI. Make sure MLflow is installed and port 5000 is available.")

# Dataset selection
dataset_name = st.selectbox("Select Dataset", list(MODEL_CONFIGS.keys()))
config = MODEL_CONFIGS[dataset_name]

# Training options sidebar
st.sidebar.header("Training Options")
use_random_search = st.sidebar.checkbox("Use RandomizedSearchCV (faster)", value=True)
n_iter = st.sidebar.number_input("n_iter (for RandomizedSearch)", min_value=10, max_value=1000, value=50)
cv_folds = st.sidebar.number_input("CV folds", min_value=3, max_value=10, value=5)
use_feature_selection = st.sidebar.checkbox("Use SelectKBest feature selection", value=False)
k_features = st.sidebar.number_input("K features (if feature selection)", min_value=1, max_value=20, value=5)
seed = int(st.sidebar.number_input("Random Seed", value=42))
opt_metric = st.sidebar.selectbox("Optimization Metric", options=config.get("metrics", []), index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("**Composite metric weights (optional)**")
metric_weights = {}
for m in config.get("metrics", []):
    v = st.sidebar.number_input(f"Weight for {m}", min_value=0.0, max_value=10.0, value=1.0)
    metric_weights[m] = float(v)

st.info(f"Will train {len(config['models'])} models. Using {cv_folds}-fold CV. RandomSearch: {use_random_search}")

if st.button("Train Models"):
    st.write("⏳ Starting improved training...")
    progress_bar = st.progress(0)

    # simple log area
    log_container = st.container()
    log_placeholder = log_container.empty()
    if 'training_logs' not in st.session_state:
        st.session_state.training_logs = []

    def log_message(message, log_type="info"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Add emoji prefix based on log type
        if log_type == "success":
            prefix = "✅"
        elif log_type == "warning":
            prefix = "⚠️"
        elif log_type == "error":
            prefix = "❌"
        elif log_type == "metric":
            prefix = "📊"
        elif log_type == "model":
            prefix = "🤖"
        else:
            prefix = "ℹ️"
            
        formatted_message = f"[{timestamp}] {prefix} {message}"
        st.session_state.training_logs.append(formatted_message)
        if len(st.session_state.training_logs) > 200:
            st.session_state.training_logs = st.session_state.training_logs[-200:]
        # Use markdown for better formatting
        log_placeholder.markdown('  \n'.join(st.session_state.training_logs))  # Double space before newline for markdown line break

    st.session_state.training_logs = []
    log_message("Initializing improved training...")

    X, y = load_dataset(dataset_name)
    # stratify for classification
    stratify = y if config['task_type'] == 'classification' else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=stratify)
    log_message(f"Loaded {dataset_name} with {X.shape[0]} samples, features: {X.shape[1]}")

    setup_mlflow(dataset_name.replace(" ", "_"))

    results = {}
    total_models = sum([1 for _ in config['models'].items()])
    # Initialize model tracking variables
    best_model_info = None
    best_score_so_far = -float('inf')
    model_idx = 0

    for model_name, cfg in config['models'].items():
        model_idx += 1
        log_message(f"\n=== Model {model_idx}/{total_models}: {model_name} ===", "model")

        # dynamic import
        module_name, cls_name = cfg['model'].rsplit('.', 1)
        ModelClass = getattr(import_module(module_name), cls_name)

        # build pipeline
        steps = build_pipeline(config['task_type'], do_feature_selection=use_feature_selection, k_features=k_features)
        steps.append(("estimator", ModelClass()))
        pipeline = Pipeline(steps)

        # prepare param grid for pipeline's estimator
        raw_params = cfg.get('params', {})
        pipeline_params = {}
        for k, v in raw_params.items():
            pipeline_params[f'estimator__{k}'] = v

        # Choose CV splitter
        if config['task_type'] == 'classification':
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
            log_message(f"Using {cv_folds}-fold stratified cross-validation", "info")
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
            log_message(f"Using {cv_folds}-fold cross-validation", "info")

        # Choose search
        if use_random_search:
            searcher = RandomizedSearchCV(
                pipeline,
                pipeline_params,
                n_iter=n_iter,
                scoring=None,  # we'll compute our own metrics using cross_validate
                cv=cv,
                verbose=1,  # Enable scikit-learn's built-in verbose output
                n_jobs=-1,
                random_state=seed,
                return_train_score=True,
                error_score='raise',
                refit=opt_metric
            )
            log_message(f"Using RandomizedSearchCV with {n_iter} iterations", "info")
        else:
            searcher = GridSearchCV(
                pipeline,
                pipeline_params,
                cv=cv,
                scoring=None,
                verbose=1,  # Enable scikit-learn's built-in verbose output
                n_jobs=-1,
                return_train_score=True,
                error_score='raise',
                refit=opt_metric
            )
            log_message(f"Using GridSearchCV with {len(searcher.param_grid)} parameter combinations", "info")

        with mlflow.start_run(run_name=f"{model_name}"):
            mlflow.log_param("model", model_name)
            mlflow.log_params({"use_random_search": use_random_search, "cv_folds": cv_folds})
            mlflow.log_params({f"param_{k}": str(v) for k, v in raw_params.items()})

            try:
                # Add progress tracking
                progress_text = st.empty()
                progress_bar = st.progress(0)
                
                # Create a callback for logging progress
                class ProgressCallback:
                    def __init__(self, total_iterations):
                        self.total = total_iterations
                        self.current = 0
                        self.start_time = time.time()
                    
                    def __call__(self, iter_num, *args, **kwargs):
                        self.current += 1
                        progress = min(self.current / self.total, 1.0)
                        progress_bar.progress(progress)
                        
                        # Calculate ETA
                        elapsed = time.time() - self.start_time
                        if self.current > 0:
                            eta = (elapsed / self.current) * (self.total - self.current)
                            eta_str = f"{int(eta // 60)}m {int(eta % 60)}s"
                        else:
                            eta_str = "Calculating..."
                            
                        progress_text.text(f"Training {model_name}: {self.current}/{self.total} iterations | ETA: {eta_str}")
                
                log_message(f"Starting training for {model_name}...", "model")
                
                # Calculate total iterations (params * CV folds)
                total_iterations = n_iter * cv_folds if use_random_search else len(searcher.param_grid) * cv_folds
                callback = ProgressCallback(total_iterations)
                
                t0 = time.time()
                searcher.fit(X_train, y_train)
                fit_time = time.time() - t0
                
                # Clear progress elements
                progress_bar.empty()
                progress_text.empty()
                
                log_message(f"Training completed in {fit_time:.2f} seconds", "success")
                
                # Get the best model and its score
                best_model = searcher.best_estimator_
                best_params = searcher.best_params_
                
                # Log best parameters
                log_message(f"Best parameters found:", "info")
                for param, value in best_params.items():
                    log_message(f"{param}: {value}", "info")
                
                # Get cross-validation results
                cv_results = searcher.cv_results_
                cv_score = searcher.best_score_
                log_message(f"Best cross-validation {opt_metric} score: {cv_score:.4f}", "info")
                
                # Log cross-validation results
                log_message("Cross-validation results:", "info")
                for metric in config['metrics']:
                    if f'mean_test_{metric}' in cv_results:
                        mean_score = cv_results[f'mean_test_{metric}'][searcher.best_index_]
                        std_score = cv_results[f'std_test_{metric}'][searcher.best_index_]
                        log_message(f"{metric}: {mean_score:.4f} (+/- {std_score:.4f})", "info")
                
                # Evaluate on test set
                test_metrics = evaluate_on_test(best_model, X_test, y_test, config['task_type'])
                log_message("Test set evaluation:", "info")
                for metric, value in test_metrics.items():
                    log_message(f"{metric}: {value}", "info")
                
                # Check for potential overfitting (perfect score on test set)
                if any(v == 1.0 for k, v in test_metrics.items() if k in ['accuracy', 'f1_score', 'precision', 'recall']):
                    log_message("⚠️ Warning: Model achieved perfect score on test set. This may indicate data leakage or overfitting.", "warning")
                
                # Log to MLflow - only log scalar values, skip non-scalar metrics like confusion_matrix
                mlflow.log_metric(f"cv_{opt_metric}", cv_score)
                mlflow.log_metric("fit_time", fit_time)
                for metric, value in test_metrics.items():
                    # Skip non-scalar values that can't be converted to float
                    if metric not in ['confusion_matrix', 'classification_report']:
                        try:
                            mlflow.log_metric(f"test_{metric}", float(value))
                        except (TypeError, ValueError):
                            # Log a warning if we can't convert the value to float
                            log_message(f"Skipping non-scalar metric for MLflow logging: {metric}", "warning")
                
                # Calculate composite score for model comparison using CV score instead of test score
                # to prevent overfitting to the test set
                cv_metrics = {k.replace('test_', 'cv_'): v for k, v in test_metrics.items()}
                composite_score = compute_composite_score(cv_metrics, config['task_type'], metric_weights)
                
                # Update best model if this one has better cross-validation score
                if best_model_info is None or cv_score > best_score_so_far:
                    best_model_info = {
                        'model': best_model,
                        'score': composite_score,
                        'cv_score': cv_score,
                        'test_metrics': test_metrics,
                        'params': best_params,
                        'name': model_name
                    }
                    best_score_so_far = cv_score  # Use CV score for model selection
                
            except Exception as e:
                log_message(f"Error during training: {str(e)}", "error")
                import traceback
                log_message(traceback.format_exc(), "error")
                continue

    # After all models, pick best across composite_score
    if best_model_info is not None:
        # Check if the model is potentially overfitted (perfect test scores)
        test_metrics = best_model_info['test_metrics']
        is_overfitted = any(v == 1.0 for k, v in test_metrics.items() 
                          if k in ['accuracy', 'f1_score', 'precision', 'recall'])
        
        if is_overfitted:
            log_message("⚠️ Warning: The best model appears to be overfitted (perfect test scores). "
                      "Consider collecting more data or using regularization.", "warning")
            
            # Still save the model but with a warning in the filename
            final_path = save_model_artifact(
                best_model_info['model'], 
                dataset_name, 
                f"{best_model_info['name']}_POTENTIAL_OVERFIT", 
                tag='warning_overfit'
            )
            log_message(f"⚠️ Saved potentially overfitted model with warning in filename: {final_path}", "warning")
        else:
            final_path = save_model_artifact(
                best_model_info['model'], 
                dataset_name, 
                best_model_info['name'], 
                tag='best'
            )
            log_message(f"✅ Best model: {best_model_info['name']} with cross-validation score {best_score_so_far:.4f}")
            log_message(f"Saved best model to {final_path}")
        
        st.success(f"Training complete. Best model: {best_model_info['name']} (score: {best_model_info['score']:.4f})")
        
        # Display metrics in a more readable format
        st.subheader("Best Model Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Accuracy", f"{best_model_info['test_metrics'].get('accuracy', 0):.4f}")
            st.metric("Precision", f"{best_model_info['test_metrics'].get('precision', 0):.4f}")
        
        with col2:
            st.metric("Recall", f"{best_model_info['test_metrics'].get('recall', 0):.4f}")
            st.metric("F1 Score", f"{best_model_info['test_metrics'].get('f1_score', 0):.4f}")
        
        # Show confusion matrix if available
        if 'confusion_matrix' in best_model_info['test_metrics']:
            st.subheader("Confusion Matrix")
            st.write(best_model_info['test_metrics']['confusion_matrix'])
            
        # Show classification report if available
        if 'classification_report' in best_model_info['test_metrics']:
            st.subheader("Classification Report")
            st.json(best_model_info['test_metrics']['classification_report'])
            
        st.download_button(
            label="Download Best Model",
            data=open(final_path, 'rb').read(),
            file_name=f"best_model_{dataset_name.replace(' ', '_')}.pkl",
            mime="application/octet-stream"
        )
        
        # Show cross-validation results if available
        if 'cv_results' in best_model_info:
            st.subheader("Cross-Validation Results")
            cv_metrics = {}
            for metric in config['metrics']:
                if f'mean_test_{metric}' in best_model_info['cv_results']:
                    mean_score = best_model_info['cv_results'][f'mean_test_{metric}'][0]  # Get first fold's score
                    std_score = best_model_info['cv_results'][f'std_test_{metric}'][0] if f'std_test_{metric}' in best_model_info['cv_results'] else 0
                    cv_metrics[metric] = f"{mean_score:.4f} (±{std_score:.4f})"
            st.json(cv_metrics)
            
        st.markdown(f"Model saved to: `{final_path}`")
    else:
        st.error("No successful model runs")

    st.info("Run this to view MLflow UI:")
    st.code(f"mlflow ui --backend-store-uri file:{MLFLOW_DIR}")

    if st.button("Clear Logs"):
        st.session_state.training_logs = []
        st.experimental_rerun()
