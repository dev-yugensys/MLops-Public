import os
import time
import itertools
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from pathlib import Path
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
)
from importlib import import_module

# ====== CONFIG ======
MLFLOW_DIR = Path("mlruns").absolute()
MLFLOW_DIR.mkdir(exist_ok=True)

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
            "Random Forest": {
                "model": "sklearn.ensemble.RandomForestClassifier",
                "params": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [5, 10, None],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2],
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
            },
            "XGBoost": {
                "model": "xgboost.XGBClassifier",
                "params": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1, 0.3],
                    "max_depth": [3, 6, 9],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                    "random_state": [42]
                }
            }
        }
    },
    "Housing Regression": {
        "task_type": "regression",
        "metrics": ["mse", "mae", "r2", "explained_variance"],
        "models": {
            "Linear Regression": {
                "model": "sklearn.linear_model.LinearRegression",
                "params": {
                    "fit_intercept": [True, False],
                    "positive": [True, False]
                }
            },
            "Random Forest": {
                "model": "sklearn.ensemble.RandomForestRegressor",
                "params": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [5, 10, 20, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "random_state": [42]
                }
            },
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
    mlflow.sklearn.autolog(log_models=True)
    mlflow.set_experiment(experiment_name)

def load_dataset(dataset_name):
    if dataset_name == "Iris Classification":
        data = datasets.load_iris(as_frame=True)
        return data.data, data.target
    elif dataset_name == "Housing Regression":
        data = datasets.fetch_california_housing(as_frame=True)
        return data.data, data.target
    else:
        raise ValueError("Unknown dataset")

def get_param_combinations(params_dict):
    keys = list(params_dict.keys())
    values = list(params_dict.values())
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))

def save_best_model(results, dataset_name):
    metric = "accuracy" if MODEL_CONFIGS[dataset_name]["task_type"] == "classification" else "r2"
    best_model_name, best_model, best_score = None, None, -np.inf
    for model_name, runs in results.items():
        for run in runs:
            if run["metrics"][metric] > best_score:
                best_model_name = model_name
                best_model = run["model"]
                best_score = run["metrics"][metric]
    save_path = Path(f"models/{dataset_name.replace(' ', '_').lower()}_model.pkl")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, save_path)
    return save_path

# ====== STREAMLIT UI ======
st.set_page_config(page_title="Model Training", layout="wide")
st.title("Model Training with MLflow")

dataset_name = st.selectbox("Select Dataset", list(MODEL_CONFIGS.keys()))

models_config = MODEL_CONFIGS[dataset_name]["models"]
num_models = len(models_config)
total_param_sets = sum(len(list(get_param_combinations(cfg["params"]))) for cfg in models_config.values())

st.info(f"📊 **{num_models} models** will be trained with a total of **{total_param_sets} hyperparameter sets**.")

if st.button("Train Models"):
    st.write("⏳ Starting training...")
    progress_bar = st.progress(0)
    
    # Create a container for logs with max height and scrollable
    log_container = st.container()
    log_placeholder = log_container.empty()
    
    # Add some CSS to make the log container scrollable with a fixed height
    st.markdown("""
    <style>
    .log-container {
        max-height: 300px;
        overflow-y: auto;
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 1rem;
        background-color: #1e1e1e;  /* Dark background */
        color: #e0e0e0;  /* Light text color */
        font-family: monospace;
        white-space: pre-wrap;
        margin-top: 1rem;
    }
    .log-entry {
        margin: 0.25rem 0;
        padding: 0.5rem;
        border-radius: 0.25rem;
        line-height: 1.4;
    }
    .log-entry:nth-child(odd) {
        background-color: #2d2d2d;  /* Slightly lighter dark for odd rows */
    }
    .log-entry:hover {
        background-color: #3a3a3a;  /* Hover effect */
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize log list in session state if it doesn't exist
    if 'training_logs' not in st.session_state:
        st.session_state.training_logs = []
    
    def log_message(message):
        """Add a message to the training logs"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        st.session_state.training_logs.append(formatted_message)
        # Keep only the last 50 log messages to prevent memory issues
        if len(st.session_state.training_logs) > 50:
            st.session_state.training_logs = st.session_state.training_logs[-50:]
        # Update the log display
        log_placeholder.markdown(
            f'<div class="log-container">' + 
            ''.join(f'<div class="log-entry">{log}</div>' for log in st.session_state.training_logs) + 
            '</div>', 
            unsafe_allow_html=True
        )
    
    # Clear previous logs when starting new training
    st.session_state.training_logs = []
    log_message("Initializing training...")
    
    X, y = load_dataset(dataset_name)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    log_message(f"Loaded {dataset_name} dataset with {X.shape[0]} samples and {X.shape[1]} features")

    setup_mlflow(dataset_name.replace(" ", "_"))
    results = {}
    total_steps = total_param_sets
    current_step = 0

    for model_name, cfg in models_config.items():
        ModelClass = getattr(import_module(cfg["model"].rsplit(".", 1)[0]), cfg["model"].rsplit(".", 1)[1])
        param_sets = list(get_param_combinations(cfg["params"]))
        results[model_name] = []

        for params in param_sets:
            current_step += 1
            progress = current_step / total_steps
            progress_bar.progress(progress)
            with mlflow.start_run(run_name=f"{model_name}"):
                mlflow.log_params(params)
                log_message(f"🚀 Training {model_name} ({current_step}/{total_steps}): {params}")
                try:
                    model = ModelClass(**params)
                    start_time = time.time()
                    model.fit(X_train, y_train)
                    training_time = time.time() - start_time
                    log_message(f"✅ Completed {model_name} in {training_time:.2f} seconds")

                    if MODEL_CONFIGS[dataset_name]["task_type"] == "classification":
                        y_pred = model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, average="weighted")
                        recall = recall_score(y_test, y_pred, average="weighted")
                        f1 = f1_score(y_test, y_pred, average="weighted")
                        
                        metrics = {
                            "accuracy": accuracy,
                            "precision": precision,
                            "recall": recall,
                            "f1_score": f1
                        }
                        
                        log_message(f"📊 {model_name} Results - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
                    else:
                        y_pred = model.predict(X_test)
                        mse = mean_squared_error(y_test, y_pred)
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        explained_var = explained_variance_score(y_test, y_pred)
                        
                        metrics = {
                            "mse": mse,
                            "mae": mae,
                            "r2": r2,
                            "explained_variance": explained_var
                        }
                        
                        log_message(f"📊 {model_name} Results - R²: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, Explained Variance: {explained_var:.4f}")

                    mlflow.log_metrics(metrics)
                    results[model_name].append({
                        "params": params,
                        "metrics": metrics,
                        "model": model
                    })
                    
                except Exception as e:
                    log_message(f"❌ Error training {model_name}: {str(e)}")
                    continue

    try:
        best_model_path = save_best_model(results, dataset_name)
        log_message(f"🏆 Training completed! Best model saved to {best_model_path}")
        st.success("✅ Training completed successfully!")
        
        # Add a button to clear logs
        if st.button("Clear Logs"):
            st.session_state.training_logs = []
            log_placeholder.markdown('<div class="log-container"></div>', unsafe_allow_html=True)
            st.rerun()
            
    except Exception as e:
        log_message(f"❌ Error saving best model: {str(e)}")
        st.error("An error occurred during training. Please check the logs for details.")
    st.info("Run this to view MLflow UI:")
    st.code(f"mlflow ui --backend-store-uri file:{MLFLOW_DIR}")
