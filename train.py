"""
Unified model training script for YugenAI project
"""

import argparse
import logging
from pathlib import Path

from src.utils.logger import setup_logger

from .train_housing import HousingModelTrainer
from .train_iris import IrisModelTrainer

logger = setup_logger("model_training")


def train_iris_model(data_path: str = "data/raw/iris.csv", experiment_name: str = None):
    """
    Train iris classification models

    Args:
        data_path: Path to iris data file
        experiment_name: MLflow experiment name
    """
    logger.info("Starting iris model training")

    # Initialize trainer
    trainer = IrisModelTrainer(experiment_name=experiment_name)

    # Train all models
    results = trainer.train_all_models(data_path)

    # Save best model
    best_model_path = trainer.save_best_model()

    # Save results summary
    summary_path = trainer.save_results_summary()

    logger.info("Iris training completed successfully!")
    logger.info(f"Best model saved to: {best_model_path}")
    logger.info(f"Results summary saved to: {summary_path}")

    return results


def train_housing_model(
    data_path: str = "data/raw/housing.csv", experiment_name: str = None
):
    """
    Train housing regression models

    Args:
        data_path: Path to housing data file
        experiment_name: MLflow experiment name
    """
    logger.info("Starting housing model training")

    # Initialize trainer
    trainer = HousingModelTrainer(experiment_name=experiment_name)

    # Train all models
    results = trainer.train_all_models(data_path)

    # Save best model
    best_model_path = trainer.save_best_model()

    # Save results summary
    summary_path = trainer.save_results_summary()

    logger.info("Housing training completed successfully!")
    logger.info(f"Best model saved to: {best_model_path}")
    logger.info(f"Results summary saved to: {summary_path}")

    return results


def main():
    """Main training function with command line interface"""
    parser = argparse.ArgumentParser(description="Train YugenAI models")
    parser.add_argument(
        "--model",
        choices=["iris", "housing", "all"],
        default="all",
        help="Model to train (default: all)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to data file (optional, uses defaults if not specified)",
    )
    parser.add_argument(
        "--experiment-name", type=str, help="MLflow experiment name (optional)"
    )
    parser.add_argument(
        "--iris-data",
        type=str,
        default="data/raw/iris.csv",
        help="Path to iris data file",
    )
    parser.add_argument(
        "--housing-data",
        type=str,
        default="data/raw/housing.csv",
        help="Path to housing data file",
    )

    args = parser.parse_args()

    logger.info("Starting YugenAI model training")
    logger.info(f"Model: {args.model}")

    results = {}

    try:
        if args.model in ["iris", "all"]:
            iris_data_path = args.data_path or args.iris_data
            iris_experiment = args.experiment_name or "iris_classification"
            results["iris"] = train_iris_model(iris_data_path, iris_experiment)

        if args.model in ["housing", "all"]:
            housing_data_path = args.data_path or args.housing_data
            housing_experiment = args.experiment_name or "housing_prediction"
            results["housing"] = train_housing_model(
                housing_data_path, housing_experiment
            )

        logger.info("All training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

    return results


if __name__ == "__main__":
    main()
