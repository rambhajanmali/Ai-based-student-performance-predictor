"""
Model training and evaluation for student performance prediction.

Scope:
- Import preprocessed data from data_preprocessing
- Train RandomForestRegressor with fixed random_state
- Evaluate using MAE and R² score
- Persist trained model to disk

No preprocessing logic is performed here.
"""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from config import MODEL_PARAMS, RANDOM_STATE
from src.data_preprocessing import preprocess_dataset


class ModelTrainer:
    """Handles training and evaluation of the student performance model."""

    def __init__(
        self,
        model_params: dict | None = None,
        model_path: str = "models/student_performance_model.pkl",
    ) -> None:
        """
        Initialize trainer with model hyperparameters.
        
        Random Forest is chosen for:
        - Robustness to mixed feature types (numerical + one-hot categorical)
        - Non-linear interactions capture (e.g., study time × failures)
        - Low risk of overfitting with moderate tuning
        """
        params = model_params or MODEL_PARAMS
        self.model = RandomForestRegressor(**params)
        self.model_path = model_path

    def train(self, X_train, y_train) -> None:
        """Fit the model on training data."""
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test) -> dict[str, float]:
        """Compute MAE and R² on test set."""
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return {"mae": mae, "r2": r2}

    def save_model(self, path: str | Path | None = None) -> None:
        """Persist trained model using joblib for reproducibility."""
        target_path = Path(path or self.model_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, target_path)

    def load_model(self, path: str | Path | None = None) -> None:
        """Load a previously trained model from disk."""
        target_path = Path(path or self.model_path)
        self.model = joblib.load(target_path)

    def predict(self, X):
        """Generate predictions for new data."""
        return self.model.predict(X)


def train_and_evaluate() -> ModelTrainer:
    """
    Full training pipeline: preprocess → train → evaluate → save.
    
    Returns trained ModelTrainer instance.
    """
    print("="*70)
    print("Model Training Pipeline")
    print("="*70)

    # Step 1: Load preprocessed data
    print("\n[1/4] Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_dataset()
    print(f"      Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"      Test:  {X_test.shape[0]} samples")

    # Step 2: Initialize and train model
    print("\n[2/4] Training RandomForestRegressor...")
    trainer = ModelTrainer()
    trainer.train(X_train, y_train)
    print("      Training complete")

    # Step 3: Evaluate on test set
    print("\n[3/4] Evaluating model...")
    metrics = trainer.evaluate(X_test, y_test)
    print(f"      MAE (Mean Absolute Error): {metrics['mae']:.3f}")
    print(f"      R² Score:                   {metrics['r2']:.3f}")

    # Step 4: Save trained model
    print("\n[4/4] Saving model...")
    trainer.save_model()
    print(f"      Model saved to {trainer.model_path}")

    print("\n" + "="*70)
    print("Training pipeline complete")
    print("="*70)

    return trainer


if __name__ == "__main__":
    train_and_evaluate()
