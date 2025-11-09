"""
Training script for housing price prediction model.

This script:
1. Loads preprocessed train/test data
2. Builds and trains a complete pipeline
3. Evaluates the model
4. Saves the trained pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PowerTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector as selector, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Import custom transformers (you can also define them here)
# For now, we'll define them inline for clarity

YES_NO_COLS = [
    "mainroad", "guestroom", "basement", "hotwaterheating", 
    "airconditioning", "prefarea"
]
TARGET = "price"


class YesNoMapper(BaseEstimator, TransformerMixin):
    """Maps yes/no values to 1/0 for binary categorical variables."""
    def __init__(self, yes_no_cols=YES_NO_COLS):
        self.yes_no_cols = yes_no_cols
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        Xc = X.copy()
        for col in self.yes_no_cols:
            if col in Xc.columns:
                Xc[col] = Xc[col].map({"yes": 1, "no": 0}).astype(int)
        return Xc


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Engineers new features from existing ones."""
    def __init__(self, target_col=TARGET):
        self.target_col = target_col
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        Xc = X.copy()
        
        # Space features
        if "bedrooms" in Xc.columns and "bathrooms" in Xc.columns:
            Xc["total_rooms"] = Xc["bedrooms"].astype(float) + Xc["bathrooms"].astype(float)
            Xc["bath_bed_ratio"] = (
                Xc["bathrooms"].astype(float) / 
                Xc["bedrooms"].replace({0: np.nan}).astype(float)
            )
        
        # Luxury features
        luxury_cols = [c for c in YES_NO_COLS if c in Xc.columns]
        if luxury_cols:
            Xc["luxury_score"] = Xc[luxury_cols].sum(axis=1)
        
        # Property features
        if "stories" in Xc.columns:
            Xc["is_multi_story"] = (Xc["stories"].astype(float) > 1).astype(int)
        
        if "area" in Xc.columns and "stories" in Xc.columns:
            Xc["area_story_interaction"] = (
                Xc["area"].astype(float) * Xc["stories"].astype(float)
            )
        
        # Price per sqft (only if target is in X - this happens during fit, not transform)
        # Note: In production, target won't be available, so we skip this feature
        # or compute it differently. For now, we'll skip it in transform.
        
        # Handle inf/nan
        Xc.replace([np.inf, -np.inf], np.nan, inplace=True)
        return Xc


def build_pipeline():
    """Builds the complete preprocessing and modeling pipeline."""
    
    # Step 1: Feature engineering pipeline
    pre_feature = Pipeline(steps=[
        ("map_yes_no", YesNoMapper()),
        ("engineer", FeatureEngineer(target_col=TARGET)),
    ])
    
    # Step 2: Column-wise transforms
    numeric_pipe = Pipeline(steps=[
        ("yj", PowerTransformer(standardize=True)),
    ])
    
    categorical_pipe = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
    ])
    
    column_pipe = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, selector(dtype_include=np.number)),
            ("cat", categorical_pipe, selector(dtype_include=object)),
        ],
        remainder="drop",
    )
    
    # Step 3: Full pipeline
    full_pipe = Pipeline(steps=[
        ("pre_feature", pre_feature),
        ("columns", column_pipe),
        ("model", Ridge(alpha=1.0)),
    ])
    
    # Step 4: Transform target too
    regressor = TransformedTargetRegressor(
        regressor=full_pipe,
        transformer=PowerTransformer(standardize=True)
    )
    
    return regressor


def load_data():
    """Loads preprocessed train/test data. Creates it if it doesn't exist."""
    data_dir = Path(__file__).parent.parent / "data" / "processed"
    raw_data_path = Path(__file__).parent.parent / "data" / "raw" / "Housing.csv"
    
    # Check if processed data exists
    if not (data_dir / "X_train.csv").exists():
        print("Processed data not found. Creating train/test split from raw data...")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load raw data
        df_raw = pd.read_csv(raw_data_path)
        
        # Split data
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            df_raw,
            test_size=0.2,
            random_state=42,
            shuffle=True
        )
        
        # Split features and target
        X_train = train_df.drop(columns=[TARGET])
        X_test = test_df.drop(columns=[TARGET])
        y_train = train_df[TARGET]
        y_test = test_df[TARGET]
        
        # Save
        X_train.to_csv(data_dir / "X_train.csv", index=False)
        X_test.to_csv(data_dir / "X_test.csv", index=False)
        y_train.to_csv(data_dir / "y_train.csv", index=False)
        y_test.to_csv(data_dir / "y_test.csv", index=False)
        print(f"✓ Saved processed data to {data_dir}")
    else:
        # Load existing processed data
        X_train = pd.read_csv(data_dir / "X_train.csv")
        X_test = pd.read_csv(data_dir / "X_test.csv")
        y_train = pd.read_csv(data_dir / "y_train.csv").squeeze()
        y_test = pd.read_csv(data_dir / "y_test.csv").squeeze()
    
    return X_train, X_test, y_train, y_test


def evaluate_model(regressor, X_train, X_test, y_train, y_test):
    """Evaluates the model and prints metrics."""
    
    # Predictions
    pred_train = regressor.predict(X_train)
    pred_test = regressor.predict(X_test)
    
    # Metrics
    metrics = {
        "Train RMSE": mean_squared_error(y_train, pred_train, squared=False),
        "Test RMSE": mean_squared_error(y_test, pred_test, squared=False),
        "Train MAE": mean_absolute_error(y_train, pred_train),
        "Test MAE": mean_absolute_error(y_test, pred_test),
        "Train R²": r2_score(y_train, pred_train),
        "Test R²": r2_score(y_test, pred_test),
    }
    
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    for metric, value in metrics.items():
        print(f"{metric:15s}: {value:10.2f}")
    print("="*50)
    
    # Check for overfitting
    train_test_gap = metrics["Train R²"] - metrics["Test R²"]
    if train_test_gap > 0.1:
        print(f"\n⚠️  Warning: Large train-test gap ({train_test_gap:.3f})")
        print("   Model may be overfitting. Consider regularization.")
    else:
        print("\n✓ Train and test performance are similar (good generalization)")
    
    return metrics


def main():
    """Main training function."""
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    print("\nBuilding pipeline...")
    regressor = build_pipeline()
    
    print("Training model...")
    regressor.fit(X_train, y_train)
    
    print("Evaluating model...")
    metrics = evaluate_model(regressor, X_train, X_test, y_train, y_test)
    
    # Save model
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "housing_price_pipeline.joblib"
    
    joblib.dump(regressor, model_path)
    print(f"\n✓ Model saved to: {model_path.resolve()}")
    
    return regressor, metrics


if __name__ == "__main__":
    regressor, metrics = main()

