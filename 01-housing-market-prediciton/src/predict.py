"""
Prediction script for housing price model.

This script loads a trained model and makes predictions on new data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib


def load_model(model_path=None):
    """Loads the trained pipeline from disk."""
    if model_path is None:
        model_path = Path(__file__).parent.parent / "models" / "housing_price_pipeline.joblib"
    else:
        model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please train the model first using train_model.py"
        )
    
    return joblib.load(model_path)


def predict_price(
    area,
    bedrooms,
    bathrooms,
    stories,
    mainroad="yes",
    guestroom="no",
    basement="no",
    hotwaterheating="no",
    airconditioning="no",
    parking=0,
    prefarea="no",
    furnishingstatus="furnished",
    model_path=None
):
    """
    Predicts house price given features.
    
    Parameters:
    -----------
    area : int or float
        Area in square feet
    bedrooms : int
        Number of bedrooms
    bathrooms : int
        Number of bathrooms
    stories : int
        Number of stories
    mainroad : str, default "yes"
        "yes" or "no"
    guestroom : str, default "no"
        "yes" or "no"
    basement : str, default "no"
        "yes" or "no"
    hotwaterheating : str, default "no"
        "yes" or "no"
    airconditioning : str, default "no"
        "yes" or "no"
    parking : int, default 0
        Number of parking spaces
    prefarea : str, default "no"
        "yes" or "no"
    furnishingstatus : str, default "furnished"
        "furnished", "semi-furnished", or "unfurnished"
    model_path : str or Path, optional
        Path to saved model. If None, uses default location.
    
    Returns:
    --------
    float
        Predicted price
    """
    # Load model
    regressor = load_model(model_path)
    
    # Create DataFrame with same structure as training data
    new_data = pd.DataFrame({
        "area": [area],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "stories": [stories],
        "mainroad": [mainroad],
        "guestroom": [guestroom],
        "basement": [basement],
        "hotwaterheating": [hotwaterheating],
        "airconditioning": [airconditioning],
        "parking": [parking],
        "prefarea": [prefarea],
        "furnishingstatus": [furnishingstatus]
    })
    
    # Predict
    predicted_price = regressor.predict(new_data)[0]
    
    return predicted_price


def predict_batch(df, model_path=None):
    """
    Predicts prices for a batch of houses.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns matching training data
    model_path : str or Path, optional
        Path to saved model
    
    Returns:
    --------
    numpy.ndarray
        Array of predicted prices
    """
    regressor = load_model(model_path)
    return regressor.predict(df)


def main():
    """Example usage."""
    print("Loading model...")
    regressor = load_model()
    
    # Example 1: Single prediction
    print("\n" + "="*50)
    print("Example 1: Single House Prediction")
    print("="*50)
    
    price = predict_price(
        area=5000,
        bedrooms=3,
        bathrooms=2,
        stories=2,
        mainroad="yes",
        guestroom="no",
        basement="yes",
        hotwaterheating="no",
        airconditioning="yes",
        parking=2,
        prefarea="yes",
        furnishingstatus="furnished"
    )
    
    print(f"Predicted price: ${price:,.2f}")
    
    # Example 2: Batch prediction
    print("\n" + "="*50)
    print("Example 2: Batch Prediction")
    print("="*50)
    
    new_houses = pd.DataFrame({
        "area": [4000, 6000, 3000],
        "bedrooms": [2, 4, 3],
        "bathrooms": [1, 3, 2],
        "stories": [1, 3, 2],
        "mainroad": ["yes", "yes", "no"],
        "guestroom": ["no", "yes", "no"],
        "basement": ["no", "yes", "yes"],
        "hotwaterheating": ["no", "no", "no"],
        "airconditioning": ["yes", "yes", "no"],
        "parking": [1, 3, 0],
        "prefarea": ["no", "yes", "no"],
        "furnishingstatus": ["semi-furnished", "furnished", "unfurnished"]
    })
    
    predictions = predict_batch(new_houses)
    
    for i, (_, house) in enumerate(new_houses.iterrows()):
        print(f"\nHouse {i+1}:")
        print(f"  Area: {house['area']} sqft, Bedrooms: {house['bedrooms']}, "
              f"Bathrooms: {house['bathrooms']}")
        print(f"  Predicted price: ${predictions[i]:,.2f}")


if __name__ == "__main__":
    main()

