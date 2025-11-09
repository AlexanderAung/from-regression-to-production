"""
Main entry point for housing market prediction.

This script demonstrates how to:
1. Train a model
2. Make predictions
"""

from src.train_model import main as train_main
from src.predict import predict_price, load_model


def main():
    """Main function demonstrating training and prediction."""
    print("=" * 60)
    print("HOUSING MARKET PREDICTION - TRAINING & PREDICTION")
    print("=" * 60)

    # Option 1: Train the model
    print("\n[1] Training the model...")
    print("-" * 60)
    regressor, metrics = train_main()

    # Option 2: Make predictions
    print("\n[2] Making predictions on new houses...")
    print("-" * 60)

    # Example predictions
    examples = [
        {
            "area": 5000,
            "bedrooms": 3,
            "bathrooms": 2,
            "stories": 2,
            "mainroad": "yes",
            "guestroom": "no",
            "basement": "yes",
            "hotwaterheating": "no",
            "airconditioning": "yes",
            "parking": 2,
            "prefarea": "yes",
            "furnishingstatus": "furnished",
        },
        {
            "area": 3000,
            "bedrooms": 2,
            "bathrooms": 1,
            "stories": 1,
            "mainroad": "no",
            "guestroom": "no",
            "basement": "no",
            "hotwaterheating": "no",
            "airconditioning": "no",
            "parking": 0,
            "prefarea": "no",
            "furnishingstatus": "unfurnished",
        },
    ]

    for i, house in enumerate(examples, 1):
        price = predict_price(**house)
        print(f"\nHouse {i}:")
        print(
            f"  Features: {house['area']} sqft, {house['bedrooms']}BR/{house['bathrooms']}BA, "
            f"{house['stories']} stories"
        )
        print(f"  Predicted Price: ${price:,.2f}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
