"""
============================================================
Project : AI Loan Eligibility System

Train Machine Learning Model

============================================================
"""

from models.loan_model import train_model


def main():

    print("=" * 60)
    print(" AI Loan Eligibility System ")
    print("=" * 60)

    print("\nTraining Model...\n")

    accuracy = train_model()

    print("Training Completed Successfully")

    print(
        f"\nModel Accuracy : {accuracy*100:.2f}%"
    )

    print("\nModel saved successfully.")


if __name__ == "__main__":
    main()