"""
=========================================================
Topic      : AI Fundamentals

Lesson     : AI Limitations

File       : 08_ai_limitations.py

Author     : Mohan Raju

Repository : GenAI-Learning-Journey

Description
-----------
Demonstrates that AI predictions have
different confidence levels and should
not always be treated as certainty.

Learning Objectives
-------------------
1. Understand AI limitations.
2. Learn the concept of confidence.
3. Understand why human verification is important.
=========================================================
"""

HIGH = 90
MEDIUM = 70


def predict_failure(cpu: int) -> tuple[str, int]:
    """
    Simulates an AI prediction.

    Args:
        cpu (int): CPU utilization percentage.

    Returns:
        tuple[str, int]:
            Prediction and confidence score.
    """

    if cpu >= HIGH:
        return "High Risk of Failure", 95

    if cpu >= MEDIUM:
        return "Medium Risk of Failure", 75

    return "Low Risk of Failure", 60


def main():
    """
    Main function.
    """

    print("=" * 55)
    print(" AI Prediction Confidence Demo ")
    print("=" * 55)

    cpu = int(input("\nEnter CPU Usage (%): ").strip())

    prediction, confidence = predict_failure(cpu)

    print("\nPrediction")
    print("-" * 20)
    print(prediction)

    print(f"\nConfidence : {confidence}%")

    print("\nImportant Note")
    print("-" * 20)

    if confidence < 80:
        print("⚠️ Human verification recommended.")
    else:
        print("✅ High confidence, but human review is still advised.")

    print("\nObservation")
    print("-" * 20)
    print("AI provides predictions, not guarantees.")


if __name__ == "__main__":
    main()