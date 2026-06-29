"""
=========================================================
Topic      : Machine Learning

Lesson     : Learning From Data

File       : 01_learning_from_data.py

Author     : Mohan Raju

Repository : GenAI-Learning-Journey

Description
-----------
Demonstrates how computers learn
patterns from historical data.

NOTE

This is NOT Machine Learning.

This is a simulation showing the
basic idea behind learning.
=========================================================
"""

# Historical data

students = [
    {"hours": 2, "result": "Fail"},
    {"hours": 3, "result": "Fail"},
    {"hours": 5, "result": "Pass"},
    {"hours": 6, "result": "Pass"},
    {"hours": 8, "result": "Pass"},
]


def learn_rule(data):
    """
    Learns the minimum study hours
    required to pass.
    """

    passed = [
        student["hours"]
        for student in data
        if student["result"] == "Pass"
    ]

    return min(passed)


def predict(hours, learned_hours):
    """
    Predicts pass/fail.
    """

    if hours >= learned_hours:
        return "Pass"

    return "Fail"


def main():

    print("=" * 60)
    print(" Learning From Data ")
    print("=" * 60)

    print("\nHistorical Data")

    for student in students:
        print(student)

    print("\nLearning...")

    rule = learn_rule(students)

    print(f"\nDiscovered Rule")

    print(f"Minimum Hours = {rule}")

    hours = int(input("\nStudy Hours : "))

    prediction = predict(hours, rule)

    print("\nPrediction")

    print(prediction)

    print("\nObservation")

    print(
        "The computer discovered the rule "
        "from historical data."
    )


if __name__ == "__main__":
    main()