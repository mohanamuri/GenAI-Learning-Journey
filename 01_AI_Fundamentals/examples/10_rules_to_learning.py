"""
=========================================================
Topic      : AI Fundamentals

Lesson     : From Rules to Learning

File       : 10_rules_to_learning.py

Author     : Mohan Raju

Repository : GenAI-Learning-Journey

Description
-----------
Demonstrates the transition from
Rule-Based Programming to Learning
using historical data.

NOTE:
This is NOT Machine Learning.
This is a simulation to explain
how learning begins.

Learning Objectives
-------------------
1. Understand why Machine Learning exists.
2. Learn the difference between rules and data.
3. Understand pattern discovery.
=========================================================
"""

# --------------------------------------------
# Historical Data
# --------------------------------------------

training_data = [
    {"age": 25, "salary": 50000, "approved": True},
    {"age": 30, "salary": 60000, "approved": True},
    {"age": 22, "salary": 35000, "approved": True},
    {"age": 17, "salary": 10000, "approved": False},
    {"age": 16, "salary": 8000, "approved": False},
    {"age": 18, "salary": 32000, "approved": True},
]


def learn_rules(data):
    """
    Simulates learning from historical data.

    This function does NOT use Machine Learning.

    It simply observes approved records
    and discovers the minimum age
    and minimum salary.
    """

    approved = [row for row in data if row["approved"]]

    minimum_age = min(person["age"] for person in approved)
    minimum_salary = min(person["salary"] for person in approved)

    return minimum_age, minimum_salary


def predict(age, salary, learned_age, learned_salary):
    """
    Makes prediction using learned values.
    """

    if age >= learned_age and salary >= learned_salary:
        return "Approved"

    return "Rejected"


def main():

    print("=" * 60)
    print(" Rules → Learning Demonstration ")
    print("=" * 60)

    print("\nTraining Data")
    print("-" * 30)

    for person in training_data:

        decision = "Approved" if person["approved"] else "Rejected"

        print(
            f"Age={person['age']:2} | "
            f"Salary={person['salary']:6} | "
            f"{decision}"
        )

    print("\nLearning...")
    print("-" * 30)

    learned_age, learned_salary = learn_rules(training_data)

    print("Pattern Discovered")

    print(f"Minimum Approved Age    : {learned_age}")
    print(f"Minimum Approved Salary : {learned_salary}")

    print("\nNow let's test a NEW customer.")

    age = int(input("\nEnter Age: "))
    salary = int(input("Enter Salary: "))

    result = predict(
        age,
        salary,
        learned_age,
        learned_salary,
    )

    print("\nPrediction")
    print("-" * 30)
    print(result)

    print("\nObservation")
    print("-" * 30)

    print("Nobody wrote:")
    print("if age >= 18")

    print("\nNobody wrote:")
    print("if salary >= 32000")

    print("\nThe computer DISCOVERED")
    print("those values from historical data.")

    print("\nThis idea becomes Machine Learning.")


if __name__ == "__main__":
    main()