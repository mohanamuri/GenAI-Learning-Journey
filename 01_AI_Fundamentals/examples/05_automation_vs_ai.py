"""
=========================================================
Topic      : AI Fundamentals

Lesson     : Automation vs Artificial Intelligence

File       : 05_automation_vs_ai.py

Author     : Mohan Raju

Repository : GenAI-Learning-Journey

Description
-----------
Demonstrates the difference between
Automation and Artificial Intelligence.

=========================================================
"""


def automation(cpu_usage: int):
    """
    Fixed rule.
    """

    if cpu_usage > 80:
        return "Restart Service"

    return "Everything Normal"


def ai_prediction(cpu_usage: int):
    """
    Simulated AI prediction.

    (In future we'll replace this
    with a Machine Learning model.)
    """

    if cpu_usage > 90:
        return "High Risk of Failure"

    elif cpu_usage > 70:
        return "Medium Risk"

    return "Low Risk"


def main():

    cpu = int(input("Enter CPU Usage (%): "))

    print("\nAutomation Result")
    print("------------------")
    print(automation(cpu))

    print("\nAI Prediction")
    print("------------------")
    print(ai_prediction(cpu))

    print("\nObservation")
    print("------------------")
    print("Automation follows rules.")
    print("AI predicts future behavior.")


if __name__ == "__main__":
    main()