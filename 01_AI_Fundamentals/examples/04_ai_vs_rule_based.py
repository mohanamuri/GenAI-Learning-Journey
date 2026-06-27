"""
=========================================================
Topic      : AI Fundamentals

Lesson     : AI vs Traditional Programming

File       : 04_ai_vs_rule_based.py

Author     : Mohan Raju

Repository : GenAI-Learning-Journey

Description
-----------
Demonstrates why rule-based programming
becomes difficult as business rules increase.

Learning Objectives
-------------------
1. Understand the limitation of rule-based systems.
2. Learn why AI was invented.
3. Compare rule-based systems with AI thinking.
=========================================================
"""


def identify_animal(legs: int, tail: bool, sound: str):
    """
    Identifies an animal using manually written rules.
    """

    if legs == 4 and tail and sound.lower() == "meow":
        return "🐱 Cat"

    elif legs == 4 and tail and sound.lower() == "bark":
        return "🐶 Dog"

    elif legs == 2 and not tail:
        return "🧑 Human"

    else:
        return "❓ Unknown"


def main():

    print("=" * 45)
    print(" Rule-Based Animal Identification ")
    print("=" * 45)

    legs = int(input("Number of Legs : "))
    tail = input("Has Tail (yes/no): ").strip().lower() == "yes"
    sound = input("Animal Sound : ")

    animal = identify_animal(legs, tail, sound)

    print("\nPrediction :", animal)

    print("\nObservation")
    print("-" * 20)
    print("This works only because WE wrote the rules.")
    print("Imagine writing rules for 10 million animals!")
    print("That is why Artificial Intelligence exists.")


if __name__ == "__main__":
    main()