"""
=========================================================
Topic      : AI Fundamentals

Lesson     : Types of Artificial Intelligence

File       : 06_types_of_ai.py

Author     : Mohan Raju

Repository : GenAI-Learning-Journey

Description
-----------
Classifies AI systems into ANI, AGI, or ASI.

Learning Objectives
-------------------
1. Understand the three types of AI.
2. Learn how AI systems are categorized.
3. Practice clean Python programming.
=========================================================
"""

AI_TYPES = {
    "calculator": "ANI - Artificial Narrow Intelligence",
    "chess engine": "ANI - Artificial Narrow Intelligence",
    "chatgpt": "ANI - Artificial Narrow Intelligence",
    "github copilot": "ANI - Artificial Narrow Intelligence",
    "siri": "ANI - Artificial Narrow Intelligence",
    "alexa": "ANI - Artificial Narrow Intelligence",
    "jarvis": "AGI - Artificial General Intelligence (Fictional)",
    "skynet": "ASI - Artificial Super Intelligence (Fictional)"
}


def classify_ai(system_name: str) -> str:
    """
    Classifies an AI system.

    Args:
        system_name (str): Name of the AI system.

    Returns:
        str: AI category.
    """
    return AI_TYPES.get(
        system_name.lower(),
        "Unknown AI System"
    )


def main():
    """
    Main function.
    """

    print("=" * 50)
    print(" AI Type Classifier ")
    print("=" * 50)

    system = input("\nEnter AI System Name: ").strip()

    result = classify_ai(system)

    print("\nClassification")
    print("-" * 20)
    print(result)


if __name__ == "__main__":
    main()