"""
=========================================================
Topic      : AI Fundamentals

Lesson     : AI Applications

File       : 07_ai_applications.py

Author     : Mohan Raju

Repository : GenAI-Learning-Journey

Description
-----------
Demonstrates various real-world AI applications
based on different industries.

Learning Objectives
-------------------
1. Understand AI applications.
2. Learn how AI is used in industries.
3. Practice Python dictionaries and functions.
=========================================================
"""

AI_APPLICATIONS = {
    "healthcare": [
        "Disease Detection",
        "Medical Imaging",
        "Drug Discovery"
    ],
    "finance": [
        "Fraud Detection",
        "Credit Scoring",
        "Loan Approval"
    ],
    "ecommerce": [
        "Product Recommendation",
        "Personalized Ads",
        "Inventory Forecasting"
    ],
    "education": [
        "AI Tutor",
        "Automatic Grading",
        "Personalized Learning"
    ],
    "cybersecurity": [
        "Threat Detection",
        "Malware Detection",
        "Network Monitoring"
    ],
    "devops": [
        "Log Analysis",
        "Root Cause Analysis",
        "Incident Prediction",
        "Capacity Planning",
        "Kubernetes Troubleshooting"
    ]
}


def get_ai_applications(domain: str) -> list:
    """
    Returns AI applications for the given domain.

    Args:
        domain (str): Industry or business domain.

    Returns:
        list: List of AI applications.
    """
    return AI_APPLICATIONS.get(domain.lower(), [])


def main():
    """
    Main function.
    """

    print("=" * 55)
    print(" AI Applications Explorer ")
    print("=" * 55)

    print("\nAvailable Domains")
    print("-----------------")

    for domain in AI_APPLICATIONS:
        print(f"- {domain}")

    domain = input("\nEnter a domain: ").strip()

    applications = get_ai_applications(domain)

    print("\nResults")
    print("-" * 20)

    if applications:
        print(f"\nAI Applications in {domain.title()}:\n")

        for index, app in enumerate(applications, start=1):
            print(f"{index}. {app}")
    else:
        print("No information available for this domain.")


if __name__ == "__main__":
    main()