"""
Dataset Generator
Generates synthetic loan eligibility data.
"""

import random
import pandas as pd

random.seed(42)

NUMBER_OF_RECORDS = 500

data = []

for _ in range(NUMBER_OF_RECORDS):

    salary = random.randint(20000, 120000)
    experience = random.randint(0, 20)
    credit_score = random.randint(350, 900)

    score = 0

    if salary >= 60000:
        score += 1

    if experience >= 5:
        score += 1

    if credit_score >= 700:
        score += 1

    approved = 1 if score >= 2 else 0

    data.append({
        "salary": salary,
        "experience": experience,
        "credit_score": credit_score,
        "approved": approved
    })

loan_dataset = pd.DataFrame(data)

loan_dataset.to_csv("loan_dataset.csv", index=False)

print("=" * 60)
print("Loan Dataset Generated Successfully")
print("=" * 60)

print(f"Total Records : {len(loan_dataset)}")
print("\nFirst 10 Records\n")
print(loan_dataset.head(10))

#python dataset_generator.py