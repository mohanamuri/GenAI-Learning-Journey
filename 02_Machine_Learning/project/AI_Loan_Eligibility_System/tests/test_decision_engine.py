from decision_engine import LoanDecisionEngine
from constants import APPROVED


def test_loan_approval():

    engine = LoanDecisionEngine()

    decision, confidence, reasons = engine.evaluate(
        age=30,
        salary=60000,
        experience=5,
        credit_score=800,
        employment="Permanent",
    )

    assert decision == APPROVED