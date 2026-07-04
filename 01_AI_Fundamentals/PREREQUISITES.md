# Prerequisites — Module 01 AI Fundamentals

## No External Libraries — Runs Out of the Box

All examples use only Python standard library. No pip installs, no downloads, no API keys.

---

## Requirements

- Python 3.8+

Verify:
```bash
python3 --version
```

---

## What Each Example Does & How to Run

| Example | What it does | Run |
|---------|-------------|-----|
| `01_rule_based_programming.py` | Hardcoded if/else rules — the baseline before AI | `python 01_rule_based_programming.py` |
| `02_pattern_matching.py` | Match text patterns using regex and rules | `python 02_pattern_matching.py` |
| `03_decision_engine.py` | Rule-based decision tree — no ML, pure logic | `python 03_decision_engine.py` |
| `04_ai_vs_rule_based.py` | Side-by-side: rules vs learning approach | `python 04_ai_vs_rule_based.py` |
| `05_automation_vs_ai.py` | Automation repeats steps; AI adapts to data | `python 05_automation_vs_ai.py` |
| `06_types_of_ai.py` | Narrow AI vs General AI vs Super AI | `python 06_types_of_ai.py` |
| `07_ai_applications.py` | Real-world AI use cases across industries | `python 07_ai_applications.py` |
| `08_ai_limitations.py` | What AI cannot do — bias, explainability, data hunger | `python 08_ai_limitations.py` |
| `09_ai_ethics.py` | Fairness, privacy, accountability in AI | `python 09_ai_ethics.py` |
| `10_rules_to_learning.py` | Bridge from rule-based to ML — the transition | `python 10_rules_to_learning.py` |

---

## Run All Examples

```bash
cd 01_AI_Fundamentals/examples/

# Run one
python 01_rule_based_programming.py

# Run all in order
for f in $(ls *.py | sort); do
    echo "=== $f ==="; python "$f"; echo
done
```

---

## No API Keys — No Cost

Everything runs locally with zero dependencies.
