"""Sample articles for the Text Intelligence Pipeline."""

SAMPLE_ARTICLES = [
    {
        "title": "AI in Healthcare",
        "text": """
Artificial intelligence is revolutionizing healthcare by enabling faster and more
accurate diagnoses. Machine learning models trained on millions of medical images
can now detect cancers, diabetic retinopathy, and cardiovascular conditions with
accuracy that matches or exceeds experienced specialists. In radiology, AI systems
analyze CT scans and MRIs in seconds, flagging anomalies for human review.
Beyond diagnostics, AI is accelerating drug discovery by predicting how molecules
will interact with biological targets, reducing the time to develop new treatments
from decades to years. Hospitals are also using AI for operational efficiency —
predicting patient admissions, optimizing bed allocation, and reducing administrative
burdens on clinical staff. However, challenges remain around data privacy, model
bias, and regulatory approval before widespread clinical deployment.
        """.strip(),
        "questions": [
            "What can AI detect in medical images?",
            "How is AI used in drug discovery?",
            "What challenges does AI face in healthcare?",
        ],
        "candidate_labels": ["healthcare", "technology", "finance", "sports", "education"],
    },
    {
        "title": "Python Programming Language",
        "text": """
Python has become the dominant programming language for data science and machine
learning due to its simplicity, readability, and rich ecosystem. Created by Guido
van Rossum and first released in 1991, Python emphasizes code readability with its
use of significant whitespace and clean syntax. The language supports multiple
programming paradigms including procedural, object-oriented, and functional styles.
Libraries like NumPy, Pandas, and Matplotlib form the data science stack, while
TensorFlow, PyTorch, and scikit-learn power machine learning workloads. Python's
package manager pip and virtual environments make dependency management
straightforward. In 2024, Python topped the TIOBE index as the world's most
popular programming language, surpassing Java and C for the first time.
        """.strip(),
        "questions": [
            "Who created Python?",
            "When was Python first released?",
            "What libraries are used for machine learning?",
        ],
        "candidate_labels": ["programming", "sports", "cooking", "finance", "travel"],
    },
]
