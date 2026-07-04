"""
Sample review data for the sentiment analyzer.
In a real project, this would be loaded from a CSV/database.
"""

SAMPLE_REVIEWS = [
    # Positive
    {"id": 1, "product": "Laptop", "review": "Absolutely love this laptop! Lightning fast, great display. Best purchase I've made this year.", "rating": 5},
    {"id": 2, "product": "Laptop", "review": "Excellent build quality. The battery lasts all day. Highly recommend!", "rating": 5},
    {"id": 3, "product": "Headphones", "review": "Amazing sound quality. Noise cancellation works perfectly. Worth every penny.", "rating": 5},
    {"id": 4, "product": "Headphones", "review": "Very comfortable for long sessions. Crystal clear audio.", "rating": 4},
    {"id": 5, "product": "Keyboard", "review": "Great tactile feedback. Keys are responsive and satisfying to type on.", "rating": 4},

    # Negative
    {"id": 6, "product": "Laptop", "review": "Terrible experience. Overheats constantly and crashes every 2 hours. Waste of money.", "rating": 1},
    {"id": 7, "product": "Laptop", "review": "Not good at all. The keyboard stopped working after a week. Very disappointed.", "rating": 1},
    {"id": 8, "product": "Headphones", "review": "Poor build quality. The ear cups fell off within a month. Do not buy.", "rating": 1},
    {"id": 9, "product": "Keyboard", "review": "Keys are sticky and unresponsive. Feels very cheap. Would not recommend.", "rating": 2},

    # Neutral / Mixed
    {"id": 10, "product": "Laptop", "review": "It's okay for the price. Nothing special but gets the job done.", "rating": 3},
    {"id": 11, "product": "Headphones", "review": "Average sound. Expected more for this price point. It works though.", "rating": 3},
    {"id": 12, "product": "Keyboard", "review": "Decent keyboard. Some keys feel a bit loose but typing is fine overall.", "rating": 3},
]
