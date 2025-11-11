import pandas as pd

data = [
    # hour, campaign, spend, clicks, conversions (arrive later)
    {"hour": 0, "campaign": "A", "spend": 100, "clicks": 50, "conversions": 0},
    {"hour": 0, "campaign": "B", "spend": 100, "clicks": 40, "conversions": 0},
    {"hour": 0, "campaign": "C", "spend": 100, "clicks": 30, "conversions": 0},

    {"hour": 1, "campaign": "A", "spend": 80, "clicks": 40, "conversions": 0},
    {"hour": 1, "campaign": "B", "spend": 120, "clicks": 50, "conversions": 0},
    {"hour": 1, "campaign": "C", "spend": 100, "clicks": 35, "conversions": 0},

    {"hour": 2, "campaign": "A", "spend": 90, "clicks": 45, "conversions": 0},
    {"hour": 2, "campaign": "B", "spend": 110, "clicks": 55, "conversions": 0},
    {"hour": 2, "campaign": "C", "spend": 100, "clicks": 30, "conversions": 0},

    # conversions arrive late (delayed feedback)
    {"hour": 3, "campaign": "A", "spend": 0, "clicks": 0, "conversions": 4},
    {"hour": 3, "campaign": "B", "spend": 0, "clicks": 0, "conversions": 2},
    {"hour": 3, "campaign": "C", "spend": 0, "clicks": 0, "conversions": 1},

    {"hour": 4, "campaign": "A", "spend": 0, "clicks": 0, "conversions": 2},
    {"hour": 4, "campaign": "B", "spend": 0, "clicks": 0, "conversions": 1},
    {"hour": 4, "campaign": "C", "spend": 0, "clicks": 0, "conversions": 0},
]


df = pd.DataFrame(data)