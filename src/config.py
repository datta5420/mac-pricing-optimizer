"""Business assumptions for the pricing model.

Every number here is an assumption for a mock client, not a real PBM contract.
Change them and re-run to see how the answer moves.
"""

# Pharmacy channels. Each has its own guarantee in a typical PBM contract.
CHANNELS = ["retail_30", "retail_90", "mail"]

# AWP discount guarantees promised to the (mock) reseller client.
# 0.80 means "on average, generics will cost AWP minus 80%".
GUARANTEES = {"retail_30": 0.80, "retail_90": 0.84, "mail": 0.86}

# Pharmacies must be paid at least acquisition cost (NADAC) plus this margin.
PHARMACY_MARGIN_FLOOR = 0.05

# No drug's price may move more than this share in one pricing cycle.
MAX_PRICE_CHANGE = 0.15

# Objective weights (all in dollars).
SHORTFALL_WEIGHT = 1.0    # $1 of missed guarantee = $1 true-up owed to client
SURPLUS_WEIGHT = 1.0      # $1 of over-delivery = $1 of margin given away
DISRUPTION_WEIGHT = 0.01  # small tie-breaker: prefer fewer / smaller price moves

N_DRUGS = 200
SEED = 42
