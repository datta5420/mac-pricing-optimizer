"""Build a book of business: many clients sharing one MAC list.

v1 priced ONE client. In a real PBM, a single MAC list pays pharmacies for
hundreds of clients at once, and every client has its own drug mix and its own
guarantees. That is what makes the problem hard: one set of prices has to work
for everyone.

Drug acquisition costs are REAL (CMS NADAC, data/raw/nadac_sample.csv).
Claim volumes, AWP, current prices and contract terms are simulated, because
they are proprietary.
"""
import numpy as np
import pandas as pd

from . import config

CHANNELS = config.CHANNELS

# Guarantee tiers a PBM might offer. Bigger books negotiate deeper discounts.
TIERS = {
    "standard": {"retail_30": 0.79, "retail_90": 0.83, "mail": 0.85},
    "mid":      {"retail_30": 0.80, "retail_90": 0.84, "mail": 0.86},
    "deep":     {"retail_30": 0.82, "retail_90": 0.855, "mail": 0.875},
}


def load_nadac_sample(path="data/raw/nadac_sample.csv", n_drugs=None, seed=config.SEED):
    """Real generic acquisition costs from the CMS NADAC weekly file."""
    df = pd.read_csv(path)
    df = df.rename(columns={"nadac_per_unit": "nadac"})
    df = df[df["nadac"] > 0].reset_index(drop=True)
    if n_drugs and n_drugs < len(df):
        # Even spread across the cost range, so cheap and costly drugs both appear.
        idx = np.linspace(0, len(df) - 1, n_drugs).astype(int)
        df = df.iloc[idx].reset_index(drop=True)
    df["drug_id"] = range(len(df))
    return df[["drug_id", "name", "unit", "nadac"]]


def build_book(drugs, n_clients=12, seed=config.SEED):
    """Simulate claim volumes per (client, drug, channel) plus AWP and current prices.

    Returns (claims, clients). Volumes are heavily skewed: a few drugs carry most
    of the spend, as in real claims data.
    """
    rng = np.random.default_rng(seed)
    d = drugs.copy()

    # Generic AWP sits far above acquisition cost; the multiple varies by drug.
    d["awp"] = np.round(d["nadac"] * rng.uniform(6, 18, len(d)), 4)

    # Drug-level popularity (skewed) and a therapeutic class used later for
    # correlated cost/volume shocks.
    d["popularity"] = rng.lognormal(mean=0, sigma=1.4, size=len(d))
    d["klass"] = rng.integers(0, 12, len(d))

    clients = pd.DataFrame({
        "client_id": range(n_clients),
        "name": [f"CLIENT_{i:02d}" for i in range(n_clients)],
        "tier": rng.choice(list(TIERS), size=n_clients, p=[0.4, 0.4, 0.2]),
        "size": rng.lognormal(mean=0, sigma=0.9, size=n_clients),
        "mail_pref": rng.uniform(0.15, 0.60, n_clients),  # how much runs through mail
    })

    share = {"retail_30": 1.0, "retail_90": 0.45}
    rows = []
    for c in clients.itertuples():
        # Each client uses most, not all, of the formulary.
        used = rng.random(len(d)) < 0.85
        part0 = d[used]
        for ch in CHANNELS:
            part = part0[["drug_id", "name", "unit", "nadac", "awp", "klass"]].copy()
            part["client_id"] = c.client_id
            part["channel"] = ch
            mix = c.mail_pref if ch == "mail" else share[ch]
            units = part0["popularity"].values * c.size * mix * 1200 * rng.uniform(0.6, 1.4, len(part0))
            part["units"] = np.maximum(np.round(units), 1.0)
            rows.append(part)
    claims = pd.concat(rows, ignore_index=True)

    # Current prices: a drug-level MAC that is deliberately a bit off target, so
    # the optimizer has something to fix. One price per (drug, channel), shared
    # by every client, which is the point of a MAC list.
    base = d["awp"] * (1 - np.clip(rng.normal(0.78, 0.06, len(d)), 0.55, 0.93))
    mult = {"retail_30": 1.0, "retail_90": 0.80, "mail": 0.72}
    price = {}
    for ch in CHANNELS:
        for i, p in zip(d["drug_id"], base * mult[ch]):
            price[(i, ch)] = max(round(p, 4), 0.0001)
    claims["current_price"] = [price[(r.drug_id, r.channel)] for r in claims.itertuples()]

    # Guarantees per client and channel.
    guarantees = {
        (c.client_id, ch): TIERS[c.tier][ch] for c in clients.itertuples() for ch in CHANNELS
    }
    return claims, clients, guarantees


def mac_table(claims):
    """The distinct (drug, channel) rows the MAC list prices, with total units."""
    t = (claims.groupby(["drug_id", "channel"], as_index=False)
                .agg(name=("name", "first"), nadac=("nadac", "first"),
                     awp=("awp", "first"), klass=("klass", "first"),
                     units=("units", "sum"), current_price=("current_price", "first")))
    return t.reset_index(drop=True)
