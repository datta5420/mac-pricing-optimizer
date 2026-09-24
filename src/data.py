"""Build the input dataset: drugs (cost, AWP) and claims (units, current price).

Acquisition cost can come from real public CMS NADAC data. AWP, claim volumes
and current prices are SIMULATED because they are proprietary.
"""
import numpy as np
import pandas as pd

from . import config


def load_nadac(path, n_drugs, seed=config.SEED):
    """Load a NADAC CSV downloaded from data.medicaid.gov and keep generics.

    Column names differ slightly between NADAC years, so we normalise them.
    """
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    df = df[df["classification_for_rate_setting"].str.upper() == "G"]
    df["nadac_per_unit"] = pd.to_numeric(df["nadac_per_unit"], errors="coerce")
    df["effective_date"] = pd.to_datetime(df["effective_date"], errors="coerce")
    df = df.dropna(subset=["nadac_per_unit", "effective_date"])
    df = df[df["nadac_per_unit"] > 0]

    # Latest price per NDC, then one row per drug description.
    df = df.sort_values("effective_date").groupby("ndc").tail(1)
    drugs = (df.groupby("ndc_description", as_index=False)["nadac_per_unit"]
               .median()
               .rename(columns={"ndc_description": "name", "nadac_per_unit": "nadac"}))

    drugs = drugs.sample(n=min(n_drugs, len(drugs)), random_state=seed)
    return drugs.reset_index(drop=True)


def synthetic_drugs(n_drugs, seed=config.SEED):
    """Fallback when no NADAC file is available: realistic-looking unit costs."""
    rng = np.random.default_rng(seed)
    nadac = np.round(rng.lognormal(mean=-1.5, sigma=1.2, size=n_drugs), 4)
    names = [f"GENERIC_{i:03d}" for i in range(n_drugs)]
    return pd.DataFrame({"name": names, "nadac": np.maximum(nadac, 0.01)})


def simulate_claims(drugs, seed=config.SEED):
    """Add simulated AWP, then one row per (drug, channel) with units and price.

    Current prices are deliberately set so some channels start above their
    guarantee and some below, giving the optimizer something to fix.
    """
    rng = np.random.default_rng(seed)
    drugs = drugs.copy()
    drugs["drug_id"] = range(len(drugs))
    # Generic AWP is usually far above acquisition cost.
    drugs["awp"] = drugs["nadac"] * rng.uniform(8, 16, len(drugs))

    # Retail 30-day price: AWP minus a drug-level discount.
    disc_30 = np.clip(rng.normal(0.78, 0.05, len(drugs)), 0.60, 0.92)
    price_30 = drugs["awp"] * (1 - disc_30)
    # 90-day and mail are cheaper per unit than retail 30 (keeps channel order).
    price_90 = price_30 * rng.uniform(0.62, 0.82, len(drugs))
    price_mail = price_90 * rng.uniform(0.88, 0.99, len(drugs))

    base_units = rng.lognormal(mean=8, sigma=1, size=len(drugs))
    share = {"retail_30": 1.0, "retail_90": 0.6, "mail": 0.4}
    prices = {"retail_30": price_30, "retail_90": price_90, "mail": price_mail}

    rows = []
    for ch in config.CHANNELS:
        part = drugs[["drug_id", "name", "nadac", "awp"]].copy()
        part["channel"] = ch
        part["units"] = np.round(base_units * share[ch] * rng.uniform(0.7, 1.3, len(drugs)))
        part["current_price"] = np.round(prices[ch].values, 4)
        rows.append(part)

    return pd.concat(rows, ignore_index=True)


def build_dataset(nadac_path=None, n_drugs=config.N_DRUGS, seed=config.SEED):
    drugs = load_nadac(nadac_path, n_drugs, seed) if nadac_path else synthetic_drugs(n_drugs, seed)
    return simulate_claims(drugs, seed)
