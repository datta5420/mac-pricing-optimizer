"""v2 data: split each claim line across pharmacy types, and simulate volume scenarios.

Pharmacy types (all simulated, illustrative assumptions):
- chain:       large chains buy generics below the NADAC survey average.
- independent: independents buy at or above NADAC, so a MAC that works for a
               chain can leave an independent under water.
- mail:        the PBM's own mail-order pharmacy (mail channel only).

NADAC itself is a survey average of retail pharmacy acquisition cost, which is
why chain cost sits below it and independent cost above it here.
"""
import numpy as np
import pandas as pd

from . import config

RETAIL = ["retail_30", "retail_90"]


def add_pharmacy_types(claims, seed=config.SEED):
    """Add per-row units and acquisition cost for chain and independent pharmacies."""
    rng = np.random.default_rng(seed + 1)
    df = claims.copy()
    drug_ids = df["drug_id"].unique()

    # Drug-level cost multipliers (same drug costs a chain the same in every channel).
    chain_mult = pd.Series(rng.uniform(0.88, 0.97, len(drug_ids)), index=drug_ids)
    indep_mult = pd.Series(rng.uniform(1.03, 1.18, len(drug_ids)), index=drug_ids)
    df["cost_chain"] = df["nadac"] * df["drug_id"].map(chain_mult)
    df["cost_indep"] = df["nadac"] * df["drug_id"].map(indep_mult)

    # Share of each retail line filled at independents; mail is all PBM mail.
    indep_share = rng.uniform(0.15, 0.35, len(df))
    is_retail = df["channel"].isin(RETAIL)
    df["units_indep"] = np.where(is_retail, np.round(df["units"] * indep_share), 0.0)
    df["units_chain"] = np.where(is_retail, df["units"] - df["units_indep"], 0.0)
    return df


def volume_scenarios(n_rows, n_scenarios, spread=0.15, seed=0):
    """Volume multipliers per claim line and scenario, uniform in [1-spread, 1+spread].

    Same noise model as the v1 README test: each line's volume moves by up to
    +/-15% after repricing. Returns an array of shape (n_rows, n_scenarios).
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(1 - spread, 1 + spread, size=(n_rows, n_scenarios))
