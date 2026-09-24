"""Linear program that sets generic drug prices to meet client guarantees.

For each (drug, channel) the model chooses a new unit price. It tries to hit
each channel's AWP discount guarantee exactly: missing it costs a true-up,
beating it gives away margin. See README for the full formulation.
"""
import gurobipy as gp
import numpy as np
from gurobipy import GRB

from . import config


def _price_bounds(df, max_change, margin_floor):
    """Lower/upper price bounds per row.

    If the pharmacy floor is above the allowed range, the floor wins
    (we never pay a pharmacy below cost) and the row is flagged.
    """
    floor = df["nadac"] * (1 + margin_floor)
    lo = np.maximum(floor, df["current_price"] * (1 - max_change))
    hi = df["current_price"] * (1 + max_change)
    floor_override = lo > hi
    hi = np.maximum(hi, lo)
    return lo, hi, floor_override


def optimize_prices(claims, guarantees=None, max_change=None, margin_floor=None, verbose=False):
    guarantees = guarantees or config.GUARANTEES
    max_change = config.MAX_PRICE_CHANGE if max_change is None else max_change
    margin_floor = config.PHARMACY_MARGIN_FLOOR if margin_floor is None else margin_floor

    df = claims.copy().reset_index(drop=True)
    lo, hi, floor_override = _price_bounds(df, max_change, margin_floor)

    m = gp.Model("mac_pricing")
    m.Params.OutputFlag = 1 if verbose else 0
    idx = list(df.index)

    # Decision: new price per (drug, channel), within its bounds.
    price = m.addVars(idx, lb=lo.values, ub=hi.values, name="price")

    # |new - current| split into up/down parts so the objective stays linear.
    up = m.addVars(idx, lb=0, name="move_up")
    down = m.addVars(idx, lb=0, name="move_down")
    m.addConstrs((price[i] - df.at[i, "current_price"] == up[i] - down[i] for i in idx), name="move")

    # Guarantee per channel: spend - target spend = shortfall - surplus.
    shortfall = m.addVars(config.CHANNELS, lb=0, name="shortfall")
    surplus = m.addVars(config.CHANNELS, lb=0, name="surplus")
    for ch in config.CHANNELS:
        rows = df.index[df["channel"] == ch]
        target_spend = (1 - guarantees[ch]) * (df.loc[rows, "awp"] * df.loc[rows, "units"]).sum()
        spend = gp.quicksum(price[i] * df.at[i, "units"] for i in rows)
        m.addConstr(spend - target_spend == shortfall[ch] - surplus[ch], name=f"guarantee_{ch}")

    # Channel consistency: per unit, mail <= retail 90 <= retail 30.
    key = {(r.drug_id, r.channel): i for i, r in df.iterrows()}
    for d in df["drug_id"].unique():
        if (d, "retail_90") in key and (d, "retail_30") in key:
            m.addConstr(price[key[d, "retail_90"]] <= price[key[d, "retail_30"]], name=f"order90_{d}")
        if (d, "mail") in key and (d, "retail_90") in key:
            m.addConstr(price[key[d, "mail"]] <= price[key[d, "retail_90"]], name=f"ordermail_{d}")

    disruption = gp.quicksum(df.at[i, "units"] * (up[i] + down[i]) for i in idx)
    m.setObjective(
        config.SHORTFALL_WEIGHT * shortfall.sum()
        + config.SURPLUS_WEIGHT * surplus.sum()
        + config.DISRUPTION_WEIGHT * disruption,
        GRB.MINIMIZE,
    )

    m.optimize()

    if m.Status == GRB.INFEASIBLE:
        m.computeIIS()
        bad = [c.ConstrName for c in m.getConstrs() if c.IISConstr]
        raise RuntimeError(f"Model infeasible. Conflicting constraints include: {bad[:10]}")
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Solver stopped with status {m.Status}")

    df["new_price"] = [price[i].X for i in idx]
    df["floor_override"] = floor_override.values
    return {
        "prices": df,
        "shortfall": {ch: shortfall[ch].X for ch in config.CHANNELS},
        "surplus": {ch: surplus[ch].X for ch in config.CHANNELS},
        "objective": m.ObjVal,
        "n_variables": m.NumVars,
        "n_constraints": m.NumConstrs,
    }
