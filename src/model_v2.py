"""v2 LP: client guarantees under volume uncertainty + pharmacy network adequacy.

Two additions to the v1 model (src/model.py, unchanged):

1. Volume scenarios. Instead of hitting each guarantee on one planned volume,
   the guarantee is written once per volume scenario, each with its own
   shortfall / surplus. The objective averages the cost over scenarios.
   `risk_weight` > 1 makes a dollar of true-up cost more than a dollar of
   margin given away, which buys a cushion above the guarantee.

2. Network adequacy. In each retail channel, total reimbursement to each
   pharmacy type must cover its total acquisition cost plus a margin. One
   MAC list pays every pharmacy, so this keeps independents (higher cost)
   whole without separate prices.

Prices stay one MAC list per (drug, channel), and the client is billed on
the same prices (pass-through), as in v1.
"""
import gurobipy as gp
import numpy as np
from gurobipy import GRB

from . import config
from .model import _price_bounds
from .network import RETAIL

# Required margin over acquisition cost, per pharmacy type, on each retail channel total.
NETWORK_MARGIN = {"chain": 0.05, "indep": 0.08}


def optimize_v2(claims, scenarios=None, risk_weight=1.0, network=True,
                network_margin=None, max_change=None, guarantees=None, verbose=False):
    guarantees = guarantees or config.GUARANTEES
    max_change = config.MAX_PRICE_CHANGE if max_change is None else max_change
    network_margin = network_margin or NETWORK_MARGIN

    df = claims.copy().reset_index(drop=True)
    if scenarios is None:
        scenarios = np.ones((len(df), 1))  # plan volumes only = v1 behaviour
    n_scen = scenarios.shape[1]
    lo, hi, floor_override = _price_bounds(df, max_change, config.PHARMACY_MARGIN_FLOOR)

    m = gp.Model("mac_pricing_v2")
    m.Params.OutputFlag = 1 if verbose else 0
    idx = list(df.index)
    price = m.addVars(idx, lb=lo.values, ub=hi.values, name="price")

    up = m.addVars(idx, lb=0, name="move_up")
    down = m.addVars(idx, lb=0, name="move_down")
    m.addConstrs((price[i] - df.at[i, "current_price"] == up[i] - down[i] for i in idx), name="move")

    key = {(r.drug_id, r.channel): i for i, r in df.iterrows()}
    for d in df["drug_id"].unique():
        if (d, "retail_90") in key and (d, "retail_30") in key:
            m.addConstr(price[key[d, "retail_90"]] <= price[key[d, "retail_30"]], name=f"order90_{d}")
        if (d, "mail") in key and (d, "retail_90") in key:
            m.addConstr(price[key[d, "mail"]] <= price[key[d, "retail_90"]], name=f"ordermail_{d}")

    # 1. Client guarantee, once per volume scenario.
    sc = [(ch, s) for ch in config.CHANNELS for s in range(n_scen)]
    shortfall = m.addVars(sc, lb=0, name="shortfall")
    surplus = m.addVars(sc, lb=0, name="surplus")
    for ch in config.CHANNELS:
        rows = df.index[df["channel"] == ch]
        units = df.loc[rows, "units"].values
        awp = df.loc[rows, "awp"].values
        for s in range(n_scen):
            q = units * scenarios[rows, s]
            target = (1 - guarantees[ch]) * float((awp * q).sum())
            spend = gp.quicksum(price[i] * float(qi) for i, qi in zip(rows, q))
            m.addConstr(spend - target == shortfall[ch, s] - surplus[ch, s], name=f"guar_{ch}_{s}")

    # 2. Network adequacy per pharmacy type on each retail channel (plan volumes).
    if network:
        for ch in RETAIL:
            rows = df.index[df["channel"] == ch]
            for ptype in ["chain", "indep"]:
                q = df.loc[rows, f"units_{ptype}"].values
                need = (1 + network_margin[ptype]) * float((df.loc[rows, f"cost_{ptype}"].values * q).sum())
                paid = gp.quicksum(price[i] * float(qi) for i, qi in zip(rows, q))
                m.addConstr(paid >= need, name=f"network_{ptype}_{ch}")

    disruption = gp.quicksum(df.at[i, "units"] * (up[i] + down[i]) for i in idx)
    m.setObjective(
        (risk_weight * shortfall.sum() + surplus.sum()) / n_scen
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
    return {"prices": df, "objective": m.ObjVal,
            "n_variables": m.NumVars, "n_constraints": m.NumConstrs}


def evaluate(prices, multipliers, guarantees=None):
    """Score a price list on volume scenarios it was NOT optimized on.

    Returns, per channel and overall: how often a true-up is owed, and the
    average true-up and margin given away per scenario.
    """
    guarantees = guarantees or config.GUARANTEES
    df = prices.reset_index(drop=True)
    gaps = {}
    for ch in config.CHANNELS:
        rows = df.index[df["channel"] == ch]
        line_gap = ((df.loc[rows, "new_price"] - (1 - guarantees[ch]) * df.loc[rows, "awp"])
                    * df.loc[rows, "units"]).values
        gaps[ch] = line_gap @ multipliers[rows, :]  # one number per scenario
    out = {}
    any_short = np.zeros(multipliers.shape[1], dtype=bool)
    for ch, g in gaps.items():
        out[ch] = {"p_true_up": float((g > 1e-6).mean()),
                   "avg_true_up_$": float(np.clip(g, 0, None).mean()),
                   "avg_given_away_$": float(np.clip(-g, 0, None).mean())}
        any_short |= g > 1e-6
    out["all"] = {"p_true_up": float(any_short.mean()),
                  "avg_true_up_$": sum(v["avg_true_up_$"] for v in out.values()),
                  "avg_given_away_$": sum(v["avg_given_away_$"] for k, v in out.items() if k != "all")}
    return out


def network_health(prices):
    """Share of independent / chain retail units paid below that pharmacy's cost,
    and each type's margin over cost on retail claims."""
    df = prices[prices["channel"].isin(RETAIL)]
    out = {}
    for ptype in ["chain", "indep"]:
        q = df[f"units_{ptype}"]
        below = (df["new_price"] < df[f"cost_{ptype}"]) * q
        paid = (df["new_price"] * q).sum()
        cost = (df[f"cost_{ptype}"] * q).sum()
        out[ptype] = {"units_paid_below_cost_%": float(100 * below.sum() / q.sum()),
                      "margin_over_cost_%": float(100 * (paid / cost - 1))}
    return out
