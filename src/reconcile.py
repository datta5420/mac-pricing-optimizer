"""Recompute every number from the claims, independently of the optimizer.

`settle()` and `pnl()` read no solver output at all. They take the final MAC
prices and the claim lines and rebuild the contract settlement from scratch,
the way a finance team would, so if they disagree with the solver, the solver
is wrong. That is the point of keeping them separate.

`check_true_up_against_solver()` is the one function here that deliberately
DOES read solver output -- the `short` decision variables -- because its whole
job is to hold the solver's own answer against the independent one. It is kept
in this module so the comparison lives next to the calculation it checks.

Three figures are kept apart on purpose:

  contribution margin   admin fee revenue - true-up payments - operating cost.
                        Ingredient and dispensing dollars pass through, so they
                        net to zero and do not appear.

  true-up payments      what the PBM owes clients whose billed ingredient cost
                        exceeded their guarantee. A real cash expense.

  guarantee             how far a client's discount BEAT its contractual
  overperformance       minimum. A benefit to the client, NOT a PBM expense:
                        under pass-through those dollars went to the pharmacy,
                        and were never the PBM's to keep.
"""
import numpy as np
import pandas as pd

from .contract import DISPENSING_FEE


def settle(claims, prices, guarantees, terms, volume_multiplier=None):
    """Settle the contract year for every client and channel, from the claims.

    `prices` is the priced MAC table with a `new_price` column.
    `volume_multiplier` optionally scales each claim line's units and scripts,
    to settle a what-if year rather than the planned one.
    """
    price_of = {(r.drug_id, r.channel): r.new_price for r in prices.itertuples()}
    admin_of = dict(zip(terms["client_id"], terms["admin_fee_per_script"]))
    opcost_of = dict(zip(terms["client_id"], terms["operating_cost_per_script"]))

    df = claims.copy()
    if volume_multiplier is not None:
        df["units"] = df["units"] * volume_multiplier
        df["scripts"] = df["scripts"] * volume_multiplier
    df["mac"] = [price_of[(r.drug_id, r.channel)] for r in df.itertuples()]
    df["ingredient_$"] = df["mac"] * df["units"]
    df["awp_$"] = df["awp"] * df["units"]
    df["dispensing_$"] = [DISPENSING_FEE[ch] * s for ch, s in zip(df["channel"], df["scripts"])]
    df["admin_$"] = [admin_of[c] * s for c, s in zip(df["client_id"], df["scripts"])]
    df["opcost_$"] = [opcost_of[c] * s for c, s in zip(df["client_id"], df["scripts"])]

    g = (df.groupby(["client_id", "channel"], as_index=False)
           .agg(ingredient=("ingredient_$", "sum"), awp=("awp_$", "sum"),
                dispensing=("dispensing_$", "sum"), admin=("admin_$", "sum"),
                opcost=("opcost_$", "sum"), scripts=("scripts", "sum")))

    g["guarantee"] = [guarantees[(c, ch)] for c, ch in zip(g["client_id"], g["channel"])]
    g["allowed_ingredient"] = (1 - g["guarantee"]) * g["awp"]
    g["achieved_discount"] = 1 - g["ingredient"] / g["awp"]
    gap = g["ingredient"] - g["allowed_ingredient"]
    g["true_up_$"] = gap.clip(lower=0)             # PBM pays the client
    g["overperformance_$"] = (-gap).clip(lower=0)  # client benefit, not a PBM cost
    g["pharmacy_paid_$"] = g["ingredient"] + g["dispensing"]
    g["client_billed_$"] = g["ingredient"] + g["dispensing"] + g["admin"]
    g["contribution_$"] = g["admin"] - g["true_up_$"] - g["opcost"]
    return g.rename(columns={"client_id": "client"})


def pnl(settled):
    """Book-level P&L. Pass-through dollars net out and are shown for context."""
    return {
        "admin_fee_revenue_$": float(settled["admin"].sum()),
        "true_up_payments_$": float(settled["true_up_$"].sum()),
        "operating_cost_$": float(settled["opcost"].sum()),
        "contribution_margin_$": float(settled["contribution_$"].sum()),
        "_passthrough_ingredient_$": float(settled["ingredient"].sum()),
        "_passthrough_dispensing_$": float(settled["dispensing"].sum()),
        "_guarantee_overperformance_$": float(settled["overperformance_$"].sum()),
    }


def check_true_up_against_solver(claims, prices, guarantees, terms, result, scenario=0):
    """Compare the SOLVER'S OWN true-up variables with an independent settlement.

    This is the check that can actually fail. The solver reports what each pool
    owes through its `short` decision variables, computed inside the model from
    its own constraint matrix. Here the same quantity is rebuilt from the claim
    lines: group the claims, apply the guarantee, take the positive part. The
    two calculations share only the price vector and the input data.

    (The earlier version of this function compared price x units aggregated two
    different ways, which was close to tautological. It was replaced.)

    Returns the largest absolute difference in dollars across all pools.
    """
    mult = result.get("scenario_multipliers")
    if mult is None:
        line_mult = None
    else:
        key = {(r.drug_id, r.channel): j for j, r in enumerate(prices.itertuples())}
        cpos = {c: i for i, c in enumerate(result["clients"])}
        rows_i = np.array([cpos[c] for c in claims["client_id"]])
        rows_j = np.array([key[(d, ch)] for d, ch in zip(claims["drug_id"], claims["channel"])])
        line_mult = mult[rows_i, rows_j, scenario]

    settled = settle(claims, prices, guarantees, terms, volume_multiplier=line_mult)
    worst = 0.0
    compared = 0
    for _, row in settled.iterrows():
        from_solver = result["solver_true_up"].get(
            (int(row["client"]), row["channel"], scenario))
        if from_solver is None:
            continue
        worst = max(worst, abs(from_solver - float(row["true_up_$"])))
        compared += 1
    if compared == 0:
        raise ValueError("No solver true-up variables matched the settled pools.")
    return worst


def scenario_pnl(claims, prices, guarantees, terms, result, scenarios):
    """Settle the year under each volume scenario, independently of the solver."""
    key = {(r.drug_id, r.channel): j for j, r in enumerate(prices.itertuples())}
    cpos = {c: i for i, c in enumerate(result["clients"])}
    rows_i = np.array([cpos[c] for c in claims["client_id"]])
    rows_j = np.array([key[(d, ch)] for d, ch in zip(claims["drug_id"], claims["channel"])])

    out = []
    for s in range(scenarios.shape[2]):
        mult = scenarios[rows_i, rows_j, s]
        settled = settle(claims, prices, guarantees, terms, volume_multiplier=mult)
        out.append(pnl(settled))
    df = pd.DataFrame(out)
    return {
        "scenarios": len(df),
        "mean_true_up_$": float(df["true_up_payments_$"].mean()),
        "p95_true_up_$": float(np.percentile(df["true_up_payments_$"], 95)),
        "cvar95_true_up_$": float(df["true_up_payments_$"]
                                  [df["true_up_payments_$"] >=
                                   np.percentile(df["true_up_payments_$"], 95)].mean()),
        "mean_contribution_$": float(df["contribution_margin_$"].mean()),
        "p05_contribution_$": float(np.percentile(df["contribution_margin_$"], 5)),
        "p_negative_contribution": float((df["contribution_margin_$"] < 0).mean()),
        "frame": df,
    }
