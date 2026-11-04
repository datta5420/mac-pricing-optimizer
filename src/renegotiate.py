"""SEPARATE EXERCISE: what would a contract have to say to be deliverable?

This is NOT part of the pricing optimization. During a contract term the
guarantee and the billing terms are fixed, and `model_v3.optimize_mac` treats
them that way. This module answers a different, renewal-time question:

    given the MAC prices we can actually afford to pay pharmacies, what
    guarantee could we have promised this client?

and the related one, for arrangements that do use a client billing multiplier:

    what multiplier would this client's contract need for the guarantee to
    hold at these prices?

Keeping it in its own file is deliberate. V2 mixed this up with pricing, let
the optimizer move billing multipliers freely, and so "met" every guarantee for
a reason no pricing analyst could reproduce.
"""
import pandas as pd

from .model_v3 import pool_index, price_bounds


def deliverable_guarantee(mac, claims, guarantees, max_change=None, margin_floor=None):
    """The deepest discount each pool could honestly be promised, two ways.

    `deepest_this_cycle` respects the price-change limit as well as the pharmacy
    floor: what could be delivered at the next repricing.
    `deepest_ever` respects only the pharmacy floor: what the drug economics
    allow once prices have had time to get there.

    The two differ whenever stability policy, not cost, is the binding
    constraint, and the remedy differs with them. A pool short only on
    `deepest_this_cycle` needs more cycles or a wider limit. A pool short on
    `deepest_ever` needs a different contract.
    """
    import numpy as np

    from . import config

    lo, _, _ = price_bounds(mac, max_change, margin_floor)
    mf = config.PHARMACY_MARGIN_FLOOR if margin_floor is None else margin_floor
    floor_only = mac["nadac"].values * (1 + mf)
    clients, units, _, chan_rows = pool_index(claims, mac)
    awp = mac["awp"].values
    rows = []
    for i, c in enumerate(clients):
        for ch, idx in chan_rows.items():
            q = units[i, idx]
            total_awp = float((awp[idx] * q).sum())
            this_cycle = 1 - float((lo[idx] * q).sum()) / total_awp
            ever = 1 - float((floor_only[idx] * q).sum()) / total_awp
            promised = guarantees[(c, ch)]
            rows.append({"client": int(c), "channel": ch,
                         "promised_guarantee": round(promised, 4),
                         "deepest_this_cycle": round(this_cycle, 4),
                         "deepest_ever": round(ever, 4),
                         "short_this_cycle_pts": round(100 * max(promised - this_cycle, 0), 3),
                         "short_ever_pts": round(100 * max(promised - ever, 0), 3),
                         "needs_more_cycles": (promised > this_cycle) and (promised <= ever),
                         "renegotiate": promised > ever})
    return pd.DataFrame(rows)


def required_multiplier(result, guarantees, settled):
    """For a spread-style contract: the billing multiplier each pool would need.

    Reported, never optimized. A multiplier far from 1.0 says the contract and
    the price list disagree, which is a commercial conversation, not a solver
    setting.
    """
    rows = []
    for r in settled.itertuples():
        need = r.allowed_ingredient / r.ingredient if r.ingredient else float("nan")
        rows.append({"client": int(r.client), "channel": r.channel,
                     "required_multiplier": round(need, 4),
                     "within_10pct": abs(need - 1) <= 0.10})
    return pd.DataFrame(rows)
