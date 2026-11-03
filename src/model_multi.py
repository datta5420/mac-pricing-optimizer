"""Price one MAC list for a whole book of clients.

THE PROBLEM
Pharmacies are paid from ONE MAC list. Each client has its own guarantees and
its own drug mix, and what a client is billed is that MAC list adjusted by a
per-client scale factor (a single multiplier applied across the list). So the
decisions are:

    p[j]  the MAC price for row j = (drug, channel)   -> what pharmacies are paid
    s[c]  the scale factor for client c               -> what client c is billed

Client c's ingredient spend is  sum_j s[c] * p[j] * q[c,j].

WHY IT IS NOT ONE LP
That spend multiplies two decision variables, so the joint problem is bilinear,
not linear. Rather than hand it to a nonlinear solver, this module alternates
two LPs, each of which Gurobi solves exactly:

    p-step: hold every s[c] fixed -> linear in p
    s-step: hold p fixed          -> linear in s

Each step can only lower the shared objective, which is bounded below by zero,
so the sequence converges. It converges to a local optimum, not a proven global
one; that trade is deliberate and is stated in the README.
"""
import gurobipy as gp
import numpy as np
from gurobipy import GRB

from . import config

PENALTY_SHORTFALL = 1.0   # $1 of missed guarantee = $1 true-up owed
PENALTY_SURPLUS = 1.0     # $1 of over-delivery = $1 of margin given away
DISRUPTION_WEIGHT = 0.01  # tie-breaker: prefer fewer, smaller price moves
SCALE_BOUNDS = (0.85, 1.15)


def _client_matrix(claims, mac):
    """units[c, j] = units client c runs through MAC row j, plus the AWP target."""
    key = {(r.drug_id, r.channel): j for j, r in enumerate(mac.itertuples())}
    clients = np.sort(claims["client_id"].unique())
    cpos = {c: i for i, c in enumerate(clients)}
    units = np.zeros((len(clients), len(mac)))
    for r in claims.itertuples():
        units[cpos[r.client_id], key[(r.drug_id, r.channel)]] += r.units
    chan_rows = {ch: np.array([j for j, r in enumerate(mac.itertuples()) if r.channel == ch])
                 for ch in config.CHANNELS}
    return clients, units, chan_rows


def _targets(mac, units, clients, guarantees, cushion=0.0):
    """Most each client may spend per channel: (1 - guarantee - cushion) * total AWP.

    cushion is either one number for the whole book or a dict keyed
    (client_id, channel), which is how a cushion is aimed only where the risk is.
    """
    awp = mac["awp"].values
    out = {}
    for i, c in enumerate(clients):
        for ch in config.CHANNELS:
            rows = np.array([j for j, r in enumerate(mac.itertuples()) if r.channel == ch])
            cu = cushion.get((int(c), ch), 0.0) if isinstance(cushion, dict) else cushion
            g = min(guarantees[(c, ch)] + cu, 0.999)
            out[(c, ch)] = (1 - g) * float((awp[rows] * units[i, rows]).sum())
    return out


def optimize_book(claims, mac, guarantees, iters=6, max_change=None, margin_floor=None,
                  cushion=0.0, scale_bounds=SCALE_BOUNDS, scale_by="client", verbose=False):
    """Alternate the p-step and the s-step until the objective stops improving.

    scale_by: "client" gives each client ONE factor across every channel (the
    simplest thing to administer); "client_channel" gives each client one factor
    per channel. A client has three channel guarantees, so one factor has a
    single degree of freedom for three targets and cannot hit them all.
    """
    max_change = config.MAX_PRICE_CHANGE if max_change is None else max_change
    margin_floor = config.PHARMACY_MARGIN_FLOOR if margin_floor is None else margin_floor

    clients, units, chan_rows = _client_matrix(claims, mac)
    target = _targets(mac, units, clients, guarantees, cushion)
    p0 = mac["current_price"].values
    awp = mac["awp"].values
    floor = mac["nadac"].values * (1 + margin_floor)
    lo = np.maximum(floor, p0 * (1 - max_change))
    hi = np.maximum(p0 * (1 + max_change), lo)
    floor_override = (floor > p0 * (1 + max_change))

    # Rows of the same drug, for the mail <= retail_90 <= retail_30 rule.
    pos = {(r.drug_id, r.channel): j for j, r in enumerate(mac.itertuples())}
    order_pairs = []
    for d in mac["drug_id"].unique():
        if (d, "mail") in pos and (d, "retail_90") in pos:
            order_pairs.append((pos[(d, "mail")], pos[(d, "retail_90")]))
        if (d, "retail_90") in pos and (d, "retail_30") in pos:
            order_pairs.append((pos[(d, "retail_90")], pos[(d, "retail_30")]))

    # s[(i, ch)] for every client position i and channel ch.
    s = {(i, ch): 1.0 for i in range(len(clients)) for ch in config.CHANNELS}
    p = p0.copy()
    history = []

    for it in range(iters):
        p = _p_step(p0, lo, hi, units, clients, chan_rows, target, s, order_pairs, verbose)
        s = _s_step(p, units, clients, chan_rows, target, scale_bounds, scale_by, verbose)
        history.append(_objective(p, p0, s, units, clients, chan_rows, target))
        if len(history) > 1 and abs(history[-2] - history[-1]) < 1e-6 * max(1.0, history[-1]):
            break

    res = mac.copy()
    res["new_price"] = p
    res["floor_override"] = floor_override
    return {
        "prices": res,
        "scale": {(int(clients[i]), ch): v for (i, ch), v in s.items()},
        "scale_by": scale_by,
        "history": history,
        "clients": clients,
        "units": units,
        "chan_rows": chan_rows,
        "target": target,
        "awp": awp,
    }


def _p_step(p0, lo, hi, units, clients, chan_rows, target, s, order_pairs, verbose):
    m = gp.Model("p_step")
    m.Params.OutputFlag = 1 if verbose else 0
    n = len(p0)
    p = m.addVars(n, lb=lo, ub=hi, name="price")
    up = m.addVars(n, lb=0, name="up")
    dn = m.addVars(n, lb=0, name="dn")
    m.addConstrs((p[j] - p0[j] == up[j] - dn[j] for j in range(n)), name="move")
    for a, b in order_pairs:
        m.addConstr(p[a] <= p[b], name=f"order_{a}_{b}")

    short, surp = {}, {}
    for i, c in enumerate(clients):
        for ch, rows in chan_rows.items():
            short[c, ch] = m.addVar(lb=0, name=f"short_{c}_{ch}")
            surp[c, ch] = m.addVar(lb=0, name=f"surp_{c}_{ch}")
            spend = gp.quicksum(float(s[(i, ch)] * units[i, j]) * p[j] for j in rows)
            m.addConstr(spend - target[(c, ch)] == short[c, ch] - surp[c, ch],
                        name=f"guar_{c}_{ch}")

    disruption = gp.quicksum(float(units[:, j].sum()) * (up[j] + dn[j]) for j in range(n))
    m.setObjective(
        PENALTY_SHORTFALL * gp.quicksum(short.values())
        + PENALTY_SURPLUS * gp.quicksum(surp.values())
        + DISRUPTION_WEIGHT * disruption,
        GRB.MINIMIZE)
    _solve(m)
    return np.array([p[j].X for j in range(n)])


def _s_step(p, units, clients, chan_rows, target, bounds, scale_by, verbose):
    m = gp.Model("s_step")
    m.Params.OutputFlag = 1 if verbose else 0
    if scale_by == "client":
        var = {i: m.addVar(lb=bounds[0], ub=bounds[1], name=f"scale_{i}")
               for i in range(len(clients))}
        pick = lambda i, ch: var[i]
    elif scale_by == "client_channel":
        var = {(i, ch): m.addVar(lb=bounds[0], ub=bounds[1], name=f"scale_{i}_{ch}")
               for i in range(len(clients)) for ch in chan_rows}
        pick = lambda i, ch: var[(i, ch)]
    else:
        raise ValueError("scale_by must be 'client' or 'client_channel'")

    short, surp = {}, {}
    for i, c in enumerate(clients):
        for ch, rows in chan_rows.items():
            short[c, ch] = m.addVar(lb=0, name=f"short_{c}_{ch}")
            surp[c, ch] = m.addVar(lb=0, name=f"surp_{c}_{ch}")
            base = float((p[rows] * units[i, rows]).sum())
            m.addConstr(base * pick(i, ch) - target[(c, ch)] == short[c, ch] - surp[c, ch],
                        name=f"guar_{c}_{ch}")
    m.setObjective(PENALTY_SHORTFALL * gp.quicksum(short.values())
                   + PENALTY_SURPLUS * gp.quicksum(surp.values()), GRB.MINIMIZE)
    _solve(m)
    return {(i, ch): pick(i, ch).X for i in range(len(clients)) for ch in chan_rows}


def _solve(m):
    m.optimize()
    if m.Status == GRB.INFEASIBLE:
        m.computeIIS()
        bad = [c.ConstrName for c in m.getConstrs() if c.IISConstr]
        raise RuntimeError(f"Infeasible. Conflicting constraints include: {bad[:10]}")
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Solver stopped with status {m.Status}")


def _objective(p, p0, s, units, clients, chan_rows, target):
    total = 0.0
    for i, c in enumerate(clients):
        for ch, rows in chan_rows.items():
            gap = float((s[(i, ch)] * p[rows] * units[i, rows]).sum()) - target[(c, ch)]
            total += abs(gap)
    total += DISRUPTION_WEIGHT * float((units.sum(axis=0) * np.abs(p - p0)).sum())
    return total


# ---------------------------------------------------------------- reporting

def client_summary(result, guarantees):
    """Per client and channel: achieved discount, true-up owed, margin given away."""
    import pandas as pd
    p = result["prices"]["new_price"].values
    awp, units = result["awp"], result["units"]
    rows = []
    for i, c in enumerate(result["clients"]):
        for ch, idx in result["chan_rows"].items():
            sc = result["scale"][(int(c), ch)]
            spend = float((sc * p[idx] * units[i, idx]).sum())
            total_awp = float((awp[idx] * units[i, idx]).sum())
            gap = spend - result["target"][(c, ch)]
            rows.append({"client": int(c), "channel": ch,
                         "guarantee": guarantees[(c, ch)],
                         "achieved": round(1 - spend / total_awp, 4),
                         "scale": round(sc, 4),
                         "true_up_$": round(max(gap, 0), 0),
                         "given_away_$": round(max(-gap, 0), 0)})
    return pd.DataFrame(rows)


def spread_dollars(result):
    """What the book bills clients minus what pharmacies are paid, per client.

    A client whose spread is negative is billed less than the pharmacies serving
    it are paid. That is a contract problem, not a pricing one: no MAC list can
    fix a guarantee priced below the cost of filling it.
    """
    import pandas as pd
    p = result["prices"]["new_price"].values
    units, chan_rows = result["units"], result["chan_rows"]
    rows = []
    for i, c in enumerate(result["clients"]):
        billed = paid = 0.0
        for ch, idx in chan_rows.items():
            sc = result["scale"][(int(c), ch)]
            billed += float((sc * p[idx] * units[i, idx]).sum())
            paid += float((p[idx] * units[i, idx]).sum())
        rows.append({"client": int(c), "billed_$": round(billed, 0),
                     "paid_to_pharmacies_$": round(paid, 0),
                     "spread_$": round(billed - paid, 0)})
    return pd.DataFrame(rows)
