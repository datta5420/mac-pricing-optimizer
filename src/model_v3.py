"""V3: price the MAC list, with the contract held fixed.

WHAT THE EARLIER VERSION GOT WRONG
An intermediate version of this model (Part 2, removed from the working tree;
still readable at commit d3d8ce7) let the optimizer choose a billing multiplier
per client, free within 0.85-1.15. That is a lever no pricing analyst has
during a contract term, and it made the guarantees easy to hit for the wrong
reason: if you may change the invoice, you can always keep the promise. It duly
reported 0 of 36 pools missed and $0 of true-up. With the contract held fixed,
3 of those 36 pools cannot be reached in a single pricing cycle under the
+/-15% price-change limit, leaving $16,146 of true-up. They ARE reachable on
drug economics alone -- see `feasibility_report`, which separates the two
causes -- so the perfect score was an artefact of the extra lever, not of the
contracts being unpriceable.

This version removes that lever. The contract is an input:

    decision  :  p[j], the MAC price for row j = (drug, channel)
    fixed     :  every client's guarantee, every fee
    assumption:  pass-through, so the client's ingredient charge IS p * units

That leaves a genuine tension. A client's guarantee caps what it may be billed,
which pushes prices DOWN. The pharmacy reimbursement floor pushes them UP, and
the price-change limit stops either side moving far in one cycle. One shared
MAC list has to serve twelve clients with different drug mixes, so the
optimizer has to decide whose guarantee gets missed, and by how much.

Note that the objective also carries a price-stability penalty, which is not a
contract cost and is NOT small: on the book in the README it is roughly half
the objective value and exceeds the expected-true-up term. It is reported
separately rather than folded into the P&L.

UNCERTAINTY IS OPTIMIZED, NOT JUST MEASURED
Volumes are not known when prices are set. The earlier version priced on plan
volumes and scored the result afterwards. This one puts scenarios inside the
objective (sample average approximation) and can also control the tail with
CVaR:

    minimise  (1 - lam) * E[L_s]  +  lam * CVaR_alpha(L_s)  +  w * disruption

    L_s = total true-up owed across all clients in scenario s

L_s carries only true-up payments. Admin-fee revenue is per script and does not
depend on the MAC price, so it is a constant in this objective and cannot change
the optimal prices; it is reported in the P&L instead. Guarantee
overperformance is likewise absent: under pass-through a deeper discount is
pharmacy revenue, not a PBM expense (see contract.py).

CVaR is linearised the standard way (Rockafellar and Uryasev):

    CVaR_alpha(L) = min over eta of  eta + 1/((1-alpha)S) * sum_s max(L_s - eta, 0)
"""
import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from . import config

DISRUPTION_WEIGHT = 0.01


def price_bounds(mac, max_change=None, margin_floor=None):
    """Lower and upper bound on each MAC price, and which rows the floor forces."""
    max_change = config.MAX_PRICE_CHANGE if max_change is None else max_change
    margin_floor = config.PHARMACY_MARGIN_FLOOR if margin_floor is None else margin_floor
    p0 = mac["current_price"].values
    floor = mac["nadac"].values * (1 + margin_floor)
    lo = np.maximum(floor, p0 * (1 - max_change))
    hi = np.maximum(p0 * (1 + max_change), lo)
    return lo, hi, (floor > p0 * (1 + max_change))


def pool_index(claims, mac):
    """units[c, j] for every client c and MAC row j, plus the rows of each channel."""
    key = {(r.drug_id, r.channel): j for j, r in enumerate(mac.itertuples())}
    clients = np.sort(claims["client_id"].unique())
    cpos = {c: i for i, c in enumerate(clients)}
    units = np.zeros((len(clients), len(mac)))
    scripts = np.zeros((len(clients), len(mac)))
    for r in claims.itertuples():
        j = key[(r.drug_id, r.channel)]
        units[cpos[r.client_id], j] += r.units
        scripts[cpos[r.client_id], j] += r.scripts
    chan_rows = {ch: np.array([j for j, r in enumerate(mac.itertuples()) if r.channel == ch])
                 for ch in config.CHANNELS}
    return clients, units, scripts, chan_rows


def feasibility_report(mac, claims, guarantees, max_change=None, margin_floor=None):
    """Which guarantee pools can be met, before any optimization.

    The cheapest a client can possibly be billed is every price at its lower
    bound, so comparing that to the guarantee target settles feasibility exactly
    with no solver involved.

    IMPORTANT: that lower bound is `max(pharmacy floor, last price x (1 - max_change))`,
    so a pool can be out of reach for two quite different reasons, and the
    difference matters commercially:

      * the PHARMACY FLOOR binds -- the guarantee is deeper than the cost of
        filling the claims, and no price list ever fixes it;
      * the CHANGE LIMIT binds -- the guarantee is reachable, but not in one
        pricing cycle without moving prices further than policy allows.

    `binding_bound` reports which, weighted by the spend sitting on each.
    """
    lo, hi, _ = price_bounds(mac, max_change, margin_floor)
    margin_floor = config.PHARMACY_MARGIN_FLOOR if margin_floor is None else margin_floor
    floor_only = mac["nadac"].values * (1 + margin_floor)
    on_floor = lo <= floor_only + 1e-12   # rows where the floor, not the limit, binds

    clients, units, _, chan_rows = pool_index(claims, mac)
    awp = mac["awp"].values
    rows = []
    for i, c in enumerate(clients):
        for ch, idx in chan_rows.items():
            g = guarantees[(c, ch)]
            q = units[i, idx]
            target = (1 - g) * float((awp[idx] * q).sum())
            cheapest = float((lo[idx] * q).sum())
            dearest = float((hi[idx] * q).sum())
            # Would it be reachable if only the pharmacy floor applied?
            floor_cheapest = float((floor_only[idx] * q).sum())
            spend_on_floor = float((lo[idx] * q)[on_floor[idx]].sum())
            share_floor = spend_on_floor / cheapest if cheapest else 0.0

            if cheapest > target:
                gap = cheapest - target
                if floor_cheapest > target:
                    status = "infeasible: pharmacy floor above guarantee"
                else:
                    status = "unreachable this cycle: price-change limit"
            elif dearest < target:
                status, gap = "always over-delivers", target - dearest
            else:
                status, gap = "feasible", 0.0
            rows.append({"client": int(c), "channel": ch, "guarantee": g,
                         "target_$": round(target, 0),
                         "min_possible_$": round(cheapest, 0),
                         "min_at_floor_only_$": round(floor_cheapest, 0),
                         "max_possible_$": round(dearest, 0),
                         "pct_spend_on_pharmacy_floor": round(100 * share_floor, 1),
                         "status": status, "gap_$": round(gap, 0)})
    return pd.DataFrame(rows)


def optimize_mac(claims, mac, guarantees, scenarios=None, cvar_alpha=0.95,
                 cvar_weight=0.0, max_change=None, margin_floor=None,
                 disruption_weight=DISRUPTION_WEIGHT, verbose=False):
    """Choose MAC prices. `scenarios` is (n_clients, n_rows, n_scen) multipliers.

    cvar_weight = 0 minimises expected true-up only; 1 minimises CVaR only.
    """
    lo, hi, floor_override = price_bounds(mac, max_change, margin_floor)
    clients, units, scripts, chan_rows = pool_index(claims, mac)
    awp = mac["awp"].values
    p0 = mac["current_price"].values
    n = len(mac)

    if scenarios is None:
        scenarios = np.ones((len(clients), n, 1))
    n_scen = scenarios.shape[2]

    m = gp.Model("mac_v3")
    m.Params.OutputFlag = 1 if verbose else 0
    p = m.addVars(n, lb=lo, ub=hi, name="price")
    up = m.addVars(n, lb=0, name="up")
    dn = m.addVars(n, lb=0, name="dn")
    m.addConstrs((p[j] - p0[j] == up[j] - dn[j] for j in range(n)), name="move")

    pos = {(r.drug_id, r.channel): j for j, r in enumerate(mac.itertuples())}
    for d in mac["drug_id"].unique():
        if (d, "mail") in pos and (d, "retail_90") in pos:
            m.addConstr(p[pos[(d, "mail")]] <= p[pos[(d, "retail_90")]], name=f"ord_m_{d}")
        if (d, "retail_90") in pos and (d, "retail_30") in pos:
            m.addConstr(p[pos[(d, "retail_90")]] <= p[pos[(d, "retail_30")]], name=f"ord_r_{d}")

    # True-up owed per client, channel and scenario. One-sided: a pool that
    # comes in under its target owes nothing, and the surplus is NOT a cost.
    short = {}
    for i, c in enumerate(clients):
        for ch, idx in chan_rows.items():
            g = guarantees[(c, ch)]
            for s in range(n_scen):
                q = units[i, idx] * scenarios[i, idx, s]
                target = (1 - g) * float((awp[idx] * q).sum())
                v = m.addVar(lb=0, name=f"short_{c}_{ch}_{s}")
                m.addConstr(v >= gp.quicksum(float(qi) * p[j] for j, qi in zip(idx, q)) - target,
                            name=f"tu_{c}_{ch}_{s}")
                short[(c, ch, s)] = v

    loss = {s: gp.quicksum(short[(c, ch, s)] for c in clients for ch in chan_rows)
            for s in range(n_scen)}
    expected = gp.quicksum(loss.values()) / n_scen

    if cvar_weight > 0:
        eta = m.addVar(lb=-GRB.INFINITY, name="cvar_eta")
        z = m.addVars(n_scen, lb=0, name="cvar_excess")
        m.addConstrs((z[s] >= loss[s] - eta for s in range(n_scen)), name="cvar")
        cvar = eta + gp.quicksum(z.values()) / ((1 - cvar_alpha) * n_scen)
    else:
        cvar = 0.0

    disruption = gp.quicksum(float(units[:, j].sum()) * (up[j] + dn[j]) for j in range(n))
    m.setObjective((1 - cvar_weight) * expected + cvar_weight * cvar
                   + disruption_weight * disruption, GRB.MINIMIZE)
    m.optimize()

    if m.Status == GRB.INFEASIBLE:
        m.computeIIS()
        bad = [c.ConstrName for c in m.getConstrs() if c.IISConstr]
        raise RuntimeError(f"Infeasible. Conflicting constraints include: {bad[:10]}")
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Solver stopped with status {m.Status}")

    out = mac.copy()
    out["new_price"] = [p[j].X for j in range(n)]
    out["floor_override"] = floor_override
    disruption_cost = disruption_weight * float(
        (units.sum(axis=0) * np.abs(out["new_price"].values - p0)).sum())
    return {
        "prices": out,
        # The solver's OWN view of what each pool owes, kept so that the
        # independent settlement in reconcile.py has something real to disagree
        # with. Comparing price x units two ways would prove nothing.
        "solver_true_up": {k: v.X for k, v in short.items()},
        "scenario_multipliers": scenarios,
        "expected_true_up": float(sum(v.X for v in short.values()) / n_scen),
        "disruption_cost": disruption_cost,
        "clients": clients,
        "units": units,
        "scripts": scripts,
        "chan_rows": chan_rows,
        "n_scenarios": n_scen,
        "cvar_weight": cvar_weight,
        "cvar_alpha": cvar_alpha,
        "objective": m.ObjVal,
        "termination": "optimal",
        "n_variables": m.NumVars,
        "n_constraints": m.NumConstrs,
    }
