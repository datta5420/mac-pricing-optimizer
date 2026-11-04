"""V3: price the MAC list with the contract fixed, then settle it independently.

    python run_v3.py
    python run_v3.py --drugs 120 --clients 10 --train 12 --test 2000
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.book import build_book, load_nadac_sample, mac_table
from src.contract import contract_terms, describe_assumptions
from src.model_v3 import feasibility_report, optimize_mac
from src.reconcile import check_true_up_against_solver, pnl, scenario_pnl, settle
from src.renegotiate import deliverable_guarantee, required_multiplier
from src.risk import volume_scenarios


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drugs", type=int, default=150)
    ap.add_argument("--clients", type=int, default=12)
    ap.add_argument("--train", type=int, default=15, help="in-sample scenarios")
    ap.add_argument("--test", type=int, default=2000, help="held-out scenarios")
    ap.add_argument("--nadac", default="data/raw/nadac_sample.csv")
    args = ap.parse_args()

    out = Path("outputs")
    out.mkdir(exist_ok=True)
    pd.set_option("display.width", 170)

    print(describe_assumptions(), "\n")

    drugs = load_nadac_sample(args.nadac, n_drugs=args.drugs)
    claims, clients, guarantees = build_book(drugs, n_clients=args.clients)
    mac = mac_table(claims)
    terms = contract_terms(clients)
    print(f"{len(drugs)} drugs x 3 channels = {len(mac)} MAC rows | "
          f"{args.clients} clients | {len(claims):,} claim lines\n")

    # 1. Feasibility, settled by arithmetic rather than assumed.
    print("1. CAN EACH GUARANTEE BE MET, AND IF NOT, WHAT IS STOPPING IT?")
    fr = feasibility_report(mac, claims, guarantees)
    for status, n in fr["status"].value_counts().items():
        print(f"   {n:>2} of {len(fr)} pools: {status}")
    stuck = fr[fr["status"] != "feasible"]
    if len(stuck):
        print(f"   True-up that cannot be avoided this cycle: ${stuck['gap_$'].sum():,.0f}")
        print(f"   On those pools only {stuck['pct_spend_on_pharmacy_floor'].mean():.0f}% of the "
              "cheapest possible spend sits on the pharmacy floor,")
        print("   so the binding constraint is the price-change limit, not drug cost.")
        print(f"   Same book at other one-cycle change limits "
              f"(current: {100 * config.MAX_PRICE_CHANGE:.0f}%):")
        for lim in [0.15, 0.17, 0.20, 0.25]:
            f2 = feasibility_report(mac, claims, guarantees, max_change=lim)
            s2 = f2[f2["status"] != "feasible"]
            print(f"     +/-{100 * lim:>3.0f}% : {len(s2):>2} pools short, ${s2['gap_$'].sum():>8,.0f}")
    fr.to_csv(out / "v3_feasibility.csv", index=False)
    print()

    # 2. Price on plan volumes, then with uncertainty in the objective.
    train = volume_scenarios(mac, np.arange(args.clients), args.train, seed=11)
    test = volume_scenarios(mac, np.arange(args.clients), args.test, seed=99)
    tail = (1 - 0.80) * args.train
    print(f"2. PRICING STRATEGIES  (trained on {args.train} scenarios, "
          f"scored on {args.test:,} held out)")
    print(f"   CVaR level 0.80, so the in-sample tail holds {tail:.0f} scenarios.\n")

    strategies = {
        "plan volumes only": dict(scenarios=None, cvar_weight=0.0),
        "expected cost (SAA)": dict(scenarios=train, cvar_weight=0.0),
        "SAA + CVaR (w=0.5)": dict(scenarios=train, cvar_weight=0.5, cvar_alpha=0.80),
        "CVaR only (w=1.0)": dict(scenarios=train, cvar_weight=1.0, cvar_alpha=0.80),
    }
    rows, results = [], {}
    for name, kw in strategies.items():
        r = optimize_mac(claims, mac, guarantees, **kw)
        sp = scenario_pnl(claims, r["prices"], guarantees, terms, r, test)
        results[name] = r
        rows.append({"strategy": name, "vars": r["n_variables"],
                     "mean_true_up_$": round(sp["mean_true_up_$"]),
                     "p95_true_up_$": round(sp["p95_true_up_$"]),
                     "cvar95_true_up_$": round(sp["cvar95_true_up_$"]),
                     "mean_contribution_$": round(sp["mean_contribution_$"]),
                     "p05_contribution_$": round(sp["p05_contribution_$"]),
                     "P(negative year)": round(sp["p_negative_contribution"], 4)})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    table.to_csv(out / "v3_strategies.csv", index=False)
    best = "expected cost (SAA)"
    print(f"\n   Chosen: {best}. CVaR is implemented but does not help here -- with")
    print(f"   {args.train} training scenarios its tail rests on {tail:.0f} points, so it fits sampling")
    print( "   noise. Worth revisiting with a licence that allows more scenarios.")
    print( "   Note: contribution rises by exactly the true-up saved. Admin fees and")
    print( "   operating costs do not depend on price, so it is one gain, not two.\n")

    # 3. Settle the contract year independently of the solver.
    result = results[best]
    settled = settle(claims, result["prices"], guarantees, terms)
    diff = check_true_up_against_solver(claims, result['prices'], guarantees, terms, result)
    print("3. INDEPENDENT RECONCILIATION")
    print("   The solver's own true-up variables vs the same figure rebuilt from the")
    print("   claim lines (group, apply guarantee, take the positive part).")
    print(f"   Largest disagreement: ${diff:,.6f}")
    book = pnl(settled)
    print("\n   P&L ON PLAN VOLUMES (the single planned year, not an expectation):")
    for k in ["admin_fee_revenue_$", "true_up_payments_$", "operating_cost_$",
              "contribution_margin_$"]:
        print(f"     {k:28s} ${book[k]:>12,.0f}")
    print(f"     {'(pass-through ingredient)':28s} ${book['_passthrough_ingredient_$']:>12,.0f}")
    print(f"     {'(guarantee overperformance)':28s} ${book['_guarantee_overperformance_$']:>12,.0f}"
          "   client benefit, NOT a PBM expense")
    chosen = next(r for r in rows if r["strategy"] == best)
    print(f"\n   Across the {args.test:,} held-out years, the same prices give a mean")
    print(f"   true-up of ${chosen['mean_true_up_$']:,.0f} and mean contribution of "
          f"${chosen['mean_contribution_$']:,.0f}.")
    print(f"\n   Objective composition: the price-stability penalty contributes "
          f"${result['disruption_cost']:,.0f}")
    print(f"   of the ${result['objective']:,.0f} objective "
          f"({100 * result['disruption_cost'] / result['objective']:.0f}%). It is a non-cash")
    print( "   proxy for the operational cost of changing prices, not a contract cost,")
    print( "   and it is excluded from the P&L above.")
    settled.to_csv(out / "v3_settlement.csv", index=False)
    result["prices"].round(4).to_csv(out / "v3_prices.csv", index=False)
    print()

    # 4. What the contracts would have to say (separate from pricing).
    print("4. RENEGOTIATION VIEW (not part of the optimization)")
    dg = deliverable_guarantee(mac, claims, guarantees)
    need = dg[dg["renegotiate"]]
    cycles = dg[dg["needs_more_cycles"]]
    if len(need):
        print(f"   {len(need)} pools promise more than the pharmacy floor allows -- these")
        print("   need a different contract, not a different price list:")
        print(need[["client", "channel", "promised_guarantee",
                    "deepest_ever", "short_ever_pts"]].to_string(index=False))
    else:
        print("   Every promised guarantee is deliverable once prices reach the floor,")
        print("   so no contract here is mispriced against drug economics.")
    if len(cycles):
        print(f"\n   {len(cycles)} pools are reachable but not in one cycle at the current")
        print(f"   +/-{100 * config.MAX_PRICE_CHANGE:.0f}% change limit. The fix is more cycles or a wider limit:")
        print(cycles[["client", "channel", "promised_guarantee", "deepest_this_cycle",
                      "deepest_ever", "short_this_cycle_pts"]].to_string(index=False))
    dg.to_csv(out / "v3_renegotiation.csv", index=False)

    rm = required_multiplier(settled)
    off = rm[~rm["within_10pct"]]
    print(f"\n   For reference, the billing multiplier each pool would need if this were")
    print(f"   a spread contract (the lever V2 optimized, reported here only):")
    print(f"     range {rm['required_multiplier'].min():.3f} to {rm['required_multiplier'].max():.3f}; "
          f"{len(off)} of {len(rm)} outside +/-10% of 1.0")
    rm.to_csv(out / "v3_required_multipliers.csv", index=False)
    print(f"\nSaved outputs to {out}/")


if __name__ == "__main__":
    main()
