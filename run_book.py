"""Price one MAC list across a book of clients, then stress-test it.

    python run_book.py                 # 200 drugs, 12 clients
    python run_book.py --drugs 120 --clients 8 --scenarios 1000
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.book import build_book, load_nadac_sample, mac_table
from src.model_multi import client_summary, optimize_book, spread_dollars
from src.risk import cushion_sweep, evaluate_book, plot_cushion, volume_scenarios


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drugs", type=int, default=200)
    ap.add_argument("--clients", type=int, default=12)
    ap.add_argument("--scenarios", type=int, default=2000)
    ap.add_argument("--nadac", default="data/raw/nadac_sample.csv")
    args = ap.parse_args()

    out = Path("outputs")
    out.mkdir(exist_ok=True)
    pd.set_option("display.width", 160)

    drugs = load_nadac_sample(args.nadac, n_drugs=args.drugs)
    claims, clients, guarantees = build_book(drugs, n_clients=args.clients)
    mac = mac_table(claims)
    print(f"{len(drugs)} drugs (real NADAC costs) x 3 channels = {len(mac)} MAC rows")
    print(f"{args.clients} clients, {len(claims):,} client-drug-channel lines\n")

    # 1. One scale factor per client vs one per client and channel.
    print("HOW MANY SCALE FACTORS DOES A CLIENT NEED?")
    runs = {}
    for mode in ["client", "client_channel"]:
        r = optimize_book(claims, mac, guarantees, scale_by=mode)
        cs = client_summary(r, guarantees)
        runs[mode] = (r, cs)
        print(f"  {mode:15s} converged in {len(r['history'])} passes | "
              f"guarantee pools missed {int((cs['true_up_$'] > 1).sum())}/{len(cs)} | "
              f"true-up ${cs['true_up_$'].sum():,.0f} | "
              f"given away ${cs['given_away_$'].sum():,.0f}")
    print("  A client has 3 channel guarantees, so 1 factor cannot hit all 3.\n")

    result, summary = runs["client_channel"]
    summary.to_csv(out / "book_client_summary.csv", index=False)
    result["prices"].round(4).to_csv(out / "book_recommended_prices.csv", index=False)

    # 2. Contracts that cannot be priced profitably, whatever the MAC list is.
    sp = spread_dollars(result)
    bad = sp[sp["spread_$"] < 0]
    print("CONTRACTS PRICED BELOW THE COST OF FILLING THEM")
    print(f"  {len(bad)} of {len(sp)} clients are billed less than their pharmacies are paid "
          f"(${-bad['spread_$'].sum():,.0f} total).")
    print("  No MAC list fixes this; it is a contract-negotiation finding.\n")
    sp.to_csv(out / "book_spread.csv", index=False)

    # 3. What the prices do when volumes do not land on plan.
    mult = volume_scenarios(mac, result["clients"], args.scenarios, seed=7)
    e = evaluate_book(result, guarantees, mult)
    print(f"UNDER CORRELATED VOLUME SHOCKS ({args.scenarios:,} scenarios)")
    print(f"  Chance at least one guarantee pool owes a true-up: {e['p_any_true_up']:.1%}")
    print(f"  Average true-up ${e['avg_true_up_$']:,.0f} | 95th percentile ${e['p95_true_up_$']:,.0f}")
    print(f"  Average margin given away ${e['avg_given_away_$']:,.0f}")

    exposure = {}
    p, awp, units = result["prices"]["new_price"].values, result["awp"], result["units"]
    for i, c in enumerate(result["clients"]):
        for ch, idx in result["chan_rows"].items():
            s, g = result["scale"][(int(c), ch)], guarantees[(c, ch)]
            gap = ((s * p[idx] - (1 - g) * awp[idx]) * units[i, idx]) @ mult[i, idx, :]
            exposure[(int(c), ch)] = float(np.clip(gap, 0, None).mean())
    total = sum(exposure.values())
    by_client = pd.Series({k[0]: 0 for k in exposure}, dtype=float)
    for (c, ch), v in exposure.items():
        by_client[c] += v
    top = by_client.sort_values(ascending=False)
    print(f"  Exposure is concentrated: client {top.index[0]} carries "
          f"{100 * top.iloc[0] / total:.0f}% of the dollars at risk.\n")

    # 4. What a cushion above the guarantee buys, and what it costs.
    print("WHAT PREDICTABILITY COSTS")
    sweep = cushion_sweep(claims, mac, guarantees, n_scen=args.scenarios)
    show = sweep.copy()
    for col in ["avg_true_up_$", "p95_true_up_$", "avg_given_away_$", "avg_net_cost_$"]:
        show[col] = show[col].map("${:,.0f}".format)
    show["P(true_up)"] = (100 * sweep["P(true_up)"]).map("{:.0f}%".format)
    print(show.to_string(index=False))
    sweep.to_csv(out / "book_cushion_sweep.csv", index=False)
    plot_cushion(sweep, out / "cushion_tradeoff.png")
    print(f"\nSaved outputs to {out}/")


if __name__ == "__main__":
    main()
