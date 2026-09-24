"""Run the full pipeline: data -> optimize -> report -> trade-off chart.

Usage:
    python run.py                              # synthetic drug costs
    python run.py --nadac data/raw/nadac.csv   # real CMS NADAC costs
"""
import argparse
from pathlib import Path

import pandas as pd

from src import config
from src.data import build_dataset
from src.model import optimize_prices
from src.report import channel_summary, current_gap, plot_tradeoff, tradeoff_sweep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nadac", help="Path to NADAC CSV from data.medicaid.gov")
    parser.add_argument("--n-drugs", type=int, default=config.N_DRUGS)
    args = parser.parse_args()

    out = Path("outputs")
    out.mkdir(exist_ok=True)
    pd.set_option("display.width", 140)

    claims = build_dataset(args.nadac, args.n_drugs)
    print(f"Loaded {claims['drug_id'].nunique()} drugs x {claims['channel'].nunique()} channels\n")

    print("BEFORE (current prices): positive = true-up owed, negative = margin given away")
    for ch, gap in current_gap(claims).items():
        print(f"  {ch:10s} ${gap:>12,.0f}")

    result = optimize_prices(claims)
    print(f"\nModel size: {result['n_variables']} variables, {result['n_constraints']} constraints")
    print("\nAFTER optimization:")
    print(channel_summary(result).to_string(index=False))
    overrides = int(result["prices"]["floor_override"].sum())
    print(f"\nPrices forced up to the pharmacy cost floor (beyond the change limit): {overrides}")

    result["prices"].round(4).to_csv(out / "recommended_prices.csv", index=False)

    sweep = tradeoff_sweep(claims)
    print("\nTrade-off: tighter price-change limits vs. cost")
    show = sweep.copy()
    show["max_price_change"] = (show["max_price_change"] * 100).map("{:.0f}%".format)
    for col in ["true_up_owed_$", "margin_given_away_$"]:
        show[col] = show[col].map("${:,.0f}".format)
    print(show.to_string(index=False))
    plot_tradeoff(sweep, out / "tradeoff.png")
    print(f"\nSaved {out/'recommended_prices.csv'} and {out/'tradeoff.png'}")


if __name__ == "__main__":
    main()
