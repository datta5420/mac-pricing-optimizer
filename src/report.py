"""Summaries and the trade-off chart."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config
from .model import optimize_prices


def effective_discount(df, price_col):
    """Achieved AWP discount = 1 - total paid / total AWP."""
    paid = (df[price_col] * df["units"]).sum()
    awp = (df["awp"] * df["units"]).sum()
    return 1 - paid / awp


def channel_summary(result, guarantees=None):
    guarantees = guarantees or config.GUARANTEES
    df = result["prices"]
    rows = []
    for ch in config.CHANNELS:
        part = df[df["channel"] == ch]
        rows.append({
            "channel": ch,
            "guarantee": guarantees[ch],
            "current_discount": round(effective_discount(part, "current_price"), 4),
            "new_discount": round(effective_discount(part, "new_price"), 4),
            "true_up_owed_$": round(result["shortfall"][ch], 0),
            "margin_given_away_$": round(result["surplus"][ch], 0),
            "prices_changed": int((abs(part["new_price"] - part["current_price"]) > 1e-4).sum()),
        })
    return pd.DataFrame(rows)


def current_gap(claims, guarantees=None):
    """True-up and give-away if we changed nothing (the baseline)."""
    guarantees = guarantees or config.GUARANTEES
    out = {}
    for ch in config.CHANNELS:
        part = claims[claims["channel"] == ch]
        target = (1 - guarantees[ch]) * (part["awp"] * part["units"]).sum()
        spend = (part["current_price"] * part["units"]).sum()
        out[ch] = spend - target  # positive = shortfall, negative = surplus
    return out


def tradeoff_sweep(claims, limits=(0.25, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02, 0.01)):
    rows = []
    for lim in limits:
        r = optimize_prices(claims, max_change=lim)
        rows.append({
            "max_price_change": lim,
            "true_up_owed_$": sum(r["shortfall"].values()),
            "margin_given_away_$": sum(r["surplus"].values()),
            "prices_changed": int((abs(r["prices"]["new_price"] - r["prices"]["current_price"]) > 1e-4).sum()),
        })
    return pd.DataFrame(rows)


def plot_tradeoff(sweep, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = sweep["max_price_change"] * 100
    ax.plot(x, sweep["true_up_owed_$"] / 1000, marker="o", label="True-up owed to client")
    ax.plot(x, sweep["margin_given_away_$"] / 1000, marker="s", label="Margin given away")
    ax.set_xlabel("Max price change allowed per cycle (%)")
    ax.set_ylabel("Cost ($ thousands)")
    ax.set_title("Price stability vs. financial cost")
    ax.invert_xaxis()
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
