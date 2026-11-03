"""How fragile is a price list when volumes do not land where they were planned?

Prices are set on planned volumes. Real utilization moves, and it does not move
independently drug by drug: a whole therapeutic class shifts together when a
guideline changes, a competitor's drug goes generic, or a client's membership
turns over. Independent noise averages out across hundreds of drugs and makes a
price list look far safer than it is, so the shocks here are CORRELATED:

    volume multiplier = client effect x therapeutic-class effect x drug noise

Nothing here is re-optimized. The question is what the prices already chosen do
when the year does not go to plan.
"""
import numpy as np

from . import config


def volume_scenarios(mac, clients, n_scen, client_sd=0.06, class_sd=0.12,
                     drug_sd=0.10, seed=0):
    """Multipliers of shape (n_clients, n_mac_rows, n_scen)."""
    rng = np.random.default_rng(seed)
    n_c, n_j = len(clients), len(mac)
    klass = mac["klass"].values
    n_k = int(klass.max()) + 1

    client_eff = rng.lognormal(0, client_sd, size=(n_c, 1, n_scen))
    class_eff = rng.lognormal(0, class_sd, size=(1, n_k, n_scen))[:, klass, :]
    drug_eff = rng.lognormal(0, drug_sd, size=(n_c, n_j, n_scen))
    return client_eff * class_eff * drug_eff


def evaluate_book(result, guarantees, mult):
    """Score a fixed price list and scale factors against volume scenarios.

    Per line the margin against the guarantee is  s * p - (1 - g) * awp,
    so a client-channel's year-end gap is that margin dotted with its volumes.
    Positive gap = true-up owed to the client; negative = margin given away.
    """
    p = result["prices"]["new_price"].values
    awp, units, chan_rows = result["awp"], result["units"], result["chan_rows"]
    n_scen = mult.shape[2]
    any_true_up = np.zeros(n_scen, dtype=bool)
    true_up = np.zeros(n_scen)
    given_away = np.zeros(n_scen)
    worst = {}

    for i, c in enumerate(result["clients"]):
        for ch, idx in chan_rows.items():
            s = result["scale"][(int(c), ch)]
            g = guarantees[(c, ch)]
            margin = s * p[idx] - (1 - g) * awp[idx]          # $ per unit
            gap = (margin * units[i, idx]) @ mult[i, idx, :]  # one gap per scenario
            any_true_up |= gap > 1e-6
            true_up += np.clip(gap, 0, None)
            given_away += np.clip(-gap, 0, None)
            worst[(int(c), ch)] = float((gap > 1e-6).mean())

    return {
        "p_any_true_up": float(any_true_up.mean()),
        "avg_true_up_$": float(true_up.mean()),
        "p95_true_up_$": float(np.percentile(true_up, 95)),
        "avg_given_away_$": float(given_away.mean()),
        "avg_net_cost_$": float((true_up + given_away).mean()),
        "worst_rows": sorted(worst.items(), key=lambda kv: -kv[1])[:5],
    }


def cushion_sweep(claims, mac, guarantees, cushions=(0, 0.0025, 0.005, 0.0075, 0.01, 0.015),
                  n_scen=2000, scale_by="client_channel", seed=7, **kw):
    """Price with a cushion above each guarantee, then score against the REAL ones."""
    import pandas as pd

    from .model_multi import optimize_book

    base = optimize_book(claims, mac, guarantees, scale_by=scale_by, **kw)
    mult = volume_scenarios(mac, base["clients"], n_scen, seed=seed)

    rows = []
    for c in cushions:
        r = optimize_book(claims, mac, guarantees, cushion=c, scale_by=scale_by, **kw)
        e = evaluate_book(r, guarantees, mult)
        rows.append({
            "cushion_pts": round(100 * c, 3),
            "P(true_up)": round(e["p_any_true_up"], 3),
            "avg_true_up_$": round(e["avg_true_up_$"], 0),
            "p95_true_up_$": round(e["p95_true_up_$"], 0),
            "avg_given_away_$": round(e["avg_given_away_$"], 0),
            "avg_net_cost_$": round(e["avg_net_cost_$"], 0),
        })
    return pd.DataFrame(rows)


def plot_cushion(sweep, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = sweep["cushion_pts"]
    ax.plot(x, 100 * sweep["P(true_up)"], marker="o", color="#2b6cb0",
            label="Chance a client is owed a true-up (%)")
    ax.set_xlabel("Cushion priced above each guarantee (percentage points)")
    ax.set_ylabel("Chance of owing a true-up (%)")
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(x, sweep["avg_given_away_$"] / 1000, marker="s", color="#999999",
             label="Margin given away ($K)")
    ax2.set_ylabel("Average margin given away ($ thousands)")

    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="center right")
    ax.set_title("Buying predictability: what a pricing cushion costs")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
