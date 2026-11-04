"""Contract terms, and how money moves under them.

WHAT IS FIXED AND WHAT IS DECIDED
A signed contract is an input, not a decision. The optimizer may not touch:

  * the AWP discount guarantee for each client and channel
  * the dispensing fee paid to the pharmacy, per script
  * the administrative fee the PBM charges the client, per script
  * any client billing multiplier (see `renegotiate.py` for the separate
    exercise of asking what multiplier a contract WOULD need)

The only decision is the MAC price for each (drug, channel).

THE PASS-THROUGH ASSUMPTION (a simplification, not an industry rule)
This model bills the client the same ingredient cost the pharmacy is paid:

    client ingredient charge = pharmacy ingredient reimbursement = MAC x units

Real pass-through contracts vary, and real arrangements may differ in how
ingredient cost, fees and other components are defined. Spread arrangements,
where the client is billed from a separate schedule, are a different structure
and are not modelled here. This assumption is stated so that nothing below is
mistaken for a universal rule.

Fees are kept separate from ingredient cost throughout, because the guarantee
is written against INGREDIENT cost only:

    client total charge = ingredient charge
                        + dispensing fee x scripts      (passed through)
                        + admin fee x scripts           (PBM revenue)

WHAT THAT MEANS FOR THE PBM
Under pass-through the ingredient dollars flow through the PBM untouched, so:

    PBM revenue   = admin fee x scripts          (does not depend on MAC price)
    PBM cost      = true-up payments             (does depend on MAC price)
    contribution  = revenue - true-ups - operating cost

Note the consequence, which is easy to get wrong: giving a client a DEEPER
discount than promised is not a PBM expense here. Those dollars were never the
PBM's; a lower MAC means the pharmacy is paid less and the client pays less.
It is reported as guarantee overperformance (a client benefit), never
subtracted from margin as if it were a loss.

So the only pricing-dependent CONTRACT cost is true-ups, which pushes MAC
prices DOWN, and the counterweight is the pharmacy reimbursement floor, which
pushes them UP.

One caveat, because it is easy to overstate the line above: the optimizer's
objective also carries a price-stability penalty, which is a non-cash proxy for
the operational cost of repricing rather than a contract cost. It is large
enough to shape the answer (about half the objective on the README's book), so
"true-ups are the only pricing-dependent cost" is true of the P&L and not of
the objective. See `model_v3`.
"""
import numpy as np

# Per-script fees. Simulated contract terms, not taken from any real contract.
DISPENSING_FEE = {"retail_30": 1.25, "retail_90": 1.75, "mail": 0.60}
ADMIN_FEE_PER_SCRIPT = 0.45
OPERATING_COST_PER_SCRIPT = 0.18


def contract_terms(clients):
    """Per-client fee terms. Guarantees come from book.build_book and are fixed.

    Deterministic: the admin fee is a function of the client's size rank, so
    this takes no seed.
    """
    out = clients.copy()
    # Larger books negotiate a lower admin fee.
    rank = out["size"].rank(pct=True).values
    out["admin_fee_per_script"] = np.round(ADMIN_FEE_PER_SCRIPT * (1.25 - 0.5 * rank), 4)
    out["operating_cost_per_script"] = OPERATING_COST_PER_SCRIPT
    out["dispensing_fee_source"] = "channel table"
    return out


def pharmacy_payment(prices, units, scripts, channels):
    """What pharmacies receive: ingredient reimbursement plus dispensing fees."""
    ingredient = float((prices * units).sum())
    fees = float(sum(DISPENSING_FEE[ch] * s for ch, s in zip(channels, scripts)))
    return {"ingredient_$": ingredient, "dispensing_fees_$": fees,
            "total_$": ingredient + fees}


def client_charge(ingredient, scripts_total, admin_fee, dispensing_fees):
    """What the client is billed, with every component kept separate."""
    admin = admin_fee * scripts_total
    return {"ingredient_$": ingredient,
            "dispensing_fees_$": dispensing_fees,
            "admin_fees_$": admin,
            "total_$": ingredient + dispensing_fees + admin}


def describe_assumptions():
    """Printed by the runners so no result is read without its caveats."""
    return "\n".join([
        "ASSUMPTIONS",
        "  Real:      generic acquisition cost benchmark (CMS NADAC, 2026-10-28).",
        "             NADAC is a national survey average, NOT any individual",
        "             pharmacy's acquisition cost.",
        "  Simulated: AWP, claim volumes, script counts, current prices, client",
        "             contracts, fees, and the `synthetic_class` grouping, which",
        "             is a random label and not a therapeutic classification.",
        "  Structure: simplified pass-through -- client ingredient charge equals",
        "             pharmacy ingredient reimbursement. Fees are tracked",
        "             separately. Spread arrangements are not modelled.",
        "  Guarantee: written against ingredient cost only, excluding fees.",
    ])
