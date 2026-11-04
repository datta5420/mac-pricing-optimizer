"""Volume scenarios: what a price list faces when utilization does not go to plan.

Prices are set on planned volumes. Real utilization moves, and it does not move
independently drug by drug: a whole therapeutic class shifts together when a
guideline changes, a competitor's drug goes generic, or a client's membership
turns over. Independent noise averages out across hundreds of drugs and makes a
price list look far safer than it is, so the shocks here are CORRELATED:

    volume multiplier = client effect x therapeutic-class effect x drug noise

These multipliers are the only thing this module produces. Part 3 uses them two
ways: inside the objective, as the sample-average scenarios the MAC list is
optimized against (`model_v3.optimize_mac`), and afterwards as a held-out set
the chosen prices are scored on (`reconcile.scenario_pnl`). Training and
scoring draw different seeds, so the second is a real test and not a replay.
"""
import numpy as np


def volume_scenarios(mac, clients, n_scen, client_sd=0.06, class_sd=0.12,
                     drug_sd=0.10, seed=0):
    """Multipliers of shape (n_clients, n_mac_rows, n_scen).

    `synthetic_class` is a random grouping label, not a therapeutic
    classification. It exists so shocks arrive correlated across groups of
    drugs instead of independently, which is the whole point of this module.
    """
    rng = np.random.default_rng(seed)
    n_c, n_j = len(clients), len(mac)
    klass = mac["synthetic_class"].values
    n_k = int(klass.max()) + 1

    client_eff = rng.lognormal(0, client_sd, size=(n_c, 1, n_scen))
    class_eff = rng.lognormal(0, class_sd, size=(1, n_k, n_scen))[:, klass, :]
    drug_eff = rng.lognormal(0, drug_sd, size=(n_c, n_j, n_scen))
    return client_eff * class_eff * drug_eff
