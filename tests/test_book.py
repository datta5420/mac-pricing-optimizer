import numpy as np
import pytest

pytest.importorskip("gurobipy", reason="Gurobi is not installed in this environment")

from src import config
from src.book import build_book, load_nadac_sample, mac_table
from src.model_multi import client_summary, optimize_book
from src.risk import evaluate_book, volume_scenarios


@pytest.fixture(scope="module")
def book():
    drugs = load_nadac_sample(n_drugs=60, seed=5)
    claims, clients, guarantees = build_book(drugs, n_clients=5, seed=5)
    return claims, mac_table(claims), guarantees


def test_nadac_costs_are_real_and_positive():
    d = load_nadac_sample(n_drugs=60)
    assert (d["nadac"] > 0).all()
    # Real NADAC descriptions, not GENERIC_000 placeholders.
    assert not d["name"].str.startswith("GENERIC_").any()


def test_one_mac_price_per_drug_and_channel(book):
    _, mac, _ = book
    assert not mac.duplicated(["drug_id", "channel"]).any()


def test_pharmacy_never_paid_below_cost_floor(book):
    claims, mac, guarantees = book
    r = optimize_book(claims, mac, guarantees, iters=2)
    floor = r["prices"]["nadac"] * (1 + config.PHARMACY_MARGIN_FLOOR)
    assert (r["prices"]["new_price"] >= floor - 1e-6).all()


def test_channel_price_order_holds(book):
    claims, mac, guarantees = book
    r = optimize_book(claims, mac, guarantees, iters=2)
    wide = r["prices"].pivot(index="drug_id", columns="channel", values="new_price")
    assert (wide["retail_90"] <= wide["retail_30"] + 1e-6).all()
    assert (wide["mail"] <= wide["retail_90"] + 1e-6).all()


def test_per_channel_scale_factors_hit_every_guarantee(book):
    claims, mac, guarantees = book
    r = optimize_book(claims, mac, guarantees, iters=4, scale_by="client_channel")
    cs = client_summary(r, guarantees)
    assert cs["true_up_$"].sum() < 1.0
    assert cs["given_away_$"].sum() < 1.0


def test_one_factor_per_client_cannot_beat_one_per_channel(book):
    claims, mac, guarantees = book
    coarse = client_summary(optimize_book(claims, mac, guarantees, iters=4,
                                          scale_by="client"), guarantees)
    fine = client_summary(optimize_book(claims, mac, guarantees, iters=4,
                                        scale_by="client_channel"), guarantees)
    coarse_cost = coarse["true_up_$"].sum() + coarse["given_away_$"].sum()
    fine_cost = fine["true_up_$"].sum() + fine["given_away_$"].sum()
    assert fine_cost <= coarse_cost + 1e-6


def test_alternating_objective_never_increases(book):
    claims, mac, guarantees = book
    h = optimize_book(claims, mac, guarantees, iters=5)["history"]
    assert all(h[i + 1] <= h[i] + 1e-6 for i in range(len(h) - 1))


def test_a_cushion_lowers_true_up_risk(book):
    claims, mac, guarantees = book
    mult = volume_scenarios(mac, np.sort(claims["client_id"].unique()), 300, seed=3)
    plain = optimize_book(claims, mac, guarantees, iters=3, scale_by="client_channel")
    padded = optimize_book(claims, mac, guarantees, iters=3, scale_by="client_channel",
                           cushion=0.01)
    a = evaluate_book(plain, guarantees, mult)
    b = evaluate_book(padded, guarantees, mult)
    assert b["avg_true_up_$"] <= a["avg_true_up_$"]
    assert b["avg_given_away_$"] >= a["avg_given_away_$"]


def test_correlated_shocks_are_more_volatile_than_independent(book):
    """Class-level shocks must not wash out the way independent noise does."""
    claims, mac, guarantees = book
    clients = np.sort(claims["client_id"].unique())
    corr = volume_scenarios(mac, clients, 400, class_sd=0.12, drug_sd=0.10, seed=1)
    indep = volume_scenarios(mac, clients, 400, class_sd=0.0, drug_sd=0.16, seed=1)
    assert corr.mean(axis=1).std() > indep.mean(axis=1).std()
