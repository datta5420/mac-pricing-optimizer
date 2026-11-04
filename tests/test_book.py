"""The shared data layer: real NADAC costs, the client book, and the correlated
volume scenarios. Part 3 is built on all three, so they are tested on their own
rather than only through the optimizer."""
import numpy as np
import pytest

from src.book import build_book, load_nadac_sample, mac_table
from src.risk import volume_scenarios


@pytest.fixture(scope="module")
def book():
    drugs = load_nadac_sample(n_drugs=60, seed=5)
    claims, _, _ = build_book(drugs, n_clients=5, seed=5)
    return claims, mac_table(claims)


def test_nadac_costs_are_real_and_positive():
    d = load_nadac_sample(n_drugs=60)
    assert (d["nadac"] > 0).all()
    # Real NADAC descriptions, not GENERIC_000 placeholders.
    assert not d["name"].str.startswith("GENERIC_").any()


def test_one_mac_price_per_drug_and_channel(book):
    _, mac = book
    assert not mac.duplicated(["drug_id", "channel"]).any()


def test_correlated_shocks_are_more_volatile_than_independent(book):
    """Class-level shocks must not wash out the way independent noise does."""
    claims, mac = book
    clients = np.sort(claims["client_id"].unique())
    corr = volume_scenarios(mac, clients, 400, class_sd=0.12, drug_sd=0.10, seed=1)
    indep = volume_scenarios(mac, clients, 400, class_sd=0.0, drug_sd=0.16, seed=1)
    assert corr.mean(axis=1).std() > indep.mean(axis=1).std()
