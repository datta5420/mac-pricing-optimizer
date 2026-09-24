import numpy as np
import pytest

from src import config
from src.data import build_dataset
from src.model import optimize_prices
from src.report import effective_discount


@pytest.fixture(scope="module")
def claims():
    return build_dataset(n_drugs=60, seed=7)


def test_loose_limits_hit_every_guarantee(claims):
    r = optimize_prices(claims, max_change=0.5)
    df = r["prices"]
    for ch, g in config.GUARANTEES.items():
        assert effective_discount(df[df["channel"] == ch], "new_price") == pytest.approx(g, abs=1e-3)


def test_pharmacy_never_paid_below_floor(claims):
    df = optimize_prices(claims)["prices"]
    floor = df["nadac"] * (1 + config.PHARMACY_MARGIN_FLOOR)
    assert (df["new_price"] >= floor - 1e-3).all()


def test_price_moves_within_limit_unless_floor_forced(claims):
    lim = 0.10
    df = optimize_prices(claims, max_change=lim)["prices"]
    change = (df["new_price"] / df["current_price"] - 1).abs()
    normal = ~df["floor_override"]
    assert (change[normal] <= lim + 1e-3).all()


def test_channel_price_order(claims):
    df = optimize_prices(claims)["prices"]
    wide = df.pivot(index="drug_id", columns="channel", values="new_price")
    assert (wide["retail_90"] <= wide["retail_30"] + 1e-4).all()
    assert (wide["mail"] <= wide["retail_90"] + 1e-4).all()


def test_tighter_limits_never_cost_less(claims):
    costs = []
    for lim in [0.3, 0.1, 0.03, 0.01]:
        r = optimize_prices(claims, max_change=lim)
        costs.append(sum(r["shortfall"].values()) + sum(r["surplus"].values()))
    assert all(np.diff(costs) >= -1e-3)
