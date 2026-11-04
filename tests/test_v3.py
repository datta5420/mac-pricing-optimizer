import numpy as np
import pytest

pytest.importorskip("gurobipy", reason="Gurobi is not installed in this environment")

from src import config
from src.book import build_book, load_nadac_sample, mac_table
from src.contract import DISPENSING_FEE, contract_terms
from src.model_v3 import feasibility_report, optimize_mac, price_bounds
from src.reconcile import check_true_up_against_solver, pnl, scenario_pnl, settle
from src.renegotiate import deliverable_guarantee
from src.risk import volume_scenarios


@pytest.fixture(scope="module")
def book():
    drugs = load_nadac_sample(n_drugs=60, seed=5)
    claims, clients, guarantees = build_book(drugs, n_clients=5, seed=5)
    return claims, mac_table(claims), guarantees, contract_terms(clients)


@pytest.fixture(scope="module")
def solved(book):
    claims, mac, guarantees, _ = book
    return optimize_mac(claims, mac, guarantees)


# ---------------------------------------------------------- contract is fixed

def test_optimizer_returns_no_billing_multipliers(solved):
    """V3's whole point: the contract is an input, so there is nothing to scale."""
    assert "scale" not in solved
    assert "scale_by" not in solved


def test_guarantees_are_never_mutated(book):
    claims, mac, guarantees, _ = book
    before = dict(guarantees)
    optimize_mac(claims, mac, guarantees)
    assert guarantees == before


# ------------------------------------------------- pricing respects the rules

def test_pharmacy_floor_and_change_limit(book, solved):
    _, mac, _, _ = book
    lo, hi, _ = price_bounds(mac)
    p = solved["prices"]["new_price"].values
    assert (p >= lo - 1e-9).all() and (p <= hi + 1e-9).all()


def test_channel_price_order(solved):
    wide = solved["prices"].pivot(index="drug_id", columns="channel", values="new_price")
    assert (wide["retail_90"] <= wide["retail_30"] + 1e-6).all()
    assert (wide["mail"] <= wide["retail_90"] + 1e-6).all()


# ------------------------------------------------------- feasibility, not guesswork

def test_feasibility_report_is_consistent_with_the_solver(book, solved):
    """True-up cannot be lower than the gap that no price list can close."""
    claims, mac, guarantees, terms = book
    fr = feasibility_report(mac, claims, guarantees)
    unreachable = fr["gap_$"][fr["status"] != "feasible"].sum()
    settled = settle(claims, solved["prices"], guarantees, terms)
    assert settled["true_up_$"].sum() >= unreachable - 1.0


def test_renegotiation_separates_cost_problems_from_stability_problems(book):
    """A pool short only because of the change limit must NOT be flagged for
    renegotiation: more cycles fix it, a new contract is not needed."""
    claims, mac, guarantees, _ = book
    fr = feasibility_report(mac, claims, guarantees)
    dg = deliverable_guarantee(mac, claims, guarantees)
    floor_bound = {(r.client, r.channel) for r in fr.itertuples()
                   if r.status == "infeasible: pharmacy floor above guarantee"}
    limit_bound = {(r.client, r.channel) for r in fr.itertuples()
                   if r.status == "unreachable this cycle: price-change limit"}
    assert {(r.client, r.channel) for r in dg.itertuples() if r.renegotiate} == floor_bound
    assert {(r.client, r.channel) for r in dg.itertuples()
            if r.needs_more_cycles} == limit_bound


def test_a_wider_change_limit_removes_stability_driven_shortfalls(book):
    """Proof that the shortfall is policy, not drug economics."""
    claims, mac, guarantees, _ = book
    tight = feasibility_report(mac, claims, guarantees, max_change=0.15)
    loose = feasibility_report(mac, claims, guarantees, max_change=0.50)
    assert loose["gap_$"].sum() <= tight["gap_$"].sum()


# --------------------------------------------- independent reconciliation

def test_reconciliation_matches_the_solver_true_up(book, solved):
    """The solver's own shortfall variables vs true-up rebuilt from claim lines."""
    claims, _, guarantees, terms = book
    assert check_true_up_against_solver(claims, solved["prices"], guarantees,
                                        terms, solved) < 1e-6


def test_reconciliation_catches_a_corrupted_price(book, solved):
    """The check must be able to FAIL; a tautological check is worthless."""
    claims, _, guarantees, terms = book
    tampered = {**solved, "prices": solved["prices"].copy()}
    tampered["prices"]["new_price"] = tampered["prices"]["new_price"] * 1.05
    assert check_true_up_against_solver(claims, tampered["prices"], guarantees,
                                        terms, tampered) > 1.0


def test_pass_through_identity_holds(book, solved):
    """Client ingredient charge equals pharmacy ingredient reimbursement."""
    claims, _, guarantees, terms = book
    s = settle(claims, solved["prices"], guarantees, terms)
    assert np.allclose(s["client_billed_$"] - s["admin"] - s["dispensing"], s["ingredient"])
    assert np.allclose(s["pharmacy_paid_$"], s["ingredient"] + s["dispensing"])


def test_contribution_margin_arithmetic(book, solved):
    claims, _, guarantees, terms = book
    s = settle(claims, solved["prices"], guarantees, terms)
    assert np.allclose(s["contribution_$"], s["admin"] - s["true_up_$"] - s["opcost"])


def test_overperformance_is_not_charged_against_margin(book, solved):
    """A deeper discount than promised is a client benefit, never a PBM expense."""
    claims, _, guarantees, terms = book
    s = settle(claims, solved["prices"], guarantees, terms)
    book_pnl = pnl(s)
    assert book_pnl["_guarantee_overperformance_$"] > 0  # it does happen here
    expected = (book_pnl["admin_fee_revenue_$"] - book_pnl["true_up_payments_$"]
                - book_pnl["operating_cost_$"])
    assert abs(book_pnl["contribution_margin_$"] - expected) < 1e-6


def test_true_up_and_overperformance_never_both_positive_in_a_pool(book, solved):
    claims, _, guarantees, terms = book
    s = settle(claims, solved["prices"], guarantees, terms)
    assert ((s["true_up_$"] > 1e-9) & (s["overperformance_$"] > 1e-9)).sum() == 0


# ------------------------------------------------------- optimizing under uncertainty

def test_scenario_pricing_beats_plan_only_out_of_sample(book):
    claims, mac, guarantees, terms = book
    clients = np.arange(claims["client_id"].nunique())
    train = volume_scenarios(mac, clients, 10, seed=2)
    test = volume_scenarios(mac, clients, 400, seed=77)
    plan = optimize_mac(claims, mac, guarantees)
    saa = optimize_mac(claims, mac, guarantees, scenarios=train)
    a = scenario_pnl(claims, plan["prices"], guarantees, terms, plan, test)
    b = scenario_pnl(claims, saa["prices"], guarantees, terms, saa, test)
    assert b["mean_true_up_$"] <= a["mean_true_up_$"]


def test_cvar_tail_is_large_enough_to_mean_something():
    """Guard the methodology: (1-alpha)*S must cover at least one scenario."""
    alpha, n_scen = 0.80, 15
    assert (1 - alpha) * n_scen >= 1


def test_more_scenarios_do_not_break_the_model(book):
    claims, mac, guarantees, _ = book
    clients = np.arange(claims["client_id"].nunique())
    sc = volume_scenarios(mac, clients, 8, seed=4)
    r = optimize_mac(claims, mac, guarantees, scenarios=sc, cvar_weight=0.5, cvar_alpha=0.75)
    assert r["termination"] == "optimal"
    assert r["n_scenarios"] == 8


# ------------------------------------------------------------------ labelling

def test_synthetic_class_is_labelled_as_synthetic(book):
    _, mac, _, _ = book
    assert "synthetic_class" in mac.columns
    assert "klass" not in mac.columns


def test_fees_are_separate_from_ingredient_cost(book, solved):
    claims, _, guarantees, terms = book
    s = settle(claims, solved["prices"], guarantees, terms)
    # The guarantee is measured on ingredient cost alone, so fees must not
    # appear in the allowed amount.
    assert np.allclose(s["allowed_ingredient"], (1 - s["guarantee"]) * s["awp"])
    assert (s["dispensing"] > 0).all() and (s["admin"] > 0).all()
