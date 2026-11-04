"""Skip the solver tests when Gurobi cannot actually solve anything.

`pytest.importorskip("gurobipy")` is not enough on its own: the package imports
fine without a licence, and only fails later at `gp.Model()`. This builds a
one-variable model up front and skips the suite cleanly if that does not work,
so a reviewer without a licence sees skips rather than a wall of errors.
"""
import pytest


def _gurobi_usable():
    try:
        import gurobipy as gp
        m = gp.Model()
        m.Params.OutputFlag = 0
        m.addVar()
        m.optimize()
        m.dispose()
        return True, ""
    except Exception as exc:  # no package, no licence, size-limited model
        return False, str(exc).strip().splitlines()[0] if str(exc) else type(exc).__name__


GUROBI_OK, GUROBI_WHY = _gurobi_usable()


def pytest_collection_modifyitems(config, items):
    if GUROBI_OK:
        return
    skip = pytest.mark.skip(reason=f"Gurobi unusable here: {GUROBI_WHY}")
    for item in items:
        item.add_marker(skip)
