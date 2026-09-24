# MAC Pricing Optimizer

**Setting generic drug prices to meet a health plan client's pricing guarantees, using linear programming (Python + Gurobi).**

> Learning project, built September 2026 to understand PBM pricing-guarantee optimization.
> Drug acquisition costs can come from public CMS NADAC data. AWP, claim volumes, current prices and contract terms are **simulated**. Nothing here reflects any real company's contracts or data.

---

## The business problem

A pharmacy benefit manager (PBM) serving a reseller health plan client promises **AWP discount guarantees**: for example, "generics at retail 30-day pharmacies will cost, on average, AWP minus 80%." Each pharmacy channel (retail 30-day, retail 90-day, mail) has its own target.

The PBM controls this through **MAC prices**, the maximum it pays per unit of each generic drug. Every pricing cycle, those prices have to be set so that:

- **The guarantee is met.** Missing it means paying the client a *true-up* for the shortfall.
- **It is not beaten by much.** Over-delivering gives away margin for nothing.
- **Pharmacies are paid at least their acquisition cost**, or the network suffers.
- **Prices don't swing wildly** between cycles.
- **Channels stay consistent.** A unit shouldn't cost more at mail order than at a retail pharmacy.

With hundreds of drugs across several channels, this is a constrained optimization problem.

## Results (200 simulated drugs, 3 channels)

Starting position with current prices:

| Channel | Guarantee | Current discount | Position |
|---|---|---|---|
| Retail 30 | 80.0% | 78.0% | **$78K true-up owed** |
| Retail 90 | 84.0% | 84.2% | $6K margin given away |
| Mail | 86.0% | 85.3% | **$12K true-up owed** |

After optimization with a ±15% price-change limit, **all three channels hit their guarantee exactly, with $0 true-up and $0 margin given away** on the planned volumes (see Limitations for what happens when volumes shift). This moved 101 of 600 prices. 10 prices had to be raised beyond the limit because they were below pharmacy acquisition cost.

### The key trade-off: price stability costs money

![Trade-off chart](outputs/tradeoff.png)

| Max price change per cycle | True-up owed | Prices changed |
|---|---|---|
| 15% | $0 | 101 |
| 10% | $0 | 228 |
| 7% | $17,934 | 342 |
| 5% | $37,964 | 412 |
| 2% | $70,872 | 454 |

**Takeaway:** down to about a 10% change limit, the guarantees can still be met at no cost. Below that, every point of added stability has a measurable dollar cost. This gives a business stakeholder a concrete basis for choosing the limit.

## Limitations (what a real system would add)

- Real AWP, utilization and contract terms. Real contracts also include brand guarantees, dispensing fees and rebates.
- **Utilization uncertainty.** Hitting each guarantee exactly is fragile. In a quick test, I let each drug's volume move randomly by about ±15% after repricing, and the "optimal" prices still owed a small true-up about 87% of the time. A next step is to optimize across several demand scenarios, or target a small cushion above the guarantee, and show what that safety costs in margin.
- Pharmacy-side network contracts, which have their own guarantees.
- Member cost-share impact.
- Thousands of drugs. The pip version of Gurobi has a size-limited license, so this demo uses 200.

---

## Technical details

### Formulation

For each drug *i* and channel *c*:

**Decision variable:** `p[i,c]`, the new unit price.

**Data:** `q` = units, `awp` = AWP per unit, `nadac` = acquisition cost, `p0` = current price, `g[c]` = guarantee.

**Objective:** minimize total dollar cost.

```
minimize   Σc shortfall[c] + Σc surplus[c] + 0.01 · Σ q · |p - p0|
```

The first two terms are real money. The third is a small tie-breaker that prefers fewer, smaller price changes.

**Constraints:**

| Constraint | Formula | Why |
|---|---|---|
| Guarantee | `Σi p·q − (1−g[c])·Σi awp·q = shortfall[c] − surplus[c]` | Measures the miss in dollars. Linear because the effective discount is total paid ÷ total AWP. |
| Pharmacy floor | `p ≥ nadac · 1.05` | Never pay below cost |
| Stability | `p0·(1−δ) ≤ p ≤ p0·(1+δ)` | Limit price swings (the floor wins if they conflict) |
| Channel order | `p[mail] ≤ p[retail_90] ≤ p[retail_30]` | Longer supply shouldn't cost more per unit |
| Absolute value | `p − p0 = up − down`, with `up, down ≥ 0` | Keeps `\|p − p0\|` linear |

Size: 1,806 variables and 1,003 constraints. It solves in well under a second.

If the model is infeasible, it computes an IIS (Irreducible Inconsistent Subsystem) and reports which constraints conflict.

### Project structure

```
src/config.py    business assumptions (guarantees, limits, weights)
src/data.py      NADAC loader + simulated AWP, claims, current prices
src/model.py     Gurobi LP
src/report.py    channel summary, trade-off sweep, chart
run.py           end-to-end pipeline
tests/           pytest checks on the model's guarantees and constraints
.github/         CI: tests run on every push
```

### Run it

```bash
pip install -r requirements.txt
python run.py                              # simulated drug costs
python run.py --nadac data/raw/nadac.csv   # real CMS acquisition costs
pytest -q
```

To use real costs, download a NADAC CSV from data.medicaid.gov (search "NADAC") and save it as `data/raw/nadac.csv`. The loader keeps generic drugs and the latest price per NDC.

### Tests

The tests check that the model:
- hits every guarantee when limits are loose,
- never prices below the pharmacy floor,
- respects the price-change limit (except floor-forced rows),
- keeps channel price order,
- never gets cheaper as limits tighten.
