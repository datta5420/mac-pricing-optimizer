# MAC Pricing Optimizer

**Setting generic drug prices to meet a health plan client's pricing guarantees, using linear programming (Python + Gurobi).**

> Learning project, built September–November 2026 to understand PBM pricing-guarantee optimization.
> Part 2 uses **real** generic acquisition costs from the public CMS NADAC file. AWP, claim volumes, current prices and contract terms are **simulated** throughout. Nothing here reflects any real company's contracts or data.

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

Part 2 below takes on the first two of these.

---

# Part 2: one MAC list, many clients

Part 1 priced a single client. A real PBM pays pharmacies from **one MAC list**
that has to satisfy every client at once, each with its own drug mix and its own
guarantees. Part 2 rebuilds the problem that way, on **real acquisition costs**.

- **Costs are real:** 200 generic drugs from the CMS NADAC weekly file
  (`data/raw/nadac_sample.csv`, as of 2026-10-28) — real names, real
  `nadac_per_unit`, sampled across the cost range.
- **Still simulated:** AWP, claim volumes, current prices, client contracts.

**Decisions:** `p[drug, channel]`, the MAC price pharmacies are paid, and
`s[client, channel]`, the scale factor applied to that list to bill each client.

```bash
python run_book.py       # 200 drugs x 3 channels, 12 clients, 6,111 claim lines
```

### The joint problem is bilinear, so it is solved as two LPs

A client's spend is `sum(s * p * units)`, which multiplies two decision
variables. Rather than hand that to a nonlinear solver, the model alternates:

| step | held fixed | solved for | linear? |
|---|---|---|---|
| p-step | scale factors | the shared MAC list | yes |
| s-step | the MAC list | scale factors | yes |

Each step can only lower the same objective, which is bounded below by zero, so
it converges — here in **2–3 passes**. This finds a local optimum, not a proven
global one. The trade is deliberate: two exact LPs are easier to explain, debug
and hand over than one nonlinear model.

### Finding 1: one scale factor per client cannot work

| scale factors | guarantee pools missed | true-up owed | margin given away |
|---|---|---|---|
| one per client | **14 of 36** | $55,128 | $5,485 |
| one per client and channel | **0 of 36** | $0 | $0 |

A client has three channel guarantees. One factor is a single degree of freedom
aimed at three targets, so it lands on retail 30 and misses 90-day and mail.
Three factors per client hit all of them exactly. The cost is operational, not
mathematical: three times as many numbers to administer and audit.

### Finding 2: some contracts cannot be priced profitably at all

**6 of 12 clients are billed less than the pharmacies serving them are paid**,
$404,657 in total. Their guarantees are deeper than the cost of filling their
claims, so no MAC list fixes them. The model surfaces these as a
contract-negotiation finding instead of quietly absorbing the loss.

### Finding 3: exact-fit pricing is fragile, and the usual headline misleads

Volumes do not land on plan, and they do not move independently — a whole
therapeutic class shifts together. Scoring the chosen prices against 2,000
**correlated** volume scenarios (client x class x drug effects):

- **96.9%** of scenarios leave at least one pool owing a true-up
- average true-up **$125,358**, 95th percentile **$406,395**
- average margin given away **$129,594**

That 96.9% sounds alarming and mostly is not: with 36 pools each near a coin
flip, *something* almost always misses. The number that matters is dollars, and
they are concentrated — **one client carries 52% of the exposure**. Reporting
the probability alone would aim the team at the wrong problem.

### Finding 4: a cushion buys predictability at a poor exchange rate

![Cushion trade-off](outputs/cushion_tradeoff.png)

Pricing a little above each guarantee, then scoring against the real ones:

| cushion | chance of a true-up | avg true-up | 95th pct | margin given away |
|---|---|---|---|---|
| 0 | 97% | $125,358 | $406,395 | $129,594 |
| 0.25 pt | 48% | $14,389 | $89,176 | $481,379 |
| 0.50 pt | 6% | $578 | $566 | $929,793 |
| 0.75 pt | 0% | $24 | $0 | $1,392,054 |
| 1.00 pt | 0% | $0 | $0 | $1,854,378 |

A quarter-point cushion removes **$111K** of expected true-up and costs
**$352K** of extra margin — roughly **$3 given up per $1 of exposure avoided**.
Aiming the cushion only at the riskiest pools did not help either: the exposure
sits with the *largest* client, and a cushion costs in proportion to a client's
spend while its benefit scales with the variance of that client's gap, which is
proportionally smaller for a big book. Cushions are worst value exactly where
the dollars are.

**What that argues for:** better volume forecasting on the few large pools, and
renegotiating the structurally unprofitable contracts — not a blanket cushion.
Whether to buy a cushion anyway is a business judgment about how much a surprise
true-up costs beyond its dollar amount. The table gives that conversation
numbers instead of adjectives.

### What Part 2 still leaves out

- **Mid-year re-pricing.** Guarantees run over a contract year; by month 7 the
  first 7 months are locked and only the rest can be steered. A year-to-date
  model is the natural next step.
- **Brand drugs, rebates, dispensing fees and specialty**, all of which sit in
  real contracts.
- **"Lesser of" logic** — many cheap generics actually pay at the pharmacy's
  cash price, which changes the effective rate.
- **Pharmacy-side guarantees** and per-network MAC lists (chain vs. independent).
- **Scale is capped by the licence.** The pip build of Gurobi is size-limited,
  so the book is 200 drugs and 12 clients rather than thousands.

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
src/config.py      business assumptions (guarantees, limits, weights)
src/data.py        NADAC loader + simulated AWP, claims, current prices
src/model.py       Part 1: Gurobi LP for one client
src/report.py      channel summary, trade-off sweep, chart
run.py             Part 1 pipeline

src/book.py        Part 2: real NADAC costs + a book of clients
src/model_multi.py Part 2: shared MAC list + per-client scale factors (two LPs)
src/risk.py        correlated volume scenarios, cushion sweep, chart
run_book.py        Part 2 pipeline

data/raw/          real CMS NADAC sample (2026-10-28)
tests/             pytest checks on both models' guarantees and constraints
.github/           CI: tests run on every push
```

### Run it

```bash
pip install -r requirements.txt
python run.py                              # Part 1: one client, simulated costs
python run.py --nadac data/raw/nadac.csv   # Part 1 with a full CMS NADAC download
python run_book.py                         # Part 2: 12 clients, real NADAC costs
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
