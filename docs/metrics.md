# Metrics

!!! note "v0.1.0 is in development"

    This page documents the gross view: EP curves, tail metrics and their Monte
    Carlo error. The financial module that produces ceded and net curves is not
    yet built.

An exceedance probability curve is the empirical distribution of a per-year
loss, read from the tail. Two of them matter, and they answer different
questions.

**OEP**, occurrence exceedance probability, is the distribution of the largest
single event in a year. It is what a per-occurrence layer responds to.
**AEP**, aggregate exceedance probability, is the distribution of total annual
loss. It is what aggregate cover and capital respond to. Both come from the
same year loss table, so one simulation gives you both.

```python
import catagg as ca

elt = ca.swiss_multiperil_elt(n_events=400, seed=0)
yelt = ca.simulate(elt, n_years=50_000, seed=1, secondary_uncertainty=True)

aggregate = ca.ep_curve(yelt, kind="AEP")
print(aggregate.at_return_period([100, 200, 250]))
print(aggregate.tvar(200))
```

`EPCurve` keeps the whole per-year loss vector rather than a summarised curve,
so any return period, tail metric or standard error can be read afterwards
without re-simulating.

## Return periods, and which plotting position

A return period $R$ is the reciprocal of an annual exceedance probability, so
the loss at $R$ is the $1 - 1/R$ quantile of the annual loss. The quantile uses
the **Weibull plotting position**, $k / (n + 1)$ for the $k$-th largest of $n$
simulated years.

That is not a cosmetic choice. For a continuous distribution $k / (n + 1)$ is
unbiased for the exceedance probability of the $k$-th order statistic. The
$k / n$ convention some vendor tools use is biased, and it cannot represent the
largest simulated year at all without assigning it a zero exceedance
probability — an infinite return period from one observation.

## Refusing to extrapolate

`at_return_period` raises rather than answering when the return period is longer
than the simulation:

```pycon
>>> curve = ca.ep_curve(ca.simulate(elt, n_years=10_000, seed=1), kind="AEP")
>>> curve.at_return_period(10_000)
CataggError: return_period 10000 exceeds the 10000 simulated years...
```

Reading a 1-in-10,000 loss off 10,000 simulated years returns the single largest
year and dresses one observation up as a return period. The library will not do
it quietly.

## TVaR

Tail value at risk is the mean loss in the years that exceed a return period, so
it answers what capital regimes actually ask: not "how bad is the 1-in-200 year"
but "how bad is the average year worse than that". It is computed as the
integrated expected shortfall over the largest $n / R$ years, with a fractional
weight on the boundary year so the figure stays continuous in $R$ rather than
jumping as the count crosses an integer.

## Standard error is not optional

Every figure this module returns carries the Monte Carlo standard error of the
estimate, because a return period read off a simulation is a statistic and
carries sampling error like any other. For the example portfolio the RP-200 AEP
estimated from 10,000 simulation years has a standard error of around 8% —
large enough to move a reinsurance decision, and almost never reported.

Neither estimator introduces a tuning constant, and both are checked against a
replication study rather than asserted.

### Quantiles

The empirical CDF at the quantile has standard deviation
$\sqrt{p(1-p)/n}$. Perturbing the probability level by exactly that amount and
reading the quantile function at both ends gives the sampling spread of the
estimate, mapped through the distribution's own shape:

$$\text{se} = \frac{Q(u + h) - Q(u - h)}{2}, \qquad u = 1 - p, \quad h = \sqrt{\frac{p\,(1-p)}{n}}$$

This is the Siddiqui density-quantile estimator with the bandwidth pinned to the
statistically natural scale rather than chosen. It needs no density estimate and
degrades correctly in the far tail, where the spacing between order statistics
is wide.

### TVaR, and the mistake worth naming

The tail mean over $k = np$ years is a mean, so the obvious estimator of its
error is $s_{\text{tail}} / \sqrt{k}$.

That estimator is wrong, and wrong in the dangerous direction. It treats the
exceedance threshold as known when the threshold is itself estimated from the
same sample. Against a replication study it **understates the true error by
between a fifth and a third**. The correction restores what it omits:

$$\text{se} = \sqrt{\frac{s_{\text{tail}}^2 + (1 - p)\,(\text{TVaR} - \text{VaR})^2}{k}}$$

### How well they do

Measured against 30 independent reruns of the same portfolio, as the ratio of
the error a single run reports to the spread actually observed across runs:

| Simulated years | Quantile | TVaR |
|---|---|---|
| 10,000 | 1.14 | 1.15 |
| 100,000 | 0.98 | 0.99 |

Both run slightly conservative at short simulations, which is the right
direction for a number whose purpose is to stop someone over-reading a tail.
The tests assert this ratio directly, so a change that quietly breaks an error
estimate fails rather than shipping a comfortable, wrong number.

## Plotting

matplotlib is **not** a runtime dependency. The engine needs numpy, pandas,
pyarrow and scipy, and a library that forces a plotting stack on everyone who
imports it has made a decision on their behalf. Install the extra:

```bash
pip install catagg[plot]
```

```python
from catagg.plot import ep_comparison

ep_comparison({"Gross": aggregate}, title="Swiss multi-peril, gross")
```

The shaded band is plus or minus one standard error, drawn by default. At the
long return periods on the right of the plot it is wide enough to change a
decision, and a curve drawn as a clean line hides that.

## API

::: catagg.metrics.ep

::: catagg.plot
