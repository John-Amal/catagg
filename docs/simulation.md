# Simulation

!!! note "v0.1.0 is in development"

    This page documents what is implemented today: ELT ingest and validation,
    frequency sampling, secondary uncertainty, and year event loss table
    generation. Dependence and the financial module are not yet built.

## The objects

| Object | One row per | Columns |
|---|---|---|
| **ELT** | modelled event | `event_id`, `rate`, `mean_loss`, `std_loss`, `exposure_value`, `peril`, `region` |
| **YELT** | event occurrence, per simulated year | `sim_year`, `event_id`, `occurrence_index`, `gross_loss` |
| **YLT** | simulated year | `sim_year`, `aggregate_loss`, `max_occurrence_loss` |

```python
import catagg as ca

elt = ca.swiss_multiperil_elt(n_events=1_000, seed=0)
yelt = ca.simulate(elt, n_years=100_000, frequency="poisson", seed=42)
ylt = ca.to_ylt(yelt)
```

## Validation is part of ingest

`validate_elt` collects every failure before raising, so a malformed table is
fixed in one pass rather than one error at a time. Beyond the obvious range
checks it enforces one bound worth stating explicitly:

$$\sigma_i^2 \le \mu_i \,(E_i - \mu_i)$$

For any loss distribution supported on $[0, E_i]$ with mean $\mu_i$, the variance
is maximised by the two-point distribution placing all its mass at the
endpoints. A row exceeding that bound describes a distribution which cannot
exist, so no severity model can honour it. Catching it at ingest is what stops
the severity sampler from silently clipping the moments it was asked to
reproduce.

## Frequency is modelled at the group level

Both frequency models act on the annual count of a **group** of events —
by default one group per peril — rather than on each event separately. The
occurrences are then allocated to individual events in proportion to their
rates.

This matters for the negative binomial. Applied per event with dispersion $k$,
an event's count variance is $\lambda_i + \lambda_i^2/k$. Real event rates are
of order $10^{-3}$, so the quadratic term is a millionth of the linear one and
the portfolio is indistinguishable from Poisson: the over-dispersion has been
specified and then thrown away.

Applied to the group total, the marginal rates stay exact, since
$\mathbb{E}[N_i] = \mathbb{E}[N]\,p_i = \lambda_i$, while the shared draw of $N$
induces

$$\operatorname{Cov}(N_i, N_j) = \frac{\lambda_i \lambda_j}{k}$$

between any two events in the group. That covariance *is* the clustering.

The Poisson case needs no special handling: multinomial thinning of a Poisson
total yields exactly independent $\text{Poisson}(\lambda_i)$ per-event counts, so
one code path serves both models.

## Secondary uncertainty is one family, read at three points

With `secondary_uncertainty=True`, loss given occurrence is drawn from a
method-of-moments fit on $[0, E_i]$ rather than fixed at $\mu_i$. Writing
$m = \mu_i / E_i$ and $s = \sigma_i / E_i$ for the damage ratio and its
standard deviation:

$$\nu = \frac{m\,(1 - m)}{s^2} - 1, \qquad \alpha = m\,\nu, \qquad \beta = (1 - m)\,\nu$$

$\nu$ is the concentration, and it is non-negative exactly when
$s^2 \le m\,(1 - m)$ — the feasibility bound enforced at ingest. Because
validation runs first, the fit never has to clip a moment it was asked to
reproduce.

The probability mass at zero and at full exposure is **not a pair of free
parameters bolted onto the beta**. It is the boundary member of the same
family. As $\nu \to 0$, the law $\text{Beta}(m\nu, (1-m)\nu)$ converges to the
Bernoulli law placing $1 - m$ at $0$ and $m$ at $1$, which is the unique
distribution on $[0, 1]$ attaining the feasibility bound. So `SeverityRegime`
reads one continuous family at three points:

| Regime | When | Fit |
|---|---|---|
| `DEGENERATE` | $\sigma_i = 0$ | All mass at $\mu_i$ |
| `BETA` | $0 < s^2 < m\,(1-m)$ | Beta on $[0, E_i]$, continuous, no atom |
| `TWO_POINT` | $s^2 = m\,(1-m)$ | Mass $1 - m$ at $0$ and $m$ at $E_i$ |

This is a deliberate choice and worth stating plainly. A beta *plus* a free
mass at each endpoint is the more familiar picture of secondary uncertainty,
and it is what a location-level damage distribution really looks like. But an
ELT row carries two moments, and two moments cannot determine four parameters.
Any interior mass would come from a convention invented here rather than from
the data, and it would propagate silently into every tail metric downstream.
The two-parameter fit is exactly determined, reproduces both moments to machine
precision in every regime, and still puts real mass on the endpoints wherever
the moments demand it. Free endpoint masses belong with a richer input schema,
not with this one.

Two numerical points fall out of this, and both are tested.

A row sitting exactly on the feasibility bound computes $\nu$ as a few
multiples of machine epsilon, **of either sign**. Testing $\nu > 0$ would
therefore label identical rows differently on the last bit, and a fit labelled
`BETA` reports no endpoint mass for a distribution that in practice produces
nothing else. The regime boundary is instead tested against the same relative
slack `validate_elt` allows on the bound, so the two cannot drift apart.

A `std_loss` that is positive but vanishing against `exposure_value` underflows
the scaled variance and sends $\nu$ to infinity. That is a `DEGENERATE` fit, not
a `TWO_POINT` one: routing it to the endpoints would replace a variance of
nearly zero with the largest the support allows.

The AAL is unchanged by the flag, since the severity mean is $\mu_i$ either way.
What changes is everything above the first moment: the variance of the annual
aggregate loss gains $\sum_i \lambda_i \sigma_i^2$, and
`analytic_annual_loss_variance(..., secondary_uncertainty=True)` is the closed
form the simulation is checked against.

## Empty years are not optional

A year in which nothing happened contributes no rows to the YELT, so the
simulated year count cannot be recovered from the table and travels in
`yelt.attrs["n_years"]`. `to_ylt` reindexes to every year in
`range(n_years)`.

Collapsing a YELT with a bare `groupby("sim_year")` instead divides by the
number of *non-empty* years and inflates the AAL by roughly

$$\frac{1}{1 - e^{-\Lambda}}$$

for a Poisson portfolio with total rate $\Lambda$. There is a test asserting
exactly that ratio against the naive computation, so the bug cannot return
unnoticed.

## Performance

The year loop is fully vectorised: counts are drawn per group in one call, and
occurrences are assigned by inverse-CDF sampling against the cumulative rate
vector. Uniforms are drawn in chunks to bound transient memory, and because
every year's count is drawn before any uniform, results are identical for any
`chunk_years`.

## API

::: catagg.io.elt
    options:
      members: [validate_elt, read_elt]

::: catagg.frequency.models

::: catagg.severity.beta

::: catagg.simulate.engine
    options:
      members: [simulate, to_ylt]

::: catagg.metrics.analytic

::: catagg.examples
