# Simulation

!!! note "v0.1.0 is in development"

    This page documents what is implemented today: ELT ingest and validation,
    frequency sampling, and year event loss table generation. Secondary
    uncertainty, dependence and the financial module are not yet built, and
    `simulate(..., secondary_uncertainty=True)` raises `NotImplementedError`
    rather than quietly returning something else.

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

::: catagg.simulate.engine
    options:
      members: [simulate, to_ylt]

::: catagg.metrics.analytic

::: catagg.examples
