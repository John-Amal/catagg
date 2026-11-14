# Validation

!!! note "Placeholder"

    This page is a stub. It is filled by the validation notebook built at the
    end of the v0.1.0 milestone; no results are published here yet.

The suite this page will report on, each check running in CI:

| Check | What it proves | Status |
|---|---|---|
| Simulated AAL against analytic `Σ λᵢ · μᵢ` | The frequency–severity loop is unbiased | In CI |
| Sampled frequency moments against Poisson / negative binomial | The sampler matches the specified distribution | In CI |
| Annual loss variance against its closed form | Clustering is present at the group level, not lost per event | In CI |
| Sampled severity moments against the ELT input | Method-of-moments beta fitting recovers its own inputs | In CI |
| Ceded loss against the aggregate reinstatement cap | Reinstatement accounting never over-recovers, on a portfolio where the cap demonstrably binds | In CI |
| Net + ceded = gross, every occurrence and every year | The financial module conserves loss, to within one unit in the last place per occurrence | In CI |
| Simulated layer AAL against analytic `Σ λᵢ · clip(yᵢ − A, 0, L)` | The layer responds to the right retained loss, behind a quota share | In CI |
| Annual ceded and net totals under reordered occurrences | Occurrence order moves cover between events, never in total | In CI |
| Tail-metric standard error against simulation count | How noisy a given return period estimate actually is | In CI |
| Reported standard error against replication | The error the library reports predicts the spread across independent reruns | In CI |

Tolerances in the implemented checks are analytic, not hand-picked: a mean of
`n` draws is compared against its own standard error `sqrt(Var / n)`, computed
in closed form from the ELT and the frequency model. A fixed percentage
tolerance would pass a biased engine on a fat-tailed portfolio and fail a
correct one on a thin portfolio.

The severity check applies the same rule at the second moment: the sample
variance of `n` draws is compared against `sqrt((mu_4 - sigma**4) / n)`, with
the fourth central moment taken from the fit's own closed-form moments. Each
assertion also carries a guard on its own detection power, so a run too short
to prove anything fails rather than passing quietly.
