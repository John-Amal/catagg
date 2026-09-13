# Validation

!!! note "Placeholder"

    This page is a stub. It is filled by the validation notebook built at the
    end of the v0.1.0 milestone; no results are published here yet.

The suite this page will report on, each check running in CI:

| Check | What it proves |
|---|---|
| Simulated AAL against analytic `Σ λᵢ · μᵢ` | The frequency–severity loop is unbiased |
| Sampled frequency moments against Poisson / negative binomial | The sampler matches the specified distribution |
| Sampled severity moments against the ELT input | Method-of-moments beta fitting recovers its own inputs |
| Ceded loss against the aggregate reinstatement cap | Reinstatement accounting never over-recovers |
| Net + ceded = gross, every simulated year | The financial module conserves loss exactly |
| Tail-metric standard error against simulation count | How noisy a given return period estimate actually is |
