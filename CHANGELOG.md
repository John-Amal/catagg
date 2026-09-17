# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Repository scaffolding generated from the shared template: hatchling build
  with src layout, ruff and mypy configuration, pytest setup, GitHub Actions
  CI (lint, type-check, five-way test matrix, docs build, distribution build),
  MkDocs Material documentation site, MIT licence, issue and pull request
  templates.
- Empty subpackages fixing the module layout: `io`, `frequency`, `severity`,
  `dependence`, `simulate`, `financial`, `metrics`.
- Packaging tests: version/metadata drift and `py.typed` installation.
- Release workflow publishing to PyPI via Trusted Publishing, with a
  tag-versus-package-version guard, and the release procedure documented in
  `docs/releasing.md`.
- ELT schema with `validate_elt`, which reports every failed check at once and
  enforces the feasibility bound `std_loss**2 <= mean_loss * (exposure_value -
  mean_loss)`; `read_elt` for Parquet and CSV.
- Frequency models `Poisson` and `NegativeBinomial`, applied to a group's annual
  event count rather than per event, so that specified over-dispersion survives
  into the aggregate.
- `simulate`, generating a year event loss table by rate-proportional allocation
  of each group's annual occurrences; `to_ylt`, collapsing it to one row per
  simulated year including years with no events.
- `write_yelt` and `read_yelt`, round-tripping the simulated year count through
  Parquet file metadata.
- Closed-form references `analytic_aal`, `analytic_annual_loss_variance` and
  `aal_standard_error`, used as the tolerances in the convergence tests.
- `swiss_multiperil_elt`, a documented synthetic generator so the repository
  runs from a clean clone with no data dependencies.
- `BetaSeverity` and `SeverityRegime` in `catagg.severity`: a method-of-moments
  beta fit on `[0, exposure_value]` with probability mass at the endpoints,
  reproducing each event's mean and standard deviation to machine precision.
  The endpoint masses are the boundary member of the beta family rather than
  free parameters, since an ELT row carries two moments and cannot determine
  four; see the simulation page for why.
- `simulate(..., secondary_uncertainty=True)`, drawing loss given occurrence
  from that fit. Severity uses a reserved child seed, so switching the flag on
  leaves the frequency and allocation streams bit-identical. The flag is a
  no-op, exactly, on an ELT whose variances are all zero.
- Severity validation: per-event moment recovery against Monte Carlo standard
  errors derived from the fit's own fourth moment, regime assignment at and
  below the feasibility bound, and end-to-end agreement between the simulated
  annual loss variance and `analytic_annual_loss_variance(...,
  secondary_uncertainty=True)`.
- Simulation documentation page.

[Unreleased]: https://github.com/John-Amal/catagg/commits/main
