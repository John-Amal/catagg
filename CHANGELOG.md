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

[Unreleased]: https://github.com/John-Amal/catagg/commits/main
