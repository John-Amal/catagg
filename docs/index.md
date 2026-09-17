# catagg

Catastrophe loss aggregation, from event loss tables to net exceedance
probability curves.

!!! warning "Pre-release"

    v0.1.0 is under development. The public API described in the README is the
    specification being built against, not yet the shipped surface.

## Install

```bash
pip install catagg
```

Python 3.11+. Runtime dependencies are numpy, pandas, pyarrow and scipy.

## Where to start

- **[Simulation](simulation.md)** — the ELT schema, frequency models and the year loop.
- **[Validation](validation.md)** — the checks that show the numbers can be trusted.
- **[API reference](api.md)** — the public surface.
- **[Changelog](changelog.md)** — what changed, and when.
