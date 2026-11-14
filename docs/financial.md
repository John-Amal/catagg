# Financial structures

!!! note "v0.1.0 is in development"

    This page documents loss-side reinsurance mechanics. Premium, ceding
    commission and reinstatement premium are not yet modelled.

A `Programme` applies reinsurance structures to a simulated YELT and returns
gross, ceded and net loss on every occurrence.

```python
import catagg as ca

elt = ca.swiss_multiperil_elt(n_events=1_000, seed=0)
yelt = ca.simulate(elt, n_years=100_000, secondary_uncertainty=True, seed=42)

programme = ca.Programme([
    ca.QuotaShare(cession=0.20),
    ca.ExcessOfLoss(attachment=50e6, limit=50e6, reinstatements=2),
    ca.AggregateStopLoss(attachment=100e6, limit=100e6),
])
result = programme.apply(yelt)

gross = ca.ep_curve(result.yelt, kind="OEP")
net = ca.ep_curve(result.yelt, kind="OEP", column="net_loss")
```

`result.yelt` holds `gross_loss`, `ceded_loss` and `net_loss` side by side, and
`result.ceded_by_structure` holds each structure's own cession, labelled by
position: `1_quota_share`, `2_excess_of_loss`, `3_aggregate_stop_loss`.

## Programme order is inuring order

Each structure protects what the structures before it left retained. In the
programme above the quota share takes 20% of every gross loss; the 50m xs 50m
layer then responds to the retained 80%, not to the gross; and the stop-loss
responds to what the layer left. Reversing two structures changes the answer,
so the order in the list is the order of application.

## Structures

| Structure | Pays, per occurrence | Running total |
|---|---|---|
| `QuotaShare(cession)` | `cession × retained` | None |
| `ExcessOfLoss(attachment, limit, reinstatements)` | Retained loss above `attachment`, up to `limit` | Annual cap `limit × (1 + reinstatements)`, or none if `reinstatements=None` |
| `AggregateStopLoss(attachment, limit)` | Its share of the annual recovery above `attachment` | Annual deductible, then annual `limit` |

`reinstatements` has no default. Market terms vary too much for any default to
be safe, and a silently assumed one would change every net figure.

`AggregateStopLoss` takes its attachment and limit in money, which also makes
it an aggregate excess of loss. A stop-loss quoted as a loss ratio converts to
this form given the subject premium.

## Order within the year

The per-occurrence layer's annual cap and the stop-loss's deductible and limit
are consumed in chronological order, using `occurrence_index` (see
[Simulation](simulation.md#occurrence-order-is-chronological)). A 50 xs 50
layer with two reinstatements pays at most 150 a year. Given layer losses of
30, 50, 50 and 40 in that order, the fourth recovers only the 20 left. In the
reverse order it is the 30 that is cut, to 10.

Order never changes a structure's annual total, so AAL and AEP of ceded and net
loss are unaffected by it. It does change which occurrence is cut off, and so
the net loss per occurrence and the **net OEP**. Allocating the stop-loss
recovery the same way is what gives a net OEP after an aggregate cover a
definite meaning.

## Conservation, in floating point

Net loss is what remains after each structure's cession is taken off in turn,
and `ceded_loss` is computed once, as `gross_loss − net_loss`. With that
construction `net_loss + ceded_loss` equals `gross_loss` to within one unit in
the last place on every occurrence: rounding one subtraction moves the result
by at most half a unit, and adding back at most doubles it.

That is the strongest form the identity can take in floating point. A bitwise
equality is not attainable: for a 20% quota share alone, a few percent of
occurrences differ from gross by one unit in the last place, whatever order the
arithmetic is done in.

`Programme.apply` also checks every structure's cession, including structures
written outside the library against the `Structure` protocol: none may be
negative, and none may exceed the loss it was offered. So net loss can never go
negative.

## API

::: catagg.financial.structures

::: catagg.financial.programme
