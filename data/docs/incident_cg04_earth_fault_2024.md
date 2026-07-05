# Incident Report: Vessel CG-04 Switchboard Earth Fault, 2024-03-15

## Summary

On 2024-03-15 at 0247 hrs, vessel CG-04 reported a sustained earth fault alarm on the 440V main switchboard. The fault persisted for 38 minutes until isolated. No injuries; no loss of propulsion. Investigation completed 2024-03-22.

## Sequence of Events

- 0247: Earth fault alarm activated at main switchboard. Engineering watch acknowledged.
- 0249: Watch attempted standard isolation procedure by sequentially opening feeder breakers.
- 0254: Earth fault persisted through all feeder isolations. Fault localized to main bus.
- 0301: Engineering officer woken; arrived at switchboard at 0307.
- 0315: Fault traced to main switchboard cable gland penetration at the deckhead.
- 0325: Cable insulation visually confirmed compromised by long-term salt water ingress through deck seal.
- 0325: Faulty cable isolated by opening upstream incomer. Earth fault cleared.

## Root Cause

The cable in question was the 440V supply to the auxiliary winch motor. The cable's deck penetration gland had degraded over time and admitted seawater spray during heavy weather. Salt deposits accumulated on the cable insulation inside the deckhead, and the combination of moisture and salt eventually compromised insulation resistance to a level where leakage to ground became continuous.

Contributing factors:
- Cable gland was last inspected during the 2021 dry-dock. Inspection interval per current SOP is 5 years. The SOP interval is inadequate for cables routed through high-spray deck zones.
- The earth fault detection system did not localize the fault automatically — it only flagged that one existed somewhere on the 440V bus. The vessel does not have feeder-level earth fault discrimination.

## Lessons Learned

1. Deck gland inspection interval for cables in spray zones should be shortened to 18 months. A vessel-wide audit of cable gland conditions should be completed before next sea trial.
2. Consider installing feeder-level earth fault detection on next major refit. Estimated cost ~SGD 45,000 per vessel; trades off against reduced fault localization time.
3. Engineering watch standing orders should be amended: if an earth fault does not clear after one round of feeder isolation, escalate to engineering officer immediately rather than attempting further isolation.

## Affected Equipment

- 440V main switchboard, vessel CG-04
- Auxiliary winch motor supply cable, P/N CBL-WNCH-440-04
- Deck penetration gland, P/N GLD-DPN-25

## Status

- Cable replaced in dry-dock 2024-04-02.
- Gland replaced with upgraded P/N GLD-DPN-25-MK2 (improved seal geometry).
- Fleet-wide gland audit complete on 2024-05-30. Three additional vessels (CG-02, CG-07, CG-11) required preventive cable replacements.
- SOP amendment ratified and effective 2024-06-01.
