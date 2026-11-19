[![Bounty](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/mietrea519-cyber/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/mietrea519-cyber/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecast

A program that forecasts every Grand Theft Auto VI PlayStation 5 trophy's
earn rate on Exophase at 2026-11-26T00:00:00Z, as five quantiles per trophy
(5%, 25%, 50%, 75%, 95%). Entry for the make.fun bounty
[gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` (Python 3.13, standard library only) reads every `*.in` case in the
input directory and writes `<id>.out` per case: `{"trophies": [{"id": n,
"q": [q05, q25, q50, q75, q95]}, ...]}`. No network, no randomness: two runs
on the same case write byte-identical output.

## How the forecast works

Exophase refreshes a game's counts on its own schedule, not with every
player's trophy, so a page can sit unchanged for hours and then jump: on
launch night the tracked base went from 638 to 3,868 players in one refresh
while almost every earn rate fell. The forecaster is built around that:

- **Distinct states.** Consecutive identical reads collapse into one state,
  so a stale page counts once, not once per read.
- **Trend.** Weighted least squares of logit(rate) on log-time, recency
  weighted (24 h half-life), slopes shrunk toward the cross-trophy mean.
- **Earners / players.** Player counts are extrapolated on the log-time
  axis; each trophy's earner count grows with an elasticity fitted from
  ln(earners) on ln(players), so the rate falls mechanically when the
  tracked base broadens faster than earners grow.
- **Stale-page dilution prior.** When the page never moved, the trend and
  elasticity are degenerate, so the median applies a dilution prior: a
  just-released game with a tiny tracked base multiplies its players
  several-fold over the horizon (about 6x on launch night), and with an
  elasticity near 0.3 the median rate is roughly the last rate times 0.3.
- **Sum calibration.** One bounded common logit shift aligns the sum of the
  medians with the extrapolated total trophies-earned per player.
- **Spread.** A random-walk term in horizon days plus fit residual and a
  staleness term for a page that never moved, so the 90% band stays honest
  across a refresh the history has not seen yet. When only one refresh
  exists in the whole history and the tracked base barely moved, half of
  the stale dilution prior is blended in rather than trusting two-point
  fits.

## Practice check

On the kit's practice cases (p00, p01), run locally with
`MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out`, the output is
shape-valid, deterministic across runs (byte-identical), and under the
refreshed-page answer implied by the launch-night refresh (earners roughly
unchanged, players 638 → 3,868) it scores a mean pinball loss around
0.27–0.33, against about 1.51 for the house persistence baseline. The
final case is graded by the server; there is no local scorer for it.

## License

MIT — see [LICENSE](LICENSE).

---

Built for [make.fun](https://make.fun/?ref=1e3qpbvy) — enter the bounty and
follow the leaderboard there.
