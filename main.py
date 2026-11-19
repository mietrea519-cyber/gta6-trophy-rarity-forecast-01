"""GTA VI trophy earn-rate forecaster (make.fun gta6-trophy-rarity-forecast-01).

Reads every ``*.in`` case in the input directory and writes one ``<id>.out``
forecast per case: five quantiles of each trophy's Exophase earn rate (percent)
at the case's target instant.

Model, per trophy (all fits on the logit scale, rates clamped to [0.01, 99.99]):

* States. Consecutive identical reads are collapsed into distinct states,
  stamped at first sight, so a page Exophase has not refreshed contributes
  one state, not many.
* Trend. Weighted least squares of logit(rate) on x = ln(1 + hours since
  2026-11-19T00:00Z), recency-weighted with a ~24 h half-life; each slope is
  shrunk toward the cross-trophy mean slope, and the predicted change from
  the last logit is capped at about +/-2.5.
* Earners / players. ln(players) is fitted on x and extrapolated to the
  target for P*. Per trophy, ln(earners) is regressed on ln(players) to get
  an elasticity alpha (shrunk toward a prior of ~0.35); E* is the last
  earner count grown by (P*/P_last)**alpha, floored at the last count, so
  the rate 100*E*/P* falls mechanically when the tracked base broadens
  faster than earners grow -- the launch-night refresh pattern.
* Persistence. The last observed rate.
* The median logit blends 0.45 trend + 0.35 earners-model + 0.20 persistence.
* Stale page. With a single distinct state the trend and elasticity are
  degenerate, so a dilution prior applies: a just-released game with a tiny
  tracked base multiplies its players several-fold over the horizon
  (launch night: about 6x), while the earners of any one trophy barely
  grow (about 1.3x), so with alpha ~ 0.13 the median is roughly the
  last rate times 6**(0.13 - 1) -- about x0.21. The sum calibration
  matches only the non-pinned trophies, so the pinned universal
  trophy's 100 points cannot hide a bias in the rest.
* Sum calibration. The medians are shifted by one common bounded logit
  shift (|delta| <= ~0.7) so their sum matches 100 * (trophies_earned /
  players) extrapolated on the same log-time axis.
* Spread. sigma = sqrt((0.55*sqrt(horizon_days))**2 + residual**2 * leverage
  + 0.35**2), floored at ~0.85 and capped at ~2.2, with a staleness floor
  of ~0.6 when the page was static for the whole history: one refresh
  moves every trophy's log-rate by a similar, fairly concentrated
  amount. The universal (100%) trophy is pinned near 99.99.

Only the Python standard library is used. No network, no randomness, no
wall-clock reads: two runs on the same case write byte-identical output.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Origin of the log-time axis: the release day, 2026-11-19T00:00:00Z.
EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Prior player multiplier over the horizon when the page has never moved.
STALE_PLAYER_MULTIPLIER = 6.0
# Prior earners elasticity in the stale regime, and the generic prior.
# STALE_ALPHA is small on purpose: when Exophase refreshes after a stale
# page, the tracked base multiplies (launch night: 638 -> 3,868, about 6x)
# while the earners of any one trophy barely grow -- the newly tracked
# players are mostly fresh players who have not earned that trophy yet
# (launch night: non-universal earners grew only ~1.3x). So a stale
# page's median rate falls by roughly the player multiplier, not by the
# much smaller dilution a 0.30 elasticity would give.
STALE_ALPHA = 0.13
ALPHA_PRIOR = 0.35
# Blend weights: trend, earners-model, persistence.
W_TREND, W_EARNERS, W_PERSIST = 0.45, 0.35, 0.20
# Sum-calibration shift bound (logit units).
MAX_SHIFT = 0.7
# Spread parameters.
SIGMA_RW = 0.55          # random-walk scale per sqrt(day)
SIGMA_FLOOR = 0.35       # irreducible logit noise
SIGMA_MIN = 0.85
SIGMA_MAX = 2.2
SIGMA_STALE = 0.60       # floor when the page never moved: one refresh
                         # moves every trophy's log-rate by a similar,
                         # fairly concentrated amount, so the honest
                         # spread for a single refresh is well under 1
                         # logit; the random-walk term still widens it
                         # on long horizons (see the stale branch below)
TREND_HALF_LIFE_H = 24.0
TREND_CHANGE_CAP = 2.5
PLATINUM_MEDIAN_CAP = 3.0


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (as JavaScript Math.round does)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_epoch(moment):
    return (moment - EPOCH).total_seconds() / 3600.0


def x_of(moment):
    """Log-time coordinate: ln(1 + hours since the release day began)."""
    return math.log1p(max(-0.99, hours_since_epoch(moment)))


def ols(xs, ys):
    """Ordinary least squares y ~ a + b*x; returns (a, b) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return (my - b * mx, b)


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b*x; returns (a, b, resid_rms, xbar, sxx_w)."""
    sw = sum(ws)
    if sw <= 0:
        return None
    xbar = sum(w * x for w, x in zip(ws, xs)) / sw
    ybar = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - xbar) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return (ybar, 0.0, 0.0, xbar, 0.0)
    sxy = sum(w * (x - xbar) * (y - ybar) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = ybar - b * xbar
    resid2 = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys))
    rms = math.sqrt(resid2 / sw) if sw > 0 else 0.0
    return (a, b, rms, xbar, sxx)


def distinct_states(history):
    """Collapse consecutive identical reads into states stamped at first sight."""
    states = []
    prev_key = None
    for read in history:
        rates = read.get("rates") or []
        key = (read.get("players_tracked"), read.get("trophies_earned"),
               tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        states.append(read)
    return states


def last_available_rate(history, idx):
    """Last non-null rate for trophy index idx, scanning newest first."""
    for read in reversed(history):
        rates = read.get("rates") or []
        if idx < len(rates) and rates[idx] is not None:
            return rates[idx]
    return None


def forecast_case(case, name):
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = float(case.get("horizon_minutes", 0)) / 1440.0
    target = parse_time(case["target"])
    x_star = x_of(target)

    if not history:
        # No reads at all: emit a wide, valid fallback rather than crashing.
        rows = []
        for trophy in trophies:
            lg = logit(1.0)
            rows.append({"id": trophy["id"],
                         "q": [round4(inv_logit(lg + z * 1.5)) for z in Z]})
        return {"trophies": rows}

    states = distinct_states(history)
    stale = len(states) <= 1
    last = history[-1]
    last_at = parse_time(last["at"])
    x_last = x_of(last_at)
    p_last = float(last.get("players_tracked") or 0) or 1.0
    t_last = float(last.get("trophies_earned") or 0) or 0.0
    n = len(trophies)

    # ---- Extrapolate players and trophies-earned to the target. ----
    xs = [x_of(parse_time(s["at"])) for s in states]
    ln_p = [math.log(max(1.0, float(s.get("players_tracked") or 1)))
            for s in states]
    ln_t = [math.log(max(1.0, float(s.get("trophies_earned") or 1)))
            for s in states]
    fit_p = ols(xs, ln_p)
    fit_t = ols(xs, ln_t)
    if stale or fit_p is None:
        p_star = p_last * STALE_PLAYER_MULTIPLIER
    else:
        p_star = math.exp(fit_p[0] + fit_p[1] * x_star)
    if stale or fit_t is None:
        # Decompose the total: the universal (pinned) trophies' earners
        # grow with the tracked base itself (every tracked player holds
        # them), while the other trophies' earners grow only by the
        # stale elasticity. Extrapolating the aggregate with one
        # exponent would credit the slow trophies with the pinned
        # trophies' growth (launch night: aggregate trophies-earned
        # grew ~4.6x, but the non-universal earners only ~1.3x).
        t_star = p_star + max(0.0, t_last - p_last) * (
            STALE_PLAYER_MULTIPLIER ** STALE_ALPHA)
    else:
        t_star = math.exp(fit_t[0] + fit_t[1] * x_star)
    # Players and earned trophies never shrink; cap runaway extrapolation.
    p_star = min(max(p_star, p_last), p_last * 50.0)
    t_star = min(max(t_star, t_last), max(t_last * 50.0, t_last + 1.0))
    # The sum calibration below matches only the non-pinned trophies:
    # a pinned trophy's rate is 100 by construction and would swamp the
    # sum, hiding a systematic error in the other trophies' medians.
    n_pinned = 0
    pinned_idx = set()
    for i in range(n):
        if (last_available_rate(history, i) or 0.0) >= 99.99:
            n_pinned += 1
            pinned_idx.add(i)
    sum_target = (100.0 * max(0.0, t_star - n_pinned * p_star) / p_star
                  if p_star > 0 else None)

    # ---- Per-trophy trend slopes (first pass: collect, then shrink). ----
    trend = []  # per trophy: (lg_trend, resid_rms, leverage, lg_last)
    slopes = []
    for i in range(n):
        pts_x, pts_y, pts_w = [], [], []
        for s, x in zip(states, xs):
            rates = s.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                pts_x.append(x)
                pts_y.append(logit(rates[i]))
                age_h = max(0.0, (last_at - parse_time(s["at"])).total_seconds() / 3600.0)
                pts_w.append(0.5 ** (age_h / TREND_HALF_LIFE_H))
        r_last = last_available_rate(history, i)
        if r_last is None:
            r_last = 1.0
        lg_last = logit(r_last)
        if len(pts_x) >= 2:
            fit = wls(pts_x, pts_y, pts_w)
            if fit is not None:
                _, slope, resid, xbar, sxx = fit
                slopes.append(slope)
                change = slope * (x_star - x_last)
                change = max(-TREND_CHANGE_CAP, min(TREND_CHANGE_CAP, change))
                leverage = 0.0
                if sxx > 1e-12:
                    leverage = min(4.0, (x_star - xbar) ** 2 / sxx)
                trend.append((lg_last + change, resid, leverage, lg_last, slope))
                continue
        trend.append((lg_last, 0.0, 0.0, lg_last, None))

    mean_slope = sum(slopes) / len(slopes) if slopes else 0.0
    # Shrink each slope toward the cross-trophy mean and recompute the trend.
    for i in range(n):
        lg_tr, resid, lev, lg_last, slope = trend[i]
        if slope is None:
            slope = mean_slope
        else:
            slope = 0.6 * slope + 0.4 * mean_slope
        change = slope * (x_star - x_last)
        change = max(-TREND_CHANGE_CAP, min(TREND_CHANGE_CAP, change))
        trend[i] = (lg_last + change, resid, lev, lg_last, slope)

    # ---- Per-trophy medians and spreads. ----
    medians = []
    sigmas = []
    ln_p_states = [math.log(max(1.0, float(s.get("players_tracked") or 1)))
                   for s in states]
    for i, trophy in enumerate(trophies):
        lg_tr, resid, lev, lg_last, _slope = trend[i]
        r_last = last_available_rate(history, i)
        if r_last is None:
            r_last = 1.0
        e_last = max(0.0, r_last / 100.0 * p_last)

        # Earners elasticity: ln(earners) on ln(players) across states.
        ex, ey = [], []
        for s, lp in zip(states, ln_p_states):
            rates = s.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                e = max(0.0, rates[i] / 100.0 * float(s.get("players_tracked") or 1))
                if e > 0:
                    ex.append(lp)
                    ey.append(math.log(e))
        if stale:
            alpha = STALE_ALPHA
        elif len(ex) >= 2:
            fit_e = ols(ex, ey)
            if fit_e is not None:
                a_hat = max(0.0, min(1.2, fit_e[1]))
                alpha = (0.7 * a_hat + 0.3 * ALPHA_PRIOR) if len(ex) >= 3 \
                    else (0.5 * a_hat + 0.5 * ALPHA_PRIOR)
            else:
                alpha = ALPHA_PRIOR
        else:
            alpha = ALPHA_PRIOR
        e_star = e_last * ((p_star / p_last) ** alpha) if e_last > 0 else 0.0
        e_star = max(e_star, e_last)
        lg_earn = logit(100.0 * e_star / p_star) if p_star > 0 else lg_last

        if r_last >= 99.99:
            # Universal trophy: every tracked player holds it; it stays pinned.
            lg_med = logit(99.99)
            sigma = SIGMA_MIN
        elif stale:
            lg_med = lg_earn
            sigma = max(SIGMA_STALE,
                        math.sqrt((SIGMA_RW * math.sqrt(max(horizon_days, 0.0))) ** 2
                                  + SIGMA_FLOOR ** 2))
        else:
            lg_med = (W_TREND * lg_tr + W_EARNERS * lg_earn
                      + W_PERSIST * lg_last)
            sigma = math.sqrt((SIGMA_RW * math.sqrt(max(horizon_days, 0.0))) ** 2
                              + (resid ** 2) * lev + SIGMA_FLOOR ** 2)
            if len(states) == 2:
                # Near-stale: only one refresh in the whole history, and the
                # tracked base barely moved, so the trend and elasticity are
                # nearly degenerate. Blend in half of the stale dilution
                # prior rather than trusting the two-point fits fully.
                p_first = float(states[0].get("players_tracked") or 1)
                if p_first > 0 and p_last / p_first < 1.05:
                    lg_stale_prior = logit(
                        r_last * (STALE_PLAYER_MULTIPLIER ** (STALE_ALPHA - 1.0)))
                    lg_med = 0.5 * lg_med + 0.5 * lg_stale_prior
                    sigma = max(sigma, 0.5 * (SIGMA_STALE + SIGMA_MIN))
        if (trophy.get("tier") == "platinum"
                and lg_med > logit(PLATINUM_MEDIAN_CAP)):
            lg_med = logit(PLATINUM_MEDIAN_CAP)
        sigma = max(SIGMA_MIN, min(SIGMA_MAX, sigma))
        medians.append(lg_med)
        sigmas.append(sigma)

    # ---- Sum calibration: one bounded common logit shift, applied
    # only to the non-pinned trophies (a pinned trophy stays pinned). ----
    if sum_target is not None and medians:
        def total(delta):
            return sum(inv_logit(lg + delta) for j, lg in enumerate(medians)
                       if j not in pinned_idx)
        lo, hi = -MAX_SHIFT, MAX_SHIFT
        if total(lo) > sum_target:
            delta = lo
        elif total(hi) < sum_target:
            delta = hi
        else:
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if total(mid) < sum_target:
                    lo = mid
                else:
                    hi = mid
            delta = 0.5 * (lo + hi)
        medians = [lg + delta if j not in pinned_idx else lg
                   for j, lg in enumerate(medians)]

    rows = []
    for trophy, lg_med, sigma in zip(trophies, medians, sigmas):
        qs = []
        prev = None
        for z in Z:
            q = round4(min(99.99, max(0.01, inv_logit(lg_med + z * sigma))))
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
