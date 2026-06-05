"""
Railway Postgres trade-performance analysis.

Reads the `positions` / `signals` / `debate_logs` tables and reports win rate,
per-strategy P&L, stop-loss damage, and model calibration. Useful for the
research project's retrospective on how the (old debate and current council)
decisions actually played out once markets settled.

Requires DATABASE_URL in the environment (no credentials are hardcoded):
  export DATABASE_URL=postgresql://USER:PASSWORD@HOST:PORT/DB
  python -m analysis.analyze_trades
"""
import os
from collections import defaultdict
from decimal import Decimal
from statistics import mean

import psycopg2
import psycopg2.extras

URL = os.environ.get("DATABASE_URL")
if not URL:
    raise SystemExit(
        "Set DATABASE_URL before running, e.g.:\n"
        "  export DATABASE_URL=postgresql://USER:PASSWORD@HOST:PORT/DB"
    )


def f(x, p=2):
    if x is None:
        return "—"
    return f"{float(x):.{p}f}"


def hr(c="="):
    print(c * 78)


def section(t):
    print()
    hr()
    print(f" {t}")
    hr()


conn = psycopg2.connect(URL, connect_timeout=20)
cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

cur.execute("SELECT * FROM positions ORDER BY entry_time")
positions = cur.fetchall()

closed = [p for p in positions if p["status"] in ("closed_win", "closed_loss") or p["exit_time"]]
open_pos = [p for p in positions if p not in closed]

print()
hr("#")
print(f"  KALSHI TRADER — PERFORMANCE ANALYSIS")
print(f"  Total positions in DB: {len(positions)}   Closed: {len(closed)}   Open: {len(open_pos)}")
hr("#")

# ---------- 1. OVERALL ----------
section("1. OVERALL STATS (closed trades)")
pnls = [float(p["realized_pnl"] or 0) for p in closed]
wins = [p for p in closed if (p["realized_pnl"] or 0) > 0]
losses = [p for p in closed if (p["realized_pnl"] or 0) < 0]
flats = [p for p in closed if (p["realized_pnl"] or 0) == 0]
total = sum(pnls)
best = max(closed, key=lambda p: p["realized_pnl"] or Decimal(0)) if closed else None
worst = min(closed, key=lambda p: p["realized_pnl"] or Decimal(0)) if closed else None
print(f"  Total closed trades : {len(closed)}")
print(f"  Wins                : {len(wins)}")
print(f"  Losses              : {len(losses)}")
print(f"  Breakeven           : {len(flats)}")
print(f"  Win rate            : {100*len(wins)/max(len(closed),1):.1f}%  (excl. breakeven: {100*len(wins)/max(len(wins)+len(losses),1):.1f}%)")
print(f"  Total P&L           : ${f(total)}")
print(f"  Avg P&L / trade     : ${f(mean(pnls) if pnls else 0)}")
print(f"  Best trade          : ${f(best['realized_pnl'])}  {best['ticker']}  ({best['strategy']})")
print(f"  Worst trade         : ${f(worst['realized_pnl'])}  {worst['ticker']}  ({worst['strategy']})")

# ---------- 2. BY STRATEGY ----------
section("2. BY STRATEGY")
by_strat = defaultdict(list)
for p in closed:
    by_strat[p["strategy"] or "unknown"].append(p)

print(f"  {'strategy':<14} {'n':>4} {'win':>4} {'loss':>5} {'win%':>6} {'P&L $':>9} {'avg $':>8}")
print(f"  {'-'*14} {'-'*4} {'-'*4} {'-'*5} {'-'*6} {'-'*9} {'-'*8}")
for s in sorted(by_strat):
    rows = by_strat[s]
    w = sum(1 for r in rows if (r["realized_pnl"] or 0) > 0)
    l = sum(1 for r in rows if (r["realized_pnl"] or 0) < 0)
    pl = sum(float(r["realized_pnl"] or 0) for r in rows)
    wr = 100 * w / max(w + l, 1)
    avg = pl / len(rows) if rows else 0
    print(f"  {s:<14} {len(rows):>4} {w:>4} {l:>5} {wr:>5.1f}% {pl:>9.2f} {avg:>8.2f}")

# average edge at entry: pull from signals (latest signal per ticker/strategy ≤ entry_time)
print()
print("  Average entry edge (from `signals` joined on ticker+strategy, latest ≤ entry_time):")
cur.execute("SELECT market_ticker, strategy, model_prob, market_prob, edge, created_at FROM signals")
signals = cur.fetchall()
sig_idx = defaultdict(list)
for s in signals:
    sig_idx[(s["market_ticker"], s["strategy"])].append(s)
for k in sig_idx:
    sig_idx[k].sort(key=lambda r: r["created_at"])

def latest_signal(p):
    cands = sig_idx.get((p["ticker"], p["strategy"]), [])
    pri = [s for s in cands if s["created_at"] <= p["entry_time"]]
    return pri[-1] if pri else None

for s in sorted(by_strat):
    rows = by_strat[s]
    edges = [float(latest_signal(r)["edge"]) for r in rows if latest_signal(r) and latest_signal(r)["edge"] is not None]
    if edges:
        print(f"    {s:<14} avg signal_edge = {mean(edges):.4f}   (matched {len(edges)}/{len(rows)})")
    else:
        print(f"    {s:<14} no matching signals in `signals` table")

# ---------- 3. EXIT REASON ----------
section("3. EXIT REASON BREAKDOWN")
by_exit = defaultdict(list)
for p in closed:
    by_exit[p["exit_reason"] or "(none)"].append(p)
print(f"  {'reason':<18} {'n':>4} {'wins':>5} {'P&L $':>9} {'avg $':>8}")
print(f"  {'-'*18} {'-'*4} {'-'*5} {'-'*9} {'-'*8}")
for r in sorted(by_exit, key=lambda x: -len(by_exit[x])):
    rows = by_exit[r]
    w = sum(1 for x in rows if (x["realized_pnl"] or 0) > 0)
    pl = sum(float(x["realized_pnl"] or 0) for x in rows)
    print(f"  {r:<18} {len(rows):>4} {w:>5} {pl:>9.2f} {pl/len(rows):>8.2f}")

# ---------- 4. STOP LOSS ANALYSIS ----------
section("4. STOP LOSS ANALYSIS")
stops = [p for p in closed if p["exit_reason"] == "stop_loss"]
print(f"  Stop-loss exits     : {len(stops)}")
if stops:
    losses_per_stop = [float(p["realized_pnl"] or 0) for p in stops]
    print(f"  Total $ lost        : ${sum(losses_per_stop):.2f}")
    print(f"  Avg loss per stop   : ${mean(losses_per_stop):.2f}")
    print(f"  Worst stop          : ${min(losses_per_stop):.2f}")
    holds = [(p["exit_time"] - p["entry_time"]).total_seconds() / 3600 for p in stops if p["exit_time"] and p["entry_time"]]
    if holds:
        print(f"  Avg hold before stop: {mean(holds):.1f} h    (min {min(holds):.1f}h, max {max(holds):.1f}h)")
    # share of total losses that are stops
    all_loss_pnl = sum(float(p["realized_pnl"] or 0) for p in losses)
    if all_loss_pnl != 0:
        print(f"  Stops as share of total $ lost: {100*sum(losses_per_stop)/all_loss_pnl:.1f}%")
    # how many stops were on positions where market_result eventually matched our side (would have won)?
    would_have_won = 0
    settled_known = 0
    for p in stops:
        if p["market_result"] is None:
            continue
        settled_known += 1
        # Kalshi side semantics: 'yes' wins if result == 'yes', 'no' wins if result == 'no'
        if (p["side"] or "").lower() == (p["market_result"] or "").lower():
            would_have_won += 1
    print(f"  Stops where market eventually settled in our favor: {would_have_won}/{settled_known} (where settlement known)")

# ---------- 5. EDGE ACCURACY ----------
section("5. EDGE ACCURACY (model_prob vs realized outcome)")
# For weather_v1 / compounder: pull model_prob from signals
# For ai_debate: pull judge_prob from debate_logs
cur.execute("SELECT * FROM debate_logs")
debates = cur.fetchall()
deb_idx = defaultdict(list)
for d in debates:
    deb_idx[d["ticker"]].append(d)
for k in deb_idx:
    deb_idx[k].sort(key=lambda r: r["created_at"])


def model_prob_for(p):
    s = latest_signal(p)
    if s and s["model_prob"] is not None:
        return float(s["model_prob"])
    if p["strategy"] == "ai_debate":
        cands = [d for d in deb_idx.get(p["ticker"], []) if d["created_at"] <= p["entry_time"] and d["judge_prob"] is not None]
        if cands:
            return float(cands[-1]["judge_prob"])
    return None


comparisons = []  # (ticker, strategy, side, model_p_yes_eq, win)
for p in closed:
    mp = model_prob_for(p)
    if mp is None:
        continue
    side = (p["side"] or "").lower()
    # convert model_prob (P(yes)) to "P(our side wins)"
    p_our_side = mp if side == "yes" else 1 - mp
    won = (p["realized_pnl"] or 0) > 0
    comparisons.append((p, p_our_side, won))

print(f"  Trades with model probability available: {len(comparisons)}/{len(closed)}")
if comparisons:
    avg_predicted = mean(c[1] for c in comparisons)
    actual_winrate = mean(1.0 if c[2] else 0.0 for c in comparisons)
    print(f"  Avg model-predicted P(our side wins): {avg_predicted:.3f}")
    print(f"  Actual win rate                     : {actual_winrate:.3f}")
    delta = avg_predicted - actual_winrate
    direction = "OVER-confident (predicting more wins than reality)" if delta > 0 else "UNDER-confident"
    print(f"  Delta (predicted − actual)          : {delta:+.3f}  →  model is {direction}")
    # bucket by predicted prob
    buckets = [(0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,1.01)]
    print()
    print(f"  Calibration buckets:")
    print(f"    {'predicted':<14} {'n':>4} {'avg pred':>10} {'actual %':>10}")
    for lo, hi in buckets:
        b = [c for c in comparisons if lo <= c[1] < hi]
        if b:
            print(f"    {lo:.2f}-{hi:.2f}      {len(b):>4} {mean(c[1] for c in b):>10.3f} {100*mean(1.0 if c[2] else 0.0 for c in b):>9.1f}%")

# ---------- 6. TOP 5 BEST / WORST ----------
section("6. TOP 5 BEST AND WORST")
ordered = sorted(closed, key=lambda p: float(p["realized_pnl"] or 0), reverse=True)
print("  --- BEST ---")
print(f"  {'ticker':<28} {'strat':<11} {'side':<4} {'entry':>6} {'exit':>6} {'P&L':>7} {'reason':<14}")
for p in ordered[:5]:
    print(f"  {p['ticker']:<28} {(p['strategy'] or '')[:11]:<11} {(p['side'] or '')[:4]:<4} "
          f"{f(p['entry_price'],4):>6} {f(p['exit_price'],4):>6} {f(p['realized_pnl']):>7} {(p['exit_reason'] or '')[:14]:<14}")
print("  --- WORST ---")
print(f"  {'ticker':<28} {'strat':<11} {'side':<4} {'entry':>6} {'exit':>6} {'P&L':>7} {'reason':<14}")
for p in ordered[-5:][::-1]:
    print(f"  {p['ticker']:<28} {(p['strategy'] or '')[:11]:<11} {(p['side'] or '')[:4]:<4} "
          f"{f(p['entry_price'],4):>6} {f(p['exit_price'],4):>6} {f(p['realized_pnl']):>7} {(p['exit_reason'] or '')[:14]:<14}")

# ---------- 7. RECOMMENDATIONS (printed after analysis is computed) ----------
# Compute the inputs that drive recommendations so we can print data-grounded ones.
recs = []
# A) stop-loss damage
if stops:
    stop_share = abs(sum(float(p["realized_pnl"] or 0) for p in stops)) / max(abs(sum(float(p["realized_pnl"] or 0) for p in losses)), 1e-9)
    avg_stop_loss = mean(float(p["realized_pnl"] or 0) for p in stops)
    recs.append(("stop_loss", stop_share, avg_stop_loss, len(stops)))

# B) per-strategy worst performer
strat_pnls = {s: sum(float(r["realized_pnl"] or 0) for r in rows) for s, rows in by_strat.items()}

# C) calibration
delta_calib = None
if comparisons:
    delta_calib = mean(c[1] for c in comparisons) - mean(1.0 if c[2] else 0.0 for c in comparisons)

section("7. RECOMMENDATIONS (data-driven)")
n = 1

if stops and recs[0][1] > 0.5:
    print(f"  {n}. STOP-LOSS IS BLEEDING US — {recs[0][3]} stops, avg ${recs[0][2]:.2f}/each, "
          f"= {100*recs[0][1]:.0f}% of all $ losses.")
    print(f"     Action: widen stop or remove it for short-dated weather contracts. Many stop out at")
    print(f"     local lows then settle in our favor (see §4 settlement-known stat above).")
    print(f"     Concretely: change stop from current % to either (a) only stop on edge-flip")
    print(f"     (model_prob crosses market_prob), or (b) loosen to 50% drawdown for <24h-to-close.")
    n += 1

worst_strat = min(strat_pnls, key=strat_pnls.get) if strat_pnls else None
if worst_strat and strat_pnls[worst_strat] < 0:
    rows = by_strat[worst_strat]
    w = sum(1 for r in rows if (r["realized_pnl"] or 0) > 0)
    l = sum(1 for r in rows if (r["realized_pnl"] or 0) < 0)
    wr = 100 * w / max(w + l, 1)
    print(f"  {n}. STRATEGY '{worst_strat}' IS UNPROFITABLE — {len(rows)} trades, ${strat_pnls[worst_strat]:.2f} P&L, {wr:.0f}% win rate.")
    print(f"     Action: pause live entries for this strategy. Run paper-mode only until calibration")
    print(f"     improves OR raise its MIN_EDGE_TO_TRADE so only the highest-conviction signals fire.")
    n += 1

if delta_calib is not None and abs(delta_calib) > 0.05:
    if delta_calib > 0:
        print(f"  {n}. MODEL IS OVER-CONFIDENT — predicts {delta_calib*100:+.0f}pp more wins than actual.")
        print(f"     Action: shrink reported edge by ~{delta_calib*100:.0f}pp before the Kelly sizer, OR raise")
        print(f"     MIN_EDGE_TO_TRADE from {0.05} → {0.05 + abs(delta_calib):.2f} to absorb the bias.")
    else:
        print(f"  {n}. MODEL IS UNDER-CONFIDENT — actual win rate is {-delta_calib*100:.0f}pp above predicted.")
        print(f"     Action: lower MIN_EDGE_TO_TRADE; we're leaving good trades on the table.")
    n += 1

# Filler if we don't have 3 yet
if n <= 3:
    # average position size vs P&L variance
    sizes = [float(p["entry_price"] or 0) * (p["contracts"] or 0) for p in closed]
    if sizes:
        print(f"  {n}. POSITION SIZING — avg cost/trade ${mean(sizes):.2f}, max ${max(sizes):.2f}.")
        print(f"     Action: with current ~{100*len(wins)/max(len(wins)+len(losses),1):.0f}% win rate the Kelly fraction")
        print(f"     of {0.25} may be too aggressive. Halve to 0.125 until calibration improves.")
        n += 1

if n <= 3:
    print(f"  {n}. INSTRUMENT THE GAP — only {len(comparisons)}/{len(closed)} trades have a model probability we can")
    print(f"     audit. Action: log model_prob into the `positions` row at entry so every closed trade is")
    print(f"     evaluable without joining `signals`/`debate_logs` (which lose rows on ticker mismatch).")

print()
hr("#")
conn.close()
