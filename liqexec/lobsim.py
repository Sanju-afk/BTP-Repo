"""Regime-switching, level-based queue-reactive limit order book (deck slides 9-11).

Background order flow
    Each side keeps its K best occupied price levels (price in ticks, size in shares). Events arrive as a
    continuous-time Markov jump process (Gillespie sampling), with intensities fitted to data per regime:
      LO   limit order joining level k          rate lam_lo[r, k, bin(q_k)]
      CX   cancellation at level k              rate lam_cx[r, k, bin(q_k)]
      MO   market order hitting the best level  rate lam_mo[r, bin(q_1)]   (may sweep several levels)
      INS  limit order inside the spread        rate lam_ins[r, bin(spread)]
      NEW  limit order opening a new price between occupied levels   rate lam_new[r]
    Event sizes, level gaps and the sizes of levels refilled from behind level K are drawn from the
    empirical distributions of the regime.

Market impact is not imposed. An agent's market order removes queues; the price moves when a level
empties; the book then refills at the rates of the current regime. Impact is therefore non-linear
in size (it depends on the depth profile) and transient (it decays as the book refills). That is the
mechanism behind the square-root law in queue-reactive and zero-intelligence models.

Agent (a seller)
    * market sell: walks the bid side, emptying levels (taker fill).
    * one resting limit sell: priced at the best ask (join, behind the queue) or one tick inside the
      spread (improve, at the front). Its queue position falls when market buys trade through the
      shares ahead of it, and when cancellations hit that queue (cancellations are assumed uniformly
      spread over the queue). Fills can be partial (maker fill).
Regime r can change between calls (switching); set_regime(..., shock_frac) removes a fraction of displayed
depth at once.

Quote revision: after every market order (background or agent) the hit side may withdraw its next level
and the opposite side may step toward the trade, with probabilities fitted to the data's millisecond-scale
quote moves after trades.

Resilience (third calibration stage): market-order signs have long memory (background metaorders with
Pareto-distributed lengths) and are tilted toward a latent fundamental price. In the data, the response
of the mid to a market order peaks within seconds and then partly reverts over minutes; without the
fundamental, a queue-reactive book refills around whatever the current price is, and impact never decays.
"""
import json

import numba as nb
import numpy as np

K = 10
FB = 4096            # fill buffer
MB = 8192            # 1-second mid samples buffer
# st (int64 state) fields
NB, NA, AG_ON, AG_PX, AG_QTY, AG_AHEAD, NFILL, BUYVOL, SELLVOL, NMID, LASTB, LASTA, NEV, META_SIGN, NMO, META_LEFT = range(16)
# fs (float state) fields
T, OFI, NEXTGRID, ANCHOR = range(4)


@nb.njit(cache=True)
def _seed(s):
    np.random.seed(s)


@nb.njit(cache=True, inline="always")
def _bin(x, edges):
    b = 0
    for i in range(1, len(edges)):
        if x >= edges[i]:
            b = i
    return b


@nb.njit(cache=True)
def _draw(arr):
    return arr[np.random.randint(len(arr))]


@nb.njit(cache=True)
def _ask_disp(bk, st):
    a = bk[2, 0] if st[NA] > 0 else 10 ** 9
    if st[AG_ON] == 1 and st[AG_PX] < a:
        return st[AG_PX]
    return a


@nb.njit(cache=True)
def _mid(bk, st):
    return 0.5 * (bk[0, 0] + _ask_disp(bk, st))


@nb.njit(cache=True)
def _refill(bk, st, side, gap, deep):
    """Append levels behind the deepest one until the side has K levels."""
    rp, rs, ci = (0, 1, NB) if side == 0 else (2, 3, NA)
    sgn = -1 if side == 0 else 1
    while st[ci] < K:
        n = st[ci]
        if n == 0:
            anchor = st[LASTB] if side == 0 else st[LASTA]
            bk[rp, 0] = anchor + sgn * _draw(gap)
        else:
            bk[rp, n] = bk[rp, n - 1] + sgn * _draw(gap)
        bk[rs, n] = _draw(deep)
        st[ci] += 1


@nb.njit(cache=True)
def _remove(bk, st, side, k):
    rp, rs, ci = (0, 1, NB) if side == 0 else (2, 3, NA)
    if side == 0:
        st[LASTB] = bk[rp, k]
    else:
        st[LASTA] = bk[rp, k]
        if st[AG_ON] == 1 and bk[rp, k] == st[AG_PX]:
            st[AG_AHEAD] = 0
    n = st[ci]
    for j in range(k, n - 1):
        bk[rp, j] = bk[rp, j + 1]
        bk[rs, j] = bk[rs, j + 1]
    st[ci] = n - 1


@nb.njit(cache=True)
def _insert(bk, st, side, pos, px, sz):
    rp, rs, ci = (0, 1, NB) if side == 0 else (2, 3, NA)
    n = st[ci]
    last = min(n, K - 1)
    for j in range(last, pos, -1):
        bk[rp, j] = bk[rp, j - 1]
        bk[rs, j] = bk[rs, j - 1]
    bk[rp, pos] = px
    bk[rs, pos] = sz
    st[ci] = min(n + 1, K)


@nb.njit(cache=True)
def _fill(st, fills, qty, px, maker):
    i = st[NFILL]
    if i < FB:
        fills[i, 0] = qty
        fills[i, 1] = px
        fills[i, 2] = maker
        st[NFILL] = i + 1


@nb.njit(cache=True)
def _sell_mo(bk, st, fs, fills, m, agent, gap, deep):
    """Market sell of m shares against the bids. Agent fills are recorded as taker fills."""
    rem = m
    while rem > 0:
        if st[NB] == 0:
            _refill(bk, st, 0, gap, deep)
        take = min(rem, bk[1, 0])
        if agent:
            _fill(st, fills, take, bk[0, 0], 0)
        bk[1, 0] -= take
        rem -= take
        if bk[1, 0] == 0:
            _remove(bk, st, 0, 0)
    _refill(bk, st, 0, gap, deep)
    fs[OFI] -= m
    if not agent:
        st[SELLVOL] += m


@nb.njit(cache=True)
def _buy_mo(bk, st, fs, fills, m, gap, deep):
    """Background market buy: walks the asks, which may include the agent's resting sell."""
    rem = m
    while rem > 0:
        if st[NA] == 0:
            _refill(bk, st, 1, gap, deep)
        bgp = bk[2, 0]
        if st[AG_ON] == 1 and st[AG_PX] < bgp:                 # agent alone at a better price
            take = min(rem, st[AG_QTY])
            _fill(st, fills, take, st[AG_PX], 1)
            st[AG_QTY] -= take
            rem -= take
            if st[AG_QTY] == 0:
                st[AG_ON] = 0
            continue
        if st[AG_ON] == 1 and st[AG_PX] == bgp:                 # agent shares this queue
            take = min(rem, st[AG_AHEAD])
            bk[3, 0] -= take
            st[AG_AHEAD] -= take
            rem -= take
            if rem > 0:
                take = min(rem, st[AG_QTY])
                _fill(st, fills, take, bgp, 1)
                st[AG_QTY] -= take
                rem -= take
                if st[AG_QTY] == 0:
                    st[AG_ON] = 0
            if rem > 0:
                take = min(rem, bk[3, 0])
                bk[3, 0] -= take
                rem -= take
            if bk[3, 0] == 0:
                _remove(bk, st, 1, 0)
            continue
        take = min(rem, bk[3, 0])
        bk[3, 0] -= take
        rem -= take
        if bk[3, 0] == 0:
            _remove(bk, st, 1, 0)
    _refill(bk, st, 1, gap, deep)
    fs[OFI] += m
    st[BUYVOL] += m


@nb.njit(cache=True)
def _revise(bk, st, sg, w_hit, w_opp, gap, deep, lo_sz, ins_frac):
    """Post-trade quote revision (Hasbrouck, 1991): after a market order of sign sg (+1 buy, -1 sell),
    with probability w_hit the new best level on the side that was hit is withdrawn, and with
    probability w_opp the opposite side inserts a quote inside the spread, toward the trade."""
    hit = 0 if sg < 0 else 1
    ci = NB if hit == 0 else NA
    if np.random.random() < w_hit and st[ci] > 1:
        if not (hit == 1 and st[AG_ON] == 1 and bk[2, 0] == st[AG_PX]):
            _remove(bk, st, hit, 0)
            _refill(bk, st, hit, gap, deep)
    if np.random.random() < w_opp:
        a = _ask_disp(bk, st)
        spread = a - bk[0, 0]
        if spread > 1:
            imp = max(1, min(spread - 1, int(_draw(ins_frac) * (spread - 1) + 0.5)))
            if sg < 0:
                _insert(bk, st, 1, 0, a - imp, _draw(lo_sz))      # sell: ask improves downward
            else:
                _insert(bk, st, 0, 0, bk[0, 0] + imp, _draw(lo_sz))   # buy: bid improves upward


@nb.njit(cache=True)
def _run(bk, st, fs, fills, mids, molog, t_end, qe, se, lam_lo, lam_cx, lam_mo, lam_ins, lam_new, p_new,
         lo_sz, cx_sz, mo_sz, gap, deep, ins_frac, pi_m, alpha, kappa, sigma_f, theta_p, kappa_mo, hawkes_n,
         w_hit, w_opp):
    """Simulate background flow of one regime from fs[T] to t_end.

    Market orders form one stream with the total rate (1 - n) * (lam_mo(q_bid1) + lam_mo(q_ask1)), and
    cluster as a Hawkes process with branching ratio n, estimated from the data by maximum likelihood.
    The data's kernel decays in about 3 ms, much faster than other book events, so the cluster
    representation is used: each market order immediately triggers Poisson(n) further market orders,
    recursively (expected cluster size 1 / (1 - n)). Their signs follow
    Lillo, Mike and Farmer (2005): with probability pi_m a market order is the next child of an ongoing
    background metaorder (same sign), whose length in child orders is Pareto with tail exponent alpha;
    this gives long-memory sign autocorrelation. Otherwise, and for the sign of each new metaorder, the
    order is a buy with probability proportional to lam_mo(q_ask1) * exp(kappa_mo * (p_f - mid)) (kappa_mo
    is 0 in the calibrated model: tilting market orders makes them informed, so impact grows instead of
    reverting).

    p_f is a latent fundamental price (as in ABIDES): it diffuses with sigma_f ticks per sqrt(second) and
    absorbs a fraction theta_p of every market order's immediate mid change (the permanent, informational
    part of impact, as in Hasbrouck's decomposition). Liquidity providers anchor to it: inserts inside
    the spread on the bid side arrive at rate lam_ins * exp(kappa * (p_f - mid)), and on the ask side at
    lam_ins * exp(-kappa * (p_f - mid)). After the price is pushed down, bids are inserted faster than
    asks, which closes the gap. That is the resilience that makes the rest of the impact transient.
    """
    rates = np.zeros(2 * (2 * K + 3))
    while True:
        # --- intensities of every possible event
        j = 0
        for side in range(2):
            rs, ci = (1, NB) if side == 0 else (3, NA)
            n = st[ci]
            for k in range(K):
                if k < n:
                    b = _bin(bk[rs, k], qe)
                    rates[j] = lam_lo[k, b]
                    rates[j + K] = lam_cx[k, b]
                else:
                    rates[j] = 0.0
                    rates[j + K] = 0.0
                j += 1
            j += K
            rates[j] = (1.0 - hawkes_n) * lam_mo[_bin(bk[rs, 0], qe)] if n > 0 else 0.0
            spread = _ask_disp(bk, st) - bk[0, 0]
            tilt = max(-3.0, min(3.0, kappa * (fs[ANCHOR] - 0.5 * (_ask_disp(bk, st) + bk[0, 0]))))
            rates[j + 1] = lam_ins[_bin(spread, se)] * np.exp(tilt if side == 0 else -tilt) if spread > 1 else 0.0
            rates[j + 2] = lam_new if n >= 2 else 0.0
            j += 3
        lam_b = rates[2 * K]                 # market sells hitting the bid
        lam_a = rates[2 * K + 3 + 2 * K]     # market buys lifting the ask
        rates[2 * K] = lam_a + lam_b
        rates[2 * K + 3 + 2 * K] = 0.0
        total = rates.sum()
        dt = -np.log(1.0 - np.random.random()) / total
        t_next = fs[T] + dt
        mid_now = _mid(bk, st)
        fs[ANCHOR] += sigma_f * np.sqrt(min(dt, t_end - fs[T])) * np.random.standard_normal()
        while fs[NEXTGRID] <= min(t_next, t_end) and st[NMID] < MB:
            mids[st[NMID]] = _mid(bk, st)
            st[NMID] += 1
            fs[NEXTGRID] += 1.0
        if t_next > t_end:
            fs[T] = t_end
            return
        fs[T] = t_next
        st[NEV] += 1
        # --- choose the event
        u = np.random.random() * total
        e = 0
        acc = rates[0]
        while acc < u and e < len(rates) - 1:
            e += 1
            acc += rates[e]
        side = 0 if e < 2 * K + 3 else 1
        loc = e - side * (2 * K + 3)
        rp, rs = (0, 1) if side == 0 else (2, 3)
        sgn = -1 if side == 0 else 1               # direction away from the spread
        osgn = 1 if side == 0 else -1              # OFI sign of adding bid depth
        if loc < K:                                # limit order at existing level
            q = _draw(lo_sz)
            bk[rs, loc] += q
            if loc == 0:
                fs[OFI] += osgn * q
        elif loc < 2 * K:                          # cancellation
            k = loc - K
            c = min(_draw(cx_sz), bk[rs, k])
            if side == 1 and st[AG_ON] == 1 and bk[rp, k] == st[AG_PX] and bk[rs, k] > 0:
                st[AG_AHEAD] -= (c * st[AG_AHEAD] + bk[rs, k] // 2) // bk[rs, k]
            bk[rs, k] -= c
            if k == 0:
                fs[OFI] -= osgn * c
            if bk[rs, k] == 0:
                _remove(bk, st, side, k)
                _refill(bk, st, side, gap, deep)
        elif loc == 2 * K:                         # market-order cluster (single stream, side chosen per order)
            pending = 1
            while pending > 0:
                pending -= 1
                pending += np.random.poisson(hawkes_n)
                mid_now = _mid(bk, st)
                m = _draw(mo_sz)
                x = max(-20.0, min(20.0, kappa_mo * (fs[ANCHOR] - mid_now)))
                wb = lam_a * np.exp(x)
                ws = lam_b * np.exp(-x)
                if np.random.random() < pi_m:
                    if st[META_LEFT] <= 0:                 # start a new background metaorder
                        st[META_SIGN] = 1 if np.random.random() * (wb + ws) < wb else -1
                        st[META_LEFT] = min(10 ** 6, int((1.0 - np.random.random()) ** (-1.0 / alpha)))
                    sg = st[META_SIGN]
                    st[META_LEFT] -= 1
                else:
                    sg = 1 if np.random.random() * (wb + ws) < wb else -1
                if st[NMO] < len(molog):
                    molog[st[NMO], 0] = fs[T]
                    molog[st[NMO], 1] = sg
                    molog[st[NMO], 2] = mid_now
                    st[NMO] += 1
                if sg < 0:
                    _sell_mo(bk, st, fs, fills, m, False, gap, deep)
                else:
                    _buy_mo(bk, st, fs, fills, m, gap, deep)
                _revise(bk, st, sg, w_hit, w_opp, gap, deep, lo_sz, ins_frac)
                fs[ANCHOR] += theta_p * (_mid(bk, st) - mid_now)
        elif loc == 2 * K + 1:                     # insert inside the spread
            a = _ask_disp(bk, st)
            spread = a - bk[0, 0]
            imp = max(1, min(spread - 1, int(_draw(ins_frac) * (spread - 1) + 0.5)))
            px = bk[0, 0] + imp if side == 0 else a - imp
            q = _draw(lo_sz)
            if side == 1 and st[AG_ON] == 1 and px == st[AG_PX]:
                px = st[AG_PX]                     # joins behind the agent (new bg level at agent's price)
            _insert(bk, st, side, 0, px, q)
            fs[OFI] += osgn * q
        else:                                      # new price between occupied levels
            u2 = np.random.random()
            slot = 1
            cum = p_new[1]
            while cum < u2 and slot < K - 1:
                slot += 1
                cum += p_new[slot]
            n = st[NB] if side == 0 else st[NA]
            if slot < n:
                g = abs(bk[rp, slot] - bk[rp, slot - 1])
                if g > 1:
                    px = bk[rp, slot - 1] + sgn * (1 + np.random.randint(g - 1))
                    _insert(bk, st, side, slot, px, _draw(lo_sz))


class Market:
    """Python wrapper: state arrays plus regime parameter tables."""

    def __init__(self, calib_path):
        p = json.load(open(calib_path)) if isinstance(calib_path, str) else calib_path
        self.calib = p
        self.regimes = p["regimes"]
        self.qe = np.array(p["q_edges"], np.int64)
        self.se = np.array(p["s_edges"], np.int64)
        self.tick = p["tick_dollars"]
        self.P = []
        for r in self.regimes:
            q = p["per_regime"][r]
            self.P.append(dict(
                lam_lo=np.array(q["lam_lo"]), lam_cx=np.array(q["lam_cx"]), lam_mo=np.array(q["lam_mo"]),
                lam_ins=np.array(q["lam_ins"]), lam_new=float(q["lam_new"]), p_new=np.array(q["p_new"]),
                lo_sz=np.array(q["lo_size"], np.int64), cx_sz=np.array(q["cx_size"], np.int64),
                mo_sz=np.array(q["mo_size"], np.int64), gap=np.array(q["gap"], np.int64),
                deep=np.array(q["deep_size"], np.int64), ins_frac=np.array(q["ins_frac"]),
                snaps=np.array(q["snapshots"], np.int64)))
            mm = p.get("msm", {}).get(r)                 # second-stage multipliers (liqexec/msm.py)
            if mm:
                self.P[-1]["lam_ins"] = self.P[-1]["lam_ins"] * mm["m_ins"]
                self.P[-1]["lam_cx"][0] *= mm["m_cx1"]
                self.P[-1]["lam_cx"][1:] *= mm["m_cxd"]
                self.P[-1]["lam_lo"] = self.P[-1]["lam_lo"] * mm.get("m_lo", 1.0)
        d = p.get("dyn", {})                          # third-stage parameters (liqexec/msm.py --stage dyn)
        self.pi_m, self.alpha = d.get("pi_m", 0.0), d.get("alpha", 1.5)
        self.kappa, self.sigma_f, self.theta_p = d.get("kappa", 0.0), d.get("sigma_f", 0.0), d.get("theta_p", 1.0)
        self.kappa_mo = d.get("kappa_mo", 0.0)
        self.w_hit, self.w_opp = d.get("w_hit", 0.0), d.get("w_opp", 0.0)
        self.hawkes_n = p.get("hawkes", {}).get("n", 0.0)          # market-order clustering, MLE from data
        self.bk = np.zeros((4, K), np.int64)
        self.st = np.zeros(16, np.int64)
        self.fs = np.zeros(4)
        self.molog = np.zeros((MB, 3))
        self.fills = np.zeros((FB, 3), np.int64)
        self.mids = np.zeros(MB)
        self.r = 0

    # ----- episode control
    def reset(self, regime, seed, base_px=58000, burn_in=60.0):
        _seed(seed)
        rng = np.random.default_rng(seed)
        self.r = regime
        sn = self.P[regime]["snaps"][rng.integers(len(self.P[regime]["snaps"]))]
        self.bk[0], self.bk[1] = base_px + sn[0], sn[1]
        self.bk[2], self.bk[3] = base_px + sn[2], sn[3]
        self.st[:] = 0
        self.st[NB] = self.st[NA] = K
        self.fs[:] = 0.0
        self.fs[ANCHOR] = self.mid()
        self.run_until(burn_in)
        self.clear_counters()

    def clear_counters(self):
        self.st[NFILL] = 0
        self.st[BUYVOL] = self.st[SELLVOL] = 0
        self.st[NMID] = 0
        self.st[NMO] = 0
        self.fs[OFI] = 0.0
        self.fs[NEXTGRID] = np.floor(self.fs[T]) + 1.0

    @property
    def t(self):
        return self.fs[T]

    def run_until(self, t_end):
        q = self.P[self.r]
        _run(self.bk, self.st, self.fs, self.fills, self.mids, self.molog, t_end, self.qe, self.se, q["lam_lo"],
             q["lam_cx"], q["lam_mo"], q["lam_ins"], q["lam_new"], q["p_new"], q["lo_sz"], q["cx_sz"], q["mo_sz"],
             q["gap"], q["deep"], q["ins_frac"], self.pi_m, self.alpha, self.kappa, self.sigma_f, self.theta_p,
             self.kappa_mo, self.hawkes_n, self.w_hit, self.w_opp)

    def set_regime(self, regime, shock_frac=0.0):
        self.r = regime
        if shock_frac > 0:
            q = self.P[regime]
            for side, (rs, ci) in enumerate(((1, NB), (3, NA))):
                k = 0
                while k < self.st[ci]:
                    self.bk[rs, k] = int(self.bk[rs, k] * (1 - shock_frac))
                    if side == 1 and self.st[AG_ON] == 1 and self.bk[2, k] == self.st[AG_PX]:
                        # withdrawn shares are spread over the queue, like cancellations
                        self.st[AG_AHEAD] = min(int(self.st[AG_AHEAD] * (1 - shock_frac)), int(self.bk[rs, k]))
                    if self.bk[rs, k] <= 0:
                        _remove(self.bk, self.st, side, k)
                    else:
                        k += 1
                _refill(self.bk, self.st, side, q["gap"], q["deep"])

    # ----- observation helpers
    def best_bid(self):
        return int(self.bk[0, 0])

    def best_ask(self):
        return int(_ask_disp(self.bk, self.st))

    def mid(self):
        return float(_mid(self.bk, self.st))

    def depth(self, side, levels):
        rs = 1 if side == 0 else 3
        return int(self.bk[rs, :levels].sum())

    # ----- agent actions
    def cancel_agent(self):
        left = int(self.st[AG_QTY]) if self.st[AG_ON] else 0
        self.st[AG_ON] = 0
        self.st[AG_QTY] = 0
        return left

    def agent_limit(self, qty, improve):
        """Rest a limit sell at the best background ask (join) or one tick inside the spread (improve)."""
        if qty <= 0:
            return
        a, b = int(self.bk[2, 0]), int(self.bk[0, 0])
        if improve and a - b > 1:
            px, ahead = a - 1, 0
        else:
            px, ahead = a, int(self.bk[3, 0])
        self.st[AG_ON], self.st[AG_PX], self.st[AG_QTY], self.st[AG_AHEAD] = 1, px, qty, ahead

    def agent_market(self, qty):
        if qty > 0:
            q = self.P[self.r]
            m0 = self.mid()
            _sell_mo(self.bk, self.st, self.fs, self.fills, qty, True, q["gap"], q["deep"])
            _revise(self.bk, self.st, -1, self.w_hit, self.w_opp, q["gap"], q["deep"], q["lo_sz"], q["ins_frac"])
            self.fs[ANCHOR] += self.theta_p * (self.mid() - m0)      # same permanent share as background trades

    def take_fills(self):
        n = int(self.st[NFILL])
        f = self.fills[:n].copy()
        self.st[NFILL] = 0
        return f                     # columns: qty, price (ticks), maker flag

    def take_flow(self):
        """Background buy and sell market-order volume, OFI and 1-s mids since the last call."""
        out = (int(self.st[BUYVOL]), int(self.st[SELLVOL]), float(self.fs[OFI]), self.mids[:int(self.st[NMID])].copy())
        self.st[BUYVOL] = self.st[SELLVOL] = 0
        self.st[NMID] = 0
        self.fs[OFI] = 0.0
        return out
