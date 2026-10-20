"""Statistics, tables and figures for the v2 risk-averse DDQN study (docs/v2/preregistration.md).

    python3 experiments/report_lam.py [--eval out_v2/eval] [--runs runs_v2] [--out out_v2/report]

Contrasts are paired by training seed (10-19) and by test episode (common random numbers). Each bootstrap
resample draws seeds, then episodes within each setting, and applies the same draw to both arms; a
deterministic baseline enters every resampled seed with the same episode draw. CVaR95 is computed per run
(worst 5% of that run's 1,000 episodes) and averaged over runs, then over the six settings.
"""
import argparse
import glob
import os
import re
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from report import COND_LABEL, CONDS, cvar   # noqa: E402

B = 2000
LAMS = [0, 0.5, 1, 2, 4, 8]
REF = "DDQN@0.5"
SESOI_CVAR, NI_MARGIN = 1.0, 0.3
RUN_RE = re.compile(r"lob_(DDQN|PPO)_lam([\d.]+)_s(\d+)$")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C_DDQN, C_PPO, C_AC, C_TWAP = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"


def lab(x):
    return f"{x:g}"


def arm_of(policy):
    m = RUN_RE.match(policy)
    if m:
        return f"{m.group(1)}@{lab(float(m.group(2)))}", int(m.group(3))
    return policy, -1


def load(d):
    df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(d, "*.csv"))], ignore_index=True)
    am = df.policy.map(arm_of)
    df["arm"] = [a for a, _ in am]
    df["seed"] = [s for _, s in am]
    return df


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def pf(p):
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


# ------------------------------------------------------------------ paired bootstrap
def matrix(df, cond, arm, col):
    g = df[(df.condition == cond) & (df.arm == arm)]
    piv = g.pivot(index="seed", columns="episode", values=col).sort_index()
    return piv.to_numpy(), list(piv.index)


def _f(stat):
    if stat == "mean":
        return lambda x: x.mean(1)
    return lambda x: np.sort(x, 1)[:, int(np.floor(0.95 * x.shape[1])):].mean(1)


def paired_stats(df, a, b, rng, conds=CONDS, stat="mean", col="is_bps"):
    """b minus a averaged over conds; a deterministic arm (one row, seed -1) is matched to every seed."""
    mats = []
    for c in conds:
        A, sa = matrix(df, c, a, col)
        Bm, sb = matrix(df, c, b, col)
        if sa == [-1]:
            A, sa = np.repeat(A, len(sb), 0), sb
        if sb == [-1]:
            Bm, sb = np.repeat(Bm, len(sa), 0), sa
        common = sorted(set(sa) & set(sb))
        mats.append((A[[sa.index(s) for s in common]], Bm[[sb.index(s) for s in common]]))
    f = _f(stat)
    point = np.mean([(f(Bm) - f(A)).mean() for A, Bm in mats])
    n_s = mats[0][0].shape[0]
    draws = np.empty(B)
    for k in range(B):
        si = rng.integers(0, n_s, n_s)
        tot = 0.0
        for A, Bm in mats:
            ei = rng.integers(0, A.shape[1], (n_s, A.shape[1]))
            tot += (f(np.take_along_axis(Bm[si], ei, 1)) - f(np.take_along_axis(A[si], ei, 1))).mean()
        draws[k] = tot / len(mats)
    p = min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean()))
    return point, np.percentile(draws, 2.5), np.percentile(draws, 97.5), p


def holm(ps):
    ps = np.asarray(ps)
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(ps) - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def summary(df, arm, rng):
    """Mean IS and CVaR95 averaged over settings, with seed-then-episode bootstrap intervals."""
    mats = [matrix(df, c, arm, "is_bps")[0] for c in CONDS]
    fm, fc = _f("mean"), _f("cvar")
    n_s = mats[0].shape[0]
    dm, dc = np.empty(B), np.empty(B)
    for k in range(B):
        si = rng.integers(0, n_s, n_s)
        sm = sc = 0.0
        for M in mats:
            ei = rng.integers(0, M.shape[1], (n_s, M.shape[1]))
            R = np.take_along_axis(M[si], ei, 1)
            sm += fm(R).mean()
            sc += fc(R).mean()
        dm[k], dc[k] = sm / len(mats), sc / len(mats)
    g = df[df.arm == arm]
    return {"arm": arm, "runs": n_s if n_s > 1 or matrix(df, CONDS[0], arm, "is_bps")[1] != [-1] else 0,
            "is": np.mean([fm(M).mean() for M in mats]), "is_lo": np.percentile(dm, 2.5), "is_hi": np.percentile(dm, 97.5),
            "cvar95": np.mean([fc(M).mean() for M in mats]), "cvar_lo": np.percentile(dc, 2.5),
            "cvar_hi": np.percentile(dc, 97.5), "objective": g.objective.mean(), "sold_before_T": g.completion.mean(),
            "forced": g.forced.mean(), "limit_share": g.limit_share.mean(),
            "inv_shock": g[g.condition == "shock"].inv_k10.mean()}


# ------------------------------------------------------------------ figures
def fig_frontier(ov, path):
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    style(ax)
    pos = lambda v: max(0.0, v)                                         # noqa: E731
    for fam, col, mk in (("DDQN", C_DDQN, "o"), ("AC", C_AC, "s")):
        t = ov[ov.arm.isin([f"{fam}@{lab(x)}" for x in LAMS])].copy()
        t["lam"] = t.arm.str.split("@").str[1].astype(float)
        t = t.sort_values("lam")
        ax.plot(t["is"], t.cvar95, "-", color=col, lw=1.5, alpha=0.7)
        for _, r in t.iterrows():
            if fam == "DDQN":
                ax.errorbar(r["is"], r.cvar95, xerr=[[pos(r["is"] - r.is_lo)], [pos(r.is_hi - r["is"])]],
                            yerr=[[pos(r.cvar95 - r.cvar_lo)], [pos(r.cvar_hi - r.cvar95)]], fmt=mk, ms=7, color=col,
                            mec="white", mew=1.2, elinewidth=1, capsize=0)
            else:
                ax.plot(r["is"], r.cvar95, mk, ms=7, color=col, mec="white", mew=1.2)
            ax.annotate(f"λ′={lab(r.lam)}", (r["is"], r.cvar95), xytext=(6, 3), textcoords="offset points", fontsize=7,
                        color=INK2)
        ax.plot([], [], mk + "-", color=col, label="DDQN, 10 seeds (95% CI)" if fam == "DDQN" else "Almgren–Chriss")
    for arm, col, mk, name in (("PPO@0.5", C_PPO, "D", "PPO, λ′=0.5"), ("TWAP", C_TWAP, "x", "TWAP")):
        r = ov[ov.arm == arm]
        if len(r):
            r = r.iloc[0]
            ax.plot(r["is"], r.cvar95, mk, ms=8, color=col, label=name)
    ax.set_xlabel("mean shortfall, bps (average of settings A–D3)", color=INK2, fontsize=9)
    ax.set_ylabel("CVaR95 of shortfall, bps", color=INK2, fontsize=9)
    ax.set_title("Mean vs tail cost as the risk penalty λ′ grows", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_forced(ov, path):
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4))
    for ax, col_name, ylab in ((axes[0], "forced", "% of X sold by the forced order at T"),
                               (axes[1], "limit_share", "% of child orders that are limit orders")):
        style(ax)
        for fam, col, mk in (("DDQN", C_DDQN, "o"), ("AC", C_AC, "s")):
            t = ov[ov.arm.isin([f"{fam}@{lab(x)}" for x in LAMS])].copy()
            if fam == "AC" and col_name == "limit_share":
                continue
            t["lam"] = t.arm.str.split("@").str[1].astype(float)
            t = t.sort_values("lam")
            ax.plot(range(len(t)), 100 * t[col_name], mk + "-", color=col, lw=2, ms=6,
                    label="DDQN" if fam == "DDQN" else "Almgren–Chriss")
        r = ov[ov.arm == "PPO@0.5"]
        if len(r):
            ax.plot(LAMS.index(0.5), 100 * r.iloc[0][col_name], "D", color=C_PPO, ms=7, label="PPO, λ′=0.5")
        ax.set_xticks(range(len(LAMS)), [lab(x) for x in LAMS])
        ax.set_xlabel("risk penalty λ′ (bps)", color=INK2, fontsize=8)
        ax.set_ylabel(ylab, color=INK2, fontsize=8)
        ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=7, frameon=False)
    axes[0].set_title("Inventory left to the forced sweep", fontsize=9.5, color=INK, loc="left")
    axes[1].set_title("Passive (limit) order use", fontsize=9.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_hist(df, arms, path, conds=("M", "shock")):
    cols = [C_DDQN, "#6250d6", C_PPO, C_AC]
    fig, axes = plt.subplots(1, len(conds), figsize=(9.6, 3.6), sharey=True)
    for ax, cond in zip(axes, conds):
        style(ax)
        d = df[df.condition == cond]
        lo, hi = np.percentile(d[d.arm.isin(arms)].is_bps, [0.5, 99.8])
        bins = np.linspace(lo, hi, 60)
        for a, c in zip(arms, cols):
            g = d[d.arm == a]
            ax.hist(g.is_bps, bins=bins, histtype="step", lw=2, color=c, density=True, label=a)
            ax.axvline(np.mean([cvar(x.is_bps.to_numpy()) for _, x in g.groupby("policy")]), color=c, lw=1, ls=":")
        ax.set_title(COND_LABEL[cond], fontsize=9, color=INK, loc="left")
        ax.set_xlabel("shortfall, bps (dotted: CVaR95)", color=INK2, fontsize=8)
    axes[0].set_ylabel("density", color=INK2, fontsize=8)
    axes[-1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_learning(runs_dir, path):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    style(ax)
    shades = ["#9ec5f4", "#2a78d6", "#1d5fb0", "#174a8a", "#113565", "#0a2142"]
    for lam, c in zip(LAMS, shades):
        curves = []
        for d in sorted(glob.glob(os.path.join(runs_dir, f"lob_DDQN_lam{lab(lam)}_s*"))):
            L = np.loadtxt(os.path.join(d, "log.csv"), delimiter=",", skiprows=1)
            grid = np.linspace(0, 250_000, 100)
            sm = pd.Series(L[:, 2]).rolling(500, min_periods=50).mean().to_numpy()
            curves.append(np.interp(grid, L[:, 0], sm))
        if curves:
            ax.plot(grid / 1000, np.mean(curves, 0), color=c, lw=2, label=f"λ′={lab(lam)}")
    ax.set_xlabel("environment steps (thousands)", color=INK2, fontsize=8)
    ax.set_ylabel("training shortfall, bps (500-episode mean)", color=INK2, fontsize=8)
    ax.set_title("DDQN learning curves by risk penalty (mean of 10 seeds)", fontsize=9.5, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="out_v2/eval")
    ap.add_argument("--runs", default="runs_v2")
    ap.add_argument("--out", default="out_v2/report")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)
    df = load(args.eval)
    fmt = lambda x: f"{x:+.2f}"     # noqa: E731
    ddqn = [f"DDQN@{lab(x)}" for x in LAMS]
    others = [a for a in ["PPO@0.5"] + [f"AC@{lab(x)}" for x in LAMS] + [f"AC-adaptive@{lab(x)}" for x in LAMS]
              + ["TWAP", "TWAP-LO", "VWAP", "POV"] if a in set(df.arm)]

    print("=" * 100)
    print(f"PRE-REGISTERED CONTRASTS 1-3: DDQN@lam' minus {REF}, averaged over settings A-D3 (Holm over 4 lam')")
    rows = []
    for lam in [1, 2, 4, 8]:
        arm = f"DDQN@{lab(lam)}"
        if arm not in set(df.arm):
            continue
        for stat in ("cvar", "mean"):
            pt, lo, hi, p = paired_stats(df, REF, arm, rng, stat=stat)
            rows.append({"lam": lam, "stat": "CVaR95" if stat == "cvar" else "mean", "diff": pt, "lo": lo, "hi": hi, "p": p})
    prim = pd.DataFrame(rows)
    for st in ("CVaR95", "mean"):
        m = prim.stat == st
        prim.loc[m, "p_holm"] = holm(prim.loc[m, "p"].to_numpy())
    head = []
    for lam, g in prim.groupby("lam"):
        t, mn = g[g.stat == "CVaR95"].iloc[0], g[g.stat == "mean"].iloc[0]
        keeps = t.hi < -SESOI_CVAR and mn.hi < NI_MARGIN
        head.append({"lam": lam, "tail_cut_beyond_SESOI": t.hi < -SESOI_CVAR, "mean_noninferior": mn.hi < NI_MARGIN,
                     "keeps_mean_cuts_tail": keeps})
        print(f"  lam'={lab(lam):>3}: CVaR95 {fmt(t['diff'])} [{fmt(t.lo)}, {fmt(t.hi)}] {pf(t.p_holm)} (Holm) | "
              f"mean {fmt(mn['diff'])} [{fmt(mn.lo)}, {fmt(mn.hi)}] {pf(mn.p_holm)} (Holm) -> "
              f"{'KEEPS MEAN AND CUTS TAIL' if keeps else 'tail cut beyond 1 bps' if t.hi < -SESOI_CVAR else 'no tail cut beyond 1 bps'}"
              f"{'' if keeps else ('; mean non-inferior' if mn.hi < NI_MARGIN else '; mean NOT non-inferior')}")
    prim.to_csv(os.path.join(args.out, "primary.csv"), index=False)
    pd.DataFrame(head).to_csv(os.path.join(args.out, "headline.csv"), index=False)

    print("\n" + "=" * 100)
    print("PRE-REGISTERED CONTRAST 4: DDQN@lam' minus AC@lam' on the objective (IS + penalty), Holm over 6 lam'")
    rows = []
    for lam in LAMS:
        a, b = f"AC@{lab(lam)}", f"DDQN@{lab(lam)}"
        if a in set(df.arm) and b in set(df.arm):
            for col in ("objective", "is_bps"):
                for stat in (("mean",) if col == "objective" else ("mean", "cvar")):
                    pt, lo, hi, p = paired_stats(df, a, b, rng, stat=stat, col=col)
                    rows.append({"lam": lam, "measure": f"{col}:{stat}", "diff": pt, "lo": lo, "hi": hi, "p": p})
    vac = pd.DataFrame(rows)
    m = vac.measure == "objective:mean"
    vac.loc[m, "p_holm"] = holm(vac.loc[m, "p"].to_numpy())
    vac.to_csv(os.path.join(args.out, "vs_ac.csv"), index=False)
    print(vac.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    print("\n" + "=" * 100)
    print("PER-ARM SUMMARY, averaged over A-D3 (agents: 10 seeds; seed-then-episode bootstrap 95% CIs)")
    ov = pd.DataFrame([summary(df, a, rng) for a in ddqn + others if a in set(df.arm)])
    ov.to_csv(os.path.join(args.out, "overall.csv"), index=False)
    print(ov.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 100)
    print("EXPLORATORY: DDQN@lam' minus PPO@0.5, and DDQN@0 minus DDQN@0.5")
    rows = []
    for lam in LAMS:
        for stat in ("cvar", "mean"):
            if "PPO@0.5" in set(df.arm) and f"DDQN@{lab(lam)}" in set(df.arm):
                pt, lo, hi, p = paired_stats(df, "PPO@0.5", f"DDQN@{lab(lam)}", rng, stat=stat)
                rows.append({"b": f"DDQN@{lab(lam)}", "a": "PPO@0.5", "stat": stat, "diff": pt, "lo": lo, "hi": hi, "p": p})
    for stat in (("cvar", "mean") if "DDQN@0" in set(df.arm) else ()):
        pt, lo, hi, p = paired_stats(df, REF, "DDQN@0", rng, stat=stat)
        rows.append({"b": "DDQN@0", "a": REF, "stat": stat, "diff": pt, "lo": lo, "hi": hi, "p": p})
    ex = pd.DataFrame(rows)
    ex.to_csv(os.path.join(args.out, "exploratory.csv"), index=False)
    print(ex.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    print("\n" + "=" * 100)
    print("PER-SETTING TABLE")
    rows = []
    for cond in CONDS:
        for a in ddqn + others:
            g = df[(df.condition == cond) & (df.arm == a)]
            if g.empty:
                continue
            runs = [x.is_bps.to_numpy() for _, x in g.groupby("policy")]
            rows.append({"condition": cond, "arm": a, "is": np.mean([x.mean() for x in runs]),
                         "cvar95": np.mean([cvar(x) for x in runs]), "objective": g.objective.mean(),
                         "forced": g.forced.mean(), "limit_share": g.limit_share.mean(), "inv_k10": g.inv_k10.mean()})
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(args.out, "per_setting.csv"), index=False)
    for cond in CONDS:
        print(f"\n{COND_LABEL[cond]}")
        print(tab[tab.condition == cond].drop(columns="condition").to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 100)
    print("MECHANISM: forced-sweep share in each run's worst-5% episodes vs the rest (mean over runs and settings)")
    rows = []
    for a in ddqn + ["PPO@0.5"]:
        for cond in CONDS:
            for _, r in df[(df.arm == a) & (df.condition == cond)].groupby("policy"):
                cut = np.quantile(r.is_bps, 0.95)
                rows.append({"arm": a, "forced_tail": r[r.is_bps >= cut].forced.mean(),
                             "forced_body": r[r.is_bps < cut].forced.mean(),
                             "corr_forced_is": np.corrcoef(r.forced, r.is_bps)[0, 1] if r.forced.std() > 0 else np.nan})
    mech = pd.DataFrame(rows).groupby("arm").mean().reindex([a for a in ddqn + ["PPO@0.5"] if a in set(df.arm)])
    mech.to_csv(os.path.join(args.out, "mechanism.csv"))
    print(mech.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 100)
    print(f"SENSITIVITY (500 episodes, settings B and D2): DDQN@lam' minus {REF}, and DDQN@lam' minus AC@lam' objective")
    rows = []
    for name, d in (("fast resilience (kappa x2)", "out_v2/eval_sens_fastres"), ("latency 0.5 s", "out_v2/eval_sens_latency"),
                    ("fees x3", "out_v2/eval_sens_fees")):
        if not glob.glob(os.path.join(d, "*.csv")):
            continue
        ds = load(d)
        for lam in [1, 2, 4, 8]:
            for stat in ("cvar", "mean"):
                pt, lo, hi, _ = paired_stats(ds, REF, f"DDQN@{lab(lam)}", rng, conds=["M", "shock"], stat=stat)
                rows.append({"variant": name, "lam": lam, "measure": f"vs {REF}: {stat}", "diff": pt, "lo": lo, "hi": hi})
            pt, lo, hi, _ = paired_stats(ds, f"AC@{lab(lam)}", f"DDQN@{lab(lam)}", rng, conds=["M", "shock"], col="objective")
            rows.append({"variant": name, "lam": lam, "measure": "vs AC: objective", "diff": pt, "lo": lo, "hi": hi})
    sens = pd.DataFrame(rows)
    if len(sens):
        sens.to_csv(os.path.join(args.out, "sensitivity.csv"), index=False)
        print(sens.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    fig_frontier(ov, os.path.join(args.out, "fig_frontier.png"))
    fig_forced(ov, os.path.join(args.out, "fig_forced_sweep.png"))
    best = prim[(prim.stat == "CVaR95")].sort_values("diff").iloc[0].lam if len(prim) else 0.5
    fig_hist(df, [REF, f"DDQN@{lab(best)}", "PPO@0.5", "AC@0.5"], os.path.join(args.out, "fig_cost_hist.png"))
    fig_learning(args.runs, os.path.join(args.out, "fig_learning.png"))
    print("\nfigures and CSVs in", args.out)


if __name__ == "__main__":
    main()
