"""Statistics, tables and figures for the v2 study (CVaR distributional RL), following docs/v2/preregistration.md.

    python3 experiments/report_v2.py [--eval out_v2/eval] [--runs runs_v2] [--out out_v2/report]

Uses the Phase-1 helpers of experiments/report.py (cvar, iqm, strat_boot) unchanged. Contrasts between
arms are paired by training seed (seeds 10-19 in every arm) and by test episode (common random numbers):
each bootstrap resample draws seeds, then episodes within each setting, and applies the same draw to both
arms. CVaR95 is computed per run (worst 5% of that run's 1,000 episodes) and then averaged over runs.
"""
import argparse
import glob
import json
import os
import re
import sys

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from report import COND_LABEL, CONDS, cvar, iqm, strat_boot   # noqa: E402
from train_v2 import load_agent                                # noqa: E402
from liqexec.env import ExecutionEnv                           # noqa: E402

B = 2000
ARMS = ["DDQN", "PPO", "QRDDQN-RN", "QRDDQN-CVaR25", "QRDDQN-CVaR10", "QRDDQN-THR25"]
BASE = ["TWAP", "TWAP-LO", "VWAP", "POV", "AC", "AC-adaptive"]
RUN_RE = re.compile(r"lob_(.+)_mixed_s(\d+)$")
PRIMARY, CONTROL = "QRDDQN-CVaR25", "DDQN"
SESOI_CVAR, NI_MARGIN = 1.0, 0.3
# reference categorical palette (dataviz skill, light mode), fixed order by arm
COL = {"DDQN": "#2a78d6", "PPO": "#eb6834", "QRDDQN-RN": "#1baf7a", "QRDDQN-CVaR25": "#eda100",
       "QRDDQN-CVaR10": "#e87ba4", "QRDDQN-THR25": "#6250d6"}
MARK = {"DDQN": "o", "PPO": "s", "QRDDQN-RN": "^", "QRDDQN-CVaR25": "D", "QRDDQN-CVaR10": "v", "QRDDQN-THR25": "P"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def arm_of(policy):
    m = RUN_RE.match(policy)
    return (m.group(1), int(m.group(2))) if m else (policy, None)


def load(eval_dir):
    df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(eval_dir, "*.csv"))], ignore_index=True)
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


# ------------------------------------------------------------------ matrices and paired bootstrap
def matrix(df, cond, arm, col="is_bps"):
    """(seeds x episodes) array, rows ordered by seed, columns by episode; and the seed list."""
    g = df[(df.condition == cond) & (df.arm == arm)]
    piv = g.pivot(index="seed", columns="episode", values=col).sort_index()
    return piv.to_numpy(), list(piv.index)


def paired_stats(df, a, b, conds, rng, stat="mean", col="is_bps"):
    """b minus a, averaged over conds. stat 'mean' or 'cvar'. Returns point, lo, hi, p (two-sided), draws."""
    mats = []
    for c in conds:
        A, sa = matrix(df, c, a, col)
        Bm, sb = matrix(df, c, b, col)
        common = sorted(set(sa) & set(sb))
        A = A[[sa.index(s) for s in common]]
        Bm = Bm[[sb.index(s) for s in common]]
        mats.append((A, Bm))
    f = (lambda x: x.mean(1)) if stat == "mean" else (lambda x: np.sort(x, 1)[:, int(np.floor(0.95 * x.shape[1])):].mean(1))
    point = np.mean([(f(Bm) - f(A)).mean() for A, Bm in mats])
    n_s = mats[0][0].shape[0]
    draws = np.empty(B)
    for k in range(B):
        si = rng.integers(0, n_s, n_s)
        tot = 0.0
        for A, Bm in mats:
            ei = rng.integers(0, A.shape[1], (n_s, A.shape[1]))
            Ar = np.take_along_axis(A[si], ei, 1)
            Br = np.take_along_axis(Bm[si], ei, 1)
            tot += (f(Br) - f(Ar)).mean()
        draws[k] = tot / len(mats)
    p = min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean()))
    return point, np.percentile(draws, 2.5), np.percentile(draws, 97.5), p


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(ps) - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj


# ------------------------------------------------------------------ tables
def arm_table(df, rng):
    rows = []
    for cond in CONDS:
        d = df[df.condition == cond]
        if d.empty:
            continue
        for arm in ARMS + BASE:
            g = d[d.arm == arm]
            if g.empty:
                continue
            runs = [x.sort_values("episode") for _, x in g.groupby("policy")]
            isr = [r.is_bps.to_numpy() for r in runs]
            if len(runs) > 1:
                m, lo, hi = strat_boot(isr, rng=rng, b=B)
                cv = [cvar(x) for x in isr]
                cvm, cvlo, cvhi = strat_boot([np.array([c]) for c in cv], rng=rng, b=B)
            else:
                m = isr[0].mean()
                lo, hi = np.percentile([rng.choice(isr[0], len(isr[0])).mean() for _ in range(B)], [2.5, 97.5])
                cvm = cvar(isr[0])
                cvlo, cvhi = np.percentile([cvar(rng.choice(isr[0], len(isr[0]))) for _ in range(B)], [2.5, 97.5])
            rows.append({"condition": cond, "arm": arm, "runs": len(runs), "is": m, "is_lo": lo, "is_hi": hi,
                         "cvar95": cvm, "cvar_lo": cvlo, "cvar_hi": cvhi, "sd": np.mean([x.std() for x in isr]),
                         "iqm_runs": iqm([x.mean() for x in isr]), "objective": g.objective.mean(),
                         "sold_before_T": g.completion.mean(), "forced": g.forced.mean(),
                         "limit_share": g.limit_share.mean(), "inv_k10": g.inv_k10.mean()})
    return pd.DataFrame(rows)


def overall_table(df, rng):
    """Per arm, averaged over the six settings: mean IS and CVaR95 with seed-then-episode bootstrap CIs."""
    rows = []
    for arm in ARMS:
        mats = [matrix(df, c, arm)[0] for c in CONDS]
        if any(m.size == 0 for m in mats):
            continue
        f_mean = lambda x: x.mean(1)                                         # noqa: E731
        f_cvar = lambda x: np.sort(x, 1)[:, int(np.floor(0.95 * x.shape[1])):].mean(1)  # noqa: E731
        n_s = mats[0].shape[0]
        dm, dc = np.empty(B), np.empty(B)
        for k in range(B):
            si = rng.integers(0, n_s, n_s)
            sm = sc = 0.0
            for M in mats:
                ei = rng.integers(0, M.shape[1], (n_s, M.shape[1]))
                R = np.take_along_axis(M[si], ei, 1)
                sm += f_mean(R).mean()
                sc += f_cvar(R).mean()
            dm[k], dc[k] = sm / len(mats), sc / len(mats)
        g = df[df.arm == arm]
        rows.append({"arm": arm, "seeds": n_s, "is": np.mean([f_mean(M).mean() for M in mats]),
                     "is_lo": np.percentile(dm, 2.5), "is_hi": np.percentile(dm, 97.5),
                     "cvar95": np.mean([f_cvar(M).mean() for M in mats]),
                     "cvar_lo": np.percentile(dc, 2.5), "cvar_hi": np.percentile(dc, 97.5),
                     "objective": g.objective.mean(), "sold_before_T": g.completion.mean(), "forced": g.forced.mean(),
                     "limit_share": g.limit_share.mean()})
    return pd.DataFrame(rows)


def tail_mechanism(df):
    """Forced-sweep share in each run's worst-5% episodes vs its other episodes (all settings pooled)."""
    rows = []
    for arm in ARMS:
        for cond in CONDS:
            g = df[(df.arm == arm) & (df.condition == cond)]
            for _, r in g.groupby("policy"):
                cut = np.quantile(r.is_bps, 0.95)
                tail, body = r[r.is_bps >= cut], r[r.is_bps < cut]
                rows.append({"arm": arm, "condition": cond, "forced_tail": tail.forced.mean(),
                             "forced_body": body.forced.mean(),
                             "corr_forced_is": np.corrcoef(r.forced, r.is_bps)[0, 1] if r.forced.std() > 0 else np.nan})
    t = pd.DataFrame(rows)
    return t.groupby("arm")[["forced_tail", "forced_body", "corr_forced_is"]].mean().reindex(ARMS).dropna(how="all")


def calibration(df, runs_dir, conds=("M", "shock"), episodes=200):
    """Learned return quantiles at t = 0 (chosen action) vs the realised return distribution (-objective)."""
    rows, curves = [], {}
    for arm in [a for a in ARMS if a.startswith("QRDDQN")]:
        for cond in conds:
            pred_all, real_all = [], []
            for d in sorted(glob.glob(os.path.join(runs_dir, f"lob_{arm}_mixed_s*"))):
                ag = load_agent(d)
                env = ExecutionEnv(schedule=cond)
                preds = []
                for i in range(episodes):
                    s, _ = env.reset(seed=10_000_000 + i)
                    ag.begin_episode(s)
                    a = ag.act(s)
                    with torch.no_grad():
                        z = ag.quantiles(torch.as_tensor(ag._aug(s, ag.c))[None])[0, a]
                    preds.append(np.sort(z.numpy()))
                g = df[(df.policy == os.path.basename(d)) & (df.condition == cond)].sort_values("episode")
                real = -g.objective.to_numpy()[:episodes]
                P = np.mean(preds, 0)
                taus = (np.arange(len(P)) + 0.5) / len(P)
                R = np.quantile(real, taus)
                pred_all.append(P)
                real_all.append(R)
                m = max(1, int(round(0.25 * len(P))))
                rows.append({"arm": arm, "condition": cond, "run": os.path.basename(d),
                             "mean_abs_quantile_err": np.abs(P - R).mean(), "pred_mean": P.mean(), "real_mean": real.mean(),
                             "pred_cvar25": P[:m].mean(), "real_cvar25": np.sort(real)[:int(0.25 * len(real))].mean()})
            if pred_all:
                curves[(arm, cond)] = (np.mean(pred_all, 0), np.mean(real_all, 0))
    t = pd.DataFrame(rows)
    return t, curves


# ------------------------------------------------------------------ figures
SHORT = {"DDQN": "DDQN", "PPO": "PPO", "QRDDQN-RN": "QR-RN", "QRDDQN-CVaR25": "QR-CVaR25 (primary)",
         "QRDDQN-CVaR10": "QR-CVaR10", "QRDDQN-THR25": "QR-THR25", "TWAP": "TWAP (baseline)", "AC": "AC (baseline)"}
OFFSET = {"QRDDQN-CVaR25": (-10, -14), "QRDDQN-CVaR10": (8, 6), "QRDDQN-THR25": (8, 6), "TWAP": (-10, -14),
          "AC": (-12, -4)}


def fig_scatter(overall, per_cond, path):
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    style(ax)
    pos = lambda v: max(0.0, v)                                         # noqa: E731
    for b, mk in (("TWAP", "x"), ("AC", "*")):
        t = per_cond[per_cond.arm == b]
        if len(t):
            x, y = t["is"].mean(), t.cvar95.mean()
            ax.plot(x, y, mk, ms=10, color=INK2)
            dx, dy = OFFSET[b]
            ax.annotate(SHORT[b], (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8, color=INK2,
                        ha="right" if dx < 0 else "left")
    for _, r in overall.iterrows():
        a = r.arm
        ax.errorbar(r["is"], r.cvar95, xerr=[[pos(r["is"] - r.is_lo)], [pos(r.is_hi - r["is"])]],
                    yerr=[[pos(r.cvar95 - r.cvar_lo)], [pos(r.cvar_hi - r.cvar95)]], fmt=MARK[a], ms=8, color=COL[a],
                    mec="white", mew=1.5, ecolor=COL[a], elinewidth=1.2, capsize=0)
        dx, dy = OFFSET.get(a, (8, 6))
        ax.annotate(SHORT[a], (r["is"], r.cvar95), xytext=(dx, dy), textcoords="offset points", fontsize=8, color=INK,
                    ha="right" if dx < 0 else "left")
    ax.set_xlabel("mean shortfall, bps (average of settings A–D3)", color=INK2, fontsize=9)
    ax.set_ylabel("CVaR95 of shortfall, bps", color=INK2, fontsize=9)
    ax.set_title("Mean vs tail cost; bars are 95% CIs (agents: 10 seeds, trained on 'mixed')", fontsize=9.5, color=INK,
                 loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_hist(df, path, arms=("DDQN", "PPO", "QRDDQN-RN", "QRDDQN-CVaR25"), conds=("M", "shock")):
    fig, axes = plt.subplots(1, len(conds), figsize=(9.6, 3.6), sharey=True)
    for ax, cond in zip(axes, conds):
        style(ax)
        d = df[df.condition == cond]
        lo, hi = np.percentile(d[d.arm.isin(arms)].is_bps, [0.5, 99.8])
        bins = np.linspace(lo, hi, 60)
        for a in arms:
            x = d[d.arm == a].is_bps
            ax.hist(x, bins=bins, histtype="step", lw=2, color=COL[a], density=True, label=a)
            ax.axvline(np.mean([cvar(r.is_bps.to_numpy()) for _, r in d[d.arm == a].groupby("policy")]),
                       color=COL[a], lw=1, ls=":")
        ax.set_title(COND_LABEL[cond], fontsize=9, color=INK, loc="left")
        ax.set_xlabel("shortfall, bps (dotted: CVaR95)", color=INK2, fontsize=8)
    axes[0].set_ylabel("density (10 seeds × 1,000 episodes)", color=INK2, fontsize=8)
    axes[-1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_sold(tab, path):
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    style(ax)
    arms = [a for a in ARMS if a in set(tab.arm)]
    w = 0.8 / len(arms)
    x = np.arange(len(CONDS))
    for i, a in enumerate(arms):
        t = tab[tab.arm == a].set_index("condition").reindex(CONDS)
        ax.bar(x + (i - (len(arms) - 1) / 2) * w, 100 * t.forced, w * 0.9, color=COL[a], label=SHORT[a],
               edgecolor="white", lw=1)
    ax.set_xticks(x, [COND_LABEL[c] for c in CONDS], fontsize=7)
    ax.set_ylabel("% of X sold by the forced order at T", color=INK2, fontsize=8)
    ax.set_title("Inventory left to the forced end-of-horizon sweep (lower = more sold before T)", fontsize=9.5,
                 color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_calib(curves, path):
    keys = list(curves)
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(2.6 * len(keys), 2.8), sharey=False)
    for ax, k in zip(np.atleast_1d(axes), keys):
        style(ax)
        P, R = curves[k]
        taus = (np.arange(len(P)) + 0.5) / len(P)
        ax.plot(taus, R, color=INK2, lw=2, label="realised")
        ax.plot(taus, P, color=COL[k[0]], lw=2, label="learned")
        ax.set_title(f"{k[0]}\n{COND_LABEL[k[1]]}", fontsize=7, color=INK, loc="left")
        ax.set_xlabel("τ", fontsize=7, color=INK2)
    np.atleast_1d(axes)[0].set_ylabel("return quantile, bps", fontsize=7, color=INK2)
    np.atleast_1d(axes)[0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_learning(runs_dir, path):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    style(ax)
    for a in ARMS:
        curves = []
        for d in sorted(glob.glob(os.path.join(runs_dir, f"lob_{a}_mixed_s*"))):
            L = np.loadtxt(os.path.join(d, "log.csv"), delimiter=",", skiprows=1)
            grid = np.linspace(0, 250_000, 100)
            sm = pd.Series(L[:, 2]).rolling(500, min_periods=50).mean().to_numpy()
            curves.append(np.interp(grid, L[:, 0], sm))
        if curves:
            C = np.array(curves)
            ax.plot(grid / 1000, C.mean(0), color=COL[a], lw=2, label=f"{a} ({len(C)} seeds)")
    ax.set_xlabel("environment steps (thousands)", color=INK2, fontsize=8)
    ax.set_ylabel("training shortfall, bps (500-episode mean)", color=INK2, fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="out_v2/eval")
    ap.add_argument("--runs", default="runs_v2")
    ap.add_argument("--out", default="out_v2/report")
    ap.add_argument("--calib-episodes", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)
    df = load(args.eval)
    pd.set_option("display.width", 200)
    fmt = lambda x: f"{x:+.2f}"     # noqa: E731

    print("=" * 100)
    print("PRIMARY CONTRASTS (pre-registered): QRDDQN-CVaR25 minus DDQN, averaged over settings A-D3")
    t_pt, t_lo, t_hi, t_p = paired_stats(df, CONTROL, PRIMARY, CONDS, rng, stat="cvar")
    m_pt, m_lo, m_hi, m_p = paired_stats(df, CONTROL, PRIMARY, CONDS, rng, stat="mean")
    verdict_t = ("meaningful reduction (CI below -SESOI)" if t_hi < -SESOI_CVAR else "reduction (CI below 0)" if t_hi < 0
                 else f"no tail benefit beyond {-t_lo:.2f} bps" if t_lo < 0 else "tail worse (CI above 0)")
    verdict_m = "non-inferior" if m_hi < NI_MARGIN else "non-inferiority NOT shown"
    pf = lambda p: "p<0.001" if p < 0.001 else f"p={p:.3f}"     # noqa: E731
    print(f"  (i)  CVaR95 of IS: {fmt(t_pt)} bps [{fmt(t_lo)}, {fmt(t_hi)}]  {pf(t_p)}  -> {verdict_t} (SESOI {SESOI_CVAR} bps)")
    print(f"  (ii) mean IS:      {fmt(m_pt)} bps [{fmt(m_lo)}, {fmt(m_hi)}]  {pf(m_p)}  -> {verdict_m} (margin +{NI_MARGIN} bps)")
    prim = pd.DataFrame([{"contrast": "CVaR95 IS, CVaR25 - DDQN", "diff": t_pt, "lo": t_lo, "hi": t_hi, "p": t_p, "verdict": verdict_t},
                         {"contrast": "mean IS, CVaR25 - DDQN", "diff": m_pt, "lo": m_lo, "hi": m_hi, "p": m_p, "verdict": verdict_m}])
    prim.to_csv(os.path.join(args.out, "primary.csv"), index=False)

    print("\n" + "=" * 100)
    print("PER-ARM SUMMARY, averaged over A-D3 (seed-then-episode bootstrap 95% CIs)")
    ov = overall_table(df, rng)
    ov.to_csv(os.path.join(args.out, "overall.csv"), index=False)
    print(ov.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 100)
    print("SECONDARY CONTRASTS (b minus a; Holm-adjusted p within this family)")
    pairs = [(CONTROL, a) for a in ARMS[2:]] + [("PPO", a) for a in ARMS[2:]] + [("QRDDQN-RN", "QRDDQN-CVaR25"),
                                                                                  ("QRDDQN-CVaR25", "QRDDQN-THR25"),
                                                                                  ("QRDDQN-CVaR25", "QRDDQN-CVaR10"),
                                                                                  ("DDQN", "PPO")]
    sec = []
    present = set(df.arm)
    for a, b in pairs:
        if a not in present or b not in present:
            continue
        for stat in ("cvar", "mean"):
            for scope in ["all"] + CONDS:
                conds = CONDS if scope == "all" else [scope]
                pt, lo, hi, p = paired_stats(df, a, b, conds, rng, stat=stat)
                sec.append({"b": b, "a": a, "stat": "CVaR95" if stat == "cvar" else "mean", "setting": scope,
                            "diff": pt, "lo": lo, "hi": hi, "p": p})
        for scope in ["all"]:
            pt, lo, hi, p = paired_stats(df, a, b, CONDS, rng, stat="mean", col="objective")
            sec.append({"b": b, "a": a, "stat": "objective", "setting": scope, "diff": pt, "lo": lo, "hi": hi, "p": p})
    sec = pd.DataFrame(sec)
    sec["p_holm"] = holm(sec.p.to_numpy())
    sec.to_csv(os.path.join(args.out, "secondary.csv"), index=False)
    print(sec[sec.setting == "all"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 100)
    print("PER-SETTING TABLE (agents: 10 seeds; baselines: deterministic)")
    tab = arm_table(df, rng)
    tab.to_csv(os.path.join(args.out, "per_setting.csv"), index=False)
    for cond in CONDS:
        t = tab[tab.condition == cond]
        print(f"\n{COND_LABEL[cond]}")
        print(t[["arm", "is", "is_lo", "is_hi", "cvar95", "sd", "objective", "sold_before_T", "limit_share", "inv_k10"]]
              .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 100)
    print("MECHANISM: forced-sweep share of X in each run's worst-5% episodes vs the rest (mean over settings)")
    mech = tail_mechanism(df)
    mech.to_csv(os.path.join(args.out, "mechanism.csv"))
    print(mech.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 100)
    print(f"CALIBRATION of learned quantiles at t = 0 vs realised return (first {args.calib_episodes} test episodes)")
    cal, curves = calibration(df, args.runs, episodes=args.calib_episodes)
    cal.to_csv(os.path.join(args.out, "calibration.csv"), index=False)
    if len(cal):
        print(cal.groupby(["arm", "condition"])[["mean_abs_quantile_err", "pred_mean", "real_mean", "pred_cvar25", "real_cvar25"]]
              .mean().to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 100)
    print("SENSITIVITY (500 episodes, settings B and D2 pooled): arm minus DDQN, paired seed-episode bootstrap")
    sens = []
    for name, d in (("fast resilience (kappa x2)", "out_v2/eval_sens_fastres"), ("latency 0.5 s", "out_v2/eval_sens_latency"),
                    ("fees x3", "out_v2/eval_sens_fees")):
        if not glob.glob(os.path.join(d, "*.csv")):
            continue
        ds = load(d)
        for arm in ARMS[1:]:
            for stat in ("cvar", "mean"):
                pt, lo, hi, p = paired_stats(ds, CONTROL, arm, ["M", "shock"], rng, stat=stat)
                sens.append({"variant": name, "arm": arm, "stat": "CVaR95" if stat == "cvar" else "mean",
                             "diff": pt, "lo": lo, "hi": hi})
    sens = pd.DataFrame(sens)
    if len(sens):
        sens.to_csv(os.path.join(args.out, "sensitivity.csv"), index=False)
        print(sens.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    if glob.glob(os.path.join("out_v2/eval_rn4", "*.csv")):
        print("\n" + "=" * 100)
        print("EXPLORATORY (not pre-registered): QRDDQN-RN4 = risk-neutral QR-DDQN with the CVaR25 arm's hyperparameters")
        dx = pd.concat([df, load("out_v2/eval_rn4")], ignore_index=True)
        ex = []
        for a, b in (("DDQN", "QRDDQN-RN4"), ("QRDDQN-RN", "QRDDQN-RN4"), ("PPO", "QRDDQN-RN4"), ("QRDDQN-RN4", "QRDDQN-CVaR25")):
            for stat in ("cvar", "mean"):
                pt, lo, hi, p = paired_stats(dx, a, b, CONDS, rng, stat=stat)
                ex.append({"b": b, "a": a, "stat": "CVaR95" if stat == "cvar" else "mean", "diff": pt, "lo": lo, "hi": hi, "p": p})
        ex = pd.DataFrame(ex)
        ex.to_csv(os.path.join(args.out, "exploratory_rn4.csv"), index=False)
        print(ex.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
        g = dx[dx.arm == "QRDDQN-RN4"]
        print(f"QRDDQN-RN4: sold before T {g.completion.mean():.3f}, limit share {g.limit_share.mean():.3f}, "
              f"objective {g.objective.mean():.3f}")

    fig_scatter(ov, tab, os.path.join(args.out, "fig_cvar_vs_mean.png"))
    fig_hist(df, os.path.join(args.out, "fig_cost_hist.png"))
    fig_sold(tab, os.path.join(args.out, "fig_forced_sweep.png"))
    fig_calib(curves, os.path.join(args.out, "fig_quantile_calibration.png"))
    fig_learning(args.runs, os.path.join(args.out, "fig_learning.png"))
    print("\nfigures and CSVs in", args.out)


if __name__ == "__main__":
    main()
