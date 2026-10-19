"""Statistics, tables and figures for the report.

    python3 experiments/report.py --eval out/eval --eval-ac out/eval_ac --runs runs --out out/report

Statistical protocol (Agarwal et al., 2021, "Deep RL at the edge of the statistical precipice"):
  * an RL method's score is computed per training run (seed); 5 runs per method and condition;
  * 95% intervals by stratified bootstrap: resample runs, then episodes within each resampled run;
  * IQM across runs; probability of improvement P(PPO run beats DDQN run) over all run pairs;
  * versus TWAP, differences are paired by episode (common random numbers).
Risk: standard deviation and CVaR_95 (mean of the worst 5% of shortfalls) across episodes.
"""
import argparse
import glob
import json
import os
import re

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

B = 2000
CONDS = ["H", "M", "L", "HML", "shock", "markov"]
# Experiment letters follow the Review-1 deck and the report: A = High, B = Medium, C = Low.
COND_LABEL = {"H": "A: High", "M": "B: Medium", "L": "C: Low", "HML": "D1: H→M→L", "shock": "D2: H→L shock",
              "markov": "D3: Markov switching"}
BASE = ["TWAP", "TWAP-LO", "VWAP", "POV", "AC", "AC-adaptive"]
RUN_RE = re.compile(r"(lob|ac)_(DDQN|PPO)_(\w+?)(_oracle)?_s(\d+)$")


def parse(policy):
    m = RUN_RE.match(policy)
    if not m:
        return None
    world, algo, train, oracle, seed = m.groups()
    return {"world": world, "algo": algo, "train": train, "oracle": bool(oracle), "seed": int(seed)}


def method_of(policy):
    p = parse(policy)
    if p is None:
        return policy
    return f"{p['algo']}[{p['train']}{'+oracle' if p['oracle'] else ''}{'|AC-world' if p['world'] == 'ac' else ''}]"


def cvar(x, a=0.95):
    x = np.sort(np.asarray(x))
    return x[int(np.floor(a * len(x))):].mean()


def iqm(x):
    x = np.sort(np.asarray(x))
    n = len(x)
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return x[lo:hi].mean() if hi > lo else x.mean()


def strat_boot(mat, stat=np.mean, rng=None, b=B):
    """mat: list of per-run arrays of episode values. Returns (point, lo, hi) of stat over run means."""
    rng = rng or np.random.default_rng(0)
    point = stat([m.mean() for m in mat])
    vals = []
    for _ in range(b):
        runs = rng.integers(0, len(mat), len(mat))
        vals.append(stat([mat[r][rng.integers(0, len(mat[r]), len(mat[r]))].mean() for r in runs]))
    return point, np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def load(eval_dir):
    dfs = [pd.read_csv(f) for f in glob.glob(os.path.join(eval_dir, "*.csv"))]
    df = pd.concat(dfs, ignore_index=True)
    df["method"] = df.policy.map(method_of)
    return df


def method_table(df, rng):
    """Per condition and method: IS, delta vs TWAP (paired), sd, CVaR, objective, completion, limit share."""
    rows = []
    for cond, d in df.groupby("condition"):
        tw = d[d.policy == "TWAP"].set_index("episode").is_bps
        for meth, g in d.groupby("method"):
            runs = [x.set_index("episode") for _, x in g.groupby("policy")]
            is_runs = [r.is_bps.to_numpy() for r in runs]
            diff_runs = [(r.is_bps - tw.reindex(r.index)).to_numpy() for r in runs]
            m, lo, hi = strat_boot(is_runs, rng=rng) if len(runs) > 1 else (is_runs[0].mean(), *np.percentile(
                [rng.choice(is_runs[0], len(is_runs[0])).mean() for _ in range(B)], [2.5, 97.5]))
            dm, dlo, dhi = strat_boot(diff_runs, rng=rng) if len(runs) > 1 else (diff_runs[0].mean(), *np.percentile(
                [rng.choice(diff_runs[0], len(diff_runs[0])).mean() for _ in range(B)], [2.5, 97.5]))
            allv = np.concatenate(is_runs)
            rows.append({"condition": cond, "method": meth, "runs": len(runs), "is": m, "is_lo": lo, "is_hi": hi,
                         "d_twap": dm, "d_lo": dlo, "d_hi": dhi, "sd": np.mean([x.std() for x in is_runs]),
                         "cvar95": np.mean([cvar(x) for x in is_runs]), "iqm_runs": iqm([x.mean() for x in is_runs]),
                         "objective": g.objective.mean(), "completion": g.completion.mean(),
                         "limit_share": g.limit_share.mean(),
                         **{c: (g[c].mean() if c in g else np.nan) for c in ("c_delay", "c_spread", "c_impact", "c_passive", "c_fees")}})
    return pd.DataFrame(rows)


def head_to_head(df, train, rng):
    """PPO minus DDQN (same training condition), per test condition."""
    out = []
    for cond, d in df.groupby("condition"):
        A = [g.set_index("episode").is_bps.to_numpy() for p, g in d.groupby("policy")
             if (q := parse(p)) and q["world"] == "lob" and not q["oracle"] and q["train"] == train and q["algo"] == "DDQN"]
        P = [g.set_index("episode").is_bps.to_numpy() for p, g in d.groupby("policy")
             if (q := parse(p)) and q["world"] == "lob" and not q["oracle"] and q["train"] == train and q["algo"] == "PPO"]
        Ao = [g.set_index("episode").objective.to_numpy() for p, g in d.groupby("policy")
              if (q := parse(p)) and q["world"] == "lob" and not q["oracle"] and q["train"] == train and q["algo"] == "DDQN"]
        Po = [g.set_index("episode").objective.to_numpy() for p, g in d.groupby("policy")
              if (q := parse(p)) and q["world"] == "lob" and not q["oracle"] and q["train"] == train and q["algo"] == "PPO"]
        if not A or not P:
            continue
        vals, vals_o = [], []
        for _ in range(B):
            a = [A[i][rng.integers(0, len(A[i]), len(A[i]))].mean() for i in rng.integers(0, len(A), len(A))]
            p = [P[i][rng.integers(0, len(P[i]), len(P[i]))].mean() for i in rng.integers(0, len(P), len(P))]
            vals.append(np.mean(p) - np.mean(a))
            ao = [Ao[i][rng.integers(0, len(Ao[i]), len(Ao[i]))].mean() for i in rng.integers(0, len(Ao), len(Ao))]
            po = [Po[i][rng.integers(0, len(Po[i]), len(Po[i]))].mean() for i in rng.integers(0, len(Po), len(Po))]
            vals_o.append(np.mean(po) - np.mean(ao))
        am, pm = [x.mean() for x in A], [x.mean() for x in P]
        aom, pom = [x.mean() for x in Ao], [x.mean() for x in Po]
        poi = np.mean([[pi < ai for ai in am] for pi in pm])
        out.append({"condition": cond, "train": train, "ddqn_is": np.mean(am), "ppo_is": np.mean(pm),
                    "ppo_minus_ddqn": np.mean(pm) - np.mean(am), "lo": np.percentile(vals, 2.5), "hi": np.percentile(vals, 97.5),
                    "obj_ppo_minus_ddqn": np.mean(pom) - np.mean(aom), "obj_lo": np.percentile(vals_o, 2.5),
                    "obj_hi": np.percentile(vals_o, 97.5), "P(PPO better)": poi,
                    "ddqn_iqm": iqm(am), "ppo_iqm": iqm(pm), "ddqn_runs": am, "ppo_runs": pm})
    return pd.DataFrame(out)


def cross_regime(df):
    rows = []
    for (cond, meth), g in df[df.condition.isin(["L", "M", "H"])].groupby(["condition", "method"]):
        q = parse(g.policy.iloc[0])
        if q and q["world"] == "lob" and not q["oracle"]:
            rows.append({"algo": q["algo"], "train": q["train"], "test": cond, "is": g.is_bps.mean(), "obj": g.objective.mean()})
    t = pd.DataFrame(rows)
    out = {}
    for algo, g in t.groupby("algo"):
        piv = g.pivot(index="train", columns="test", values="obj").reindex(index=["L", "M", "H", "mixed"], columns=["L", "M", "H"])
        diag = np.array([piv.loc[c, c] for c in ["L", "M", "H"]])
        out[algo] = {"objective": piv, "regret_vs_specialist": piv - diag}
    return out


def adaptation(df, cond, switch_step, N=20):
    """Mean shares sold per interval (in TWAP slices) in the 3 intervals before and after a switch."""
    rows = []
    for meth, g in df[df.condition == cond].groupby("method"):
        P = np.array([[int(v) for v in s.split()] for s in g.path])
        X = P[:, 0].mean()
        sold = -np.diff(P, axis=1) / (X / N)
        rows.append({"method": meth, "before": sold[:, switch_step - 3:switch_step].mean(),
                     "after": sold[:, switch_step:switch_step + 3].mean(),
                     "inv_at_switch": P[:, switch_step].mean() / X, "path": (P.mean(0) / X).tolist()})
    r = pd.DataFrame(rows)
    r["change"] = r.after - r.before
    return r


def fig_paths(df, cond, switch_steps, methods, path):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for meth in methods:
        g = df[(df.condition == cond) & (df.method == meth)]
        if len(g) == 0:
            continue
        P = np.array([[int(v) for v in s.split()] for s in g.path])
        ax.plot(np.arange(P.shape[1]), P.mean(0) / P[:, 0].mean(), label=meth, lw=2 if "[" in meth else 1.2,
                ls="-" if "[" in meth else "--")
    for s in switch_steps:
        ax.axvline(s, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("decision step (30 s each)")
    ax.set_ylabel("remaining inventory / X")
    ax.set_title(COND_LABEL[cond])
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_learning(runs_dir, path, train="mixed"):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for algo, col in (("DDQN", "C0"), ("PPO", "C1")):
        curves = []
        for d in sorted(glob.glob(os.path.join(runs_dir, f"lob_{algo}_{train}_s*"))):
            L = np.loadtxt(os.path.join(d, "log.csv"), delimiter=",", skiprows=1)
            grid = np.linspace(L[0, 0], L[-1, 0], 100)
            sm = pd.Series(L[:, 2]).rolling(500, min_periods=50).mean().to_numpy()
            curves.append(np.interp(grid, L[:, 0], sm))
        if curves:
            C = np.array(curves)
            ax.plot(grid / 1000, C.mean(0), color=col, label=f"{algo} (mean of {len(C)} seeds)")
            ax.fill_between(grid / 1000, C.min(0), C.max(0), color=col, alpha=0.2)
    ax.set_xlabel("environment steps (thousands)")
    ax.set_ylabel("training shortfall, bps (500-episode mean)")
    ax.set_title(f"Learning curves, trained on '{train}' liquidity")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_bars(tab, methods, path):
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), sharey=False)
    for ax, cond in zip(axes.ravel(), CONDS):
        if cond not in set(tab.condition):
            ax.axis("off")
            continue
        t = tab[(tab.condition == cond)].set_index("method").reindex(methods).dropna(subset=["is"])
        y = np.arange(len(t))
        ax.errorbar(t["is"], y, xerr=[t["is"] - t.is_lo, t.is_hi - t["is"]], fmt="o", ms=4, capsize=2)
        ax.set_yticks(y)
        ax.set_yticklabels(t.index, fontsize=7)
        ax.invert_yaxis()
        ax.set_title(COND_LABEL[cond], fontsize=9)
        ax.set_xlabel("shortfall, bps (95% CI)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_heat(cr, path):
    fig, axes = plt.subplots(1, len(cr), figsize=(4.2 * len(cr), 3.2))
    vmax = max(np.nanmax(np.abs(d["regret_vs_specialist"].to_numpy())) for d in cr.values())
    for ax, (algo, d) in zip(np.atleast_1d(axes), cr.items()):
        R = d["regret_vs_specialist"]
        im = ax.imshow(R.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(R.shape[1]), [f"test {c}" for c in R.columns])
        ax.set_yticks(range(R.shape[0]), [f"train {c}" for c in R.index])
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                ax.text(j, i, f"{R.iloc[i, j]:+.2f}", ha="center", va="center", fontsize=8)
        ax.set_title(f"{algo}: regret vs specialist (objective, bps; same scale)", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_risk_return(tab, conds, methods, path):
    fig, axes = plt.subplots(1, len(conds), figsize=(4.6 * len(conds), 3.6))
    for ax, cond in zip(np.atleast_1d(axes), conds):
        t = tab[tab.condition == cond].set_index("method").reindex(methods).dropna(subset=["is"])
        for m, r in t.iterrows():
            rl = "[" in m
            ax.scatter(r.cvar95, r["is"], s=40 if rl else 25, marker="o" if rl else "s",
                       color="C3" if m.startswith("PPO") else "C0" if m.startswith("DDQN") else "grey")
            ax.annotate(m, (r.cvar95, r["is"]), fontsize=7, xytext=(4, 2), textcoords="offset points")
        ax.set_xlabel("CVaR95 of shortfall, bps (tail risk)")
        ax.set_ylabel("mean shortfall, bps")
        ax.set_title(COND_LABEL[cond], fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="out/eval")
    ap.add_argument("--eval-ac", default="out/eval_ac")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="out/report")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)
    df = load(args.eval)
    tab = method_table(df, rng)
    tab.to_csv(os.path.join(args.out, "methods.csv"), index=False)
    h2h = pd.concat([head_to_head(df, tr, rng) for tr in ("mixed", "L", "M", "H")], ignore_index=True)
    h2h.to_csv(os.path.join(args.out, "ddqn_vs_ppo.csv"), index=False)
    cr = cross_regime(df)
    for algo, d in cr.items():
        d["objective"].to_csv(os.path.join(args.out, f"cross_regime_{algo}.csv"))
    have = set(df.condition)
    adapt = {c: adaptation(df, c, s) for c, s in (("shock", 10), ("HML", 7)) if c in have}
    for c, a in adapt.items():
        a.drop(columns="path").to_csv(os.path.join(args.out, f"adaptation_{c}.csv"), index=False)

    # robustness: degradation ratio and shock surge per method
    piv = tab.pivot(index="method", columns="condition", values="is")
    cv = tab.pivot(index="method", columns="condition", values="cvar95")
    get = lambda t, c: t[c] if c in t else np.nan
    rob = pd.DataFrame({"IS_H": get(piv, "H"), "IS_L": get(piv, "L"), "degradation_L_over_H": get(piv, "L") / get(piv, "H"),
                        "shock_surge": get(piv, "shock") - get(piv, "H"), "CVaR95_shock": get(cv, "shock"),
                        "CVaR95_H": get(cv, "H")})
    rob.to_csv(os.path.join(args.out, "robustness.csv"))

    main_methods = BASE + ["DDQN[mixed]", "PPO[mixed]"]
    fig_bars(tab, main_methods, os.path.join(args.out, "fig_is_by_condition.png"))
    fig_learning(args.runs, os.path.join(args.out, "fig_learning.png"))
    fig_risk_return(tab, [c for c in ("M", "shock") if c in set(tab.condition)], main_methods,
                    os.path.join(args.out, "fig_risk_return.png"))
    if cr:
        fig_heat(cr, os.path.join(args.out, "fig_cross_regime.png"))
    if "shock" in have:
        fig_paths(df, "shock", [10], ["TWAP", "AC", "AC-adaptive", "POV", "DDQN[mixed]", "PPO[mixed]"],
              os.path.join(args.out, "fig_paths_shock.png"))
    if "HML" in have:
        fig_paths(df, "HML", [7, 14], ["TWAP", "AC", "AC-adaptive", "POV", "DDQN[mixed]", "PPO[mixed]"],
              os.path.join(args.out, "fig_paths_hml.png"))

    res = {"methods": tab.to_dict("records"), "ddqn_vs_ppo": h2h.to_dict("records"),
           "cross_regime": {a: {k: v.to_dict() for k, v in d.items()} for a, d in cr.items()},
           "adaptation": {c: a.to_dict("records") for c, a in adapt.items()}, "robustness": rob.reset_index().to_dict("records")}

    # sub-question 4: idealised AC world vs data-calibrated LOB
    if os.path.isdir(args.eval_ac) and glob.glob(os.path.join(args.eval_ac, "*.csv")):
        dac = load(args.eval_ac)
        tac = method_table(dac, rng)
        tac.to_csv(os.path.join(args.out, "methods_acworld.csv"), index=False)
        res["ac_world"] = tac.to_dict("records")
    json.dump(res, open(os.path.join(args.out, "results.json"), "w"), indent=1, default=float)

    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.2f}".format):
        for cond in [c for c in CONDS if c in have]:
            t = tab[tab.condition == cond].set_index("method").reindex(
                main_methods + [m for m in tab.method.unique() if m not in main_methods]).dropna(subset=["is"])
            print(f"\n=== {COND_LABEL[cond]}")
            print(t[["runs", "is", "is_lo", "is_hi", "d_twap", "d_lo", "d_hi", "sd", "cvar95", "objective", "completion", "limit_share"]])
        print("\n=== PPO minus DDQN (negative = PPO better)")
        print(h2h.drop(columns=["ddqn_runs", "ppo_runs"]))
        for algo, d in cr.items():
            print(f"\n=== cross-regime objective, {algo}\n", d["objective"])
        for c, a in adapt.items():
            print(f"\n=== adaptation around the switch ({c})\n", a.drop(columns="path"))
        print("\n=== robustness\n", rob)


if __name__ == "__main__":
    main()
