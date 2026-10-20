"""Build the v2 report section (.docx) and slides (.pptx) from the generated result files.

    python3 docs/v2/build_docs.py --report-docx <Phase-1 report .docx> --deck <Phase-1 deck .pptx>

Every number is read from out_v2/report/*.csv; nothing is typed by hand. The section reuses the Phase-1
report's own paragraph, caption and table formatting (cloned from that file), and the slides are copies of
Phase-1 deck slides 3, 13 and 14 with new content, so both match the existing documents.
"""
import argparse
import copy
import os
import subprocess
import sys

import pandas as pd

R = "out_v2/report"
OUT = "docs/v2"
SHORT = {"DDQN": "DDQN", "PPO": "PPO", "QRDDQN-RN": "QR-DDQN, risk-neutral", "QRDDQN-CVaR25": "QR-DDQN-CVaR(0.25)",
         "QRDDQN-CVaR10": "QR-DDQN-CVaR(0.1)", "QRDDQN-THR25": "QR-DDQN, threshold CVaR(0.25)", "QRDDQN-RN4": "QR-DDQN-RN (DDQN update rate)"}
ARMS = ["DDQN", "PPO", "QRDDQN-RN", "QRDDQN-CVaR25", "QRDDQN-CVaR10", "QRDDQN-THR25"]


def s(x, d=2):
    return f"{x:+.{d}f}".replace("-", "−")


def u(x, d=2):
    return f"{x:.{d}f}".replace("-", "−")


def ci(r):
    return f"{s(r['diff'])} [{s(r['lo'])}, {s(r['hi'])}]"


class Data:
    def __init__(self):
        self.ov = pd.read_csv(f"{R}/overall.csv").set_index("arm")
        self.prim = pd.read_csv(f"{R}/primary.csv")
        self.sec = pd.read_csv(f"{R}/secondary.csv")
        self.tab = pd.read_csv(f"{R}/per_setting.csv")
        self.mech = pd.read_csv(f"{R}/mechanism.csv").set_index("arm")
        self.cal = pd.read_csv(f"{R}/calibration.csv")
        self.sens = pd.read_csv(f"{R}/sensitivity.csv")
        self.rn4 = pd.read_csv(f"{R}/exploratory_rn4.csv")

    def pair(self, b, a, stat, setting="all"):
        r = self.sec[(self.sec.b == b) & (self.sec.a == a) & (self.sec.stat == stat) & (self.sec.setting == setting)]
        return r.iloc[0]

    def x4(self, b, a, stat):
        return self.rn4[(self.rn4.b == b) & (self.rn4.a == a) & (self.rn4.stat == stat)].iloc[0]

    def base(self, arm, col):
        return self.tab[self.tab.arm == arm][col].mean()


def numbers(D):
    P = D.prim.set_index("contrast")
    t, m = P.iloc[0], P.iloc[1]
    o = D.ov
    cal = D.cal.groupby(["arm", "condition"])[["pred_cvar25", "real_cvar25"]].mean()
    sens = D.sens[D.sens.arm == "QRDDQN-CVaR25"]
    return dict(
        t=ci(t), m=ci(m), t_hi=s(t["hi"]), m_lo=s(m["lo"]),
        obj=ci(D.pair("QRDDQN-CVaR25", "DDQN", "objective")),
        rn_t=ci(D.pair("QRDDQN-RN", "DDQN", "CVaR95")), rn_m=ci(D.pair("QRDDQN-RN", "DDQN", "mean")),
        rn4_t=ci(D.x4("QRDDQN-RN4", "DDQN", "CVaR95")), rn4_m=ci(D.x4("QRDDQN-RN4", "DDQN", "mean")),
        risk_t=ci(D.x4("QRDDQN-CVaR25", "QRDDQN-RN4", "CVaR95")), risk_m=ci(D.x4("QRDDQN-CVaR25", "QRDDQN-RN4", "mean")),
        thr_t=ci(D.pair("QRDDQN-THR25", "QRDDQN-CVaR25", "CVaR95")),
        thr_p=D.pair("QRDDQN-THR25", "QRDDQN-CVaR25", "CVaR95")["p_holm"],
        a10_t=ci(D.pair("QRDDQN-CVaR10", "QRDDQN-CVaR25", "CVaR95")),
        ppo_t=ci(D.pair("QRDDQN-CVaR25", "PPO", "CVaR95")), ppo_m=ci(D.pair("QRDDQN-CVaR25", "PPO", "mean")),
        forced_d=100 * o.loc["DDQN", "forced"], forced_c=100 * o.loc["QRDDQN-CVaR25", "forced"],
        forced_p=100 * o.loc["PPO", "forced"], forced_thr=100 * o.loc["QRDDQN-THR25", "forced"],
        lim_d=100 * o.loc["DDQN", "limit_share"], lim_c=100 * o.loc["QRDDQN-CVaR25", "limit_share"],
        mt_d=100 * D.mech.loc["DDQN", "forced_tail"], mb_d=100 * D.mech.loc["DDQN", "forced_body"],
        mt_c=100 * D.mech.loc["QRDDQN-CVaR25", "forced_tail"], mb_c=100 * D.mech.loc["QRDDQN-CVaR25", "forced_body"],
        r_d=D.mech.loc["DDQN", "corr_forced_is"],
        shock_d=100 * D.tab[(D.tab.arm == "DDQN") & (D.tab.condition == "shock")].inv_k10.iloc[0],
        shock_c=100 * D.tab[(D.tab.arm == "QRDDQN-CVaR25") & (D.tab.condition == "shock")].inv_k10.iloc[0],
        shock_p=100 * D.tab[(D.tab.arm == "PPO") & (D.tab.condition == "shock")].inv_k10.iloc[0],
        ac_is=D.base("AC", "is"), ac_cv=D.base("AC", "cvar95"),
        c_is=o.loc["QRDDQN-CVaR25", "is"], c_cv=o.loc["QRDDQN-CVaR25", "cvar95"],
        d_is=o.loc["DDQN", "is"], d_cv=o.loc["DDQN", "cvar95"],
        cal_pred=cal.loc[("QRDDQN-CVaR25", "M"), "pred_cvar25"], cal_real=cal.loc[("QRDDQN-CVaR25", "M"), "real_cvar25"],
        sens_t_lo=sens[sens.stat == "CVaR95"]["diff"].max(), sens_t_hi=sens[sens.stat == "CVaR95"]["diff"].min(),
        sens_m_lo=sens[sens.stat == "mean"]["diff"].min(), sens_m_hi=sens[sens.stat == "mean"]["diff"].max(),
    )


# ====================================================================================== report section
def build_docx(D, N, src, out):
    from docx import Document
    from docx.shared import Inches
    doc = Document(src)
    paras = doc.paragraphs

    def find(style, startswith=None):
        for p in paras:
            st = p.style.name if p.style is not None else None
            if st == style and (startswith is None or p.text.startswith(startswith)):
                return p
        raise KeyError(style)

    tpl = {"h2": find("Heading 2", "5.3"), "h3": find("Heading 3"), "tcap": find("Table Caption", "Table 5.4"),
           "fcap": find("Figure Caption", "Figure 5.3")}
    body_p = [p for p in paras if (p.style is None or p.style.name in (None, "Normal")) and p.text.startswith("Figure 5.3 and Table 5.4")][0]
    fig_p = [p for p in paras if p._p.xpath(".//w:drawing")][3]
    tbl_tpl = copy.deepcopy(doc.tables[12]._tbl)
    lp = [p for p in paras if p.style is not None and p.style.name == "List Paragraph"][0]
    tpl.update(body=body_p, fig=fig_p, li=lp)
    tpl = {k: copy.deepcopy(v._p) for k, v in tpl.items()}

    body = doc.element.body
    sect = body[-1]
    for el in list(body)[:-1]:
        body.remove(el)

    from docx.text.paragraph import Paragraph
    W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

    def add(kind, text):
        p = copy.deepcopy(tpl[kind])
        runs = p.findall(f"{W}r")
        for r in runs[1:]:
            p.remove(r)
        for el in p.findall(f"{W}hyperlink") + p.findall(f"{W}bookmarkStart") + p.findall(f"{W}bookmarkEnd"):
            p.remove(el)
        sect.addprevious(p)
        para = Paragraph(p, doc._body)
        if runs:
            r = para.runs[0]
            for t in r._r.findall(f"{W}t")[1:]:
                r._r.remove(t)
            r.text = text
        else:
            para.add_run(text)
        return para

    def figure(path, caption, width=6.0):
        p = copy.deepcopy(tpl["fig"])
        for r in p.findall(f"{W}r"):
            p.remove(r)
        sect.addprevious(p)
        Paragraph(p, doc._body).add_run().add_picture(path, width=Inches(width))
        add("fcap", caption)

    def table(caption, header, rows, widths):
        add("tcap", caption)
        t = copy.deepcopy(tbl_tpl)
        trs = t.findall(f"{W}tr")
        h_tc = copy.deepcopy(trs[0].findall(f"{W}tc")[0])
        b_tc = copy.deepcopy(trs[1].findall(f"{W}tc")[1])
        b_tc0 = copy.deepcopy(trs[1].findall(f"{W}tc")[0])
        h_tr = copy.deepcopy(trs[0])
        b_tr = copy.deepcopy(trs[1])
        for tr in trs:
            t.remove(tr)
        grid = t.find(f"{W}tblGrid")
        for gc in list(grid):
            grid.remove(gc)
        tot = sum(widths)
        tw = [int(9026 * w / tot) for w in widths]
        for w in tw:
            gc = copy.deepcopy(tbl_tpl.find(f"{W}tblGrid")[0])
            gc.set(f"{W}w", str(w))
            grid.append(gc)

        def row(tr_tpl, cells, first_tpl, other_tpl):
            tr = copy.deepcopy(tr_tpl)
            for tc in tr.findall(f"{W}tc"):
                tr.remove(tc)
            for j, txt in enumerate(cells):
                tc = copy.deepcopy(first_tpl if j == 0 else other_tpl)
                tc.find(f"{W}tcPr").find(f"{W}tcW").set(f"{W}w", str(tw[j]))
                ps = tc.findall(f"{W}p")
                for extra in ps[1:]:
                    tc.remove(extra)
                para = Paragraph(ps[0], doc._body)
                rs = para.runs
                for r in rs[1:]:
                    r._r.getparent().remove(r._r)
                if rs:
                    rs[0].text = txt
                else:
                    para.add_run(txt)
                tr.append(tc)
            t.append(tr)
        row(h_tr, header, h_tc, h_tc)
        for r in rows:
            row(b_tr, r, b_tc0, b_tc)
        sect.addprevious(t)

    o = D.ov
    add("h2", "5.12  Tail-Risk-Aware Distributional Agents")
    add("body", "Section 5.3 found a mean-versus-tail trade-off between the two algorithms: DDQN had the lower mean "
        "shortfall, while PPO had the lower CVaR95 because it left less inventory to the forced market order at T. "
        "This section tests whether an agent that learns the whole cost distribution and acts on its lower tail can "
        "change that trade-off, and whether risk-sensitivity, rather than the value-based versus policy-gradient "
        "split, explains the tail difference. The design, contrasts and thresholds were fixed in a pre-registration "
        "file (docs/v2/preregistration.md) committed before any run.")

    add("h3", "5.12.1  Position in the Literature")
    add("body", "No published deep-RL execution study that we could find optimises a tail risk measure: the execution "
        "agents in Section 2.4, including the queue-reactive DDQN of Espana et al. (2025), minimise expected cost. "
        "CVaR-optimal execution has been solved by dynamic programming in an Almgren–Chriss-type model (Min, Moallemi "
        "and Maglaras, 2022), and DDQN and PPO have been compared for limit-order placement on replayed data "
        "(Schnaubelt, 2022). The contribution is therefore the combination: to our knowledge, this is the first "
        "controlled comparison of value-based, policy-gradient and CVaR-sensitive distributional agents for optimal "
        "execution in a data-calibrated queue-reactive order book with liquidity shocks, that links tail-cost "
        "differences to forced end-of-horizon liquidation. The literature check behind this wording, with every "
        "citation tagged as verified or not, is in docs/v2/literature_check.md.")

    add("h3", "5.12.2  Agents and Protocol")
    add("body", "QR-DDQN (Dabney, Rowland, Bellemare and Munos, 2018) replaces the Q-head of DDQN with K = 40 quantiles "
        "of the return per action, trained with the quantile Huber loss (κ = 1) and a Double-DQN target in which the "
        "online network chooses the next action with the agent's own criterion. The body, replay, ε schedule, target "
        "sync and 250k-step budget are those of DDQN. Three criteria were compared: the mean of the quantiles "
        "(risk-neutral); CVaR_α-greedy, the mean of the lowest αK quantiles with α = 0.25 (primary) or 0.1; and the "
        "threshold rule of Lim and Malik (2022), which carries a threshold c in the state (c ← c − r after each "
        "reward) and acts on E[−(c − Z)⁺]. The threshold rule was added because CVaR-greedy selection at every step "
        "optimises neither the static nor the nested CVaR of the total cost (Lim and Malik, 2022, Proposition 1); "
        "the CVaR-greedy agent is therefore described as CVaR-sensitive, not CVaR-optimal. The two tuned QR arms got "
        "the DDQN tuning grid and budget; the α = 0.1 and threshold arms reuse the CVaR(0.25) settings. DDQN and PPO "
        "were re-run with their Phase-1 settings, and every arm was trained on 'mixed' with the same ten fresh seeds "
        "(10–19), so all contrasts are paired by seed and by test episode.")

    add("h3", "5.12.3  Pre-Registered Results")
    add("body", f"Against DDQN, averaged over the six settings, the CVaR(0.25) agent changed CVaR95 of shortfall by "
        f"{N['t']} bps (difference, 95% interval). The whole interval lies beyond the pre-registered smallest effect "
        f"of interest (1.0 bps). Its mean shortfall changed by {N['m']} bps, so non-inferiority at the +0.3 bps margin "
        f"was not shown, and its risk-adjusted objective changed by {N['obj']} bps (higher is worse). The agent buys tail protection with mean cost; "
        f"it does not achieve DDQN-like mean cost with lower tail risk. Against PPO, the differences were "
        f"{N['ppo_t']} bps in CVaR95 and {N['ppo_m']} bps in mean.")
    table("Table 5.10: Agents trained on 'mixed', averaged over settings A–D3 (10 seeds each; 95% intervals by "
          "seed-then-episode bootstrap)",
          ["Agent", "Mean IS (bps)", "CVaR95 (bps)", "Objective", "Sold before T", "Limit-order share"],
          [[SHORT[a], f"{u(o.loc[a, 'is'])} [{u(o.loc[a, 'is_lo'])}, {u(o.loc[a, 'is_hi'])}]",
            f"{u(o.loc[a, 'cvar95'])} [{u(o.loc[a, 'cvar_lo'])}, {u(o.loc[a, 'cvar_hi'])}]", u(o.loc[a, "objective"]),
            f"{100 * o.loc[a, 'sold_before_T']:.1f}%", f"{100 * o.loc[a, 'limit_share']:.0f}%"] for a in ARMS],
          [2.4, 1.6, 1.7, 1.0, 1.1, 1.2])
    figure(f"{R}/fig_cvar_vs_mean.png", "Figure 5.8: Mean shortfall against CVaR95, averaged over the six settings, with "
           "95% intervals. TWAP and Almgren–Chriss are shown for reference.")
    add("body", f"The CVaR agent does not dominate the classical benchmarks either. The Almgren–Chriss schedule has a "
        f"mean of {u(N['ac_is'])} bps and a CVaR95 of {u(N['ac_cv'])} bps averaged over the settings, against "
        f"{u(N['c_is'])} and {u(N['c_cv'])} bps for the CVaR(0.25) agent: a front-loaded deterministic schedule still "
        f"has the thinnest tail in this simulator. In the per-setting results (out_v2/report/per_setting.csv) the "
        f"Almgren–Chriss CVaR95 is below the CVaR(0.25) agent's in every one of the six settings. Lowering α to 0.1 changed CVaR95 by {N['a10_t']} bps relative to α = 0.25.")

    add("h3", "5.12.4  Mechanism: the Forced Sweep")
    add("body", f"Acting on the lower tail removes the forced sweep. DDQN left {N['forced_d']:.1f}% of the order to the "
        f"market order at T on average, PPO {N['forced_p']:.1f}% and the CVaR(0.25) agent {N['forced_c']:.1f}%. To do "
        f"so, the CVaR agent switched from passive to marketable orders (limit-order share {N['lim_d']:.0f}% for DDQN, "
        f"{N['lim_c']:.0f}% for CVaR(0.25)) and front-loaded: at the D2 shock it held {N['shock_c']:.0f}% of the order, "
        f"against {N['shock_d']:.0f}% for DDQN and {N['shock_p']:.0f}% for PPO. This is the risk-averse behaviour "
        f"Almgren–Chriss predicts, and paying the spread on more of the order is where the extra mean cost comes from. "
        f"The tail is where the sweep lands: in DDQN's worst 5% of episodes the forced order sold {N['mt_d']:.0f}% of X, "
        f"against {N['mb_d']:.0f}% in its other episodes, and the episode-level correlation between the forced share "
        f"and the shortfall was {u(N['r_d'])}. For the CVaR(0.25) agent the same shares were {N['mt_c']:.0f}% and "
        f"{N['mb_c']:.1f}%.")
    figure(f"{R}/fig_forced_sweep.png", "Figure 5.9: Share of the order sold by the forced market order at T, by setting "
           "and agent.")
    figure(f"{R}/fig_cost_hist.png", "Figure 5.10: Distribution of shortfall in settings B and D2 (10 seeds × 1,000 "
           "episodes per agent); dotted lines mark each agent's CVaR95.")

    add("h3", "5.12.5  What Learning a Distribution Adds")
    add("body", f"The risk-neutral QR-DDQN, which isolates the effect of learning a distribution, matched DDQN's mean "
        f"({N['rn_m']} bps) with a lower CVaR95 ({N['rn_t']} bps). That arm's tuning, however, selected four times "
        f"as many gradient updates as DDQN (train_freq 1 instead of 4), so the comparison is confounded. An "
        f"exploratory control, not pre-registered, re-ran it with DDQN's update rate on the same seeds: the tail "
        f"difference to DDQN shrank to {N['rn4_t']} bps and the mean difference was {N['rn4_m']} bps. At equal "
        f"hyperparameters, the CVaR criterion alone moved CVaR95 by {N['risk_t']} bps and the mean by {N['risk_m']} "
        f"bps. The tail reduction therefore comes from the risk criterion, not from learning a distribution; the "
        f"extra updates are a separate, untested explanation for the first arm's gain.")
    add("body", f"The threshold rule, the method that targets static CVaR, did not beat the simpler CVaR-greedy rule: "
        f"its CVaR95 differed by {N['thr_t']} bps (Holm-adjusted p = {N['thr_p']:.2f}), it left {N['forced_thr']:.1f}% "
        f"of the order to the sweep, and its objective was the worst of the QR arms. We do not have an explanation "
        f"we could test; a likely factor is that the learned quantiles that set its threshold are miscalibrated. At "
        f"t = 0 in setting B, the CVaR(0.25) agent's learned lower-quartile mean of the return was {u(N['cal_pred'])} "
        f"bps against {u(N['cal_real'])} bps realised, so the learned distribution understates the tail, as Prasad "
        f"(2026) reports for distributional agents in games.")
    add("body", f"Across the three simulator variants of Section 5.8 (faster resilience, 0.5 s latency, three times the "
        f"fees; settings B and D2, 500 episodes), the CVaR(0.25) agent's CVaR95 change against DDQN ranged from "
        f"{s(N['sens_t_hi'])} to {s(N['sens_t_lo'])} bps and its mean change from {s(N['sens_m_lo'])} to "
        f"{s(N['sens_m_hi'])} bps, so the trade-off holds under each (out_v2/report/sensitivity.csv).")

    add("h3", "5.12.6  Answer and Limitations")
    add("body", "The open question of Section 5.11 was whether the DDQN/PPO tail gap is algorithmic or caused by the "
        "setup. Within one value-based agent, changing only the action-selection criterion moved the forced-sweep "
        "share and CVaR95 by more than the DDQN–PPO gap, so the tail is governed by how risk-averse the policy is, "
        "not by value-based versus policy-gradient learning as such. The trade-off itself did not go away: both "
        "CVaR-greedy agents and the threshold agent cut the tail and paid for it in mean shortfall. The one arm that "
        "cut the tail at no mean cost, the risk-neutral QR-DDQN, lost most of that gain at DDQN's update rate.")
    for li in ["One stock and one day, in a simulator whose volatility is 14–37% too high (Section 5.1); tail "
               "estimates at the 95% level rest on 50 episodes per run.",
               "Ten seeds per arm and four tuning configurations per tuned arm; α and K were fixed, not tuned.",
               "CVaR-greedy selection is a heuristic for static CVaR; the threshold rule that targets it was weaker "
               "here, possibly because its learned quantiles understate the tail.",
               "The risk-neutral QR-DDQN control at DDQN's update rate was added after the main results and is "
               "exploratory.",
               "The objective's λ′ was not changed; a risk-averse DDQN with a larger λ′ is the obvious missing "
               "comparator and is left for the next phase."]:
        para = add("li", li)
        for r in para.runs:
            r.bold = False

    add("h3", "References added in this section")
    for ref in ["Bäuerle, N. and Ott, J. (2011). Markov decision processes with average-value-at-risk criteria. "
                "Mathematical Methods of Operations Research, 74(3), 361–379.",
                "Dabney, W., Rowland, M., Bellemare, M. G. and Munos, R. (2018). Distributional reinforcement learning "
                "with quantile regression. AAAI 2018. arXiv:1710.10044.",
                "Espana, T., Hafsi, Y., Lillo, F. and Vittori, E. (2025). Reinforcement learning in queue-reactive "
                "models: application to optimal execution. arXiv:2511.15262.",
                "Lim, S. H. and Malik, I. (2022). Distributional reinforcement learning for risk-sensitive policies. "
                "Advances in Neural Information Processing Systems 35 (NeurIPS 2022).",
                "Min, S., Moallemi, C. C. and Maglaras, C. (2022). Risk-sensitive optimal execution via a conditional "
                "value-at-risk objective. Working paper, Columbia University.",
                "Prasad, H. (2026). Auditing the risk claims of distributional reinforcement learning. "
                "arXiv:2607.11607.",
                "Schnaubelt, M. (2022). Deep reinforcement learning for the optimal placement of cryptocurrency limit "
                "orders. European Journal of Operational Research, 296(3), 993–1006."]:
        add("body", ref)
    doc.save(out)


# ====================================================================================== slides
def build_pptx(D, N, src, out, skill):
    from pptx import Presentation
    from pptx.util import Inches
    tmp = out + ".tmp.pptx"
    subprocess.run([sys.executable, f"{skill}/scripts/add_slide.py", src, "slide3.xml", "-o", tmp], check=True,
                   capture_output=True)
    for sl in ("slide13.xml", "slide14.xml"):
        subprocess.run([sys.executable, f"{skill}/scripts/add_slide.py", tmp, sl, "-o", tmp], check=True,
                       capture_output=True)
    prs = Presentation(tmp)
    n0 = len(prs.slides) - 3
    sldIdLst = prs.slides._sldIdLst
    for sid in list(sldIdLst)[:n0]:
        prs.part.drop_rel(sid.rId)
        sldIdLst.remove(sid)
    s1, s2, s3 = prs.slides[0], prs.slides[1], prs.slides[2]

    def by_name(slide, name):
        return [sh for sh in slide.shapes if sh.name == name][0]

    def set_text(shape, lines):
        """Replace the text of a shape paragraph by paragraph, keeping each paragraph's run formatting."""
        tf = shape.text_frame
        ps = tf.paragraphs
        while len(ps) < len(lines):
            ps[-1]._p.addnext(copy.deepcopy(ps[-1]._p))
            ps = tf.paragraphs
        for p in ps[len(lines):]:
            p._p.getparent().remove(p._p)
        for p, line in zip(tf.paragraphs, lines):
            runs = p.runs
            for r in runs[1:]:
                r._r.getparent().remove(r._r)
            runs[0].text = line

    def set_table(tbl, rows, highlight=None):
        """Rows after the header copy the template's first body row; row index `highlight` copies its last
        (highlighted) row instead."""
        trs = tbl._tbl.tr_lst
        head, plain, hi = trs[0], copy.deepcopy(trs[1]), copy.deepcopy(trs[-1])
        for tr in trs[1:]:
            tbl._tbl.remove(tr)
        for i in range(1, len(rows)):
            tbl._tbl.append(copy.deepcopy(hi if i == highlight else plain))
        for i, r in enumerate(rows):
            for j, txt in enumerate(r):
                cell = tbl.cell(i, j)
                p = cell.text_frame.paragraphs[0]
                for extra in cell.text_frame.paragraphs[1:]:
                    extra._p.getparent().remove(extra._p)
                runs = p.runs
                for rr in runs[1:]:
                    rr._r.getparent().remove(rr._r)
                if runs:
                    runs[0].text = txt
                else:
                    p.add_run().text = txt

    def replace_picture(slide, name, path):
        pic = by_name(slide, name)
        x, y, w, h = pic.left, pic.top, pic.width, pic.height
        pic._element.getparent().remove(pic._element)
        new = slide.shapes.add_picture(path, x, y, height=h)
        if new.width > w:
            new.width, new.height = w, int(new.height * w / new.width)
        return new

    o = D.ov
    # ---- slide 1: novelty and design (from slide 3)
    set_text(by_name(s1, "Title 1"), ["Extension: CVaR Distributional Agent"])
    tb = [sh for sh in s1.shapes if sh.has_table][0].table
    set_table(tb, [["Work", "Approach", "Gap left open", "How this extension responds"],
                   ["Schnaubelt (2022)", "DDQN vs PPO, limit orders, replay", "No reactive book, no tail metrics",
                    "Same pair in a calibrated reactive book, CVaR95 reported"],
                   ["Espana et al. (2025)", "DDQN in a queue-reactive model", "Risk-neutral, no shocks, no PPO",
                    "Risk-sensitive agents; shocks; DDQN, PPO and QR-DDQN"],
                   ["Min, Moallemi & Maglaras (2022)", "CVaR-optimal execution by dynamic programming",
                    "Stylised model, no learning", "CVaR-sensitive RL in a data-calibrated book"],
                   ["Dabney et al. (2018); Lim & Malik (2022)", "QR-DQN; CVaR-greedy is not CVaR-optimal",
                    "Never applied to execution", "QR-DDQN, CVaR(0.25)-greedy and threshold-CVaR arms"],
                   ["Design (pre-registered)", "6 arms × 10 paired seeds, 250k steps, 'mixed'",
                    "Primary: CVaR95 and mean IS vs DDQN", "Smallest tail effect 1.0 bps; mean margin +0.3 bps"]])
    tshape = [sh for sh in s1.shapes if sh.has_table][0]
    by_name(s1, "TextBox 12").top = tshape.top + Inches(3.2)
    set_text(by_name(s1, "TextBox 12"), [
        "To our knowledge, the first controlled comparison of value-based, policy-gradient and CVaR-sensitive "
        "distributional agents for optimal execution in a data-calibrated queue-reactive book with liquidity shocks, "
        "linking tail cost to forced end-of-horizon liquidation",
        "Novelty is the combination; each ingredient has precedents (literature check: docs/v2/literature_check.md)"])
    s1.notes_slide.notes_text_frame.text = (
        "QR-DDQN keeps DDQN's body, replay, epsilon schedule and budget; only the head (40 quantiles) and the action "
        "rule change. CVaR-greedy is described as CVaR-sensitive because per-step CVaR selection is not static-CVaR "
        "optimal (Lim and Malik, 2022); the threshold arm is the method that targets static CVaR.")

    # ---- slide 2: primary results (from slide 13)
    set_text(by_name(s2, "Title 1"), ["Results: Tail Down, Mean Up"])
    replace_picture(s2, "Picture 10", f"{R}/fig_cvar_vs_mean.png")
    set_table([sh for sh in s2.shapes if sh.has_table][0].table,
              [["Agent", "Mean IS", "CVaR95", "Forced"]] +
              [[{"DDQN": "DDQN", "PPO": "PPO", "QRDDQN-RN": "QR-RN", "QRDDQN-CVaR25": "QR-CVaR25",
                 "QRDDQN-CVaR10": "QR-CVaR10", "QRDDQN-THR25": "QR-THR25"}[a], u(o.loc[a, "is"]),
                u(o.loc[a, "cvar95"]), f"{100 * o.loc[a, 'forced']:.1f}%"] for a in ARMS], highlight=1 + ARMS.index("QRDDQN-CVaR25"))
    set_text(by_name(s2, "TextBox 12"), [
        f"Mean IS and CVaR95 in bps, averaged over A–D3; Forced = share of X sold by the forced order at T",
        f"AC baseline still has the thinnest tail: {u(N['ac_is'])} mean, {u(N['ac_cv'])} CVaR95"])
    set_text(by_name(s2, "TextBox 13"), ["Pre-registered contrasts, QR-CVaR25 minus DDQN, 10 paired seeds, 95% CI"])
    set_text(by_name(s2, "TextBox 15"), [f"CVaR95 {N['t'].split(' [')[0]} bps"])
    set_text(by_name(s2, "TextBox 16"), [f"[{N['t'].split('[')[1]}; beyond the 1.0 bps threshold"])
    set_text(by_name(s2, "TextBox 18"), [f"Mean {N['m'].split(' [')[0]} bps"])
    set_text(by_name(s2, "TextBox 19"), [f"[{N['m'].split('[')[1]}; not within the +0.3 bps margin"])
    s2.notes_slide.notes_text_frame.text = (
        "The CVaR agent buys tail protection with mean cost: both pre-registered tests are decisive, in opposite "
        "directions. It does not reach the Almgren-Chriss schedule's tail either. Numbers from out_v2/report.")

    # ---- slide 3: mechanism (from slide 14)
    set_text(by_name(s3, "Title 1"), ["Mechanism: The Forced Sweep"])
    replace_picture(s3, "Picture 10", f"{R}/fig_forced_sweep.png")
    set_text(by_name(s3, "TextBox 12"), ["Tail = forced sweep"])
    set_text(by_name(s3, "TextBox 13"), [
        f"DDQN's worst 5% of episodes sell {N['mt_d']:.0f}% of X at T, vs {N['mb_d']:.0f}% in the rest",
        f"QR-CVaR25 leaves {N['forced_c']:.1f}% to the sweep: limit orders {N['lim_d']:.0f}% → {N['lim_c']:.0f}%, "
        f"inventory at the shock {N['shock_d']:.0f}% → {N['shock_c']:.0f}%"])
    set_text(by_name(s3, "TextBox 15"), ["Risk rule, not distribution"])
    set_text(by_name(s3, "TextBox 16"), [
        f"Same hyperparameters: CVaR rule moves CVaR95 by {N['risk_t'].split(' [')[0]} bps",
        f"Risk-neutral QR at DDQN's update rate: {N['rn4_t'].split(' [')[0]} bps, CI spans 0 (exploratory)"])
    set_text(by_name(s3, "TextBox 18"), ["Answer to the open question, and limits"])
    set_text(by_name(s3, "TextBox 19"), [
        "Changing only the action rule moves the tail more than the DDQN–PPO gap: the tail follows risk-aversion, "
        "not the algorithm family",
        "Limits: one stock, one day; 10 seeds; threshold rule weaker; learned quantiles understate the tail"])
    s3.notes_slide.notes_text_frame.text = (
        "Front-loading and marketable orders are the correct risk-averse response (Almgren-Chriss), and paying the "
        "spread on more of the order is where the extra mean cost comes from.")
    prs.save(out)
    os.remove(tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-docx", required=True)
    ap.add_argument("--deck", required=True)
    ap.add_argument("--pptx-skill", required=True, help="directory holding scripts/add_slide.py")
    args = ap.parse_args()
    D = Data()
    N = numbers(D)
    build_docx(D, N, args.report_docx, f"{OUT}/Report_Section_5.12_CVaR.docx")
    build_pptx(D, N, args.deck, f"{OUT}/Slides_CVaR_extension.pptx", args.pptx_skill)
    print("wrote", f"{OUT}/Report_Section_5.12_CVaR.docx", f"{OUT}/Slides_CVaR_extension.pptx")


if __name__ == "__main__":
    main()
