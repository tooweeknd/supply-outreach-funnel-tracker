"""
Supply Outreach Funnel Tracker.

Funnel health, exception alerts, cohort-aware measurement and an experiment
log for a high-volume outbound talent-acquisition pipeline.

This deployment runs on generated sample data (see make_leads below). No real
candidate, company or campaign data is used.
"""

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Supply Outreach Funnel Tracker", layout="wide")

# ---------------------------------------------------------------- sample data
SOURCES = ["LinkedIn Outreach", "Apollo Database", "Referrals"]
WEEKS = pd.date_range("2026-09-07", periods=8, freq="W-MON")  # 8 weekly cohorts
RNG = np.random.default_rng(42)

@st.cache_data
def make_leads() -> pd.DataFrame:
    """Generate one row per synthetic lead with dated stage transitions."""
    rows = []
    lid = 1
    for w, week in enumerate(WEEKS):
        for src in SOURCES:
            base = {"LinkedIn Outreach": 260, "Apollo Database": 240, "Referrals": 60}[src]
            n = int(base * RNG.uniform(0.9, 1.1))
            # Per-source performance. The Apollo source degrades from week 6
            # (a "bad data source") so the Exceptions tab has something to find.
            if src == "Apollo Database" and w >= 5:
                p_valid, p_bounce, p_reply, p_pos, p_pass = 0.55, 0.09, 0.020, 0.30, 0.10
            elif src == "Apollo Database":
                p_valid, p_bounce, p_reply, p_pos, p_pass = 0.80, 0.015, 0.045, 0.50, 0.22
            elif src == "LinkedIn Outreach":
                p_valid, p_bounce, p_reply, p_pos, p_pass = 0.85, 0.010, 0.052, 0.55, 0.25
            else:  # Referrals: small volume, high quality
                p_valid, p_bounce, p_reply, p_pos, p_pass = 0.95, 0.005, 0.090, 0.65, 0.40
            for _ in range(n):
                r = {"lead_id": f"L{lid:05d}", "cohort_week": week, "source": src}
                lid += 1
                r["enriched"] = RNG.random() < 0.92
                r["verified_valid"] = r["enriched"] and RNG.random() < p_valid / 0.92
                r["contacted"] = r["verified_valid"]
                r["delivered"] = r["contacted"] and RNG.random() > p_bounce
                if r["delivered"] and RNG.random() < p_reply:
                    r["reply_day"] = int(RNG.choice([0, 1, 1, 2, 2, 3, 4, 5, 7, 9]))
                    r["reply_type"] = "positive" if RNG.random() < p_pos else "negative"
                else:
                    r["reply_day"], r["reply_type"] = None, None
                r["screened"] = r["reply_type"] == "positive" and RNG.random() < 0.9
                r["qualified"] = r["screened"] and RNG.random() < (p_pass / 0.9 if p_pass < 0.9 else 1.0)
                rows.append(r)
    df = pd.DataFrame(rows)
    return df

leads = make_leads()

# ------------------------------------------------------- as-of and maturity
AS_OF = pd.Timestamp.today().normalize()  # simulated "data current as of" date

def cohort_age_days(week) -> int:
    return max((AS_OF - week).days, 0)

def window_cap(week, day: int) -> int:
    """Days of reply data actually observable for a cohort at window `day`."""
    return min(day, cohort_age_days(week))

# ------------------------------------------------------------------- sidebar
st.sidebar.header("Controls")
week_labels = [w.strftime("%d %b") for w in WEEKS]
week_idx = st.sidebar.selectbox("Cohort week", range(len(WEEKS)), index=len(WEEKS) - 1,
                                format_func=lambda i: week_labels[i])
sel_week = WEEKS[week_idx]
src_sel = st.sidebar.multiselect("Sources", SOURCES, default=SOURCES)
reply_floor = st.sidebar.slider("Reply-rate alert floor (%)", 0.5, 10.0, 2.0, 0.1) / 100
bounce_ceiling = st.sidebar.slider("Bounce-rate alert ceiling (%)", 0.5, 15.0, 3.0, 0.5) / 100
maturity_day = st.sidebar.radio("Measure replies by day", [3, 7, 10], index=2)
st.sidebar.caption("Thresholds are adjustable tripwires, not industry rules.")
st.sidebar.divider()
st.sidebar.caption("Demo workspace · generated sample data")

wk = leads[(leads["cohort_week"] == sel_week) & (leads["source"].isin(src_sel))]

if wk.empty:
    st.title("Supply Outreach Funnel Tracker")
    st.info("No leads match the current filters - select at least one source in the sidebar.")
    st.stop()

# ------------------------------------------------------------------ funnel math
def stage_counts(d: pd.DataFrame, day: int, week=None) -> dict:
    cap = window_cap(week, day) if week is not None else day
    replied = d["reply_day"].notna() & (d["reply_day"] <= cap)
    return {
        "Sourced": len(d),
        "Enriched": int(d["enriched"].sum()),
        "Verified valid": int(d["verified_valid"].sum()),
        "Contacted": int(d["contacted"].sum()),
        "Delivered": int(d["delivered"].sum()),
        f"Replied (day-{day})": int(replied.sum()),
        "Positive replies": int((d["reply_type"].eq("positive") & d["reply_day"].le(cap)).sum()),
        "Screened": int((d["screened"] & d["reply_day"].le(cap)).sum()),
        "Qualified": int((d["qualified"] & d["reply_day"].le(cap)).sum()),
    }

counts = stage_counts(wk, maturity_day, sel_week)
stages = list(counts.keys())

# -------------------------------------------------------------------- header
st.title("Supply Outreach Funnel Tracker")
st.markdown("Funnel health, exceptions, cohorts and experiments for a high-volume outbound talent pipeline.")
prov = cohort_age_days(sel_week) < maturity_day
prov_note = " · provisional, cohort still maturing" if prov else ""
st.caption(f"Cohort week of {sel_week.strftime('%d %b %Y')} · sources: {', '.join(src_sel) or 'none selected'} · replies measured by day {maturity_day}{prov_note} · data as of {AS_OF.strftime('%d %b %Y')}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Funnel overview", "Exceptions", "Cohort maturity", "Experiment log", "How to read this"])

# ------------------------------------------------------------- funnel overview
with tab1:
    rows = []
    prev = None
    for s in stages:
        conv = counts[s] / prev if prev else 1.0
        rows.append({"Stage": s, "Volume": counts[s],
                     "Conversion vs prior stage": f"{conv:.1%}" if prev else "-"})
        prev = counts[s]
    delivered = counts["Delivered"]
    replied = counts[f"Replied (day-{maturity_day})"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Contacted", counts["Contacted"])
    c2.metric("Delivery rate", f"{delivered / counts['Contacted']:.1%}" if counts["Contacted"] else "-")
    c3.metric(f"Reply rate (per delivered, day-{maturity_day}{', provisional' if prov else ''})",
              f"{replied / delivered:.1%}" if delivered else "-")
    c4.metric("Qualified", counts["Qualified"])
    if prov:
        st.caption(f"This cohort is {cohort_age_days(sel_week)} days old, so day-{maturity_day} replies are still maturing - figures count only replies observable as of {AS_OF.strftime('%d %b %Y')}.")
    st.subheader("Stage table")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.subheader("Funnel")
    st.bar_chart(pd.DataFrame({"Volume": counts.values()}, index=stages))
    st.caption("Every rate is computed per delivered email unless labelled otherwise - denominators are stated on purpose.")

# ----------------------------------------------------------------- exceptions
with tab2:
    st.subheader("Flagged anomalies this cohort")
    flags = []
    # per-source metrics
    for src in src_sel:
        d = wk[wk["source"] == src]
        sent = int(d["contacted"].sum())
        if sent == 0:
            continue
        bounced = sent - int(d["delivered"].sum())
        brate = bounced / sent
        cap = window_cap(sel_week, maturity_day)
        rep = d["reply_day"].notna() & (d["reply_day"] <= cap)
        rrate = rep.sum() / int(d["delivered"].sum()) if d["delivered"].sum() else None
        if brate > bounce_ceiling:
            flags.append({"Source": src, "Metric": "Bounce rate", "Value": f"{brate:.1%}",
                          "Tripwire": f"> {bounce_ceiling:.1%}",
                          "Likely cause": "List data quality - check verification gate and list vintage"})
        if rrate is not None and rrate < reply_floor:
            flags.append({"Source": src, "Metric": f"Reply rate (day-{maturity_day}{', provisional' if prov else ''})", "Value": f"{rrate:.1%}",
                          "Tripwire": f"< {reply_floor:.1%}",
                          "Likely cause": "Wrong audience or stale list - cut by list before touching copy"})
    if flags:
        st.error(f"{len(flags)} flag(s) raised")
        st.dataframe(pd.DataFrame(flags), use_container_width=True, hide_index=True)
    else:
        st.success("No metrics beyond tripwires in this view.")

    st.subheader("Mix vs within-source decomposition")
    st.caption("Overall reply rate is a share-weighted mix of the sources. Comparing this cohort with the trailing 4-week baseline at the same day-window splits the gap into a mix effect (volume shifting between sources) and a within-source effect (a source itself getting better or worse).")
    base_weeks = list(WEEKS[WEEKS < sel_week])[-4:]
    base = leads[(leads["cohort_week"].isin(base_weeks)) & (leads["source"].isin(src_sel))]
    tot_del_t, tot_del_b = int(wk["delivered"].sum()), int(base["delivered"].sum())
    cap_t = window_cap(sel_week, maturity_day)
    if not base_weeks or tot_del_b == 0 or tot_del_t == 0:
        st.info("No earlier cohorts to compare against for this view.")
    else:
        rows, mix_eff = [], 0.0
        for src in src_sel:
            dt, db = wk[wk["source"] == src], base[base["source"] == src]
            del_t, del_b = int(dt["delivered"].sum()), int(db["delivered"].sum())
            if del_t == 0 and del_b == 0:
                continue
            rt = (dt["reply_day"].le(cap_t)).sum() / del_t if del_t else 0.0
            rb = (db["reply_day"].le(maturity_day)).sum() / del_b if del_b else 0.0
            sh_t = del_t / tot_del_t if tot_del_t else 0.0
            sh_b = del_b / tot_del_b if tot_del_b else 0.0
            mix_eff += (sh_t - sh_b) * rb
            pos = dt["reply_type"].eq("positive") & dt["reply_day"].le(cap_t)
            rep_t = dt["reply_day"].le(cap_t).sum()
            rows.append({
                "Source": src,
                "Delivered share": f"{sh_t:.0%}",
                "Share (baseline)": f"{sh_b:.0%}" if del_b else "-",
                f"Reply rate (day-{maturity_day}{', prov.' if prov else ''})": f"{rt:.1%}" if del_t else "-",
                "Reply rate (baseline)": f"{rb:.1%}" if del_b else "-",
                "Positive share": f"{pos.sum() / rep_t:.0%}" if rep_t else "-",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        overall_t = (wk["reply_day"].le(cap_t)).sum() / tot_del_t
        overall_b = (base["reply_day"].le(maturity_day)).sum() / tot_del_b
        gap = overall_t - overall_b
        within_eff = gap - mix_eff  # within-source change including any interaction
        st.markdown(
            f"**Overall day-{maturity_day} reply rate: {overall_t:.1%} vs {overall_b:.1%} trailing 4-week baseline "
            f"({gap * 100:+.1f} pts).** Of that gap, {mix_eff * 100:+.1f} pts comes from mix shift between sources "
            f"and {within_eff * 100:+.1f} pts is within-source change. Cut a dip by source and against baseline before "
            f"recommending action.")

# -------------------------------------------------------------- cohort view
with tab3:
    st.subheader("Reply-rate maturation by cohort")
    st.caption("Replies lag sends. Compare cohorts at the same age; young cohorts can look worse than they are.")
    mat, chart_rows, prov_cells = [], [], []
    for w, week in enumerate(WEEKS):
        d = leads[(leads["cohort_week"] == week) & (leads["source"].isin(src_sel))]
        deliv = int(d["delivered"].sum())
        age = cohort_age_days(week)
        row, crow = {"Cohort week": week.strftime("%d %b")}, {"Cohort week": week.strftime("%d %b")}
        for day in (3, 7, 10):
            col = f"Day-{day} reply rate"
            if deliv == 0:
                row[col], crow[col] = "-", None
            elif age >= day:
                v = round(100 * (d["reply_day"].le(day)).sum() / deliv, 2)
                row[col], crow[col] = f"{v:.2f}", v
            else:
                v = round(100 * (d["reply_day"].le(age)).sum() / deliv, 2)
                row[col], crow[col] = f"{v:.2f}*", None
                prov_cells.append(f"{week.strftime('%d %b')} day-{day} (only {age} days old)")
        mat.append(row)
        chart_rows.append(crow)
    mdf = pd.DataFrame(mat).set_index("Cohort week")
    st.dataframe(mdf, use_container_width=True)
    st.line_chart(pd.DataFrame(chart_rows).set_index("Cohort week"))
    if prov_cells:
        st.caption("* provisional - the window has not fully elapsed as of "
                   f"{AS_OF.strftime('%d %b %Y')} ({'; '.join(prov_cells)}). Shown at observed age; "
                   "excluded from the chart so immature windows are not read as regressions.")

# ------------------------------------------------------------ experiment log
with tab4:
    st.subheader("Experiment log")
    st.caption("Structured tests: one variable, a pre-defined metric, and a written kill-or-scale decision.")
    exp = pd.DataFrame([
        {"Test": "Referral ask in screen calls", "Variable": "New source", "Metric": "Qualified/week", "Result": "+6 qualified/wk", "Decision": "Scale"},
        {"Test": "Apollo list, finance-controllers segment", "Variable": "List source", "Metric": f"Day-7 reply rate", "Result": "0.9% vs 4.5% baseline", "Decision": "Kill"},
        {"Test": "Send window Tue-Thu morning", "Variable": "Send time", "Metric": "Day-3 reply rate", "Result": "+0.4 pts, within noise", "Decision": "Keep watching"},
        {"Test": "First-line personalisation at scale", "Variable": "Message variable", "Metric": "Positive share", "Result": "Running", "Decision": "Running"},
    ])
    st.dataframe(exp, use_container_width=True, hide_index=True)

# --------------------------------------------------------------- how to read
with tab5:
    st.subheader("How to read this")
    st.markdown("""
- **What it does:** tracks a high-volume outbound talent pipeline end to end - one row per lead, stage-by-stage conversion, tripwire alerts, cohort-aware measurement, and a structured experiment log.
- **Funnel overview:** weekly volume and conversion per stage, with denominators stated. Reply rate is always per delivered email.
- **Exceptions:** flags any source past its tripwire, then compares the cohort with a trailing 4-week baseline at the same day-window and splits the overall-rate gap into mix shift vs within-source change before anyone acts.
- **Cohort maturity:** replies lag sends, so cohorts are compared at the same age (day 3 / 7 / 10). Windows that have not fully elapsed as of the stated as-of date are marked provisional and kept out of the chart, so young cohorts are not read as regressions.
- **Experiment log:** every test has one variable, a pre-defined metric, and a written kill-or-scale decision.
- **Data:** this workspace runs on generated sample data for demonstration. Thresholds are adjustable illustrations, not industry rules.
- **Built by:** Durgesh Kumar.
""")
