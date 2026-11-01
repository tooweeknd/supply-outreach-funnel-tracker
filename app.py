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

# ------------------------------------------------------------------ funnel math
def stage_counts(d: pd.DataFrame, day: int) -> dict:
    replied = d["reply_day"].notna() & (d["reply_day"] <= day)
    return {
        "Sourced": len(d),
        "Enriched": int(d["enriched"].sum()),
        "Verified valid": int(d["verified_valid"].sum()),
        "Contacted": int(d["contacted"].sum()),
        "Delivered": int(d["delivered"].sum()),
        f"Replied (day-{day})": int(replied.sum()),
        "Positive replies": int((d["reply_type"] == "positive").sum() if day >= 10
                                else (d["reply_type"].eq("positive") & d["reply_day"].le(day)).sum()),
        "Screened": int(d["screened"].sum()) if day >= 10 else int((d["screened"] & d["reply_day"].le(day)).sum()),
        "Qualified": int(d["qualified"].sum()) if day >= 10 else int((d["qualified"] & d["reply_day"].le(day)).sum()),
    }

counts = stage_counts(wk, maturity_day)
stages = list(counts.keys())

# -------------------------------------------------------------------- header
st.title("Supply Outreach Funnel Tracker")
st.markdown("Funnel health, exceptions, cohorts and experiments for a high-volume outbound talent pipeline.")
st.caption(f"Cohort week of {sel_week.strftime('%d %b %Y')} · sources: {', '.join(src_sel) or 'none selected'} · replies measured by day {maturity_day}")

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
    c3.metric(f"Reply rate (per delivered, day-{maturity_day})",
              f"{replied / delivered:.1%}" if delivered else "-")
    c4.metric("Qualified", counts["Qualified"])
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
        rep = d["reply_day"].notna() & (d["reply_day"] <= maturity_day)
        rrate = rep.sum() / int(d["delivered"].sum()) if d["delivered"].sum() else 0
        if brate > bounce_ceiling:
            flags.append({"Source": src, "Metric": "Bounce rate", "Value": f"{brate:.1%}",
                          "Tripwire": f"> {bounce_ceiling:.1%}",
                          "Likely cause": "List data quality - check verification gate and list vintage"})
        if rrate < reply_floor:
            flags.append({"Source": src, "Metric": f"Reply rate (day-{maturity_day})", "Value": f"{rrate:.1%}",
                          "Tripwire": f"< {reply_floor:.1%}",
                          "Likely cause": "Wrong audience or stale list - cut by list before touching copy"})
    if flags:
        st.error(f"{len(flags)} flag(s) raised")
        st.dataframe(pd.DataFrame(flags), use_container_width=True, hide_index=True)
    else:
        st.success("No metrics beyond tripwires in this view.")

    st.subheader("Per-source decomposition")
    st.caption("Overall rates are share-weighted mixes. This table separates a mix shift from a within-source change before anyone acts.")
    decomp = []
    for src in src_sel:
        d = wk[wk["source"] == src]
        sent = int(d["contacted"].sum())
        if sent == 0:
            continue
        rep = d["reply_day"].notna() & (d["reply_day"] <= maturity_day)
        pos = d["reply_type"].eq("positive") & d["reply_day"].le(maturity_day)
        decomp.append({
            "Source": src, "Contacted": sent,
            "Bounce rate": f"{(sent - int(d['delivered'].sum())) / sent:.1%}",
            f"Reply rate (day-{maturity_day})": f"{rep.sum() / max(int(d['delivered'].sum()), 1):.1%}",
            "Positive share": f"{pos.sum() / max(rep.sum(), 1):.0%}" if rep.sum() else "-",
            "Qualified": int((d["qualified"] & d["reply_day"].le(maturity_day)).sum()) if maturity_day < 10
                         else int(d["qualified"].sum()),
        })
    st.dataframe(pd.DataFrame(decomp), use_container_width=True, hide_index=True)

# -------------------------------------------------------------- cohort view
with tab3:
    st.subheader("Reply-rate maturation by cohort")
    st.caption("Replies lag sends. Compare cohorts at the same age; young cohorts always look worse than they are.")
    mat = []
    for w, week in enumerate(WEEKS):
        d = leads[(leads["cohort_week"] == week) & (leads["source"].isin(src_sel))]
        deliv = int(d["delivered"].sum())
        row = {"Cohort week": week.strftime("%d %b")}
        for day in (3, 7, 10):
            r = d["reply_day"].notna() & (d["reply_day"] <= day)
            row[f"Day-{day} reply rate"] = round(100 * r.sum() / deliv, 2) if deliv else 0
        mat.append(row)
    mdf = pd.DataFrame(mat).set_index("Cohort week")
    st.dataframe(mdf, use_container_width=True)
    st.line_chart(mdf)
    latest_age = (pd.Timestamp("2026-11-01") - WEEKS[-1]).days
    st.info(f"The most recent cohort is only {latest_age} days old - its day-10 figure is still provisional. Same-window comparison (day-3 vs day-3) is the honest read for young cohorts.")

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
- **Exceptions:** flags any source past its tripwire and decomposes overall rates by source, separating a mix shift from a within-source change before anyone acts.
- **Cohort maturity:** replies lag sends, so cohorts are compared at the same age (day 3 / 7 / 10). Young cohorts are marked provisional instead of being read as regressions.
- **Experiment log:** every test has one variable, a pre-defined metric, and a written kill-or-scale decision.
- **Data:** this workspace runs on generated sample data for demonstration. Thresholds are adjustable illustrations, not industry rules.
- **Built by:** Durgesh Kumar.
""")
