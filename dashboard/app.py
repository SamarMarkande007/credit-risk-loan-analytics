"""
app.py  -  Credora Finance | Credit Risk & Loan Approval Dashboard (Streamlit)

Run from the project root (after data_load.py, train.py and risk_score.py):

    streamlit run dashboard/app.py

All eight PRD components are here
    1 Loan approval rate ........ KPI + monthly trend        (Overview)
    2 Default rate .............. KPI + by-loan-type bar      (Overview)
    3 Customer risk split ....... donut                       (Risk & credit)
    4 Credit score view ......... histogram / monthly trend   (Risk & credit)
    5 Geographic distribution ... state bar chart             (Geography)
    6 Monthly applications ...... column chart                (Overview)
    7 Approval vs rejection ..... stacked area / by loan type (Overview)
    8 High-risk customer list ... filterable table + CSV      (High-risk watchlist)
plus a "Model check" tab that shows whether the risk score really separates defaulters.

How the filters work
    * Application charts use all four sidebar filters
      (loan type, state, risk category of the application, application month).
    * Customer charts use state + customer risk category. If a loan type or a month
      range is chosen, they only include customers who have a matching application.
    * No names, emails or phone numbers are ever loaded or shown.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data_load import get_engine  # noqa: E402  (same .env based MySQL connection as the pipeline)
from risk_score import CATEGORIES, LOW_MAX, MEDIUM_MAX  # noqa: E402

RISK_COLORS = {"Low": "#2E9E6B", "Medium": "#E0A82E", "High": "#D64545"}
DECISION_COLORS = {"Approved": "#2E9E6B", "Rejected": "#D64545", "Under Review": "#9AA0A6"}
ACCENT = "#3B6FB6"
MIN_N = 20  # smallest group for which a default RATE is shown (smaller groups are too noisy)

st.set_page_config(page_title="Credora Finance - Credit Risk", page_icon="📊", layout="wide")


# =============================================================================
# Small helpers
# =============================================================================
def show_chart(fig: go.Figure, key: str) -> None:
    """Plotly chart that stretches to the column width on old and new Streamlit versions."""
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), legend_title_text="")
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except Exception:  # Streamlit < 1.49
        st.plotly_chart(fig, use_container_width=True, key=key)


def show_table(df: pd.DataFrame, **kwargs) -> None:
    try:
        st.dataframe(df, hide_index=True, width="stretch", **kwargs)
    except Exception:  # Streamlit < 1.49
        st.dataframe(df, hide_index=True, use_container_width=True, **kwargs)


def pct(x) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:.1f}%"


# =============================================================================
# Data (read once, cached 10 minutes)
# =============================================================================
APPS_SQL = """
SELECT la.application_id, la.customer_id, la.application_date, la.loan_type,
       la.requested_amount, la.approval_status, la.approved_amount, la.disbursed_flag, la.loan_default,
       sa.default_probability, sa.risk_score, sa.risk_category AS app_risk_category,
       c.state, c.city, c.employment_type, ch.credit_score
FROM loan_applications la
JOIN scored_applications sa ON sa.application_id = la.application_id
JOIN customers c            ON c.customer_id     = la.customer_id
JOIN credit_history ch      ON ch.customer_id    = la.customer_id
"""
CUST_SQL = """
SELECT sc.customer_id, sc.n_applications, sc.default_probability, sc.risk_score,
       sc.risk_category, sc.scoring_basis,
       c.city, c.state, c.employment_type, c.annual_income, ch.credit_score
FROM scored_customers sc
JOIN customers c       ON c.customer_id  = sc.customer_id
JOIN credit_history ch ON ch.customer_id = sc.customer_id
"""


@st.cache_resource(show_spinner=False)
def _engine():
    return get_engine()


@st.cache_data(ttl=600, show_spinner="Loading data from MySQL ...")
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    eng = _engine()
    apps = pd.read_sql(APPS_SQL, eng)
    custs = pd.read_sql(CUST_SQL, eng)
    apps["application_date"] = pd.to_datetime(apps["application_date"])
    apps["month"] = apps["application_date"].dt.strftime("%Y-%m")
    return apps, custs


def load_model_summary() -> dict | None:
    path = ROOT / "models" / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


# =============================================================================
# Sidebar filters
# =============================================================================
def sidebar_filters(apps: pd.DataFrame, custs: pd.DataFrame) -> dict:
    """Empty selection = ALL.  Pick one or more values to narrow the dashboard."""
    types = sorted(apps["loan_type"].unique())
    states = sorted(custs["state"].unique())
    months = sorted(apps["month"].unique())
    full_range = (months[0], months[-1])
    for k in ("f_types", "f_states", "f_risk"):
        st.session_state.setdefault(k, [])            # nothing selected = everything shown

    def reset():
        for k in ("f_types", "f_states", "f_risk"):
            st.session_state[k] = []
        st.session_state.pop("f_months", None)        # deleting the key puts the slider back to its default range

    st.sidebar.header("Filters")
    st.sidebar.caption("Leave a filter empty to include everything.")
    st.sidebar.multiselect("Loan type", types, key="f_types", placeholder="All loan types")
    st.sidebar.multiselect("Region (state)", states, key="f_states", placeholder="All states")
    st.sidebar.multiselect("Risk category", list(CATEGORIES), key="f_risk", placeholder="All risk categories")
    # value=(start, end) is what makes this a two-handle RANGE slider
    m0, m1 = st.sidebar.select_slider("Application month", options=months, value=full_range, key="f_months")
    st.sidebar.button("Reset filters", on_click=reset)

    return {"types": st.session_state["f_types"] or types,        # empty -> all
            "states": st.session_state["f_states"] or states,
            "risk": st.session_state["f_risk"] or list(CATEGORIES),
            "m0": m0, "m1": m1, "all_types": types, "all_months": full_range}


def apply_filters(apps: pd.DataFrame, custs: pd.DataFrame, f: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    in_range = (apps["month"] >= f["m0"]) & (apps["month"] <= f["m1"])
    a = apps[apps["loan_type"].isin(f["types"]) & apps["state"].isin(f["states"])
             & apps["app_risk_category"].isin(f["risk"]) & in_range]

    c = custs[custs["state"].isin(f["states"]) & custs["risk_category"].isin(f["risk"])]
    narrowed = set(f["types"]) != set(f["all_types"]) or (f["m0"], f["m1"]) != f["all_months"]
    if narrowed:  # only customers who have an application matching the loan type / month filters
        match = apps.loc[apps["loan_type"].isin(f["types"]) & in_range, "customer_id"].unique()
        c = c[c["customer_id"].isin(match)]
    return a, c


# =============================================================================
# KPI calculations
# =============================================================================
def kpis(a: pd.DataFrame, c: pd.DataFrame) -> dict:
    decided = a[a["approval_status"].isin(["Approved", "Rejected"])]
    disbursed = a[a["loan_default"].notna()]
    return {
        "applications": len(a),
        "approval_rate": (decided["approval_status"] == "Approved").mean() * 100 if len(decided) else np.nan,
        "n_decided": len(decided),
        "default_rate": disbursed["loan_default"].mean() * 100 if len(disbursed) else np.nan,
        "n_disbursed": len(disbursed),
        "disbursed_cr": a.loc[a["disbursed_flag"] == 1, "approved_amount"].sum() / 1e7,
        "high_risk": int((c["risk_category"] == "High").sum()),
        "high_risk_pct": (c["risk_category"] == "High").mean() * 100 if len(c) else np.nan,
        "customers": len(c),
    }


# =============================================================================
# Chart builders (pure functions: data in, figure out)
# =============================================================================
def fig_approval_trend(a: pd.DataFrame) -> go.Figure | None:
    d = a[a["approval_status"].isin(["Approved", "Rejected"])]
    if d.empty:
        return None
    g = d.groupby("month").agg(decided=("approval_status", "size"),
                               approved=("approval_status", lambda s: int((s == "Approved").sum())))
    g["rate"] = g["approved"] / g["decided"] * 100
    overall = d["approval_status"].eq("Approved").mean() * 100
    fig = go.Figure(go.Scatter(x=g.index, y=g["rate"], mode="lines+markers", line=dict(color=ACCENT, width=3),
                               customdata=g["decided"], name="approval rate",
                               hovertemplate="%{x}<br>approval rate %{y:.1f}%<br>decided applications: %{customdata}<extra></extra>"))
    fig.add_hline(y=overall, line_dash="dash", line_color="grey",
                  annotation_text=f"average {overall:.1f}%", annotation_position="top left")
    fig.update_layout(title="Approval rate by month (decided applications)", yaxis_title="approval rate (%)",
                      yaxis_ticksuffix="%", showlegend=False)
    return fig


def fig_default_by_type(a: pd.DataFrame) -> go.Figure | None:
    d = a[a["loan_default"].notna()]
    if d.empty:
        return None
    g = d.groupby("loan_type").agg(rate=("loan_default", "mean"), n=("loan_default", "size")).reset_index()
    g["rate"] *= 100
    g = g.sort_values("rate", ascending=False)
    g["shown"] = g["rate"].where(g["n"] >= MIN_N)
    fig = px.bar(g, x="loan_type", y="shown", text=g["shown"].map(lambda v: "" if pd.isna(v) else f"{v:.1f}%"),
                 custom_data=["n"], color_discrete_sequence=[ACCENT])
    fig.update_traces(hovertemplate="%{x}<br>default rate %{y:.1f}%<br>disbursed loans: %{customdata[0]}<extra></extra>",
                      textposition="outside")
    fig.add_hline(y=d["loan_default"].mean() * 100, line_dash="dash", line_color="grey",
                  annotation_text="all selected loans", annotation_position="top right")
    fig.update_layout(title="Observed default rate by loan type (disbursed loans)", xaxis_title="",
                      yaxis_title="default rate (%)", yaxis_ticksuffix="%")
    return fig


def fig_monthly_volume(a: pd.DataFrame) -> go.Figure | None:
    if a.empty:
        return None
    g = a.groupby("month").size().rename("applications").reset_index()
    fig = px.bar(g, x="month", y="applications", color_discrete_sequence=[ACCENT])
    fig.update_layout(title="Loan applications per month", xaxis_title="", yaxis_title="applications")
    return fig


def fig_decisions_over_time(a: pd.DataFrame) -> go.Figure | None:
    if a.empty:
        return None
    g = a.groupby(["month", "approval_status"]).size().rename("applications").reset_index()
    fig = px.area(g, x="month", y="applications", color="approval_status", color_discrete_map=DECISION_COLORS,
                  category_orders={"approval_status": ["Approved", "Rejected", "Under Review"]})
    fig.update_layout(title="Approvals vs rejections over time", xaxis_title="", yaxis_title="applications")
    return fig


def fig_decisions_by_type(a: pd.DataFrame) -> go.Figure | None:
    d = a[a["approval_status"].isin(["Approved", "Rejected"])]
    if d.empty:
        return None
    g = d.groupby(["loan_type", "approval_status"]).size().rename("applications").reset_index()
    fig = px.bar(g, x="loan_type", y="applications", color="approval_status", barmode="group",
                 color_discrete_map=DECISION_COLORS)
    fig.update_layout(title="Approvals vs rejections by loan type", xaxis_title="", yaxis_title="applications")
    return fig


def fig_risk_donut(c: pd.DataFrame) -> go.Figure | None:
    if c.empty:
        return None
    counts = c["risk_category"].value_counts().reindex(list(CATEGORIES)).fillna(0)
    fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.58, sort=False,
                           marker=dict(colors=[RISK_COLORS[k] for k in counts.index]),
                           textinfo="percent+value"))
    fig.add_annotation(text=f"<b>{int(counts.sum()):,}</b><br>customers", showarrow=False, font=dict(size=16))
    fig.update_layout(title="Customer risk distribution")
    return fig


def fig_credit_hist(c: pd.DataFrame) -> go.Figure | None:
    if c.empty:
        return None
    fig = px.histogram(c, x="credit_score", color="risk_category", nbins=35, color_discrete_map=RISK_COLORS,
                       category_orders={"risk_category": list(CATEGORIES)})
    fig.update_layout(title="Credit score distribution (customers, coloured by model risk category)",
                      xaxis_title="bureau credit score", yaxis_title="customers", barmode="stack")
    return fig


def fig_credit_trend(a: pd.DataFrame) -> go.Figure | None:
    if a.empty:
        return None
    allm = a.groupby("month")["credit_score"].mean().rename("All applicants")
    dec = (a[a["approval_status"].isin(["Approved", "Rejected"])]
           .groupby(["month", "approval_status"])["credit_score"].mean().unstack())
    t = pd.concat([allm, dec], axis=1).reset_index().melt(id_vars="month", var_name="group", value_name="avg_score").dropna()
    fig = px.line(t, x="month", y="avg_score", color="group", markers=True,
                  color_discrete_map={**DECISION_COLORS, "All applicants": ACCENT})
    fig.update_layout(title="Average credit score of applicants, by application month",
                      xaxis_title="", yaxis_title="average bureau score")
    return fig


def state_table(a: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
    dec = a[a["approval_status"].isin(["Approved", "Rejected"])]
    t = pd.DataFrame({
        "Applications": a.groupby("state").size(),
        "Approval rate (%)": dec.groupby("state")["approval_status"].apply(lambda s: (s == "Approved").mean() * 100),
        "Disbursed value (Rs crore)": a[a["disbursed_flag"] == 1].groupby("state")["approved_amount"].sum() / 1e7,
    })
    d = a[a["loan_default"].notna()].groupby("state")["loan_default"].agg(["mean", "size"])
    t["Disbursed loans"] = d["size"]
    t["Default rate (%)"] = (d["mean"] * 100).where(d["size"] >= MIN_N)
    t["Avg customer risk score"] = c.groupby("state")["risk_score"].mean()
    t["High-risk customers (%)"] = c.groupby("state")["risk_category"].apply(lambda s: (s == "High").mean() * 100)
    return t.reset_index().rename(columns={"state": "State"})


def fig_geography(t: pd.DataFrame, metric: str) -> go.Figure:
    d = t[["State", metric]].dropna().sort_values(metric)
    fig = px.bar(d, x=metric, y="State", orientation="h", color_discrete_sequence=[ACCENT],
                 text=d[metric].map(lambda v: f"{v:,.1f}"))
    fig.update_traces(textposition="outside")
    fig.update_layout(title=f"{metric} by state", yaxis_title="", height=max(380, 34 * len(d) + 80))
    return fig


def fig_band_validation(a: pd.DataFrame) -> go.Figure | None:
    d = a[a["loan_default"].notna()]
    if d.empty:
        return None
    g = d.groupby("app_risk_category").agg(rate=("loan_default", "mean"), n=("loan_default", "size")).reindex(list(CATEGORIES))
    g = g.dropna()
    g["rate"] *= 100
    fig = go.Figure(go.Bar(x=g.index, y=g["rate"], marker_color=[RISK_COLORS[k] for k in g.index],
                           text=[f"{r:.1f}%<br>n={int(n):,}" for r, n in zip(g["rate"], g["n"])], textposition="outside"))
    fig.add_hline(y=d["loan_default"].mean() * 100, line_dash="dash", line_color="grey",
                  annotation_text="all selected loans", annotation_position="top left")
    fig.update_layout(title="Actual default rate by model risk category (disbursed loans)",
                      yaxis_title="actual default rate (%)", yaxis_ticksuffix="%", xaxis_title="", showlegend=False)
    return fig


def fig_mix_by_decision(a: pd.DataFrame) -> go.Figure | None:
    if a.empty:
        return None
    mix = (pd.crosstab(a["approval_status"], a["app_risk_category"], normalize="index") * 100)
    mix = mix.reindex(columns=list(CATEGORIES), fill_value=0).reset_index().melt(id_vars="approval_status", var_name="risk", value_name="share")
    fig = px.bar(mix, x="approval_status", y="share", color="risk", color_discrete_map=RISK_COLORS,
                 category_orders={"risk": list(CATEGORIES), "approval_status": ["Approved", "Rejected", "Under Review"]})
    fig.update_layout(title="Model risk category by human decision", xaxis_title="", yaxis_title="% of applications",
                      yaxis_ticksuffix="%")
    return fig


def chart_or_info(fig: go.Figure | None, key: str, msg: str = "No data for the current filters.") -> None:
    if fig is None:
        st.info(msg)
    else:
        show_chart(fig, key)


# =============================================================================
# Tabs
# =============================================================================
def tab_overview(a: pd.DataFrame) -> None:
    col1, col2 = st.columns(2)
    with col1:
        chart_or_info(fig_approval_trend(a), "approval_trend")
    with col2:
        chart_or_info(fig_default_by_type(a), "default_by_type")
    col3, col4 = st.columns(2)
    with col3:
        chart_or_info(fig_monthly_volume(a), "monthly_volume")
    with col4:
        view = st.radio("Approvals vs rejections", ["Over time", "By loan type"], horizontal=True, key="decision_view")
        chart_or_info(fig_decisions_over_time(a) if view == "Over time" else fig_decisions_by_type(a), "decisions")
    st.caption("Approval rate = approved / (approved + rejected); applications still under review are excluded. "
               "Default rate is observed on disbursed loans only, because that is the only group whose outcome is known. "
               "Default rates are hidden for groups under 20 loans.")


def tab_risk(a: pd.DataFrame, c: pd.DataFrame) -> None:
    col1, col2 = st.columns([1, 1.4])
    with col1:
        chart_or_info(fig_risk_donut(c), "donut")
        basis = c["scoring_basis"].value_counts()
        if "profile only" in basis.index:
            st.caption(f"{int(basis['profile only']):,} of these customers never applied for a loan and are scored from "
                       "their bureau, demographic and transaction profile alone.")
    with col2:
        view = st.radio("Credit score view", ["Distribution", "Trend by application month"], horizontal=True, key="credit_view")
        chart_or_info(fig_credit_hist(c) if view == "Distribution" else fig_credit_trend(a), "credit")
        if view != "Distribution":
            st.caption("The bureau file holds one snapshot per customer (30 Jun 2024), so a true history of each customer's score "
                       "does not exist. This chart shows the average score of the people who applied in each month.")
    st.caption(f"Risk score = 100 x calibrated probability of default. Bands: Low <= {LOW_MAX}, Medium {LOW_MAX + 1}-{MEDIUM_MAX}, "
               f"High >= {MEDIUM_MAX + 1}. A customer's score is the average across their applications.")


def tab_geography(a: pd.DataFrame, c: pd.DataFrame) -> None:
    if a.empty:
        st.info("No data for the current filters.")
        return
    t = state_table(a, c)
    metric = st.selectbox("Metric", [m for m in t.columns if m != "State"], index=0, key="geo_metric")
    show_chart(fig_geography(t, metric), "geo")
    with st.expander("Show the full state table"):
        show_table(t.round(1))


def tab_watchlist(c: pd.DataFrame, f: dict) -> None:
    col1, col2, col3 = st.columns([1.2, 2, 1])
    risk_pick = col1.multiselect("Risk category", list(CATEGORIES), key="wl_risk", placeholder="All categories")
    state_opts = sorted(c["state"].unique())
    state_pick = col2.multiselect("Region (state)", state_opts, key="wl_states", placeholder="All states")
    top_n = col3.number_input("Rows to show", min_value=10, max_value=1000, value=50, step=10, key="wl_n")

    w = c[c["risk_category"].isin(risk_pick or list(CATEGORIES)) & c["state"].isin(state_pick or state_opts)]
    w = w.sort_values(["risk_score", "default_probability"], ascending=False)
    st.markdown(f"**{len(w):,} customers** match. Showing the {min(int(top_n), len(w)):,} highest scores.")
    if w.empty:
        st.info("No customers match these filters.")
        return
    out = w.head(int(top_n)).assign(default_probability=lambda d: (d["default_probability"] * 100).round(1))[
        ["customer_id", "city", "state", "employment_type", "n_applications", "credit_score",
         "risk_score", "risk_category", "default_probability", "scoring_basis"]]
    out.columns = ["Customer ID", "City", "State", "Employment", "Applications", "Credit score",
                   "Risk score", "Risk category", "Default prob. (%)", "Scored on"]
    show_table(out, column_config={
        "Risk score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=100, format="%d")})
    st.download_button("Download this list (CSV)", out.to_csv(index=False).encode("utf-8"),
                       file_name="high_risk_customers.csv", mime="text/csv")
    st.caption("Customer IDs and non-sensitive attributes only. Names, emails and phone numbers are never loaded.")


def tab_model_check(a: pd.DataFrame, summary: dict | None) -> None:
    st.markdown("Does the score really separate good loans from bad ones? For disbursed loans, each score comes from a model "
                "that **never saw that loan** (out-of-fold), so this check is honest.")
    col1, col2 = st.columns(2)
    with col1:
        chart_or_info(fig_band_validation(a), "band_validation")
    with col2:
        chart_or_info(fig_mix_by_decision(a), "mix_by_decision")
    st.caption("If the model works, the actual default rate rises from Low to High. Rejected applications were never in the "
               "model's training data, so their scores are extrapolations whose calibration cannot be verified.")
    if summary:
        m = summary["test_at_tuned_threshold"]
        st.subheader("Model summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Champion model", summary["champion"])
        c2.metric("Test ROC-AUC", f"{m['roc_auc']:.3f}", help="PRD target was 0.80; see the model card for the discussion.")
        c3.metric("Recall (defaulters caught)", f"{m['recall']:.2f}")
        c4.metric("Precision", f"{m['precision']:.2f}")
        st.caption(f"Held-out test set of {summary['n_test']:,} loans. Treat the model as a second opinion for underwriters, "
                   "not an automatic decision-maker.")
    else:
        st.info("models/metrics.json not found. Run  python src/train.py  to create it.")


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    st.title("📊 Credora Finance - Credit Risk & Loan Approval Dashboard")
    st.caption("Retail lending book: decisions, default outcomes and model risk scores. Synthetic data.")

    try:
        apps, custs = load_data()
    except SystemExit as e:  # get_engine() exits when .env is incomplete
        st.error(f"Database settings are missing: {e}. Copy .env.example to .env and fill in your MySQL login.")
        st.stop()
    except Exception as e:
        st.error("Could not read the data from MySQL. Have you run  python src/data_load.py  and  python src/risk_score.py ?")
        st.exception(e)
        st.stop()

    f = sidebar_filters(apps, custs)
    a, c = apply_filters(apps, custs, f)

    summary = load_model_summary()
    with st.sidebar.expander("How to read this dashboard"):
        st.markdown(
            "- **Approval rate** counts decided applications only.\n"
            "- **Default rate** is observed on disbursed loans only.\n"
            "- **Risk score** (0-100) is a calibrated probability of default.\n"
            "- Scores for rejected or under-review applications are extrapolations.\n"
            "- Some customers never applied; they are scored on profile only.\n"
            "- Customer charts follow the state and risk filters. Choosing a loan type or month range keeps only "
            "customers with a matching application.")

        if a.empty:
            st.warning("No applications match this combination of filters. Remove a filter or press **Reset filters**.")
        

    k = kpis(a, c)
    cols = st.columns(5)
    cols[0].metric("Applications", f"{k['applications']:,}")
    cols[1].metric("Approval rate", pct(k["approval_rate"]), help=f"Approved / decided applications ({k['n_decided']:,} decided).")
    cols[2].metric("Default rate", pct(k["default_rate"]),
                   help=f"Observed on {k['n_disbursed']:,} disbursed loans with a known outcome.")
    cols[3].metric("Disbursed value", f"Rs {k['disbursed_cr']:,.0f} Cr")
    cols[4].metric("High-risk customers", f"{k['high_risk']:,}", delta=f"{pct(k['high_risk_pct'])} of {k['customers']:,}",
                   delta_color="off")

    t1, t2, t3, t4, t5 = st.tabs(["Overview", "Risk & credit", "Geography", "High-risk watchlist", "Model check"])
    with t1:
        tab_overview(a)
    with t2:
        tab_risk(a, c)
    with t3:
        tab_geography(a, c)
    with t4:
        tab_watchlist(c, f)
    with t5:
        tab_model_check(a, summary)


if __name__ == "__main__":
    main()
