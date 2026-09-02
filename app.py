"""
Vendor Reliability Scoring Dashboard
--------------------------------------
Streamlit demo app for the vendor reliability scoring project.

Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor

from modeling import (
    get_feature_names,
    BEHAVIORAL_FEATURES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
    build_pipeline,
)

st.set_page_config(page_title="Vendor Reliability Scoring", page_icon=None, layout="wide")

DATA_PATH = "vendor_data.csv"

# ----------------------------------------------------------------------
# Design tokens
# ----------------------------------------------------------------------
COLOR_BG = "#14181D"
COLOR_PANEL = "#1C222B"
COLOR_PANEL_LINE = "#2B323C"
COLOR_TEXT = "#E7E3DA"
COLOR_TEXT_MUTED = "#8B93A0"

COLOR_STEEL = "#2E6F8E"      # operational / reliable
COLOR_AMBER = "#C98A3B"      # behavioral / caution
COLOR_RUST = "#B5453D"       # high risk
COLOR_PATINA = "#4C8C6B"     # preferred

TIER_COLORS = {
    "Preferred": COLOR_PATINA,
    "Approved": COLOR_STEEL,
    "Watchlist": COLOR_AMBER,
    "High Risk": COLOR_RUST,
}

FONT_SANS = "IBM Plex Sans"
FONT_MONO = "IBM Plex Mono"

DISPLAY_NAMES = {
    "vendor_id": "Vendor ID",
    "category": "Category",
    "country": "Country",
    "years_as_vendor": "Years Approved",
    "num_orders_completed": "Orders Completed",
    "avg_order_value_usd": "Avg Order Value",
    "on_time_delivery_rate": "On-Time Delivery",
    "defect_rejection_rate": "Defect Rate",
    "lead_time_variance_days": "Lead Time Variance (days)",
    "price_competitiveness": "Price vs. Market",
    "payment_terms_compliance": "Payment Compliance",
    "rfq_response_time_hours": "RFQ Response (hrs)",
    "communication_responsiveness_score": "Responsiveness Score",
    "price_volatility_across_quotes": "Quote Volatility",
    "dispute_frequency_per_year": "Disputes / Year",
    "has_iso_certification": "ISO Certified",
    "reliability_score": "Reliability Score",
    "risk_tier": "Risk Tier",
}

# ----------------------------------------------------------------------
# Global styling
# ----------------------------------------------------------------------
st.markdown(
    f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="css"] {{
            font-family: '{FONT_SANS}', sans-serif;
            color: {COLOR_TEXT};
        }}
        .stApp {{
            background-color: {COLOR_BG};
        }}
        h1, h2, h3 {{
            font-family: '{FONT_SANS}', sans-serif;
            font-weight: 600;
            letter-spacing: -0.01em;
        }}
        .app-header {{
            border-bottom: 1px solid {COLOR_PANEL_LINE};
            padding-bottom: 20px;
            margin-bottom: 28px;
        }}
        .app-header .title {{
            font-size: 1.9rem;
            font-weight: 600;
            color: {COLOR_TEXT};
            margin-bottom: 4px;
        }}
        .app-header .subtitle {{
            font-size: 0.95rem;
            color: {COLOR_TEXT_MUTED};
            max-width: 640px;
            line-height: 1.5;
        }}
        .tile-row {{
            display: flex;
            gap: 14px;
            margin-bottom: 8px;
            flex-wrap: wrap;
        }}
        .tile {{
            background-color: {COLOR_PANEL};
            border-left: 3px solid {COLOR_STEEL};
            border-radius: 2px;
            padding: 14px 18px;
            flex: 1;
            min-width: 150px;
        }}
        .tile .tile-label {{
            font-size: 0.78rem;
            color: {COLOR_TEXT_MUTED};
            margin-bottom: 6px;
        }}
        .tile .tile-value {{
            font-family: '{FONT_MONO}', monospace;
            font-size: 1.6rem;
            font-weight: 500;
            color: {COLOR_TEXT};
        }}
        .section-label {{
            font-size: 0.8rem;
            color: {COLOR_TEXT_MUTED};
            margin-top: 6px;
            margin-bottom: 10px;
        }}
        .field-group-label {{
            font-family: '{FONT_MONO}', monospace;
            font-size: 0.78rem;
            color: {COLOR_TEXT_MUTED};
            border-bottom: 1px solid {COLOR_PANEL_LINE};
            padding-bottom: 6px;
            margin-bottom: 10px;
        }}
        .verdict-panel {{
            background-color: {COLOR_PANEL};
            border-radius: 2px;
            padding: 20px 24px;
            border-left: 3px solid var(--accent, {COLOR_STEEL});
        }}
        .verdict-score {{
            font-family: '{FONT_MONO}', monospace;
            font-size: 2.6rem;
            font-weight: 600;
            line-height: 1;
        }}
        .verdict-tier {{
            font-size: 0.95rem;
            color: {COLOR_TEXT_MUTED};
            margin-top: 6px;
        }}
        .flag-note {{
            font-size: 0.85rem;
            color: {COLOR_AMBER};
            border-left: 2px solid {COLOR_AMBER};
            padding: 8px 12px;
            margin-top: 14px;
            background-color: rgba(201, 138, 59, 0.08);
        }}
        div[data-testid="stTabs"] button {{
            font-family: '{FONT_SANS}', sans-serif;
            font-size: 0.92rem;
        }}
        [data-testid="stDataFrame"] {{
            font-family: '{FONT_MONO}', monospace;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model(_df):
    # Trained here (instead of loaded from a pickle) so the app never
    # breaks from a scikit-learn/Python version mismatch between where
    # it was trained and where it's deployed. The dataset is small, so
    # this takes well under a second and Streamlit caches the result
    # for the life of the deployment.
    X = _df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = _df[TARGET]
    pipe = build_pipeline(
        GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42)
    )
    pipe.fit(X, y)
    return pipe


df = load_data()
model = load_model(df)

plotly_layout = dict(
    paper_bgcolor=COLOR_BG,
    plot_bgcolor=COLOR_BG,
    font=dict(family=FONT_SANS, color=COLOR_TEXT, size=13),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)
axis_style = dict(gridcolor=COLOR_PANEL_LINE, zerolinecolor=COLOR_PANEL_LINE, linecolor=COLOR_PANEL_LINE)

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <div class="title">Vendor Reliability Scoring</div>
        <div class="subtitle">
            Scores oil &amp; gas procurement vendors on delivery, quality, and payment history,
            plus a smaller set of behavioral signals — RFQ responsiveness, quote consistency,
            and dispute history — that standard scorecards leave out.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_overview, tab_explore, tab_predict, tab_about = st.tabs(
    ["Overview", "Vendor explorer", "Score a vendor", "About this model"]
)

# ----------------------------------------------------------------------
# TAB 1 — Portfolio overview
# ----------------------------------------------------------------------
with tab_overview:
    st.markdown(
        f"""
        <div class="tile-row">
            <div class="tile"><div class="tile-label">Vendors tracked</div><div class="tile-value">{len(df)}</div></div>
            <div class="tile" style="border-left-color:{COLOR_STEEL}"><div class="tile-label">Average score</div><div class="tile-value">{df.reliability_score.mean():.1f}</div></div>
            <div class="tile" style="border-left-color:{COLOR_RUST}"><div class="tile-label">High risk</div><div class="tile-value">{int((df.risk_tier == "High Risk").sum())}</div></div>
            <div class="tile" style="border-left-color:{COLOR_PATINA}"><div class="tile-label">Preferred</div><div class="tile-value">{int((df.risk_tier == "Preferred").sum())}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-label">Score distribution by tier</div>', unsafe_allow_html=True)
        fig = px.histogram(
            df, x="reliability_score", color="risk_tier",
            color_discrete_map=TIER_COLORS, nbins=28,
            category_orders={"risk_tier": list(TIER_COLORS.keys())},
        )
        fig.update_layout(**plotly_layout, xaxis=dict(title="Reliability score", **axis_style),
                           yaxis=dict(title="Vendors", **axis_style), legend_title_text="")
        st.plotly_chart(fig, width='stretch')

    with c2:
        st.markdown('<div class="section-label">Vendors by tier</div>', unsafe_allow_html=True)
        tier_counts = df.risk_tier.value_counts().reindex(TIER_COLORS.keys())
        fig2 = go.Figure(go.Bar(
            x=tier_counts.index, y=tier_counts.values,
            marker_color=[TIER_COLORS[t] for t in tier_counts.index],
        ))
        fig2.update_layout(**plotly_layout, xaxis=dict(title="", **axis_style),
                            yaxis=dict(title="Vendors", **axis_style), showlegend=False)
        st.plotly_chart(fig2, width='stretch')

    st.write("")
    st.markdown('<div class="section-label">What drives the predicted score</div>', unsafe_allow_html=True)

    feature_names = get_feature_names(model)
    importances = model.named_steps["model"].feature_importances_
    order = np.argsort(importances)[::-1][:12]
    imp_names = [DISPLAY_NAMES.get(feature_names[i], feature_names[i]) for i in order]
    imp_vals = importances[order]
    imp_colors = [
        COLOR_AMBER if feature_names[i] in BEHAVIORAL_FEATURES else COLOR_STEEL
        for i in order
    ]

    fig3 = go.Figure(go.Bar(
        x=imp_vals[::-1], y=imp_names[::-1], orientation="h",
        marker_color=imp_colors[::-1],
    ))
    fig3.update_layout(
        **plotly_layout,
        xaxis=dict(title="Relative importance", **axis_style),
        yaxis=dict(title="", **axis_style),
        height=380,
    )
    st.plotly_chart(fig3, width='stretch')
    st.markdown(
        f'<div class="section-label">'
        f'<span style="color:{COLOR_STEEL}">■</span> Operational / financial &nbsp;&nbsp;'
        f'<span style="color:{COLOR_AMBER}">■</span> Behavioral'
        f"</div>",
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------
# TAB 2 — Explorer
# ----------------------------------------------------------------------
with tab_explore:
    f1, f2, f3 = st.columns(3)
    with f1:
        tiers = st.multiselect("Risk tier", options=list(TIER_COLORS.keys()), default=list(TIER_COLORS.keys()))
    with f2:
        categories = st.multiselect("Category", options=sorted(df.category.unique()), default=[])
    with f3:
        countries = st.multiselect("Country", options=sorted(df.country.unique()), default=[])

    filtered = df[df.risk_tier.isin(tiers)]
    if categories:
        filtered = filtered[filtered.category.isin(categories)]
    if countries:
        filtered = filtered[filtered.country.isin(countries)]

    st.markdown(f'<div class="section-label">{len(filtered)} of {len(df)} vendors</div>', unsafe_allow_html=True)

    display_df = filtered.sort_values("reliability_score", ascending=False).rename(columns=DISPLAY_NAMES)
    # Pre-scale fractional columns to 0-100 so the percent formatting renders correctly
    for col in ["On-Time Delivery", "Defect Rate", "Payment Compliance"]:
        display_df[col] = (display_df[col] * 100).round(1)

    st.dataframe(
        display_df,
        width='stretch',
        height=380,
        column_config={
            "Avg Order Value": st.column_config.NumberColumn(format="$%d"),
            "On-Time Delivery": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
            "Defect Rate": st.column_config.NumberColumn(format="%.1f%%"),
            "Payment Compliance": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
            "Reliability Score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
        },
        hide_index=True,
    )

    st.write("")
    st.markdown('<div class="section-label">Score against a chosen feature</div>', unsafe_allow_html=True)
    numeric_cols = [c for c in df.columns if df[c].dtype != object and c != "reliability_score"]
    x_feature = st.selectbox(
        "Feature", numeric_cols, index=numeric_cols.index("defect_rejection_rate"),
        format_func=lambda c: DISPLAY_NAMES.get(c, c), label_visibility="collapsed",
    )
    fig4 = px.scatter(
        filtered, x=x_feature, y="reliability_score", color="risk_tier",
        color_discrete_map=TIER_COLORS, category_orders={"risk_tier": list(TIER_COLORS.keys())},
        hover_data=["vendor_id", "category", "country"],
    )
    fig4.update_layout(
        **plotly_layout,
        xaxis=dict(title=DISPLAY_NAMES.get(x_feature, x_feature), **axis_style),
        yaxis=dict(title="Reliability score", **axis_style),
        legend_title_text="",
    )
    st.plotly_chart(fig4, width='stretch')

# ----------------------------------------------------------------------
# TAB 3 — Live predictor / what-if tool
# ----------------------------------------------------------------------
with tab_predict:
    st.markdown(
        '<div class="section-label">Build a vendor profile to see a predicted score '
        "before enough order history exists to calculate one directly.</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="field-group-label">Operational</div>', unsafe_allow_html=True)
        on_time = st.slider("On-time delivery rate", 0.0, 1.0, 0.85, 0.01)
        defect = st.slider("Defect / rejection rate", 0.0, 0.30, 0.08, 0.005)
        lead_var = st.slider("Lead time variance (days)", 0.0, 20.0, 5.0, 0.5)
        years = st.slider("Years as approved vendor", 0.0, 20.0, 3.0, 0.5)
        orders = st.slider("Orders completed", 1, 150, 20)

    with c2:
        st.markdown('<div class="field-group-label">Financial</div>', unsafe_allow_html=True)
        price_comp = st.slider("Price vs. market (1.0 = average)", 0.6, 1.5, 1.0, 0.01)
        payment_comp = st.slider("Payment terms compliance", 0.0, 1.0, 0.85, 0.01)
        order_value = st.number_input("Avg order value (USD)", 500, 300000, 20000, step=500)
        iso = st.checkbox("ISO certified", value=True)

    with c3:
        st.markdown('<div class="field-group-label">Behavioral</div>', unsafe_allow_html=True)
        rfq_time = st.slider("RFQ response time (hours)", 1.0, 120.0, 24.0, 1.0)
        comm_score = st.slider("Communication responsiveness (0-100)", 0.0, 100.0, 75.0, 1.0)
        price_vol = st.slider("Price volatility across quotes", 0.01, 0.60, 0.10, 0.01)
        disputes = st.slider("Disputes / renegotiations per year", 0.0, 8.0, 1.0, 0.1)

    c4, c5 = st.columns(2)
    with c4:
        category = st.selectbox("Category", sorted(df.category.unique()))
    with c5:
        country = st.selectbox("Country", sorted(df.country.unique()))

    input_row = pd.DataFrame([{
        "years_as_vendor": years,
        "num_orders_completed": orders,
        "avg_order_value_usd": order_value,
        "on_time_delivery_rate": on_time,
        "defect_rejection_rate": defect,
        "lead_time_variance_days": lead_var,
        "price_competitiveness": price_comp,
        "payment_terms_compliance": payment_comp,
        "rfq_response_time_hours": rfq_time,
        "communication_responsiveness_score": comm_score,
        "price_volatility_across_quotes": price_vol,
        "dispute_frequency_per_year": disputes,
        "has_iso_certification": int(iso),
        "category": category,
        "country": country,
    }])

    predicted = float(np.clip(model.predict(input_row)[0], 0, 100))

    if predicted >= 75:
        tier, tier_color = "Preferred", COLOR_PATINA
    elif predicted >= 55:
        tier, tier_color = "Approved", COLOR_STEEL
    elif predicted >= 35:
        tier, tier_color = "Watchlist", COLOR_AMBER
    else:
        tier, tier_color = "High Risk", COLOR_RUST

    st.write("")
    r1, r2 = st.columns([1, 1.4])

    with r1:
        st.markdown(
            f"""
            <div class="verdict-panel" style="--accent:{tier_color}">
                <div class="verdict-score" style="color:{tier_color}">{predicted:.1f}</div>
                <div class="verdict-tier">{tier} &nbsp;·&nbsp; predicted reliability score</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if defect > 0.15:
            st.markdown(
                '<div class="flag-note">Defect rate is past the 15% tolerance threshold. '
                "The model applies a steep additional penalty here rather than a gradual one — "
                "once quality problems repeat, trust drops sharply.</div>",
                unsafe_allow_html=True,
            )

    with r2:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=predicted,
            number={"font": {"family": FONT_MONO, "size": 1, "color": COLOR_BG}},  # hide duplicate number
            gauge={
                "axis": {"range": [0, 100], "tickcolor": COLOR_TEXT_MUTED, "tickfont": {"family": FONT_MONO, "size": 11}},
                "bar": {"color": tier_color, "thickness": 0.28},
                "bgcolor": COLOR_PANEL,
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 35], "color": "rgba(181,69,61,0.18)"},
                    {"range": [35, 55], "color": "rgba(201,138,59,0.18)"},
                    {"range": [55, 75], "color": "rgba(46,111,142,0.18)"},
                    {"range": [75, 100], "color": "rgba(76,140,107,0.18)"},
                ],
            },
        ))
        gauge_layout = {**plotly_layout, "margin": dict(l=20, r=20, t=10, b=0)}
        gauge.update_layout(**gauge_layout, height=220)
        st.plotly_chart(gauge, width='stretch')

# ----------------------------------------------------------------------
# TAB 4 — About
# ----------------------------------------------------------------------
with tab_about:
    st.markdown(
        f"""
        <div style="max-width:640px; line-height:1.7; color:{COLOR_TEXT};">
        <p><strong>What this scores.</strong> Each vendor gets a reliability score from
        0–100, built from delivery, quality, and payment data plus four lighter
        behavioral signals: RFQ response time, communication responsiveness, price
        volatility across quotes, and dispute frequency.</p>

        <p><strong>Why behavioral signals.</strong> Two vendors can look identical on
        paper — same on-time rate, same defect rate — and still differ in how much
        friction they create: slow RFQ turnarounds, price gamesmanship, frequent
        renegotiation. Those patterns are measurable, and they show up before a
        delivery problem does.</p>

        <p><strong>How the score behaves.</strong> Once a vendor's defect rate passes
        roughly 15%, the score takes a disproportionate hit rather than a gradual one —
        modeled on loss aversion: trust erodes in a cliff, not a slope, once quality
        problems repeat. That non-linearity is also where the gradient boosting model
        earns a real edge over a plain linear one.</p>

        <p><strong>Data.</strong> Every vendor here is synthetic, generated from a
        hidden reliability trait that drives the observable features with noise —
        no confidential vendor or company data is used.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
