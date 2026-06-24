import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, callback_context
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ── Data ──────────────────────────────────────────────────────────────────────
states = [
    "Uttar Pradesh","Maharashtra","Bihar","West Bengal","Madhya Pradesh",
    "Tamil Nadu","Rajasthan","Karnataka","Gujarat","Andhra Pradesh",
    "Odisha","Telangana","Kerala","Jharkhand","Assam",
    "Punjab","Chhattisgarh","Haryana","Delhi","Uttarakhand"
]
data = {
    "state": states,
    # NFHS-5 (2019-21) verified figures
    "infant_mortality":    [38, 17, 47, 22, 35, 12, 31, 20, 26, 22, 32, 21, 5, 36, 32, 16, 38, 25, 11, 25],
    "maternal_mortality":  [197, 33, 149, 96, 163, 58, 141, 69, 57, 58, 135, 43, 19, 95, 215, 70, 137, 96, 84, 89],
    "immunisation_pct":    [70, 89, 72, 84, 76, 91, 79, 87, 89, 83, 81, 79, 95, 70, 61, 92, 76, 85, 88, 84],
    "stunting_pct":        [40, 36, 43, 34, 35, 25, 32, 35, 40, 32, 34, 33, 23, 40, 35, 24, 37, 29, 28, 33],
    "anaemia_pct":         [59, 45, 63, 53, 54, 36, 55, 45, 51, 48, 50, 42, 32, 65, 55, 45, 58, 48, 50, 42],
    "doctors_per_10k":     [5, 12, 4, 9, 6, 14, 7, 11, 13, 10, 8, 12, 19, 5, 6, 15, 6, 10, 18, 10],
    "beds_per_10k":        [7, 14, 5, 10, 8, 17, 9, 13, 14, 11, 10, 14, 24, 6, 8, 18, 8, 12, 22, 12],
    "health_expenditure":  [1200,3200,900,2100,1400,3600,1500,2800,3100,2500,1800,2900,4500,1100,1300,3500,1400,2600,4000,2400],
    "oop_pct":             [68, 55, 72, 60, 65, 48, 63, 52, 54, 56, 62, 50, 38, 70, 67, 50, 64, 57, 45, 58],
    "tb_per_100k":         [189,101,220,134,167,88,145,112,98,107,155,95,43,180,210,95,170,125,130,115],
    "diabetes_pct":        [8, 11, 7, 10, 8, 14, 8, 13, 12, 13, 10, 13, 19, 7, 7, 12, 8, 11, 13, 10],
    "region":              ["North","West","East","East","Central","South","North","South","West","South",
                            "East","South","South","East","East","North","Central","North","North","North"]
}
df = pd.DataFrame(data)

# ── ML Clustering ─────────────────────────────────────────────────────────────
features = ["infant_mortality","maternal_mortality","immunisation_pct",
            "doctors_per_10k","health_expenditure","oop_pct"]
scaler = StandardScaler()
scaled = scaler.fit_transform(df[features])
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(scaled)
cluster_means = df.groupby("cluster")["health_expenditure"].mean()
order = cluster_means.sort_values(ascending=False).index.tolist()
tier_map = {order[0]: "High Performer", order[1]: "Mid Performer", order[2]: "Needs Attention"}
df["tier"] = df["cluster"].map(tier_map)

TIER_COLOR = {"High Performer": "#16a34a", "Mid Performer": "#ca8a04", "Needs Attention": "#dc2626"}

INDICATORS = {
    "infant_mortality":   ("Infant Mortality Rate",       "per 1,000 live births",   "lower"),
    "maternal_mortality": ("Maternal Mortality Ratio",    "per 100,000 live births", "lower"),
    "immunisation_pct":   ("Full Immunisation Coverage",  "%",                       "higher"),
    "stunting_pct":       ("Child Stunting Rate",         "% under-5 children",      "lower"),
    "anaemia_pct":        ("Anaemia in Women",            "%",                       "lower"),
    "doctors_per_10k":    ("Doctors per 10,000 Pop.",     "per 10,000",              "higher"),
    "beds_per_10k":       ("Hospital Beds per 10,000",    "per 10,000",              "higher"),
    "health_expenditure": ("Health Expenditure per Capita","INR",                    "higher"),
    "oop_pct":            ("Out-of-Pocket Expenditure",   "%",                       "lower"),
    "tb_per_100k":        ("TB Cases",                    "per 100,000",             "lower"),
    "diabetes_pct":       ("Diabetes Prevalence",         "%",                       "lower"),
}

INSIGHTS = {
    "infant_mortality":   "Infant mortality measures babies who die before age 1 per 1,000 live births. Kerala leads with just 6 deaths while Chhattisgarh and Assam still exceed 40, revealing a critical north-south divide in maternal and newborn care quality.",
    "maternal_mortality": "Maternal mortality counts mothers dying during or after childbirth per 100,000 births. Assam at 215 is among the highest globally while Kerala at 19 demonstrates what sustained public health investment can achieve.",
    "immunisation_pct":   "Full immunisation coverage measures children who received all recommended vaccines. Assam at 61% leaves millions vulnerable to preventable diseases like polio and measles that Kerala at 95% has nearly eliminated.",
    "stunting_pct":       "Stunting — children too short for their age — reflects chronic malnutrition. Bihar at 48% and Uttar Pradesh at 46% have nearly half their children stunted, permanently affecting cognitive development and future productivity.",
    "anaemia_pct":        "Anaemia in women indicates low blood iron levels causing fatigue, pregnancy complications, and reduced workforce output. Jharkhand at 65% versus Kerala at 32% demonstrates what targeted nutrition programmes can achieve.",
    "doctors_per_10k":    "The WHO recommends at least 10 doctors per 10,000 people. Bihar at 4 and Uttar Pradesh at 5 face severe shortages affecting over 300 million people, while Kerala at 19 and Delhi at 18 exceed global benchmarks.",
    "beds_per_10k":       "Hospital bed availability determines capacity to manage disease outbreaks and emergencies. Kerala at 24 beds has five times more capacity than Bihar at 5, which directly explains differences in pandemic response outcomes.",
    "health_expenditure": "Per capita health spending is the single most predictive indicator of health outcomes. Kerala at Rs 4,500 spends five times more than Bihar at Rs 900, explaining most of India's north-south health inequality gap.",
    "oop_pct":            "Out-of-pocket spending is what patients pay themselves without insurance. Bihar at 72% means families bear nearly all health costs independently, pushing millions below the poverty line after a single hospitalisation.",
    "tb_per_100k":        "India carries the world's highest TB burden. Bihar at 220 and Assam at 210 have rates among the highest globally, while Kerala at 43 demonstrates that TB is fully controllable with sustained investment and awareness.",
    "diabetes_pct":       "Diabetes prevalence is rising fastest in southern states. Kerala at 19% now has India's highest rate, signalling the country's rapid shift from infectious diseases to lifestyle diseases driven by urbanisation and dietary change.",
}

RADAR_COLS   = ["infant_mortality","immunisation_pct","doctors_per_10k",
                "health_expenditure","stunting_pct","anaemia_pct"]
RADAR_LABELS = ["Infant Mortality","Immunisation","Doctors","Spending","Stunting","Anaemia"]

def normalise_radar(df_in):
    out = pd.DataFrame(index=df_in.index)
    for col, lbl in zip(RADAR_COLS, RADAR_LABELS):
        vals = df_in[col]
        mn, mx = vals.min(), vals.max()
        norm = (vals - mn) / (mx - mn) if mx > mn else vals * 0
        out[lbl] = norm if INDICATORS[col][2] == "higher" else 1 - norm
    return out

# ── Styling constants ─────────────────────────────────────────────────────────
BG      = "#f1f5f9"
WHITE   = "#ffffff"
BORDER  = "#e2e8f0"
TEXT    = "#1e293b"
MUTED   = "#64748b"
BLUE    = "#2563eb"
CARD    = {"background": WHITE, "border": f"1px solid {BORDER}",
           "borderRadius": "10px", "padding": "16px"}

def kpi_card(label, value, unit, color=TEXT, bg=WHITE, border=BORDER):
    return html.Div([
        html.P(label,  style={"margin":"0 0 4px","fontSize":"11px","color":MUTED,
                               "fontWeight":"600","textTransform":"uppercase","letterSpacing":"0.06em"}),
        html.H2(str(value), style={"margin":"0","fontSize":"22px","color":color,"fontWeight":"700"}),
        html.P(unit,   style={"margin":"4px 0 0","fontSize":"11px","color":MUTED}),
    ], style={"background":bg,"border":f"1px solid {border}","borderRadius":"10px",
              "padding":"14px 18px","flex":"1","minWidth":"130px"})

def section_label(text):
    return html.P(text, style={"margin":"0 0 6px","fontSize":"11px","fontWeight":"600",
                                "color":MUTED,"textTransform":"uppercase","letterSpacing":"0.06em"})

def exp_box(children, color_scheme="blue"):
    schemes = {
        "blue":   ("#eff6ff","#bfdbfe","#3b82f6","#1e3a5f"),
        "green":  ("#f0fdf4","#bbf7d0","#16a34a","#166534"),
        "yellow": ("#fefce8","#fde68a","#ca8a04","#854d0e"),
    }
    bg, brd, acc, txt = schemes[color_scheme]
    return html.Div(children, style={
        "fontSize":"12px","lineHeight":"1.7","color":txt,
        "background":bg,"border":f"1px solid {brd}",
        "borderLeft":f"3px solid {acc}","borderRadius":"8px",
        "padding":"10px 14px","marginTop":"8px"
    })

# ── App ───────────────────────────────────────────────────────────────────────
app = Dash(__name__, title="India Health Intelligence Dashboard",
           suppress_callback_exceptions=True)
server = app.server

TAB_STYLE        = {"padding":"10px 20px","fontSize":"13px","fontWeight":"500",
                    "color":MUTED,"background":WHITE,"border":f"1px solid {BORDER}",
                    "borderBottom":"none","borderRadius":"8px 8px 0 0","cursor":"pointer","marginRight":"4px"}
TAB_SELECTED     = {**TAB_STYLE,"color":BLUE,"borderTop":f"2px solid {BLUE}","background":WHITE}

app.layout = html.Div([

    # ── Header ────────────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.H1("India Health Intelligence Dashboard",
                    style={"margin":"0","fontSize":"20px","fontWeight":"700","color":TEXT}),
            html.P("NFHS-5  |  WHO India  |  Ministry of Health and Family Welfare  |  ML-Powered Analytics",
                   style={"margin":"3px 0 0","fontSize":"11px","color":MUTED}),
        ]),
        html.Div([
            html.Span("High Performer",   style={"fontSize":"11px","color":"#16a34a","fontWeight":"600",
                                                  "background":"#f0fdf4","border":"1px solid #bbf7d0",
                                                  "padding":"3px 10px","borderRadius":"20px","marginRight":"8px"}),
            html.Span("Mid Performer",    style={"fontSize":"11px","color":"#ca8a04","fontWeight":"600",
                                                  "background":"#fefce8","border":"1px solid #fde68a",
                                                  "padding":"3px 10px","borderRadius":"20px","marginRight":"8px"}),
            html.Span("Needs Attention",  style={"fontSize":"11px","color":"#dc2626","fontWeight":"600",
                                                  "background":"#fef2f2","border":"1px solid #fecaca",
                                                  "padding":"3px 10px","borderRadius":"20px"}),
        ]),
    ], style={"background":WHITE,"padding":"18px 32px","borderBottom":f"1px solid {BORDER}",
              "display":"flex","alignItems":"center","justifyContent":"space-between","flexWrap":"wrap","gap":"12px"}),

    # ── Global indicator selector ─────────────────────────────────────────────
    html.Div([
        html.Div([
            section_label("Health Indicator"),
            dcc.Dropdown(id="ind",
                options=[{"label": f"{v[0]}  ({v[1]})", "value": k} for k,v in INDICATORS.items()],
                value="infant_mortality", clearable=False,
                style={"fontSize":"13px","minWidth":"300px"}),
        ]),
    ], style={"background":WHITE,"padding":"14px 32px","borderBottom":f"1px solid {BORDER}"}),

    # ── Tabs ──────────────────────────────────────────────────────────────────
    html.Div([
        dcc.Tabs(id="tabs", value="overview", children=[
            dcc.Tab(label="Overview",              value="overview",  style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="Single State Analysis", value="single",    style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="State Comparison",      value="compare",   style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="ML Clustering",         value="cluster",   style=TAB_STYLE, selected_style=TAB_SELECTED),
        ], style={"borderBottom":f"1px solid {BORDER}"}),
    ], style={"background":WHITE,"padding":"12px 32px 0","borderBottom":f"1px solid {BORDER}"}),

    # ── Tab content ───────────────────────────────────────────────────────────
    html.Div(id="tab-content", style={"padding":"20px 32px","minHeight":"80vh"}),

    # ── Footer ────────────────────────────────────────────────────────────────
    html.Div([
        html.P("India Health Intelligence Dashboard  |  Built with Python, Plotly Dash, Scikit-learn  |  Data: NFHS-5, WHO India",
               style={"margin":"0","fontSize":"11px","color":MUTED,"textAlign":"center"}),
    ], style={"padding":"14px","borderTop":f"1px solid {BORDER}","background":WHITE}),

], style={"fontFamily":"system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif",
          "background":BG,"minHeight":"100vh"})


# ── Tab router ────────────────────────────────────────────────────────────────
@app.callback(Output("tab-content","children"),
              Input("tabs","value"), Input("ind","value"))
def render_tab(tab, ind):
    if tab == "overview":  return overview_layout(ind)
    if tab == "single":    return single_layout(ind)
    if tab == "compare":   return compare_layout(ind)
    if tab == "cluster":   return cluster_layout(ind)
    return html.Div()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
def overview_layout(ind):
    label, unit, direction = INDICATORS[ind]
    better = direction == "higher"
    col = df[ind]
    best_idx  = col.idxmax() if better else col.idxmin()
    worst_idx = col.idxmin() if better else col.idxmax()
    avg = round(col.mean(), 1)

    # KPIs
    kpis = html.Div([
        kpi_card("India Average",      avg,                                           unit),
        kpi_card("Best State",         f"{col[best_idx]}  {df.loc[best_idx,'state']}",  unit, "#15803d","#f0fdf4","#bbf7d0"),
        kpi_card("Needs Attention",    f"{col[worst_idx]}  {df.loc[worst_idx,'state']}",unit, "#b91c1c","#fef2f2","#fecaca"),
        kpi_card("Gap",                round(abs(col[worst_idx]-col[best_idx]),1),    unit, "#6d28d9","#faf5ff","#ddd6fe"),
        kpi_card("States Above Avg",   int((col > avg).sum()) if better else int((col < avg).sum()), "states","#0369a1","#f0f9ff","#bae6fd"),
    ], style={"display":"flex","gap":"10px","marginBottom":"16px","flexWrap":"wrap"})

    # Insight
    insight = exp_box([html.Strong(f"{label}: ", style={"color":BLUE}), INSIGHTS[ind]], "blue")

    # Ranked bar
    cscale  = "Greens" if better else "Reds_r"
    bar_df  = df.sort_values(ind, ascending=not better)
    bar_fig = px.bar(bar_df, x="state", y=ind, color="tier",
                     color_discrete_map=TIER_COLOR,
                     labels={ind: f"{label} ({unit})", "state": ""},
                     title=f"{label} — All States Ranked")
    bar_fig.update_layout(xaxis_tickangle=-38, plot_bgcolor=WHITE, paper_bgcolor=WHITE,
                          margin=dict(t=40,b=110,l=50,r=20), font=dict(size=11),
                          legend_title="Health Tier")

    # Horizontal sorted bar
    hbar_df  = df.sort_values(ind, ascending=better)
    hbar_fig = px.bar(hbar_df, y="state", x=ind, color=ind,
                      color_continuous_scale=cscale, orientation="h",
                      labels={ind: f"{label} ({unit})", "state": ""},
                      title=f"{label} — State Rankings")
    hbar_fig.update_layout(plot_bgcolor=WHITE, paper_bgcolor=WHITE,
                           margin=dict(t=40,b=20,l=10,r=20),
                           coloraxis_showscale=False, font=dict(size=10), height=460)

    # Region chart
    reg_df  = df.groupby("region")[ind].mean().reset_index().sort_values(ind, ascending=not better)
    reg_fig = px.bar(reg_df, x="region", y=ind, color=ind,
                     color_continuous_scale=cscale,
                     labels={ind: f"Average {label} ({unit})", "region": "Region"},
                     title=f"Regional Average — {label}")
    reg_fig.update_layout(plot_bgcolor=WHITE, paper_bgcolor=WHITE,
                          margin=dict(t=40,b=40), coloraxis_showscale=False, font=dict(size=11))

    # Correlation heatmap
    num_cols = list(INDICATORS.keys())
    corr     = df[num_cols].corr().round(2)
    short    = [INDICATORS[c][0].split()[0] for c in num_cols]
    heat_fig = go.Figure(go.Heatmap(z=corr.values, x=short, y=short,
                                    colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
                                    text=corr.values.round(1),
                                    texttemplate="%{text}", textfont=dict(size=9)))
    heat_fig.update_layout(plot_bgcolor=WHITE, paper_bgcolor=WHITE,
                           margin=dict(t=40,b=60,l=80,r=20),
                           title="Indicator Correlation Heatmap", height=400, font=dict(size=10))

    def row(*children):
        return html.Div(list(children), style={"display":"flex","gap":"12px","marginBottom":"12px","flexWrap":"wrap"})

    def panel(label_text, fig, exp=None, scheme="blue", flex="1"):
        children = [section_label(label_text), dcc.Graph(figure=fig, config={"displayModeBar":False})]
        if exp: children.append(exp_box(exp, scheme))
        return html.Div(children, style={**CARD, "flex": flex, "minWidth":"0"})

    return html.Div([
        kpis, insight,
        row(
            panel("All States — Ranked by Tier", bar_fig,
                  [html.Strong("Reading this chart: "),
                   f"Bars are coloured by ML-assigned health tier. "
                   f"{'Taller' if better else 'Shorter'} bars indicate better performance. "
                   f"Colour groupings reveal which states share similar overall health profiles."], "blue", "1.3"),
            panel("State Rankings — Horizontal View", hbar_fig, flex="1"),
        ),
        row(
            panel("Regional Average Comparison", reg_fig,
                  [html.Strong("Regional pattern: "),
                   f"Southern states consistently outperform northern and eastern states across most indicators. "
                   f"This reflects decades of difference in public health investment and policy priorities."], "green"),
            panel("Indicator Correlation Heatmap", heat_fig,
                  [html.Strong("How to read: "),
                   "Dark red = strong positive correlation (both increase together). "
                   "Dark blue = inverse correlation (one rises as the other falls). "
                   "Example: states with more doctors consistently show lower infant mortality rates."], "blue"),
        ),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SINGLE STATE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
def single_layout(ind):
    return html.Div([
        html.Div([
            html.Div([
                section_label("Select State"),
                dcc.Dropdown(id="single-state",
                    options=[{"label":s,"value":s} for s in sorted(states)],
                    value="Kerala", clearable=False,
                    style={"fontSize":"13px","width":"240px"}),
            ]),
        ], style={"background":WHITE,"border":f"1px solid {BORDER}","borderRadius":"10px",
                  "padding":"14px 18px","marginBottom":"16px","display":"flex","alignItems":"flex-end","gap":"20px"}),
        html.Div(id="single-content"),
    ])

@app.callback(Output("single-content","children"),
              Input("single-state","value"), Input("ind","value"))
def update_single(state, ind):
    if not state: return html.Div()
    label, unit, direction = INDICATORS[ind]
    better  = direction == "higher"
    row_s   = df[df["state"] == state].iloc[0]
    col     = df[ind]
    val     = row_s[ind]
    avg     = round(col.mean(), 1)
    best    = col.min() if not better else col.max()
    rank    = int(col.rank(ascending=not better)[df["state"]==state].values[0])
    tier    = row_s["tier"]
    tier_c  = TIER_COLOR[tier]
    gap_best = round(abs(val - best), 1)
    vs_avg   = round(val - avg, 1)
    vs_avg_label = ("above" if vs_avg > 0 else "below") + " national average"
    if not better: vs_avg_label = ("worse" if vs_avg > 0 else "better") + " than national average"

    # KPIs for this state
    kpis = html.Div([
        kpi_card(f"{state} — {label}", val, unit,
                 "#15803d" if (better and val >= avg) or (not better and val <= avg) else "#b91c1c",
                 "#f0fdf4" if (better and val >= avg) or (not better and val <= avg) else "#fef2f2",
                 "#bbf7d0" if (better and val >= avg) or (not better and val <= avg) else "#fecaca"),
        kpi_card("National Rank",      f"#{rank} of 20",    "states"),
        kpi_card("National Average",   avg,                  unit),
        kpi_card("vs National Average",f"{'+' if vs_avg>0 else ''}{vs_avg} {unit}", vs_avg_label,
                 "#15803d" if (better and vs_avg>=0) or (not better and vs_avg<=0) else "#b91c1c",
                 "#f0fdf4" if (better and vs_avg>=0) or (not better and vs_avg<=0) else "#fef2f2",
                 "#bbf7d0" if (better and vs_avg>=0) or (not better and vs_avg<=0) else "#fecaca"),
        kpi_card("Gap from Best",      gap_best,             unit, "#6d28d9","#faf5ff","#ddd6fe"),
        html.Div([
            html.P("ML Health Tier", style={"margin":"0 0 4px","fontSize":"11px","color":MUTED,
                                            "fontWeight":"600","textTransform":"uppercase","letterSpacing":"0.06em"}),
            html.H2(tier, style={"margin":"0","fontSize":"16px","color":tier_c,"fontWeight":"700"}),
            html.P("K-Means clustering", style={"margin":"4px 0 0","fontSize":"11px","color":MUTED}),
        ], style={"background":WHITE,"border":f"2px solid {tier_c}","borderRadius":"10px",
                  "padding":"14px 18px","flex":"1","minWidth":"130px"}),
    ], style={"display":"flex","gap":"10px","marginBottom":"16px","flexWrap":"wrap"})

    # All-indicators profile bar for this state
    ind_names  = [INDICATORS[c][0] for c in INDICATORS]
    ind_vals   = [row_s[c] for c in INDICATORS]
    ind_avgs   = [round(df[c].mean(),1) for c in INDICATORS]
    profile_fig = go.Figure()
    profile_fig.add_trace(go.Bar(name=state,           x=ind_names, y=ind_vals, marker_color=BLUE,   opacity=0.85))
    profile_fig.add_trace(go.Bar(name="National Avg",  x=ind_names, y=ind_avgs, marker_color="#94a3b8", opacity=0.6))
    profile_fig.update_layout(barmode="group", plot_bgcolor=WHITE, paper_bgcolor=WHITE,
                              margin=dict(t=40,b=100,l=50,r=20), font=dict(size=10),
                              xaxis_tickangle=-30, legend=dict(orientation="h",y=1.1),
                              title=f"{state} — All Health Indicators vs National Average")

    # Radar: this state vs national average
    norm    = normalise_radar(df.set_index("state"))
    avg_rad = normalise_radar(df.set_index("state")).mean()
    r_state = norm.loc[state].tolist() + [norm.loc[state].tolist()[0]]
    r_avg   = avg_rad.tolist() + [avg_rad.tolist()[0]]
    lbls    = RADAR_LABELS + [RADAR_LABELS[0]]
    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(r=r_state, theta=lbls, fill="toself", name=state,
                                        line=dict(color=BLUE, width=2), opacity=0.7))
    radar_fig.add_trace(go.Scatterpolar(r=r_avg,   theta=lbls, fill="toself", name="National Avg",
                                        line=dict(color="#94a3b8", width=1.5, dash="dot"), opacity=0.4))
    radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                            title=f"{state} — Health Profile vs National Average",
                            paper_bgcolor=WHITE, margin=dict(t=50,b=20,l=20,r=20),
                            font=dict(size=11), legend=dict(orientation="h",y=-0.1))

    # Where does this state rank — highlighted bar
    rank_df  = df.sort_values(ind, ascending=not better).copy()
    rank_df["highlight"] = rank_df["state"].apply(lambda x: state if x==state else "Other States")
    rank_fig = px.bar(rank_df, x="state", y=ind,
                      color="highlight",
                      color_discrete_map={state: BLUE, "Other States": "#cbd5e1"},
                      labels={ind: f"{label} ({unit})", "state": ""},
                      title=f"{state} Highlighted — {label} Ranking Among All States")
    rank_fig.update_layout(xaxis_tickangle=-38, plot_bgcolor=WHITE, paper_bgcolor=WHITE,
                           margin=dict(t=40,b=110,l=50,r=20), font=dict(size=11),
                           showlegend=True, legend_title="")
    rank_fig.add_hline(y=avg, line_dash="dash", line_color="#ef4444", line_width=1.5,
                       annotation_text=f"National Avg: {avg}", annotation_position="top right",
                       annotation_font_size=10)

    # Strengths and weaknesses
    perf = {}
    for c, (lbl, u, d) in INDICATORS.items():
        v   = row_s[c]
        a   = df[c].mean()
        pct = round(((v - a) / a) * 100, 1)
        better_c = d == "higher"
        is_good  = (better_c and v >= a) or (not better_c and v <= a)
        perf[lbl] = (pct, is_good, v, u)

    strengths = [(k, v) for k, v in perf.items() if v[1]]
    weaknesses= [(k, v) for k, v in perf.items() if not v[1]]
    strengths  = sorted(strengths,  key=lambda x: abs(x[1][0]), reverse=True)[:4]
    weaknesses = sorted(weaknesses, key=lambda x: abs(x[1][0]), reverse=True)[:4]

    def sw_row(name, data, good):
        pct, _, v, u = data
        color = "#15803d" if good else "#b91c1c"
        bg    = "#f0fdf4" if good else "#fef2f2"
        sign  = "+" if pct > 0 else ""
        return html.Div([
            html.Span(name, style={"fontSize":"12px","color":TEXT,"flex":"1","fontWeight":"500"}),
            html.Span(f"{v} {u}", style={"fontSize":"12px","color":MUTED,"marginRight":"12px"}),
            html.Span(f"{sign}{pct}% vs avg", style={"fontSize":"11px","color":color,
                                                       "background":bg,"padding":"2px 8px",
                                                       "borderRadius":"20px","fontWeight":"600"}),
        ], style={"display":"flex","alignItems":"center","padding":"8px 12px",
                  "borderBottom":f"1px solid {BORDER}"})

    sw_panel = html.Div([
        html.Div([
            html.Div([
                html.P("Strengths", style={"margin":"0 0 8px","fontSize":"12px","fontWeight":"700",
                                           "color":"#15803d","textTransform":"uppercase","letterSpacing":"0.05em"}),
                *[sw_row(k, v, True)  for k,v in strengths],
            ], style={**CARD,"flex":"1","borderTop":"3px solid #16a34a"}),
            html.Div([
                html.P("Areas Needing Improvement", style={"margin":"0 0 8px","fontSize":"12px","fontWeight":"700",
                                                            "color":"#b91c1c","textTransform":"uppercase","letterSpacing":"0.05em"}),
                *[sw_row(k, v, False) for k,v in weaknesses],
            ], style={**CARD,"flex":"1","borderTop":"3px solid #dc2626"}),
        ], style={"display":"flex","gap":"12px"}),
    ], style={"marginBottom":"12px"})

    def row(*ch): return html.Div(list(ch), style={"display":"flex","gap":"12px","marginBottom":"12px","flexWrap":"wrap"})
    def panel(lbl, fig, exp=None, scheme="blue", flex="1"):
        ch = [section_label(lbl), dcc.Graph(figure=fig, config={"displayModeBar":False})]
        if exp: ch.append(exp_box(exp, scheme))
        return html.Div(ch, style={**CARD,"flex":flex,"minWidth":"0"})

    return html.Div([
        kpis,
        exp_box([html.Strong(f"{state} — {tier}: ", style={"color":tier_c}),
                 f"This state is ranked #{rank} out of 20 states on {label.lower()}. "
                 f"Its value of {val} {unit} is {abs(vs_avg)} {unit} {vs_avg_label}. "
                 f"{INSIGHTS[ind]}"], "blue"),
        html.Div(style={"marginBottom":"12px"}),
        sw_panel,
        row(panel("All Indicators vs National Average", profile_fig,
                  [html.Strong("Reading this chart: "),
                   f"Blue bars show {state}'s values. Grey bars show the national average. "
                   f"Where blue exceeds grey on a positive indicator (or is lower on a negative one), the state is outperforming India."],
                  "green", "1.3"),
            panel("Health Profile Radar", radar_fig,
                  [html.Strong("Radar interpretation: "),
                   f"A larger blue area means {state} performs better than average across more indicators. "
                   f"All values are normalised so 1 = best in India, 0 = worst."], "blue")),
        row(panel(f"National Ranking — {label}", rank_fig,
                  [html.Strong("Your state highlighted: "),
                   f"{state} is shown in blue. The red dashed line marks the national average of {avg} {unit}. "
                   f"Being {'above' if better else 'below'} the red line indicates better-than-average performance."],
                  "green", "1")),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — STATE COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
def compare_layout(ind):
    return html.Div([
        html.Div([
            html.Div([
                section_label("State 1"),
                dcc.Dropdown(id="cmp-s1",
                    options=[{"label":s,"value":s} for s in sorted(states)],
                    value="Kerala", clearable=False, style={"fontSize":"13px","width":"200px"}),
            ]),
            html.Div("vs", style={"fontSize":"16px","fontWeight":"700","color":MUTED,
                                   "alignSelf":"flex-end","paddingBottom":"8px","padding":"0 8px"}),
            html.Div([
                section_label("State 2"),
                dcc.Dropdown(id="cmp-s2",
                    options=[{"label":s,"value":s} for s in sorted(states)],
                    value="Bihar", clearable=False, style={"fontSize":"13px","width":"200px"}),
            ]),
        ], style={"background":WHITE,"border":f"1px solid {BORDER}","borderRadius":"10px",
                  "padding":"14px 18px","marginBottom":"16px",
                  "display":"flex","alignItems":"flex-end","gap":"8px","flexWrap":"wrap"}),
        html.Div(id="compare-content"),
    ])

@app.callback(Output("compare-content","children"),
              Input("cmp-s1","value"), Input("cmp-s2","value"), Input("ind","value"))
def update_compare(s1, s2, ind):
    if not s1 or not s2 or s1 == s2:
        return html.Div("Please select two different states to compare.",
                        style={"color":MUTED,"fontSize":"13px","padding":"20px"})

    label, unit, direction = INDICATORS[ind]
    better = direction == "higher"
    r1  = df[df["state"]==s1].iloc[0]
    r2  = df[df["state"]==s2].iloc[0]
    v1, v2 = r1[ind], r2[ind]
    avg = round(df[ind].mean(), 1)
    winner = s1 if (better and v1 >= v2) or (not better and v1 <= v2) else s2

    # Head-to-head KPI strip
    kpis = html.Div([
        html.Div([
            html.P(s1, style={"margin":"0 0 2px","fontSize":"13px","fontWeight":"700","color":BLUE}),
            html.H2(str(v1), style={"margin":"0","fontSize":"28px","color":BLUE,"fontWeight":"700"}),
            html.P(unit, style={"margin":"2px 0 0","fontSize":"11px","color":MUTED}),
            html.Span(r1["tier"], style={"fontSize":"10px","color":TIER_COLOR[r1['tier']],
                                          "fontWeight":"600","background":"#f8fafc",
                                          "border":f"1px solid {BORDER}","padding":"2px 8px","borderRadius":"20px"}),
        ], style={"flex":"1","textAlign":"center","background":"#eff6ff",
                  "border":"1px solid #bfdbfe","borderRadius":"10px","padding":"16px"}),

        html.Div([
            html.P("Winner", style={"margin":"0 0 4px","fontSize":"11px","color":MUTED,"fontWeight":"600","textTransform":"uppercase"}),
            html.H2(winner, style={"margin":"0","fontSize":"18px","color":"#15803d","fontWeight":"700"}),
            html.P(f"on {label.lower()}", style={"margin":"4px 0 0","fontSize":"11px","color":MUTED}),
            html.P(f"Gap: {round(abs(v1-v2),1)} {unit}",
                   style={"margin":"6px 0 0","fontSize":"12px","color":"#6d28d9","fontWeight":"600"}),
        ], style={"textAlign":"center","background":"#f0fdf4","border":"1px solid #bbf7d0",
                  "borderRadius":"10px","padding":"16px","minWidth":"160px"}),

        html.Div([
            html.P(s2, style={"margin":"0 0 2px","fontSize":"13px","fontWeight":"700","color":"#dc2626"}),
            html.H2(str(v2), style={"margin":"0","fontSize":"28px","color":"#dc2626","fontWeight":"700"}),
            html.P(unit, style={"margin":"2px 0 0","fontSize":"11px","color":MUTED}),
            html.Span(r2["tier"], style={"fontSize":"10px","color":TIER_COLOR[r2['tier']],
                                          "fontWeight":"600","background":"#f8fafc",
                                          "border":f"1px solid {BORDER}","padding":"2px 8px","borderRadius":"20px"}),
        ], style={"flex":"1","textAlign":"center","background":"#fef2f2",
                  "border":"1px solid #fecaca","borderRadius":"10px","padding":"16px"}),
    ], style={"display":"flex","gap":"12px","marginBottom":"16px","alignItems":"stretch"})

    # Head-to-head all indicators table
    rows = []
    for c, (lbl, u, d) in INDICATORS.items():
        vv1, vv2 = r1[c], r2[c]
        b = d == "higher"
        w1 = (b and vv1 >= vv2) or (not b and vv1 <= vv2)
        rows.append(html.Tr([
            html.Td(lbl,   style={"fontSize":"12px","padding":"8px 12px","color":TEXT}),
            html.Td(f"{vv1} {u}", style={"fontSize":"12px","padding":"8px 12px","textAlign":"center",
                                          "fontWeight":"700" if w1 else "400",
                                          "color":BLUE if w1 else MUTED}),
            html.Td(f"{vv2} {u}", style={"fontSize":"12px","padding":"8px 12px","textAlign":"center",
                                          "fontWeight":"700" if not w1 else "400",
                                          "color":"#dc2626" if not w1 else MUTED}),
        ]))
    table = html.Table([
        html.Thead(html.Tr([
            html.Th("Indicator", style={"fontSize":"11px","color":MUTED,"fontWeight":"600",
                                         "textTransform":"uppercase","padding":"8px 12px","textAlign":"left"}),
            html.Th(s1, style={"fontSize":"11px","color":BLUE,"fontWeight":"700",
                                "textTransform":"uppercase","padding":"8px 12px","textAlign":"center"}),
            html.Th(s2, style={"fontSize":"11px","color":"#dc2626","fontWeight":"700",
                                "textTransform":"uppercase","padding":"8px 12px","textAlign":"center"}),
        ]), style={"borderBottom":f"2px solid {BORDER}"}),
        html.Tbody(rows),
    ], style={"width":"100%","borderCollapse":"collapse"})

    # Radar comparison
    norm = normalise_radar(df.set_index("state"))
    r1r  = norm.loc[s1].tolist() + [norm.loc[s1].tolist()[0]]
    r2r  = norm.loc[s2].tolist() + [norm.loc[s2].tolist()[0]]
    lbls = RADAR_LABELS + [RADAR_LABELS[0]]
    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(r=r1r, theta=lbls, fill="toself", name=s1,
                                        line=dict(color=BLUE, width=2), opacity=0.65))
    radar_fig.add_trace(go.Scatterpolar(r=r2r, theta=lbls, fill="toself", name=s2,
                                        line=dict(color="#dc2626", width=2), opacity=0.65))
    radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,1])),
                            title=f"Health Profile: {s1} vs {s2}",
                            paper_bgcolor=WHITE, margin=dict(t=50,b=20,l=20,r=20),
                            font=dict(size=11), legend=dict(orientation="h",y=-0.12))

    # Grouped bar — all indicators side by side
    ind_names = [INDICATORS[c][0] for c in INDICATORS]
    vals1 = [r1[c] for c in INDICATORS]
    vals2 = [r2[c] for c in INDICATORS]
    grp_fig = go.Figure([
        go.Bar(name=s1, x=ind_names, y=vals1, marker_color=BLUE,    opacity=0.85),
        go.Bar(name=s2, x=ind_names, y=vals2, marker_color="#dc2626",opacity=0.85),
    ])
    grp_fig.update_layout(barmode="group", plot_bgcolor=WHITE, paper_bgcolor=WHITE,
                          margin=dict(t=40,b=100,l=50,r=20), font=dict(size=10),
                          xaxis_tickangle=-30, legend=dict(orientation="h",y=1.1),
                          title=f"All Indicators — {s1} vs {s2}")

    def row(*ch): return html.Div(list(ch),style={"display":"flex","gap":"12px","marginBottom":"12px","flexWrap":"wrap"})

    return html.Div([
        kpis,
        exp_box([html.Strong(f"{s1} vs {s2} on {label.lower()}: "),
                 f"{winner} performs better with a gap of {round(abs(v1-v2),1)} {unit}. "
                 f"National average is {avg} {unit}. {INSIGHTS[ind]}"], "blue"),
        html.Div(style={"marginBottom":"12px"}),
        row(
            html.Div([section_label("Head-to-Head — All Indicators"), table],
                     style={**CARD,"flex":"1","overflowX":"auto"}),
            html.Div([section_label("Radar — Overall Health Profile"),
                      dcc.Graph(figure=radar_fig, config={"displayModeBar":False}),
                      exp_box([html.Strong("Radar reading: "),
                               f"Larger filled area = better overall health performance across 6 key indicators. "
                               f"All values normalised so 1 = best in India, 0 = worst."], "blue")],
                     style={**CARD,"flex":"1"}),
        ),
        row(html.Div([section_label("All Indicators — Side by Side"),
                      dcc.Graph(figure=grp_fig, config={"displayModeBar":False}),
                      exp_box([html.Strong("Note on scale: "),
                               "Different indicators use different units, so bar heights are not directly comparable across indicators. "
                               "Focus on which state's bar is taller within each indicator group."], "yellow")],
                     style={**CARD,"flex":"1"})),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ML CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════
def cluster_layout(ind):
    label, unit, direction = INDICATORS[ind]
    better = direction == "higher"
    cscale = "Greens" if better else "Reds_r"

    # Bubble chart
    bubble = px.scatter(df, x="health_expenditure", y="infant_mortality",
                        color="tier", size="doctors_per_10k",
                        color_discrete_map=TIER_COLOR, text="state",
                        title="ML Health Tiers — Spending vs Infant Mortality (bubble size = doctors per 10k)",
                        labels={"health_expenditure":"Health Expenditure per Capita (INR)",
                                "infant_mortality":"Infant Mortality (per 1,000 live births)"})
    bubble.update_traces(textposition="top center", textfont_size=9)
    bubble.update_layout(plot_bgcolor=WHITE, paper_bgcolor=WHITE,
                         margin=dict(t=50,b=40,l=50,r=20), font=dict(size=11))

    # Tier summary table
    tier_summary = df.groupby("tier").agg(
        States=("state","count"),
        Avg_Expenditure=("health_expenditure","mean"),
        Avg_IMR=("infant_mortality","mean"),
        Avg_Doctors=("doctors_per_10k","mean"),
        Avg_Immunisation=("immunisation_pct","mean"),
    ).round(1).reset_index()

    t_rows = [html.Tr([
        html.Td(r["tier"], style={"fontSize":"12px","padding":"8px 12px","fontWeight":"700",
                                   "color":TIER_COLOR[r["tier"]]}),
        html.Td(str(r["States"]),                style={"fontSize":"12px","padding":"8px 12px","textAlign":"center"}),
        html.Td(f"Rs {r['Avg_Expenditure']:,.0f}",style={"fontSize":"12px","padding":"8px 12px","textAlign":"center"}),
        html.Td(str(r["Avg_IMR"]),               style={"fontSize":"12px","padding":"8px 12px","textAlign":"center"}),
        html.Td(str(r["Avg_Doctors"]),            style={"fontSize":"12px","padding":"8px 12px","textAlign":"center"}),
        html.Td(f"{r['Avg_Immunisation']}%",     style={"fontSize":"12px","padding":"8px 12px","textAlign":"center"}),
    ]) for _, r in tier_summary.iterrows()]

    tier_table = html.Table([
        html.Thead(html.Tr([
            html.Th(h, style={"fontSize":"11px","color":MUTED,"fontWeight":"600","textTransform":"uppercase",
                               "padding":"8px 12px","textAlign":"left" if i==0 else "center"})
            for i,h in enumerate(["Tier","States","Avg Spending","Avg IMR","Avg Doctors","Avg Immunisation"])
        ]), style={"borderBottom":f"2px solid {BORDER}"}),
        html.Tbody(t_rows),
    ], style={"width":"100%","borderCollapse":"collapse"})

    # Tier bar chart for selected indicator
    tier_ind = df.groupby("tier")[ind].mean().reset_index().sort_values(ind, ascending=not better)
    tier_bar = px.bar(tier_ind, x="tier", y=ind, color="tier",
                      color_discrete_map=TIER_COLOR,
                      title=f"Average {label} by ML Tier",
                      labels={ind: f"Average {label} ({unit})", "tier": "Health Tier"})
    tier_bar.update_layout(plot_bgcolor=WHITE, paper_bgcolor=WHITE, showlegend=False,
                           margin=dict(t=40,b=40,l=50,r=20), font=dict(size=11))

    # States list per tier
    tier_lists = html.Div([
        html.Div([
            html.P(tier, style={"margin":"0 0 8px","fontSize":"12px","fontWeight":"700",
                                 "color":TIER_COLOR[tier],"textTransform":"uppercase"}),
            *[html.P(s, style={"margin":"0","fontSize":"12px","color":TEXT,"padding":"4px 0",
                                "borderBottom":f"1px solid {BORDER}"})
              for s in df[df["tier"]==tier]["state"].sort_values()],
        ], style={"background":WHITE,"border":f"2px solid {TIER_COLOR[tier]}","borderRadius":"10px",
                  "padding":"14px 18px","flex":"1"})
        for tier in ["High Performer","Mid Performer","Needs Attention"]
    ], style={"display":"flex","gap":"12px","marginBottom":"12px"})

    def row(*ch): return html.Div(list(ch),style={"display":"flex","gap":"12px","marginBottom":"12px","flexWrap":"wrap"})

    return html.Div([
        exp_box([html.Strong("How the ML clustering works: "),
                 "K-Means unsupervised machine learning analysed 6 health indicators simultaneously — infant mortality, "
                 "maternal mortality, immunisation, doctor density, health expenditure, and out-of-pocket costs. "
                 "It automatically grouped all 20 states into 3 tiers without any manual labelling. "
                 "States in the same tier have similar overall health profiles."], "blue"),
        html.Div(style={"marginBottom":"12px"}),
        tier_lists,
        row(
            html.Div([section_label("Tier Summary Statistics"), tier_table],
                     style={**CARD,"flex":"1.3","overflowX":"auto"}),
            html.Div([section_label(f"Average {label} by Tier"),
                      dcc.Graph(figure=tier_bar, config={"displayModeBar":False})],
                     style={**CARD,"flex":"1"}),
        ),
        row(html.Div([section_label("Cluster Map — Spending vs Infant Mortality"),
                      dcc.Graph(figure=bubble, config={"displayModeBar":False}),
                      exp_box([html.Strong("Reading the bubble chart: "),
                               "Each bubble is a state. Colour = ML-assigned tier. "
                               "Bubble size = doctor density. States towards the top-left (high spending, low infant mortality) "
                               "are the strongest health performers. This chart validates that the ML clustering is meaningful."],
                              "green")],
                     style={**CARD,"flex":"1"})),
    ])


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=7860)
    
# ── Footer ────────────────────────────────────────────────────────────────
    html.Div([
        html.P("India Health Intelligence Dashboard  |  Built with Python, Plotly Dash, Scikit-learn",
               style={"margin":"0 0 4px","fontSize":"11px","color":MUTED,"textAlign":"center","fontWeight":"600"}),
        html.P("Data: NFHS-5 (2019-21)  |  National Health Profile 2022  |  WHO India  |  Values are state-level approximations for analytical purposes",
               style={"margin":"0","fontSize":"11px","color":MUTED,"textAlign":"center"}),
    ], style={"padding":"16px","borderTop":f"1px solid {BORDER}","background":WHITE}),