import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from scipy.cluster.hierarchy import linkage, leaves_list
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, ctx
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import warnings
from datetime import datetime, date
import io 
import base64



warnings.filterwarnings("ignore", message="Workbook contains no default style")

# ---- Configuration ----


excluded_target_companies = [
    "MOTION INDUSTRIES",
    "FORWARD TECHNOLOGY",
    "VALLEN DISTRIBUTION",
    "VAN METER",
    "US TOOL",
    "FARNSWORTH ELECTRONICS",
    "GM",
    "3M"
]

def should_display_company(company, territory):
    company_upper = company.upper().strip()
    if company_upper in excluded_target_companies:
        return False
    if company_upper == "BAUSCH AND LOMB":
        if territory == "EICHHORN":
            return False
    if company_upper == "FOX VALLEY METROLOGY":
        if territory == "EICHHORN":
            return False
    return True


# Instrumentation Resources colors
IR_RED    = "#D32F2F"
IR_ORANGE = "#F57C00"
IR_YELLOW = "#FBC02D"
IR_GREEN  = "#388E3C"
IR_WHITE  = "#FFFFFF"

# Use a neutral Bootstrap theme since we'll override colors inline
BOOTSTRAP = dbc.themes.FLATLY

# ---- Load & Clean Data ----
GLOBAL_DF = None

def clean_uploaded_df(df):
    # (Paste your cleaning logic here)
    df['ShipTo_Company'] = df['ShipTo_Company'].astype(str).str.upper().str.strip()
    df['Territory']      = df['Name_Territory_aet'].astype(str).str.upper().str.strip()
    df['Principal']      = df['Name_Principal_aet'].astype(str).str.upper().str.strip()
    df['Net_Sales']      = pd.to_numeric(df['Amount_Net_cn'], errors='coerce').fillna(0)
    date_col = next(c for c in df.columns if 'date' in c.lower())
    df['Order_Date'] = pd.to_datetime(df[date_col], errors='coerce')
    df['Quarter']    = df['Order_Date'].dt.to_period('Q')
    df = df.dropna(subset=['Order_Date']).copy()
    if not df.empty:
        most_recent_quarter = df['Quarter'].max()
        df = df[df['Quarter'] != most_recent_quarter].copy()

    territories = ['ALL'] + sorted(df['Territory'].dropna().unique())
    companies   = ['ALL'] + sorted(df['ShipTo_Company'].dropna().unique())
    principles  = ['ALL'] + sorted(df['Principal'].dropna().unique())

    # Determine min and max dates for the DatePickerRange
    min_date_allowed = df['Order_Date'].min().date() if not df.empty else date(2000, 1, 1)
    max_date_allowed = df['Order_Date'].max().date() if not df.empty else date.today()

    return df

except FileNotFoundError:
    print(f"Error: Excel file not found at {EXCEL_PATH}. Please ensure the path is correct.")
    # Create empty DataFrames to allow the app to run without data
    df = pd.DataFrame(columns=['ShipTo_Company', 'Territory', 'Principal', 'Net_Sales', 'Order_Date', 'Quarter'])
    territories = ['ALL']
    companies = ['ALL']
    principles = ['ALL']
    min_date_allowed = date(2000, 1, 1)
    max_date_allowed = date.today()
except Exception as e:
    print(f"An error occurred during data loading or initial processing: {e}")
    df = pd.DataFrame(columns=['ShipTo_Company', 'Territory', 'Principal', 'Net_Sales', 'Order_Date', 'Quarter'])
    territories = ['ALL']
    companies = ['ALL']
    principles = ['ALL']
    min_date_allowed = date(2000, 1, 1)
    max_date_allowed = date.today()


# ---- Define Limited List of Principals (exact match from Excel) ----
allowed_principals = [
    "CTS",
    "DEWESOFT",
    "FLOW TECHNOLOGY",
    "FUTEK",
    "INTEST",
    "MENSOR",
    "MICRO-EPSILON",
    "PCB PIEZO",
    "PROMESS",
    "REXGEAR",
    "SCIEMETRIC",
    "WIKA"
]

# ---- Filtered DataFrame for Allowed Principals (initial, will be filtered by callbacks) ----
df_pr = df[df['Principal'].isin(allowed_principals)].copy()


allowed_territories = [
    "BAILEY", "COOPER", "EICHHORN", "FAUSER", "GREEN",
    "GRIMES", "HILL", "HOLLAND", "KORUS", "MARQUETTE"
]
df_terr = df[df["Territory"].isin(allowed_territories)]
df_terr = df_terr[(df_terr["ShipTo_Company"] != "THORSON")]


# ---- Compute Principle Ranking Metrics ----
# These are computed once globally, not intended to be reactive to filters
# Clean missing key fields
df_cleaned = df.dropna(subset=["Principal", "Territory", "ShipTo_Company"])
df_cleaned = df_cleaned[df_cleaned["Principal"].isin(allowed_principals)].copy()


# 1) Total net sales per principal
sales_by_pr = df_cleaned.groupby('Principal')['Net_Sales'].sum() if not df_cleaned.empty else pd.Series(dtype=float)
companies_by_pr = df_cleaned.groupby('Principal')['ShipTo_Company'].nunique() if not df_cleaned.empty else pd.Series(dtype=float)
territories_by_pr = df_cleaned.groupby('Principal')['Territory'].nunique() if not df_cleaned.empty else pd.Series(dtype=float)

# 4) Growth momentum per principal (average derivative over quarters)
momentum_by_pr = {}
if not df_cleaned.empty:
    for pr in df_cleaned['Principal'].unique():
        dfp = df_cleaned[df_cleaned['Principal'] == pr].copy()
        if dfp.empty:
            momentum_by_pr[pr] = 0.0
            continue
        dfp['Period'] = dfp['Order_Date'].dt.to_period('Q')
        sales_q = dfp.groupby('Period')['Net_Sales'].sum().sort_index()
        sales_vals = sales_q.values
        if len(sales_vals) < 2:
            momentum_by_pr[pr] = 0.0
        else:
            x_idx = np.arange(len(sales_vals))
            deg   = min(4, len(x_idx) - 1)
            coeffs = np.polyfit(x_idx, sales_vals, deg)
            deriv_vals = np.polyval(np.polyder(coeffs), x_idx)
            momentum_by_pr[pr] = float(np.mean(deriv_vals))

pr_df = pd.DataFrame({
    'principal': list(sales_by_pr.index),
    'total_sales': sales_by_pr.values,
    'num_companies': companies_by_pr.reindex(sales_by_pr.index).values,
    'num_territories': territories_by_pr.reindex(sales_by_pr.index).values,
    'momentum': [momentum_by_pr.get(p, 0.0) for p in sales_by_pr.index]
}) if not sales_by_pr.empty else pd.DataFrame(columns=['principal', 'total_sales', 'num_companies', 'num_territories', 'momentum'])

pr_df = pr_df[pr_df['principal'].isin(allowed_principals)].copy()

if not pr_df.empty:
    for col in ['total_sales', 'num_companies', 'num_territories', 'momentum']:
        mn, mx = pr_df[col].min(), pr_df[col].max()
        pr_df[f'{col}_norm'] = (pr_df[col] - mn) / (mx - mn) if mx > mn else 0.0

    pr_df['composite_score'] = pr_df[['total_sales_norm','num_companies_norm','num_territories_norm','momentum_norm']].mean(axis=1)
    pr_df = pr_df.sort_values(by='composite_score', ascending=False).reset_index(drop=True)
    pr_df['rank'] = pr_df.index + 1
else:
    pr_df['composite_score'] = 0.0
    pr_df['rank'] = 0


# ---- Compute Territory Ranking Metrics ----
# These are computed once globally, not intended to be reactive to filters
# Exclude NaN territories and exclude company 'THORSON' only for territory rankings
df_terr_cleaned = df[df['Territory'].notna() & (df['ShipTo_Company'] != 'THORSON')]

# 1) Total net sales per territory
sales_by_terr = df_terr_cleaned.groupby('Territory')['Net_Sales'].sum() if not df_terr_cleaned.empty else pd.Series(dtype=float)

# 2) Unique companies per territory
companies_by_terr = df_terr_cleaned.groupby('Territory')['ShipTo_Company'].nunique() if not df_terr_cleaned.empty else pd.Series(dtype=float)

# 3) Growth momentum per territory (average derivative over quarters)
momentum_by_terr = {}
if not df_terr_cleaned.empty:
    for terr in df_terr_cleaned['Territory'].unique():
        dft = df_terr_cleaned[df_terr_cleaned['Territory'] == terr].copy()
        if dft.empty:
            momentum_by_terr[terr] = 0.0
            continue
        dft['Period'] = dft['Order_Date'].dt.to_period('Q')
        sales_q = dft.groupby('Period')['Net_Sales'].sum().sort_index()
        sales_vals = sales_q.values
        if len(sales_vals) < 2:
            momentum_by_terr[terr] = 0.0
        else:
            x_idx = np.arange(len(sales_vals))
            deg = min(4, len(x_idx) - 1)
            coeffs = np.polyfit(x_idx, sales_vals, deg)
            deriv_vals = np.polyval(np.polyder(coeffs), x_idx)
            momentum_by_terr[terr] = float(np.mean(deriv_vals))

# Assemble territory DataFrame without NaN territories
valid_territories_in_df_terr_cleaned = [t for t in sales_by_terr.index if pd.notna(t)] if not sales_by_terr.empty else []
terr_df = pd.DataFrame({
    'territory': valid_territories_in_df_terr_cleaned,
    'total_sales': [sales_by_terr[t] for t in valid_territories_in_df_terr_cleaned],
    'num_companies': [companies_by_terr.get(t, 0) for t in valid_territories_in_df_terr_cleaned],
    'momentum': [momentum_by_terr.get(t, 0.0) for t in valid_territories_in_df_terr_cleaned]
}) if not sales_by_terr.empty else pd.DataFrame(columns=['territory', 'total_sales', 'num_companies', 'momentum'])

# Normalize each metric to 0–1 scale
if not terr_df.empty:
    for col in ['total_sales', 'num_companies', 'momentum']:
        mn, mx = terr_df[col].min(), terr_df[col].max()
        terr_df[f'{col}_norm'] = (terr_df[col] - mn) / (mx - mn) if mx > mn else 0.0

    # Compute composite score as average of three normalized metrics
    terr_df['composite_score'] = terr_df[['total_sales_norm','num_companies_norm','momentum_norm']].mean(axis=1)
    terr_df = terr_df.sort_values(by='composite_score', ascending=False).reset_index(drop=True)
    terr_df['rank'] = terr_df.index + 1
else:
    terr_df['composite_score'] = 0.0
    terr_df['rank'] = 0


# ---- Prepare HTML tables ----
def make_territory_ranking_table(df_rank):
    if df_rank.empty:
        return html.P("No territory data available for ranking.", style={'color': IR_ORANGE})
    header = [
        html.Tr([
            html.Th("Rank"),
            html.Th("Territory"),
            html.Th("Total Sales"),
            html.Th("Unique Companies"),
            html.Th("Growth Momentum"),
            html.Th("Composite Score")
        ], style={'backgroundColor': IR_ORANGE, 'color': IR_WHITE})
    ]
    rows = []
    for _, row in df_rank.iterrows():
        rows.append(html.Tr([
            html.Td(int(row['rank'])),
            html.Td(row['territory']), # Corrected to show territory name
            html.Td(f"${row['total_sales']:,.0f}"),
            html.Td(int(row['num_companies'])),
            html.Td(f"{row['momentum']:.2f}"),
            html.Td(f"{row['composite_score']:.2f}")
        ]))
    return html.Table(header + rows, style={'width': '100%', 'borderCollapse': 'collapse'})


def make_principal_ranking_table(df_rank):
    if df_rank.empty:
        return html.P("No principal data available for ranking.", style={'color': IR_ORANGE})
    header = [
        html.Tr([
            html.Th("Rank"),
            html.Th("Principal"),
            html.Th("Total Sales"),
            html.Th("Unique Companies"),
            html.Th("Unique Territories"),
            html.Th("Growth Momentum"),
            html.Th("Composite Score")
        ], style={'backgroundColor': IR_ORANGE, 'color': IR_WHITE})
    ]
    rows = []
    for _, row in df_rank.iterrows():
        rows.append(html.Tr([
            html.Td(int(row['rank'])),
            html.Td(row['principal']),
            html.Td(f"${row['total_sales']:,.0f}"),
            html.Td(int(row['num_companies'])),
            html.Td(int(row['num_territories'])),
            html.Td(f"{row['momentum']:.2f}"),
            html.Td(f"{row['composite_score']:.2f}")
        ]))
    return html.Table(header + rows, style={'width': '100%', 'borderCollapse': 'collapse'})

terr_df = terr_df[terr_df['territory'].notna()].copy()
territory_ranking_table = make_territory_ranking_table(terr_df)
principle_ranking_table = make_principal_ranking_table(pr_df)

# ---- Dash App Setup ----
app = dash.Dash(__name__, external_stylesheets=[BOOTSTRAP], title="Instrumentation Resources Sales Dashboard")

# Professional Style Definitions
BODY_STYLE = {
    'backgroundColor': '#F8F9FA', # Light grey background
    'fontFamily': '"Segoe UI", Helvetica, Arial, sans-serif'
}

NAVBAR_STYLE = {
    'background': 'linear-gradient(90deg, rgba(211,47,47,1) 0%, rgba(180,40,40,1) 100%)', # Gradient IR_RED
    'padding': '10px 25px',
    'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
}

CARD_STYLE = {
    'borderRadius': '15px',
    'boxShadow': '0 8px 16px rgba(0,0,0,0.1)',
    'border': 'none',
    'transition': 'all 0.3s ease'
}

CARD_HEADER_STYLE_ = {
    'backgroundColor': IR_RED,
    'color': IR_WHITE,
    'fontSize': '20px',
    'fontWeight': 'bold',
    'borderTopLeftRadius': '15px',
    'borderTopRightRadius': '15px',
    'padding': '15px 20px'
}

CARD_HEADER_STYLE_ACCENT = {
    'backgroundColor': IR_ORANGE,
    'color': IR_WHITE,
    'fontSize': '20px',
    'fontWeight': 'bold',
    'borderTopLeftRadius': '15px',
    'borderTopRightRadius': '15px',
    'padding': '15px 20px'
}


HEADER_WRAPPER_STYLE = {
    'position': 'sticky',
    'top': '0',
    'zIndex': 1100,
    'background': 'linear-gradient(90deg, rgba(211,47,47,1) 0%, rgba(180,40,40,1) 100%)',
    'boxShadow': '0 4px 12px rgba(0,0,0,0.07)'
}

app.layout = dbc.Container(
    fluid=True,
    style=BODY_STYLE,
    children=[
        # Stores (hidden, non-visual)
        dcc.Store(id='flow-data-principal'),
        dcc.Store(id='bar-data-principal'),
        dcc.Store(id="selected-principal-node"),

        # Sticky Header: Navbar + Global Filters
        html.Div([
            dcc.Upload(
    id='upload-data',
    children=html.Div(['Drag and Drop or ', html.A('Select an Excel File')]),
    style={
        'width': '100%',
        'height': '60px',
        'lineHeight': '60px',
        'borderWidth': '2px',
        'borderStyle': 'dashed',
        'borderRadius': '5px',
        'textAlign': 'center',
        'margin': '10px 0',
        'backgroundColor': '#FAFAFA'
    },
    multiple=False
),
html.Div(id='output-data-upload', className="text-center text-muted mt-2"),

            # Navbar
            dbc.Navbar(
                dbc.Container([
                    html.A(
                        dbc.Row([
                            dbc.Col(html.Img(src="/assets/LOGO.jpg", height="80px", style={'borderRadius': '8px'})),
                            dbc.Col(html.H2("Sales Dashboard", className="ms-3 my-auto text-white"))
                        ],
                        align="center",
                        className="g-0",
                    ),
                    href="#",
                    style={"textDecoration": "none"},
                    ),
                ]),
                style=NAVBAR_STYLE,
                className="mb-0",   # Remove margin to bring card flush to nav
                dark=True,
                # Remove sticky="top"; wrapper handles it
            ),

            # Global Filters Card
            dbc.Container([
                dbc.Card(
                    dbc.CardBody([
                        dbc.Row(
                            [
                                dbc.Col([
                                    html.Label("Territory Filter", className="fw-bold mb-2"),
                                    dcc.Dropdown(
                                        id="global-territory-selector",
                                        options=[{"label": t.title(), "value": t} for t in territories],
                                        value="ALL",
                                        clearable=False,
                                    )
                                ], lg=3, md=6),
                                dbc.Col([
                                    html.Label("Date Range Filter", className="fw-bold mb-2"),
                                    dcc.DatePickerRange(
                                        id='global-date-range-selector',
                                        min_date_allowed=min_date_allowed,
                                        max_date_allowed=max_date_allowed,
                                        start_date=min_date_allowed,
                                        end_date=max_date_allowed,
                                        display_format='YYYY-MM-DD',
                                        className="w-100"
                                    )
                                ], lg=7, md=12),
                                dbc.Col(
                                    dbc.Button(
                                        "Reset Date",
                                        id="reset-date-button",
                                        color="light",
                                        outline=True,
                                        className="w-100 mt-auto"
                                    ),
                                    lg=2,
                                    md=6,
                                    className="d-flex"
                                )
                            ],
                            align="center",
                            className="g-1"
                        ),
                    ]),
                    className="mb-0",  # No bottom margin, keep content flush
                    style=CARD_STYLE  # Remove sticky/top here
                ),
            ],
            fluid=True,
            className="py-2",  # Some space below nav
            ),
        ],
        style=HEADER_WRAPPER_STYLE,
        className="sticky-top"
        ),

        dbc.Container([
            # Sales Trend & Momentum
            dbc.Card(
                [
                    dbc.CardHeader("Sales Trend & Growth Momentum", style=CARD_HEADER_STYLE_),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Company", className="fw-bold"),
                                dcc.Dropdown(id="company-selector", options=[{"label": c, "value": c} for c in companies], value="ALL", clearable=False),
                            ], md=6),
                            dbc.Col([
                                html.Label("Principal", className="fw-bold"),
                                dcc.Dropdown(id="principal-selector", options=[{"label": p.title(), "value": p} for p in principles], value="ALL", clearable=False),
                            ], md=6),
                        ], className="mb-4"),
                        dcc.Graph(id="company-trend-chart", style={"height": "450px"}),
                        html.Hr(className="my-4"),
                        html.H4(id="derivative-score-output", className="text-center mt-2 fw-bold", style={'color': IR_RED}),
                        dcc.Graph(id="derivative-sparkline", config={"displayModeBar": False}, style={"height": "250px"}),
                        html.Div(id="territory-momentum-ranking", className="mt-4")
                    ], className="p-4")
                ],
                style=CARD_STYLE, className="mb-4"
            ),

            # Principal Cross-Pollination Flow
            dbc.Card([
                dbc.CardHeader("Principal Cross-Pollination Flow", style=CARD_HEADER_STYLE_ACCENT),
                dbc.CardBody(dcc.Graph(id="principal-flow-chart", style={"height": "600px"}))
            ], style=CARD_STYLE, className="mb-4"),

            # Cross-Pollination Score
            dbc.Card([
                dbc.CardHeader("Cross-Pollination Score by Principal", style=CARD_HEADER_STYLE_ACCENT),
                dbc.CardBody(dcc.Graph(id="xpoll-bar-chart", style={'height': '600px'}))
            ], style=CARD_STYLE, className="mb-4"),

            # Company Connectivity & Comparison
            dbc.Card(
                [
                    dbc.CardHeader("Company Connectivity & Comparison (Top 50)", style=CARD_HEADER_STYLE_),
                    dbc.CardBody([
                        dcc.Graph(id="company-heatmap", style={"height": "600px"}),
                        html.Hr(className="my-4"),
                        html.H5("Comparison Details", className="fw-bold"),
                        html.P("Click a cell on the heatmap or select two companies below to compare.", className="text-muted"),
                        dbc.Row([
                            dbc.Col(dcc.Dropdown(id="company-compare-a", options=[{"label": c, "value": c} for c in sorted(df['ShipTo_Company'].unique())], placeholder="Select Company A"), md=6),
                            dbc.Col(dcc.Dropdown(id="company-compare-b", options=[{"label": c, "value": c} for c in sorted(df['ShipTo_Company'].unique())], placeholder="Select Company B"), md=6),
                        ], className="mb-3"),
                        html.Div(id="company-compare-output")
                    ], className="p-4")
                ], style=CARD_STYLE, className="mb-4"
            ),

            # Company-Principal Heatmap
            dbc.Card(
                [
                    dbc.CardHeader("Principal vs. Company Heatmap (Top 50)", style=CARD_HEADER_STYLE_ACCENT),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([html.Label("Isolate Company A", className="fw-bold"), dcc.Dropdown(id="heatmap-cp-company-a", options=[{"label": "ALL", "value": "ALL"}] + [{"label": c, "value": c} for c in sorted(df['ShipTo_Company'].unique())], value="ALL")], md=6),
                            dbc.Col([html.Label("Isolate Company B", className="fw-bold"), dcc.Dropdown(id="heatmap-cp-company-b", options=[{"label": "ALL", "value": "ALL"}] + [{"label": c, "value": c} for c in sorted(df['ShipTo_Company'].unique())], value="ALL")], md=6),
                        ], className="mb-3"),
                        dcc.Graph(id="company-principal-heatmap", style={"height": "600px"})
                    ], className="p-4")
                ], style=CARD_STYLE, className="mb-4"
            ),

            # Target Company Identification
            dbc.Card(
                [
                    dbc.CardHeader("Target Company Identification", style=CARD_HEADER_STYLE_),
                    dbc.CardBody([
                        html.Label("Select a Principal to Find Targets", className="fw-bold"),
                        dcc.Dropdown(
                            id="target-principal-selector",
                            options=[{"label": p, "value": p} for p in sorted(allowed_principals)],
                            placeholder="Select a Principal...",
                            className="mb-4"
                        ),
                        html.H5(id="target-companies-heading", className="text-center fw-bold", style={'color': IR_RED}),
                        html.Div(id="target-companies-output"),
                        dbc.Button("Export Top 10 Targets by Territory", id="export-targets-btn", color= IR_RED, className="mt-3"),
                        dcc.Download(id="download-targets-csv"),
                    ], className="p-4")
                ], style=CARD_STYLE, className="mb-4"
            ),

            # Territory Ranking
            dbc.Card([
                dbc.CardHeader("Territory Ranking", style=CARD_HEADER_STYLE_ACCENT),
                dbc.CardBody([
                    territory_ranking_table,
                    html.P("Score based on normalized sales, unique companies, and growth momentum.", className="text-muted fst-italic mt-3", style={'fontSize': '0.9em'})
                ], className="p-4")
            ], style=CARD_STYLE, className="mb-4"),

            # Principal Ranking
            dbc.Card([
                dbc.CardHeader("Principal Ranking", style=CARD_HEADER_STYLE_ACCENT),
                dbc.CardBody([
                    principle_ranking_table,
                    html.P("Score based on normalized sales, companies, territories, and growth momentum.", className="text-muted fst-italic mt-3", style={'fontSize': '0.9em'})
                ], className="p-4")
            ], style=CARD_STYLE, className="mb-4"),

            # Footer
            html.Footer(
                "© 2025 Instrumentation Resources",
                className="text-center text-muted mt-5 mb-3"
            ),
        ])
    ]
)

# ---------- Callbacks ----------

@app.callback(
    [
        Output("global-date-range-selector", "start_date"),
        Output("global-date-range-selector", "end_date"),
    ],
    [Input("reset-date-button", "n_clicks")],
    [
        State("global-date-range-selector", "min_date_allowed"),
        State("global-date-range-selector", "max_date_allowed"),
    ],
    prevent_initial_call=True
)
def reset_date_range(n_clicks, min_allowed, max_allowed):
    if n_clicks:
        return min_allowed, max_allowed
    return dash.no_update, dash.no_update


@app.callback(
    [
        Output("company-trend-chart",        "figure"),
        Output("derivative-score-output",    "children"),
        Output("derivative-sparkline",       "figure"),
        Output("territory-momentum-ranking", "children")
    ],
    [
        Input("global-territory-selector", "value"),
        Input('global-date-range-selector', 'start_date'),
        Input('global-date-range-selector', 'end_date'),
        Input("company-selector",   "value"),
        Input("principal-selector", "value")
    ]
)
def update_trend_and_momentum(selected_territory, start_date, end_date, selected_company, selected_principal):
    dff = df.copy()

    # Apply global territory filter
    if selected_territory != "ALL":
        dff = dff[dff["Territory"] == selected_territory]

    # Apply global date filter
    if start_date and end_date:
        start_date_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_date_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
        dff = dff[(dff['Order_Date'].dt.date >= start_date_dt) & (dff['Order_Date'].dt.date <= end_date_dt)]

    # Apply specific company/principal filters
    if selected_company != "ALL":
        dff = dff[dff["ShipTo_Company"] == selected_company]
    if selected_principal != "ALL":
        dff = dff[dff["Principal"] == selected_principal]

    trend_fig = go.Figure()
    deriv_fig = go.Figure()
    score_text = ""
    ranking_div = html.Div()

    if dff.empty:
        no_data_fig = go.Figure().add_annotation(
            text="No data for this filter",
            x=0.5, y=0.5,
            showarrow=False, font_size=18,
            xref="paper", yref="paper"
        )
        score_text = "No data for this filter"
        return no_data_fig, score_text, no_data_fig, html.Div()

    # Determine time granularity based on date range with NEW thresholds
    date_min = dff["Order_Date"].min()
    date_max = dff["Order_Date"].max()
    num_days = (date_max - date_min).days

    if num_days <= 30: # Up to ~1 month
        period_type = "D"
        x_label_trend = "Day"
    elif num_days <= 180: # Up to ~6 months
        period_type = "W"
        x_label_trend = "Week (start)"
    elif num_days <= 1095: # Up to ~3 year
        period_type = "M"
        x_label_trend = "Month (start)"
    else: # More than 3 years
        period_type = "Q"
        x_label_trend = "Quarter (start)"

    # ── Sales Trend ──
    dff_trend = dff.copy()
    dff_trend["Period"] = dff_trend["Order_Date"].dt.to_period(period_type)
    agg_sales = dff_trend.groupby("Period")["Net_Sales"].sum().sort_index()
    x_ts_trend = agg_sales.index.to_timestamp()
    y_sales = agg_sales.values
    x_idx_tr = np.arange(len(x_ts_trend))

    if len(x_ts_trend) > 1:
        deg_tr = min(4, len(x_idx_tr) - 1)
        fit_vals_tr = np.polyval(np.polyfit(x_idx_tr, y_sales, deg_tr), x_idx_tr)
        trend_fig = go.Figure([
            go.Scatter(
                x=x_ts_trend, y=y_sales,
                mode="markers+lines", name="Actual Sales",
                line=dict(width=2, dash="dash", color=IR_RED)
            ),
            go.Scatter(
                x=x_ts_trend, y=fit_vals_tr,
                mode="lines", name="Trend Line", line=dict(width=2, color=IR_ORANGE)
            )
        ])
    else:
        trend_fig = go.Figure(
            go.Scatter(
                x=x_ts_trend, y=y_sales,
                mode="markers+lines", name="Sales", line=dict(color=IR_RED)
            )
        )
    trend_fig.update_layout(
        title=dict(text="Sales Trend", font=dict(size=18, color=IR_RED)),
        xaxis_title=x_label_trend,
        yaxis_title="Net Sales ($)",
        template="plotly_white",
        margin=dict(l=50, r=50, t=50, b=50),
        xaxis=dict(range=[x_ts_trend.min(), x_ts_trend.max()], tickangle=-45),
        plot_bgcolor=IR_WHITE,
        paper_bgcolor=IR_WHITE
    )

    # ── Growth Momentum ──
    dff_deriv = dff.copy()
    dff_deriv["Period"] = dff_deriv["Order_Date"].dt.to_period(period_type)
    sales_by_p = dff_deriv.groupby("Period")["Net_Sales"].sum().sort_index()
    x_ts_p, y_vals_p = sales_by_p.index.to_timestamp(), sales_by_p.values
    x_idx_p = np.arange(len(x_ts_p))

    if len(x_idx_p) < 2:
        score_text = "Insufficient data"
        deriv_fig = go.Figure().add_annotation(
            text="Insufficient data",
            x=0.5, y=0.5,
            showarrow=False, font_size=18,
            xref="paper", yref="paper"
        )
    else:
        deg_p = 4 if len(x_idx_p) >= 5 else (len(x_idx_p) - 1)
        coeffs = np.polyfit(x_idx_p, y_vals_p, deg_p)
        deriv_coeffs = np.polyder(coeffs)
        deriv_vals = np.polyval(deriv_coeffs, x_idx_p)

        weights = np.arange(1, len(deriv_vals) + 1)
        weighted_avg_slope = (deriv_vals * weights).sum() / weights.sum()
        score_text = f"Growth Momentum Score: {weighted_avg_slope:.1f}"


        deriv_fig = go.Figure([
            go.Scatter(
                x=x_ts_p, y=deriv_vals,
                mode="markers+lines",
                name="Slope",
                line=dict(width=2, color=IR_RED),
                marker=dict(size=6, color=IR_ORANGE),
                hovertemplate="%{x|%Y-%m-%d}<br>Rate of Change: %{y:.2f}<extra></extra>"
            )
        ])
        deriv_fig.update_layout(
            title=dict(text="Polynomial Derivative (Slope)", font=dict(size=18, color=IR_RED)),
            xaxis_title=x_label_trend,
            yaxis_title="Rate of Change",
            template="plotly_white",
            margin=dict(l=50, r=50, t=50, b=50),
            xaxis=dict(range=[x_ts_p.min(), x_ts_p.max()], tickangle=-45),
            plot_bgcolor=IR_WHITE,
            paper_bgcolor=IR_WHITE
        )

    # ── Territory Momentum Ranking (UNFILTERED) ──
    if selected_territory != "ALL": # Only show if a specific territory is selected
        df_terr_momentum = dff.copy() # Use the already filtered dff
        company_scores = []
        for comp in sorted(df_terr_momentum["ShipTo_Company"].unique()):
            dff_comp = df_terr_momentum[df_terr_momentum["ShipTo_Company"] == comp].copy()

            dff_comp["Period"] = dff_comp["Order_Date"].dt.to_period(period_type)

            sbc = dff_comp.groupby("Period")["Net_Sales"].sum().sort_index()
            x_i, y_i = np.arange(len(sbc)), sbc.values
            if len(x_i) < 2:
                continue
            deg_c = 4 if len(x_i) >= 5 else (len(x_i) - 1)
            coeffs_c = np.polyfit(x_i, y_i, deg_c)
            deriv_vals_c = np.polyval(np.polyder(coeffs_c), x_i)
            company_scores.append((comp, deriv_vals_c.mean()))

        if not company_scores:
            ranking_div = html.Div("Not enough data in this territory for company momentum ranking.")
        else:
            company_scores.sort(key=lambda x: x[1], reverse=True)
            top5 = company_scores[:5]
            bottom5 = company_scores[-5:] if len(company_scores) >= 5 else company_scores[::-1]
            top_list = html.Ul([html.Li(f"{c}: {s:.2f}", style={'color': IR_RED}) for c, s in top5])
            bottom_list = html.Ul([html.Li(f"{c}: {s:.2f}", style={'color': IR_RED}) for c, s in bottom5[::-1]])
            ranking_div = dbc.Row(
                [
                    dbc.Col([html.H6(f"Top 5 Momentum in {selected_territory}", className="mb-2", style={'color': IR_ORANGE}), top_list], width=6),
                    dbc.Col([html.H6(f"Bottom 5 Momentum in {selected_territory}", className="mb-2", style={'color': IR_ORANGE}), bottom_list], width=6),
                ]
            )
    else:
        ranking_div = html.Div("Company momentum ranking is available when a specific territory is selected.")

    return trend_fig, score_text, deriv_fig, ranking_div


@app.callback(
    Output('principal-flow-chart', 'figure'),
    Output('selected-principal-node', 'data'),
    [
        Input('global-territory-selector', 'value'),
        Input('global-date-range-selector', 'start_date'),
        Input('global-date-range-selector', 'end_date'),
        Input('principal-flow-chart', 'clickData')
    ],
    [State('selected-principal-node', 'data')]
)
def update_principal_flow(selected_territory, start_date, end_date, clickData, stored_node):
    dff = df_pr.copy() # Start with the df already filtered for allowed_principals

    # Apply global territory filter
    if selected_territory != "ALL":
        dff = dff[dff['Territory'] == selected_territory].copy()

    # Apply global date filter
    if start_date and end_date:
        dff = dff[(dff['Order_Date'] >= start_date) & (dff['Order_Date'] <= end_date)].copy()

    if dff.empty:
        return go.Figure().add_annotation(
            text="No data for this filter",
            x=0.5, y=0.5,
            showarrow=False, font_size=18,
            xref="paper", yref="paper"
        ), None


    # Build presence_pr including all allowed_principals (zero if missing)
    # Ensure that the index for pivot table uses the existing companies in dff, not all companies in df
    unique_companies_in_dff = sorted(dff['ShipTo_Company'].unique())

    cust_prin_pr = (
        dff.groupby(['ShipTo_Company', 'Principal'])
           .size()
           .reindex(pd.MultiIndex.from_product([unique_companies_in_dff, allowed_principals]),
                    fill_value=0)
           .unstack(fill_value=0)
    )
    presence_pr  = (cust_prin_pr > 0).astype(int)
    cooccur_prin = presence_pr.T.dot(presence_pr)

    # Ensure every allowed principal appears (even if no cooccurrence)
    for p in allowed_principals:
        if p not in cooccur_prin.index:
            cooccur_prin.loc[p, :] = 0
            cooccur_prin.loc[:, p] = 0
    cooccur_prin = cooccur_prin.reindex(index=allowed_principals, columns=allowed_principals, fill_value=0)

    cust_count   = presence_pr.sum(axis=0).reindex(allowed_principals, fill_value=0)

    # Build graph with all allowed principals as nodes
    G = nx.Graph()
    G.add_nodes_from(allowed_principals)
    for i, p in enumerate(allowed_principals):
        for q in allowed_principals[i+1:]:
            w = int(cooccur_prin.loc[p, q])
            if w > 0:
                G.add_edge(p, q, weight=w)

    pos = nx.spring_layout(G, k=18, seed=42)
    max_w = max((d['weight'] for _, _, d in G.edges(data=True)), default=1)

    # --- NEW LOGIC FOR NODE SELECTION/DESELECTION ---
    clicked_node = None
    if clickData and 'points' in clickData and clickData['points']:
        point = clickData['points'][0]
        # Check if the click was on a node marker which has 'text'
        if 'text' in point and point['text'] in allowed_principals:
            clicked_node = point['text']

    # Decide the new state of the selected node
    new_stored_node = None
    if stored_node is None:
        # Nothing was selected, so select the newly clicked node
        new_stored_node = clicked_node
    else:
        # Something was selected
        if clicked_node == stored_node:
            # User clicked the same node, so deselect it
            new_stored_node = None
        else:
            # User clicked a different node or the background, so switch/deselect
            new_stored_node = clicked_node # This will be the new node, or None if background was clicked

    # Build subgraph if a principal is selected based on the new state
    if new_stored_node:
        H = nx.Graph()
        H.add_node(new_stored_node)
        for nbr in G.neighbors(new_stored_node):
            H.add_edge(new_stored_node, nbr, weight=G[new_stored_node][nbr]['weight'])
    else:
        H = G

    edge_traces, node_list, node_x, node_y, texts, sizes, node_colors = [], [], [], [], [], [], []
    palette = px.colors.qualitative.Plotly

    # Create edges with stoplight colors based on relative weight
    for u, v, d in H.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        w      = d['weight']
        norm   = w / max_w if max_w > 0 else 0
        if norm < 0.33:
            color = IR_RED
        elif norm < 0.66:
            color = IR_YELLOW
        else:
            color = IR_GREEN
        width  = 1 + (w / max_w) * 9 if max_w > 0 else 1
        edge_traces.append(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode='lines',
                line=dict(width=width, color=color),
                hoverinfo='text',
                text=f"{u} ↔ {v}: {w} shared customers"
            )
        )

    # Create node trace including all or subgraph nodes
    for idx, n in enumerate(H.nodes()):
        node_list.append(n)
        x0, y0 = pos[n]
        node_x.append(x0)
        node_y.append(y0)
        texts.append(f"{n} ({int(cust_count.get(n,0))} customers)")
        sizes.append(20 + H.degree(n) * 5)
        node_colors.append(palette[idx % len(palette)])

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=node_list,
        textposition='middle center',
        hoverinfo='text',
        hovertext=texts,
        marker=dict(size=sizes, color=node_colors, line=dict(width=1, color=IR_RED))
    )

    fig = go.Figure(edge_traces + [node_trace])
    fig.update_layout(
        title=dict(text=f'Limited Principal Cross-Pollination Flow – {selected_territory}', font=dict(size=20, color=IR_RED)),
        template='plotly_white',
        width=1200,
        height=600,
        margin=dict(l=40, r=40, t=60, b=40),
        clickmode='event',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor=IR_WHITE,
        paper_bgcolor=IR_WHITE,
        showlegend=False
    )
    return fig, new_stored_node


@app.callback(
    Output('xpoll-bar-chart', 'figure'),
    [
        Input('global-territory-selector', 'value'),
        Input('global-date-range-selector', 'start_date'),
        Input('global-date-range-selector', 'end_date')
    ]
)
def update_xpoll_bar(selected_territory, start_date, end_date):
    dff = df_pr.copy() # Start with the df already filtered for allowed_principals

    # Apply global territory filter
    if selected_territory != "ALL":
        dff = dff[dff['Territory'] == selected_territory].copy()

    # Apply global date filter
    if start_date and end_date:
        dff = dff[(dff['Order_Date'] >= start_date) & (dff['Order_Date'] <= end_date)].copy()

    if dff.empty:
        return go.Figure().add_annotation(
            text="No data for this filter",
            x=0.5, y=0.5,
            showarrow=False, font_size=18,
            xref="paper", yref="paper"
        )

    unique_companies_in_dff = sorted(dff['ShipTo_Company'].unique())

    cust_prin_pr = (
        dff.groupby(['ShipTo_Company', 'Principal'])
           .size()
           .reindex(pd.MultiIndex.from_product([unique_companies_in_dff, allowed_principals]),
                    fill_value=0)
           .unstack(fill_value=0)
    )
    presence_pr  = (cust_prin_pr > 0).astype(int)

    cooccur_prin = presence_pr.T.dot(presence_pr)
    cust_count   = presence_pr.sum(axis=0).reindex(allowed_principals, fill_value=0)
    cross_count  = cooccur_prin.sum(axis=1).reindex(allowed_principals, fill_value=0) - cust_count
    cross_score  = (cross_count / cust_count).fillna(0).reindex(allowed_principals, fill_value=0)

    sorted_principals = sorted(allowed_principals, key=lambda p: cross_score[p], reverse=True)
    sorted_scores     = [cross_score[p] for p in sorted_principals]
    avg_score = np.mean(sorted_scores) if sorted_scores else 0

    y_labels = ["Overall Average"] + sorted_principals[::-1]
    x_values = [avg_score] + [cross_score[p] for p in sorted_principals[::-1]]

    max_val = max(x_values) if x_values else 1
    bar_colors = []
    for val in x_values:
        norm = val / max_val if max_val > 0 else 0
        if norm < 0.33:
            bar_colors.append(IR_RED)
        elif norm < 0.66:
            bar_colors.append(IR_YELLOW)
        else:
            bar_colors.append(IR_GREEN)

    fig = go.Figure(
        go.Bar(
            x=x_values,
            y=y_labels,
            orientation='h',
            marker_color=bar_colors
        )
    )
    fig.update_layout(
        template='plotly_white',
        margin=dict(l=100, r=50, t=60, b=50),
        title=dict(text=f'Limited Cross-Pollination Score by Principal – {selected_territory}', font=dict(size=20, color=IR_RED), x=0.5),
        xaxis_title='Avg # Other Principals per Customer',
        yaxis_title='Principal',
        plot_bgcolor=IR_WHITE,
        paper_bgcolor=IR_WHITE
    )
    return fig


@app.callback(
    Output('company-heatmap', 'figure'),
    [
        Input('global-territory-selector', 'value'),
        Input('global-date-range-selector', 'start_date'),
        Input('global-date-range-selector', 'end_date')
    ]
)
def update_company_heatmap(selected_territory, start_date, end_date):
    dff = df.copy()

    # Apply global territory filter
    if selected_territory != "ALL":
        dff = dff[dff['Territory'] == selected_territory].copy()

    # Apply global date filter
    if start_date and end_date:
        dff = dff[(dff['Order_Date'] >= start_date) & (dff['Order_Date'] <= end_date)].copy()

    dff_filtered = dff[dff['Principal'].isin(allowed_principals)]

    if dff_filtered.empty:
        return go.Figure().add_annotation(
            text="No data for this filter",
            x=0.5, y=0.5,
            showarrow=False, font_size=18,
            xref="paper", yref="paper"
        )

    unique_companies_in_dff_filtered = sorted(dff_filtered['ShipTo_Company'].unique())

    cust_prin_full = (
        dff_filtered.groupby(['ShipTo_Company', 'Principal'])
                    .size()
                    .reindex(pd.MultiIndex.from_product([unique_companies_in_dff_filtered, allowed_principals]),
                             fill_value=0)
                    .unstack(fill_value=0)
    )
    presence_full  = (cust_prin_full > 0).astype(int)
    cooccur_comp   = presence_full.dot(presence_full.T)

    company_connection_count = cooccur_comp.sum(axis=1)
    top_comps = company_connection_count.nlargest(50).index.tolist()
    comp_matrix = cooccur_comp.reindex(index=top_comps, columns=top_comps, fill_value=0)

    # Hierarchical Clustering (Original Logic)
    if not comp_matrix.empty and comp_matrix.shape[0] > 1:
        # Perform clustering on both rows and columns
        Z = linkage(comp_matrix, method='ward')
        clustered_indices = leaves_list(Z)
        ordered_comps = [comp_matrix.index[i] for i in clustered_indices]
        clustered = comp_matrix.loc[ordered_comps, ordered_comps]
    else:
        clustered = comp_matrix # Handle case with 0 or 1 company

    fig = go.Figure(
        go.Heatmap(
            z=clustered.values,
            x=clustered.columns,
            y=clustered.index,
            colorscale="YlOrRd",
            hovertemplate="<b>%{y}</b> & <b>%{x}</b><br>Shared Principals: %{z}<extra></extra>"
        )
    )
    title_text = f"Company-to-Company Shared Limited Principals (Top 50) – {selected_territory}"
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=20, color=IR_RED), x=0.5),
        xaxis=dict(title="Company", tickangle=45, tickfont=dict(size=10)),
        yaxis=dict(title="Company", tickfont=dict(size=10)),
        template="plotly_white",
        width=1200,
        height=600,
        margin=dict(l=40, r=40, t=60, b=40),
        plot_bgcolor=IR_WHITE,
        paper_bgcolor=IR_WHITE,
        showlegend=False
    )
    return fig


@app.callback(
    Output('company-compare-a', 'value'),
    Output('company-compare-b', 'value'),
    Input('company-heatmap', 'clickData'),
    prevent_initial_call=True
)
def update_dropdowns_from_heatmap(clickData):
    if not clickData:
        raise PreventUpdate

    company_a = clickData['points'][0]['y']
    company_b = clickData['points'][0]['x']

    return company_a, company_b


@app.callback(
    Output('company-compare-output', 'children'),
    [
        Input('global-territory-selector', 'value'),
        Input('global-date-range-selector', 'start_date'),
        Input('global-date-range-selector', 'end_date'),
        Input('company-compare-a', 'value'),
        Input('company-compare-b', 'value')
    ]
)
def compare_companies(selected_territory, start_date, end_date, company_a, company_b):
    if not company_a or not company_b or company_a == company_b:
        return html.Div("Please select two distinct companies.", style={'color': IR_RED})

    dff = df_pr.copy()

    # Apply global territory filter
    if selected_territory != "ALL":
        dff = dff[dff['Territory'] == selected_territory].copy()

    # Apply global date filter
    if start_date and end_date:
        dff = dff[(dff['Order_Date'] >= start_date) & (dff['Order_Date'] <= end_date)].copy()

    if dff.empty:
        return html.Div("No data available for company comparison in the selected territory and date range.", style={'color': IR_ORANGE})

    principals_a = set(dff.loc[dff['ShipTo_Company'] == company_a, 'Principal'].unique())
    principals_b = set(dff.loc[dff['ShipTo_Company'] == company_b, 'Principal'].unique())

    shared = sorted(principals_a & principals_b)
    only_a = sorted(principals_a - principals_b)
    only_b = sorted(principals_b - principals_a)

    def make_list(items):
        if not items:
            return html.P("None", style={'color': IR_ORANGE})
        return html.Ul([html.Li(item) for item in items])

    return dbc.Container( # Use dbc.Container for better control of internal rows/cols
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H5(f"Shared Principals ({len(shared)})", style={'color': IR_RED}),
                            make_list(shared),
                        ],
                        md=4 # Takes 4 out of 12 columns
                    ),
                    dbc.Col(
                        [
                            html.H5(f"Principals only for {company_a} ({len(only_a)})", style={'color': IR_RED}),
                            make_list(only_a),
                        ],
                        md=4 # Takes 4 out of 12 columns
                    ),
                    dbc.Col(
                        [
                            html.H5(f"Principals only for {company_b} ({len(only_b)})", style={'color': IR_RED}),
                            make_list(only_b),
                        ],
                        md=4 # Takes 4 out of 12 columns
                    )
                ]
            )
        ]
    )


@app.callback(
    Output('company-principal-heatmap', 'figure'),
    [
        Input('global-territory-selector', 'value'),
        Input('global-date-range-selector', 'start_date'),
        Input('global-date-range-selector', 'end_date'),
        Input('heatmap-cp-company-a', 'value'),
        Input('heatmap-cp-company-b', 'value')
    ]
)
def update_company_principal_heatmap(selected_territory, start_date, end_date, company_a, company_b):
    dff = df_pr.copy() # Start with the df already filtered for allowed_principals

    # Apply global territory filter
    if selected_territory != "ALL":
        dff = dff[dff['Territory'] == selected_territory].copy()

    # Apply global date filter
    if start_date and end_date:
        dff = dff[(dff['Order_Date'] >= start_date) & (dff['Order_Date'] <= end_date)].copy()

    if dff.empty:
        return go.Figure().add_annotation(
            text="No data for this filter",
            x=0.5, y=0.5,
            showarrow=False, font_size=18,
            xref="paper", yref="paper"
        )

    pivot = (
        dff.pivot_table(index="Principal", columns="ShipTo_Company", values="Amount_Net_cn", aggfunc="sum", fill_value=0)
           .reindex(index=allowed_principals, fill_value=0)
    )

    pivot = pivot.loc[:, pivot.sum(axis=0) > 0]
    pivot = pivot.apply(pd.to_numeric, errors="coerce").fillna(0)

    if company_a != "ALL" and company_b != "ALL":
        selected_cols = [c for c in [company_a, company_b] if c in pivot.columns]
        if not selected_cols:
            top_cols = pivot.sum(axis=0).nlargest(min(50, pivot.shape[1])).index
            data_slice = pivot.loc[allowed_principals, top_cols]
        else:
            data_slice = pivot.loc[allowed_principals, selected_cols]
    else:
        top_cols = pivot.sum(axis=0).nlargest(min(50, pivot.shape[1])).index
        data_slice = pivot.loc[allowed_principals, top_cols]

    col_totals = data_slice.sum(axis=0).replace(0, np.nan)
    data_norm  = data_slice.div(col_totals, axis=1).fillna(0)

    # Hierarchical Clustering (Original Logic)
    if not data_norm.empty and data_norm.shape[1] > 1:
        Z_cols = linkage(data_norm.T, method='ward')
        ordered_cols = [data_norm.columns[i] for i in leaves_list(Z_cols)]
    else:
        ordered_cols = data_norm.columns.tolist()

    if not data_norm.empty and data_norm.shape[0] > 1:
        Z_rows = linkage(data_norm, method='ward')
        ordered_rows = [data_norm.index[i] for i in leaves_list(Z_rows)]
    else:
        ordered_rows = data_norm.index.tolist()

    clustered = data_norm.loc[ordered_rows, ordered_cols]

    fig = go.Figure(
        go.Heatmap(
            z=clustered.values,
            x=ordered_cols,
            y=ordered_rows,
            colorscale="YlOrRd",
            hovertemplate="<b>%{y}</b><br>%{x}<br>%{z:.1%}<extra></extra>"
        )
    )
    title_text = f"Limited Principal vs. Company Heatmap – {selected_territory}"
    if company_a != "ALL" and company_b != "ALL":
        title_text += f" (Isolated: {company_a}, {company_b})"

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=20, color=IR_RED), x=0.5),
        xaxis=dict(title="Company", tickangle=45, tickfont=dict(size=10)),
        yaxis=dict(title="Principal", tickfont=dict(size=10)),
        template="plotly_white",
        width=1200,
        height=600,
        margin=dict(l=40, r=40, t=60, b=40),
        plot_bgcolor=IR_WHITE,
        paper_bgcolor=IR_WHITE,
        showlegend=False
    )
    return fig


@app.callback(
    [Output('target-companies-output', 'children'),
     Output('target-companies-heading', 'children')],
    [Input('target-principal-selector', 'value'),
     Input('global-territory-selector', 'value'),
     Input('global-date-range-selector', 'start_date'),
     Input('global-date-range-selector', 'end_date')]
)
def update_target_companies(selected_principal, selected_territory, start_date, end_date):
    if not selected_principal:
        return html.P("Please select a principal to find target companies.", style={'color': IR_ORANGE}), ""

    try:
        # Filter df_pr based on the global territory and date selection
        dff_filtered_by_global_filters = df_pr.copy()

        if selected_territory != "ALL":
            dff_filtered_by_global_filters = dff_filtered_by_global_filters[dff_filtered_by_global_filters['Territory'] == selected_territory].copy()

        if start_date and end_date:
            dff_filtered_by_global_filters = dff_filtered_by_global_filters[(dff_filtered_by_global_filters['Order_Date'] >= start_date) & (dff_filtered_by_global_filters['Order_Date'] <= end_date)].copy()

        # Ensure df_pr is available and not empty after all filters
        if dff_filtered_by_global_filters.empty:
            return html.P("No data available for principal analysis given the selected filters.", style={'color': IR_ORANGE}), ""

        # Create a unique company-principal relationship DataFrame from the filtered data
        df_company_principal = dff_filtered_by_global_filters[['ShipTo_Company', 'Principal']].drop_duplicates()

        # Create a company-principal matrix
        company_principal_matrix = df_company_principal.pivot_table(
            index='ShipTo_Company',
            columns='Principal',
            aggfunc=lambda x: 1, # Mark 1 if company buys the principal
            fill_value=0 # 0 if company does not buy the principal
        )

        # Check if the selected principal is actually a column in the matrix
        if selected_principal not in company_principal_matrix.columns:
            return html.P(f"No sales data found for principal '{selected_principal}' given the current filters. Please select another principal, territory, or date range.", style={'color': IR_ORANGE}), f"Target Companies for {selected_principal}"

        # Identify companies that currently buy the selected principal
        buyers_of_principal = company_principal_matrix[company_principal_matrix[selected_principal] == 1].index.tolist()

        # Identify companies that do NOT buy the selected principal
        non_buyers_of_principal = company_principal_matrix[company_principal_matrix[selected_principal] == 0].index.tolist()

        if not buyers_of_principal:
            return html.P(f"No companies found currently buying '{selected_principal}' given the current filters. Cannot determine similar buying habits.", style={'color': IR_ORANGE}), f"Target Companies for {selected_principal}"

        if not non_buyers_of_principal:
            return html.P(f"All companies found given the current filters currently buy '{selected_principal}'. No new targets found.", style={'color': IR_ORANGE}), f"Target Companies for {selected_principal}"

        # Build buying habit profile based on companies that buy the selected principal
        # This counts how often other principals are bought by current buyers of the selected principal
        principal_co_occurrence_counts = {}
        for buyer_company in buyers_of_principal:
            # Get all principals bought by the current buyer, excluding the selected principal
            bought_principals_by_buyer = company_principal_matrix.loc[buyer_company]
            # Filter for principals actually bought (value == 1) and not the selected one
            other_principals = [
                p for p in bought_principals_by_buyer.index
                if bought_principals_by_buyer[p] == 1 and p != selected_principal
            ]
            for other_principal in other_principals:
                principal_co_occurrence_counts[other_principal] = principal_co_occurrence_counts.get(other_principal, 0) + 1

        if not principal_co_occurrence_counts:
            return html.P(f"Current buyers of '{selected_principal}' given the current filters do not consistently buy other principals. Cannot determine similar buying habits.", style={'color': IR_ORANGE}), f"Target Companies for {selected_principal}"

        # Score non-buying companies based on their overlap with the buying habit profile
        target_companies_scores = []
        for non_buyer_company in non_buyers_of_principal:
            principals_bought_by_non_buyer = company_principal_matrix.loc[non_buyer_company]
            non_buyer_other_principals = [
                p for p in principals_bought_by_non_buyer.index
                if principals_bought_by_non_buyer[p] == 1 and p != selected_principal # Exclude the selected principal from non-buyer's list too
            ]

            similarity_score = 0
            for principal_nb in non_buyer_other_principals:
                # Sum the co-occurrence counts for principals bought by the non-buyer
                similarity_score += principal_co_occurrence_counts.get(principal_nb, 0)

            if similarity_score > 0: # Only include companies with some similarity
                target_companies_scores.append({'Company': non_buyer_company, 'Score': similarity_score})

        if not target_companies_scores:
            return html.P(f"No companies with similar buying habits found that don't already buy '{selected_principal}' given the current filters.", style={'color': IR_ORANGE}), f"Target Companies for {selected_principal}"

        # Sort by score in descending order
        target_companies_df = pd.DataFrame(target_companies_scores).sort_values(by='Score', ascending=False).reset_index(drop=True)

        # Display top N target companies
        top_n = 20 # You can adjust this number
        display_df = target_companies_df.head(top_n)
        display_df = display_df[display_df["Company"].apply(lambda x: should_display_company(x, selected_territory))]

        if display_df.empty:
             return html.P(f"No companies with similar buying habits found that don't already buy '{selected_principal}' given the current filters.", style={'color': IR_ORANGE}), f"Target Companies for {selected_principal}"

        table_rows = []
        for index, row in display_df.iterrows():
            table_rows.append(html.Tr([
                html.Td(row['Company']),
                html.Td(f"{row['Score']:.0f}")
            ]))

        table_header = html.Thead(html.Tr([
            html.Th("Target Company", style={'backgroundColor': IR_RED, 'color': IR_WHITE}),
            html.Th("Similarity Score", style={'backgroundColor': IR_RED, 'color': IR_WHITE})
        ]))

        return [
            html.Div([
                html.P(f"This feature identifies companies that do NOT currently purchase '{selected_principal}' "
                       f"but show a high likelihood of doing so based on their existing buying patterns "
                       f"compared to companies that already buy '{selected_principal}'.",
                       style={'fontStyle': 'italic', 'color': IR_RED, 'marginBottom': '5px'}),
                html.P(f"**How the Similarity Score is Calculated:**",
                       style={'fontWeight': 'bold', 'color': IR_RED, 'marginBottom': '2px'}),
                html.Ul([
                    html.Li(html.Span("1. Identify Existing Buyers:", style={'fontWeight': 'bold'}),
                            f" The system first identifies all companies within the current territory and date range that currently purchase '{selected_principal}'."),
                    html.Li(html.Span("2. Build a 'Buying Habit Profile':", style={'fontWeight': 'bold'}),
                            f" For these existing buyers, the system tallies how frequently they also purchase other principals (e.g., if a buyer of '{selected_principal}' also buys 'DEWESOFT' and 'FUTEK', then 'DEWESOFT' and 'FUTEK' get a count increase). This creates a 'profile' of common co-purchases associated with '{selected_principal}'."),
                    html.Li(html.Span("3. Score Non-Buyers:", style={'fontWeight': 'bold'}),
                            f" Next, the system looks at companies that do NOT currently purchase '{selected_principal}' but ARE in the current territory and date range. For each of these non-buyers, it checks which other principals they *do* buy."),
                    html.Li(html.Span("4. Calculate Similarity:", style={'fontWeight': 'bold'}),
                            f" The 'Similarity Score' for a non-buyer is the sum of the co-occurrence counts (from step 2) for all the other principals that non-buyer already purchases. A higher score means the non-buyer's existing buying habits align more strongly with the typical buying habits of companies that already buy '{selected_principal}'.")
                ], style={'color': IR_RED, 'fontSize': '0.9em', 'marginBottom': '10px'}),
                html.P("This table ranks the non-buying companies by their similarity score, suggesting them as potential targets for cross-selling.",
                       style={'fontStyle': 'italic', 'color': IR_RED, 'marginBottom': '10px'}),
                dbc.Table([table_header, html.Tbody(table_rows)], bordered=True, hover=True, responsive=True, className="mt-3")
            ])
        ], f"Top {len(display_df)} Target Companies for {selected_principal}"

    except Exception as e:
        # Catch any unexpected errors within the callback and provide a user-friendly message
        return html.Div([
            html.P("An unexpected error occurred while processing target companies.", style={'color': IR_RED}),
            html.P(f"Error details: {e}", style={'color': IR_ORANGE, 'fontSize': '0.9em'})
        ]), f"Target Companies for {selected_principal} (Error)"


@app.callback(
    Output("download-targets-csv", "data"),
    Input("export-targets-btn", "n_clicks"),
    State("target-principal-selector", "value"),
    State("global-date-range-selector", "start_date"),
    State("global-date-range-selector", "end_date"),
    prevent_initial_call=True
)
def export_top_targets_by_territory_excel(n_clicks, selected_principal, start_date, end_date):
    if not selected_principal:
        # No principal selected
        return dash.no_update

    territory_sheets = {}

    for territory in allowed_territories:
        dff = df_pr[df_pr["Territory"] == territory].copy()
        if start_date and end_date:
            dff = dff[(dff["Order_Date"] >= start_date) & (dff["Order_Date"] <= end_date)]
        if dff.empty:
            continue

        df_company_principal = dff[['ShipTo_Company', 'Principal']].drop_duplicates()
        company_principal_matrix = df_company_principal.pivot_table(
            index='ShipTo_Company', columns='Principal', aggfunc=lambda x: 1, fill_value=0
        )

        if selected_principal not in company_principal_matrix.columns:
            continue

        buyers = company_principal_matrix[company_principal_matrix[selected_principal] == 1].index.tolist()
        non_buyers = company_principal_matrix[company_principal_matrix[selected_principal] == 0].index.tolist()

        profile = {}
        for company in buyers:
            for p in company_principal_matrix.columns:
                if p != selected_principal and company_principal_matrix.loc[company, p] == 1:
                    profile[p] = profile.get(p, 0) + 1

        results = []
        for company in non_buyers:
            score = sum(
                profile.get(p, 0)
                for p in company_principal_matrix.columns
                if p != selected_principal and company_principal_matrix.loc[company, p] == 1
            )
            if score > 0:
                results.append({'Company': company, 'SimilarityScore': score})

        if results:
            df_targets = pd.DataFrame(results).sort_values('SimilarityScore', ascending=False)
            df_targets = df_targets[df_targets["Company"].apply(lambda x: should_display_company(x, territory))]
            df_targets = df_targets.head(10)
            territory_sheets[territory[:31]] = df_targets


    if not territory_sheets:
        df_none = pd.DataFrame([{'Message': 'No targets found for selected principal and date range.'}])
        territory_sheets['No Targets'] = df_none

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet, df_sheet in territory_sheets.items():
            df_sheet.to_excel(writer, sheet_name=sheet, index=False)
    output.seek(0)

    # -------- BASE64 ENCODE FOR DOWNLOAD --------
    b64 = base64.b64encode(output.getvalue()).decode()
    return {
        "content": b64,
        "filename": f"Top10Targets_{selected_principal}.xlsx",
        "base64": True
    }

@app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def handle_upload(contents, filename):
    global GLOBAL_DF
    if contents is None:
        return html.Div("No file uploaded", style={'color': 'red'})
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'xls' in filename.lower():
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return html.Div(f"Unsupported file format: '{filename}'", style={'color': 'red'})
        df = clean_uploaded_df(df)
        if df.empty:
            return html.Div("No valid data after cleaning.", style={'color': 'orange'})
        GLOBAL_DF = df
        return html.Div(f"Successfully uploaded and processed '{filename}'", style={'color': 'green'})
    except Exception as e:
        return html.Div(f"Error processing file: {e}", style={'color': 'red'})



# ---- Run the app ----

