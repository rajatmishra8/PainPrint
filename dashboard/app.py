"""
PainPrint - Interactive Dashboard
Author: Rajat Mishra

Run: python dashboard/app.py
Open: http://127.0.0.1:8050
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# ── Load / Generate Data ─────────────────────────────────────────────────────

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'behavioral_logs.csv')

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
else:
    # Inline mini-generator for demo if data not present
    np.random.seed(42)
    n = 500
    pain = np.clip(np.cumsum(np.random.normal(0, 0.3, n)) + 5, 0, 10)
    df = pd.DataFrame({
        'user_id': ['user_000'] * n,
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='h'),
        'day': np.arange(n) // 16,
        'hour': np.tile(np.arange(8, 24), n // 16 + 1)[:n],
        'pain_score': pain,
        'typing_speed_wpm': np.clip(45 - 2 * pain + np.random.normal(0, 3, n), 5, 80),
        'backspace_rate': np.clip(0.05 + 0.02 * pain + np.random.normal(0, 0.01, n), 0, 0.5),
        'scroll_velocity': np.clip(800 - 40 * pain + np.random.normal(0, 50, n), 50, 1200),
        'app_switch_rate': np.clip(3 + 0.4 * pain + np.random.normal(0, 0.5, n), 0, 12),
        'session_duration_min': np.clip(25 - pain + np.random.normal(0, 3, n), 1, 60),
        'night_screen_usage': (pain > 7).astype(int),
    })

users = df['user_id'].unique().tolist()

# ── App Layout ────────────────────────────────────────────────────────────────

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "PainPrint Dashboard | Rajat Mishra"

CARD_STYLE = {'backgroundColor': '#1e1e2e', 'border': '1px solid #333', 'borderRadius': '12px'}

app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("🧠 PainPrint", className="text-center mb-0",
                    style={'color': '#7EB8F7', 'fontWeight': '800', 'fontSize': '2.5rem'}),
            html.P("Chronic Pain Prediction from Digital Behavior | Rajat Mishra",
                   className="text-center text-muted mb-4"),
        ])
    ]),

    # Controls
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Label("Select User", style={'color': '#aaa'}),
                    dcc.Dropdown(options=[{'label': u, 'value': u} for u in users],
                                 value=users[0], id='user-select',
                                 style={'backgroundColor': '#2a2a3e', 'color': '#fff'}),
                ])
            ], style=CARD_STYLE)
        ], md=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Label("Behavioral Feature", style={'color': '#aaa'}),
                    dcc.Dropdown(
                        options=[
                            {'label': 'Typing Speed (WPM)', 'value': 'typing_speed_wpm'},
                            {'label': 'Backspace Rate', 'value': 'backspace_rate'},
                            {'label': 'Scroll Velocity', 'value': 'scroll_velocity'},
                            {'label': 'App Switch Rate', 'value': 'app_switch_rate'},
                            {'label': 'Session Duration (min)', 'value': 'session_duration_min'},
                        ],
                        value='typing_speed_wpm', id='feature-select',
                        style={'backgroundColor': '#2a2a3e', 'color': '#fff'}
                    ),
                ])
            ], style=CARD_STYLE)
        ], md=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Label("Day Range", style={'color': '#aaa'}),
                    dcc.RangeSlider(0, 89, 1, value=[0, 30], id='day-range',
                                    marks={0: '0', 30: '30', 60: '60', 89: '89'},
                                    tooltip={'placement': 'bottom'}),
                ])
            ], style=CARD_STYLE)
        ], md=4),
    ], className="mb-4"),

    # KPI Cards
    dbc.Row(id='kpi-row', className="mb-4"),

    # Charts
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([dcc.Graph(id='pain-timeline', style={'height': '300px'})])
            ], style=CARD_STYLE)
        ], md=8),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([dcc.Graph(id='pain-dist', style={'height': '300px'})])
            ], style=CARD_STYLE)
        ], md=4),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([dcc.Graph(id='feature-vs-pain', style={'height': '300px'})])
            ], style=CARD_STYLE)
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([dcc.Graph(id='heatmap', style={'height': '300px'})])
            ], style=CARD_STYLE)
        ], md=6),
    ]),

], fluid=True, style={'backgroundColor': '#13131f', 'minHeight': '100vh', 'padding': '2rem'})


# ── Callbacks ─────────────────────────────────────────────────────────────────

@callback(
    Output('kpi-row', 'children'),
    Output('pain-timeline', 'figure'),
    Output('pain-dist', 'figure'),
    Output('feature-vs-pain', 'figure'),
    Output('heatmap', 'figure'),
    Input('user-select', 'value'),
    Input('feature-select', 'value'),
    Input('day-range', 'value'),
)
def update_dashboard(user_id, feature, day_range):
    udf = df[
        (df['user_id'] == user_id) &
        (df['day'] >= day_range[0]) &
        (df['day'] <= day_range[1])
    ].copy()

    if udf.empty:
        empty = go.Figure()
        return [], empty, empty, empty, empty

    avg_pain   = udf['pain_score'].mean()
    max_pain   = udf['pain_score'].max()
    flare_days = int((udf.groupby('day')['pain_score'].mean() > 7).sum())
    night_pct  = udf['night_screen_usage'].mean() * 100 if 'night_screen_usage' in udf else 0

    def kpi(title, val, color):
        return dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(val, style={'color': color, 'fontWeight': '800'}),
                    html.P(title, className='text-muted mb-0', style={'fontSize': '0.85rem'}),
                ])
            ], style=CARD_STYLE)
        ], md=3)

    kpis = [
        kpi("Avg Pain Score", f"{avg_pain:.1f}/10", '#7EB8F7'),
        kpi("Peak Pain", f"{max_pain:.1f}/10", '#F47E7E'),
        kpi("Flare Days (>7)", str(flare_days), '#F4A77E'),
        kpi("Night Screen %", f"{night_pct:.0f}%", '#A77EF4'),
    ]

    LAYOUT = dict(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ccc'), margin=dict(l=40, r=20, t=40, b=40),
    )

    # Timeline
    daily = udf.groupby('day').agg(pain_mean=('pain_score', 'mean')).reset_index()
    t_fig = go.Figure()
    t_fig.add_trace(go.Scatter(x=daily['day'], y=daily['pain_mean'],
                               mode='lines+markers', line=dict(color='#7EB8F7', width=2),
                               fill='tozeroy', fillcolor='rgba(126,184,247,0.1)',
                               name='Daily Avg Pain'))
    t_fig.add_hline(y=7, line_dash='dash', line_color='#F47E7E',
                    annotation_text='Flare threshold')
    t_fig.update_layout(title='Pain Score Over Time', xaxis_title='Day',
                        yaxis_title='Pain (0-10)', **LAYOUT)

    # Distribution
    d_fig = go.Figure()
    d_fig.add_trace(go.Histogram(x=udf['pain_score'], nbinsx=20,
                                 marker_color='#7EB8F7', opacity=0.8))
    d_fig.update_layout(title='Pain Score Distribution', xaxis_title='Score', **LAYOUT)

    # Feature vs Pain
    f_fig = go.Figure()
    f_col = feature if feature in udf.columns else 'typing_speed_wpm'
    f_fig.add_trace(go.Scatter(x=udf['pain_score'], y=udf[f_col],
                               mode='markers',
                               marker=dict(color=udf['pain_score'], colorscale='RdBu_r',
                                           size=4, opacity=0.6),
                               name=f_col))
    f_fig.update_layout(title=f'{f_col} vs Pain Score',
                        xaxis_title='Pain Score', yaxis_title=f_col, **LAYOUT)

    # Heatmap: hour vs day
    pivot = udf.pivot_table(index='hour', columns='day', values='pain_score', aggfunc='mean')
    h_fig = go.Figure(go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index,
                                 colorscale='RdBu_r', reversescale=True))
    h_fig.update_layout(title='Pain Heatmap: Hour × Day',
                        xaxis_title='Day', yaxis_title='Hour', **LAYOUT)

    return kpis, t_fig, d_fig, f_fig, h_fig


if __name__ == '__main__':
    print("🚀 PainPrint Dashboard | Rajat Mishra")
    print("   Open: http://127.0.0.1:8050")
    app.run(debug=True)
