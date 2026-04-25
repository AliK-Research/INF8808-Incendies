import json
import re
import unicodedata

import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html

# --- CHEMINS DES FICHIERS ---
INTERVENTIONS_FILE = "cleaned_data/cleaned_interventions.csv"
CASERNES_FILE = "cleaned_data/cleaned_casernes.csv"
ZONE_STATS_FILE = "cleaned_data/merged_zone_stats.csv"
LIMITES_GEOJSON_FILE = "data/limites-administratives-agglomeration.geojson"

# --- CONSTANTES GLOBALES ---
YEARS = list(range(2020, 2025))
GRAPH_CONFIG = {"displayModeBar": False, "responsive": True, "scrollZoom": False}

MONTHS = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sept", "Oct", "Nov", "Déc"]
DAYS = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]

SEVERITY_ORDER = [
    "Mineure <= 1 unité",
    "Standard <= 5 unités",
    "Confirmée <= 15 unités",
    "Majeure <= 30 unités",
    "Critique > 30 unités",
]

# --- COULEURS ---
SEVERITY_COLORS = ["#ffcaca", "#ff7070", "#ff0000", "#990000", "#330000"]

FIRE_COLORS = {
    "Déchets en feu": "#636efa",
    "Feu de champ *": "#ef553b",
    "10-22 avec feu": "#00cc96",
    "Feu de véhicule extérieur": "#ab63fa",
    "Feu de bâtiment": "#ffa15a",
    "Autres": "#c7c7c7",
}


RED_STEPS = [
    [0.0, "#fee5d9"], [0.2, "#fee5d9"],
    [0.2, "#fcae91"], [0.4, "#fcae91"],
    [0.4, "#fb6a4a"], [0.6, "#fb6a4a"],
    [0.6, "#de2d26"], [0.8, "#de2d26"],
    [0.8, "#a50f15"], [1.0, "#a50f15"],
]

DESC_BUTTON_STYLE = style={
    "marginTop": "18px",
    "padding": "10px 18px",
    "border": "1px solid #d1d5db",
    "borderRadius": "1000px",
    "backgroundColor": "#e3e2e2e4",
    "color": "black",
    "fontSize": "14px",
    "fontWeight": "600",
    "cursor": "pointer",
}

    
# --- FONCTIONS DE CHARGEMENT ---
def normalize_key(value):
    value = "" if pd.isna(value) else str(value)
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").upper()
    value = value.replace("SAINTE", "ST").replace("SAINT", "ST")
    value = re.sub(r"\b(LE|LA|LES|DES|DE|DU|D|L|VILLE)\b", "", value)
    return re.sub(r"[^A-Z0-9]", "", value)

def load_geojson(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)

def prepare_geojson(geojson):
    for feature in geojson.get("features", []):
        props = feature.setdefault("properties", {})
        name = props.get("NOM") or props.get("nom") or props.get("LIBELLE") or props.get("MUNICIPALITE")
        if not name:
            name = next((v for v in props.values() if isinstance(v, str)), None)
        props["JOIN_KEY"] = normalize_key(name)
        props["NOM_AFFICHAGE"] = name
    return geojson

def load_all_data():
    try:
        # Interventions
        interventions = pd.read_csv(INTERVENTIONS_FILE, low_memory=False)
        if "ZONE" in interventions.columns:
            interventions = interventions.rename(columns={"ZONE": "zone"})
        interventions["date"] = pd.to_datetime(interventions["date"], errors="coerce")
        interventions["severity"] = pd.Categorical(interventions["severity"], categories=SEVERITY_ORDER, ordered=True)

        # Casernes
        casernes = pd.read_csv(CASERNES_FILE, low_memory=False)

        # Zone Stats
        zone_stats = pd.read_csv(ZONE_STATS_FILE, low_memory=False)
        if "ZONE" in zone_stats.columns:
            zone_stats = zone_stats.rename(columns={"ZONE": "zone"})

        # GeoJSON
        geojson_data = prepare_geojson(load_geojson(LIMITES_GEOJSON_FILE))

        return interventions, casernes, zone_stats, geojson_data, None

    except Exception as exc:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None, str(exc)

# --- FONCTIONS D'AIDE ET DE MISE EN FORME ---
def format_int(value):
    if pd.isna(value):
        return "Non disponible"
    return f"{int(value):,}".replace(",", " ")

def top_type_columns(zone_stats_df):
    excluded = {"zone", "Incendies", "Nb_Casernes", "Population", "Ratio", "join_key", "geo_name"}
    return [c for c in zone_stats_df.columns if c not in excluded]

def ordered_type_groups(interventions_df):
    return [c for c in interventions_df["type_grouped"].value_counts().index if c != "Autres"] + ["Autres"]

def finish_figure(fig, height=520):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=70, r=35, t=70, b=70),
        font=dict(family="Inter, sans-serif", size=12, color="#2c3e50"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee", zeroline=False, automargin=True)
    fig.update_yaxes(showgrid=False, zeroline=False, automargin=True)
    return fig

def empty_figure(message, height=420):
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False)
    return finish_figure(fig, height)

# --- COMPOSANTS DASH REUTILISABLES ---
def stat_card(label, value):
    return html.Div([
        html.Div(label, className="stat-label"),
        html.Div(value, className="stat-value"),
    ], className="stat-card")

def graph_card(graph_id=None, figure=None, height=540, class_name="chart-card"):
    props = {
        "config": GRAPH_CONFIG,
        "style": {"height": f"{height}px", "width": "100%"},
        "className": "graph",
    }
    if graph_id:
        props["id"] = graph_id
    if figure is not None:
        props["figure"] = figure
    return html.Div(dcc.Graph(**props), className=class_name)

# --- Boutton générique pour afficher les descriptions ---
def button_description(button_id, description):
    return html.Div([
        html.Button(
            id={"type": "toggle-button", "section": button_id},
            n_clicks=0,
            style=DESC_BUTTON_STYLE
        ),

        html.Div(
            html.P(description),
            id={"type": "toggle-text", "section": button_id},
            style={"display" : "none"}
        )
    ])