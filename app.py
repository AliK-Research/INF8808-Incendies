from pathlib import Path
from urllib.parse import urlparse
import json
import re
import unicodedata

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from dash import Dash, Input, Output, dcc, html
from plotly.subplots import make_subplots


INTERVENTIONS_FILE = "data/interventions.csv"
CASERNES_FILE = "data/casernes.csv"
LIMITES_GEOJSON_FILE = "data/limites-administratives-agglomeration.geojson"

CACHE_DIR = Path("data/cache")
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


def normalize_key(value):
    value = "" if pd.isna(value) else str(value)
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").upper()
    value = value.replace("SAINTE", "ST").replace("SAINT", "ST")
    value = re.sub(r"\b(LE|LA|LES|DES|DE|DU|D|L|VILLE)\b", "", value)
    return re.sub(r"[^A-Z0-9]", "", value)


def clean_text(series):
    return series.fillna("").astype(str).str.replace("\uFFFD", "-", regex=False).str.strip()


def find_col(df, candidates):
    columns = {normalize_key(c): c for c in df.columns}
    for candidate in candidates:
        key = normalize_key(candidate)
        if key in columns:
            return columns[key]
    raise KeyError(f"Colonne manquante: {', '.join(candidates)}")


def resource_id_from_url(url):
    match = re.search(r"/resource/([0-9a-fA-F-]+)", url)
    return match.group(1) if match else None


def direct_download_url(resource_page_url):
    resource_id = resource_id_from_url(resource_page_url)
    if not resource_id:
        return resource_page_url

    parsed = urlparse(resource_page_url)
    api_url = f"{parsed.scheme}://{parsed.netloc}/api/3/action/resource_show?id={resource_id}"
    response = requests.get(api_url, timeout=30)
    response.raise_for_status()

    payload = response.json()
    if not payload.get("success"):
        raise RuntimeError(f"Ressource illisible: {resource_page_url}")
    return payload["result"]["url"]


def load_csv(path):
    return pd.read_csv(path, low_memory=False)


def load_geojson(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def zone_from_area(arrondissement, ville):
    arrondissement = "" if pd.isna(arrondissement) else str(arrondissement).strip()
    ville = "" if pd.isna(ville) else str(ville).strip()

    if arrondissement and arrondissement.lower() not in {"indéterminé", "indetermine", "nan"}:
        return arrondissement
    return ville or "Indéterminé"


def prepare_interventions():
    raw = load_csv(INTERVENTIONS_FILE)
    raw = raw.rename(columns={c: normalize_key(c) for c in raw.columns})

    date_col = find_col(raw, ["CREATION_DATE_TIME", "DATE"])
    type_col = find_col(raw, ["INCIDENT_TYPE_DESC", "TYPE_INCIDENT"])
    group_col = find_col(raw, ["DESCRIPTION_GROUPE", "GROUPE"])
    station_col = find_col(raw, ["CASERNE"])
    city_col = find_col(raw, ["NOM_VILLE", "VILLE"])
    borough_col = find_col(raw, ["NOM_ARROND", "ARRONDISSEMENT"])
    units_col = find_col(raw, ["NOMBRE_UNITES", "NB_UNITES", "UNITES"])
    lat_col = find_col(raw, ["LATITUDE", "LAT"])
    lon_col = find_col(raw, ["LONGITUDE", "LON", "LONG"])

    df = raw.copy()
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[df["date"].dt.year.isin(YEARS)].copy()

    groups = clean_text(df[group_col]).str.upper()
    df = df[groups.isin(["INCENDIE", "AUTREFEU"])].copy()

    df["incident_type"] = clean_text(df[type_col])
    df["description_groupe"] = clean_text(df[group_col])
    df["caserne"] = pd.to_numeric(df[station_col], errors="coerce")
    df["ville"] = clean_text(df[city_col])
    df["arrondissement"] = clean_text(df[borough_col])
    df["nombre_unites"] = pd.to_numeric(df[units_col], errors="coerce")
    df["latitude"] = pd.to_numeric(df[lat_col], errors="coerce")
    df["longitude"] = pd.to_numeric(df[lon_col], errors="coerce")
    df["zone"] = [zone_from_area(a, v) for a, v in zip(df["arrondissement"], df["ville"])]

    dt = pd.DatetimeIndex(df["date"])
    df["year"] = dt.year.astype(str)
    df["month"] = dt.month
    df["month_label"] = df["month"].map(dict(enumerate(MONTHS, start=1)))
    df["weekday"] = dt.weekday
    df["day_label"] = df["weekday"].map(dict(enumerate(DAYS)))
    df["hour"] = dt.hour

    top5 = df["incident_type"].value_counts().head(5).index
    df["type_grouped"] = np.where(df["incident_type"].isin(top5), df["incident_type"], "Autres")
    df["severity"] = pd.cut(
        df["nombre_unites"].fillna(0),
        bins=[-np.inf, 1, 5, 15, 30, np.inf],
        labels=SEVERITY_ORDER,
    ).astype(str)

    return df.dropna(subset=["date"])


def prepare_casernes():
    raw = load_csv(CASERNES_FILE)
    raw = raw.rename(columns={c: normalize_key(c) for c in raw.columns})

    station_col = find_col(raw, ["CASERNE"])
    lat_col = find_col(raw, ["LATITUDE", "LAT"])
    lon_col = find_col(raw, ["LONGITUDE", "LON", "LONG"])
    borough_col = find_col(raw, ["ARRONDISSEMENT", "NOM_ARROND"])
    city_col = find_col(raw, ["VILLE", "NOM_VILLE"])

    df = raw.copy()
    if "DATEFIN" in df.columns:
        df = df[clean_text(df["DATEFIN"]).eq("")].copy()

    df["caserne"] = pd.to_numeric(df[station_col], errors="coerce")
    df["latitude"] = pd.to_numeric(df[lat_col], errors="coerce")
    df["longitude"] = pd.to_numeric(df[lon_col], errors="coerce")
    df["arrondissement"] = clean_text(df[borough_col])
    df["ville"] = clean_text(df[city_col])
    df["zone"] = [zone_from_area(a, v) for a, v in zip(df["arrondissement"], df["ville"])]

    df = df[~df["caserne"].isin([2, 79])].copy()
    return df.dropna(subset=["latitude", "longitude"])


def prepare_geojson(geojson):
    for feature in geojson.get("features", []):
        props = feature.setdefault("properties", {})
        name = props.get("NOM") or props.get("nom") or props.get("LIBELLE") or props.get("MUNICIPALITE")

        if not name:
            name = next((v for v in props.values() if isinstance(v, str)), None)

        props["JOIN_KEY"] = normalize_key(name)
        props["NOM_AFFICHAGE"] = name
    return geojson


def add_geojson_join_key(df_zones, geojson):
    available = {
        f["properties"].get("JOIN_KEY"): f["properties"].get("NOM_AFFICHAGE")
        for f in geojson.get("features", [])
    }

    manual = {
        normalize_key("Rivière-des-Prairies-Pointe-aux-Trembles"): normalize_key("Rivière-des-Prairies - Pointe-aux-Trembles"),
        normalize_key("Côte-des-Neiges-Notre-Dame-de-Grâce"): normalize_key("Côte-des-Neiges - Notre-Dame-de-Grâce"),
        normalize_key("Villeray-Saint-Michel-Parc-Extension"): normalize_key("Villeray - Saint-Michel - Parc-Extension"),
        normalize_key("Mercier-Hochelaga-Maisonneuve"): normalize_key("Mercier - Hochelaga-Maisonneuve"),
        normalize_key("Rosemont-La Petite-Patrie"): normalize_key("Rosemont - La Petite-Patrie"),
        normalize_key("L'Île-Bizard-Sainte-Geneviève"): normalize_key("L'Île-Bizard - Sainte-Geneviève"),
    }

    df = df_zones.copy()
    df["join_key"] = df["zone"].map(normalize_key).replace(manual)
    df["geo_name"] = df["join_key"].map(available)
    return df


def build_zone_stats():
    totals = interventions.groupby("zone").size().reset_index(name="Incendies")
    top10 = interventions["incident_type"].value_counts().head(10).index

    pivot = (
        interventions[interventions["incident_type"].isin(top10)]
        .pivot_table(index="zone", columns="incident_type", aggfunc="size", fill_value=0)
        .reset_index()
    )

    stats = totals.merge(pivot, on="zone", how="left").fillna(0)
   
    try:
        df_final = pd.read_csv('data/stats_zones_final.csv')
        df_extra = df_final[['ZONE', 'Population', 'Nb_Casernes']]

        stats = stats.merge(df_extra, left_on="zone", right_on="ZONE", how="left")
        stats = stats.drop(columns=['ZONE']) 
        stats["Nb_Casernes"] = stats["Nb_Casernes"].fillna(0).astype(int)
        stats['Ratio'] = round(((stats['Incendies'] / 5) / stats['Population']) * 1000, 2)
        
    except Exception as e:
        print(f"Erreur de chargement des données locales : {e}")
        stats["Population"] = np.nan
        stats["Ratio"] = np.nan
        stats["Nb_Casernes"] = 0
    
    return stats


def top_type_columns():
    excluded = {"zone", "Incendies", "Nb_Casernes", "Population", "Ratio", "join_key", "geo_name"}
    return [c for c in zone_stats.columns if c not in excluded]


def format_int(value):
    if pd.isna(value):
        return "Non disponible"
    return f"{int(value):,}".replace(",", " ")


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


def ordered_type_groups():
    return [c for c in interventions["type_grouped"].value_counts().index if c != "Autres"] + ["Autres"]


def make_type_bar_by_zone():
    df = interventions[interventions["zone"] != "Indéterminé"].copy()
    top5 = df["incident_type"].value_counts().head(5).index.tolist()
    df["Type"] = np.where(df["incident_type"].isin(top5), df["incident_type"], "Autres")

    zones = df.groupby("zone").size().sort_values(ascending=False).head(22).index.tolist()
    grouped = df[df["zone"].isin(zones)].groupby(["zone", "Type"]).size().reset_index(name="Nombre")

    fig = px.bar(
        grouped,
        x="zone",
        y="Nombre",
        color="Type",
        title="Incendies par arrondissement et type (Top 5 + Autres)",
        labels={"zone": "Arrondissement", "Nombre": "Nombre d'incidents"},
        category_orders={"zone": zones, "Type": top5 + ["Autres"]},
        color_discrete_map=FIRE_COLORS,
    )
    fig.update_layout(
        title_x=0.5,
        barmode="stack",
        legend_title_text="Types d'incidents",
        legend=dict(x=1.02, y=1, xanchor="left", yanchor="top"),
        margin=dict(l=65, r=210, t=65, b=135),
    )
    fig.update_xaxes(tickangle=-45)
    return finish_figure(fig, 600)


def make_map(mode="Incendies"):
    if geojson_data is None:
        return make_fallback_point_map()

    df_map = add_geojson_join_key(zone_stats, geojson_data)
    value_col = "Incendies" if mode == "Ratio" and df_map["Ratio"].isna().all() else mode

    raw_max = df_map[value_col].max()
    raw_max = 1 if pd.isna(raw_max) or raw_max <= 0 else raw_max
    v_max = int(np.ceil(raw_max / 100) * 100) if value_col == "Incendies" else float(np.ceil(raw_max))
    bins = np.linspace(0, v_max, 6)
    labels = [str(int(v)) for v in bins] if value_col == "Incendies" else [f"{v:.1f}" for v in bins]

    fig = go.Figure()
    fig.add_trace(go.Choroplethmapbox(
        geojson=geojson_data,
        locations=df_map["join_key"],
        z=df_map[value_col],
        featureidkey="properties.JOIN_KEY",
        colorscale=RED_STEPS,
        zmin=0,
        zmax=v_max,
        marker_opacity=0.82,
        marker_line_width=0.6,
        marker_line_color="#4f4f4f",
        customdata=np.stack([df_map["zone"], df_map["Incendies"], df_map["Nb_Casernes"]], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b><br>Interventions: %{customdata[1]}<br>Casernes: %{customdata[2]}<extra></extra>",
        colorbar=dict(
            x=1.02,
            xanchor="left",
            len=0.66,
            thickness=18,
            ticks="outside",
            tickmode="array",
            tickvals=bins,
            ticktext=labels,
            outlinecolor="black",
            outlinewidth=1,
            title=dict(text=""),
        ),
        name="Nombre d'incidents",
    ))

    casernes_geojson = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]}}
            for lon, lat in zip(casernes["longitude"], casernes["latitude"])
        ],
    }

    fig.add_trace(go.Scattermapbox(
        lat=[0],
        lon=[0],
        mode="markers",
        marker=dict(size=8, color="black"),
        name="Casernes",
        hoverinfo="skip",
    ))

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=9.60,
        mapbox_center={"lat": 45.55, "lon": -73.735},
        mapbox_layers=[{
            "sourcetype": "geojson",
            "source": casernes_geojson,
            "type": "circle",
            "color": "black",
            "circle_radius": 3,
            "below": "",
        }],
        margin=dict(l=0, r=80, t=15, b=0),
        height=610,
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.78)", bordercolor="black", borderwidth=1),
        paper_bgcolor="white",
        uirevision="constant",
    )
    return fig


def make_fallback_point_map():
    summary = (
        interventions.dropna(subset=["latitude", "longitude"])
        .groupby("zone")
        .agg(Incendies=("incident_type", "size"), latitude=("latitude", "mean"), longitude=("longitude", "mean"))
        .reset_index()
    )
    max_count = max(summary["Incendies"].max(), 1)

    fig = go.Figure(go.Scattermapbox(
        lat=summary["latitude"],
        lon=summary["longitude"],
        mode="markers",
        marker=dict(
            size=8 + 32 * np.sqrt(summary["Incendies"] / max_count),
            color=summary["Incendies"],
            colorscale="Reds",
            showscale=True,
        ),
        customdata=np.stack([summary["zone"], summary["Incendies"]], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b><br>Interventions: %{customdata[1]}<extra></extra>",
    ))

    fig.add_trace(go.Scattermapbox(
        lat=casernes["latitude"],
        lon=casernes["longitude"],
        mode="markers",
        marker=dict(size=7, color="black"),
        name="Casernes",
    ))

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=9.3,
        mapbox_center={"lat": 45.54, "lon": -73.66},
        margin=dict(l=0, r=0, t=15, b=0),
        height=610,
        paper_bgcolor="white",
    )
    return fig


def selected_zone(click_data):
    if click_data and click_data.get("points"):
        point = click_data["points"][0]

        if point.get("customdata") is not None:
            return point["customdata"][0]

        if point.get("location") is not None and "join_key" in zone_stats.columns:
            row = zone_stats[zone_stats["join_key"].eq(point["location"])]
            if not row.empty:
                return row.iloc[0]["zone"]

    if "Ville-Marie" in set(zone_stats["zone"]):
        return "Ville-Marie"
    return zone_stats.sort_values("Incendies", ascending=False).iloc[0]["zone"]


def make_detail_bar(zone):
    row = zone_stats[zone_stats["zone"] == zone]
    if row.empty:
        return empty_figure("Aucune donnée pour ce secteur.", 300)

    values = pd.Series({col: row.iloc[0][col] for col in top_type_columns()}).sort_values()
    fig = px.bar(
        x=values.values,
        y=values.index,
        orientation="h",
        text=values.values.astype(int),
        labels={"x": "", "y": ""},
    )

    fig.update_traces(
        marker_color="#4a6984",
        marker_line_color="white",
        marker_line_width=1,
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Nombre: %{x}<extra></extra>",
    )
    fig.update_layout(
        height=305,
        margin=dict(l=8, r=45, t=10, b=20),
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=10),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee", range=[0, max(values.max() * 1.25, 10)])
    fig.update_yaxes(automargin=True)
    return fig


def make_temporal(scale):
    if scale == "week":
        heat = interventions.groupby(["day_label", "hour"]).size().reset_index(name="Valeur")
        pivot = heat.pivot(index="day_label", columns="hour", values="Valeur").reindex(DAYS).reindex(columns=range(24)).fillna(0)
        vmax = max(float(pivot.max().max()), 1)
        bins = np.linspace(0, vmax, 6)

        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=[f"{h}h" for h in pivot.columns],
            y=pivot.index,
            colorscale=RED_STEPS,
            zmin=0,
            zmax=vmax,
            xgap=3,
            ygap=3,
            colorbar=dict(
                title=dict(text="Nombre d'incidents", side="top"),
                tickmode="array",
                tickvals=bins,
                ticktext=[str(int(b)) for b in bins],
                thickness=18,
                len=0.75,
                ticks="outside",
                outlinecolor="black",
                outlinewidth=1,
            ),
            hovertemplate="Jour: %{y}<br>Heure: %{x}<br>Interventions: %{z}<extra></extra>",
        ))

        fig.update_layout(
            title={"text": "Analyse des périodes critiques sur la semaine", "x": 0.5},
            xaxis=dict(title="Heure de la journée", showline=True, linecolor="black", ticks="outside"),
            yaxis=dict(title="Jour de la semaine", autorange="reversed", showline=True, linecolor="black", ticks="outside"),
            margin=dict(l=90, r=95, t=70, b=70),
        )
        return finish_figure(fig, 540)

    if scale == "year":
        df = interventions.groupby(["year", "type_grouped"]).size().reset_index(name="Valeur")
        title, x_title, order = "Répartition annuelle par type d'incident", "Année", sorted(interventions["year"].unique())
    else:
        df = interventions.groupby(["month", "month_label", "type_grouped"]).size().reset_index(name="Valeur")
        df["Valeur"] = (df["Valeur"] / max(len(YEARS), 1)).round().astype(int)
        df = df.rename(columns={"month_label": "year"})
        title, x_title, order = "Saisonnalité des types d'incidents (Moyenne)", "Mois", MONTHS

    fig = go.Figure()
    for category in ordered_type_groups():
        sub = df[df["type_grouped"] == category].copy()
        if scale == "month":
            sub["year"] = pd.Categorical(sub["year"], categories=MONTHS, ordered=True)
            sub = sub.sort_values("year")

        fig.add_trace(go.Bar(
            name=category,
            x=sub["year"],
            y=sub["Valeur"],
            marker_color=FIRE_COLORS.get(category),
            hovertemplate=f"<b>{category}</b><br>Interventions: %{{y}}<extra></extra>",
        ))

    totals = df.groupby("year", observed=False)["Valeur"].sum().reindex(order).dropna().reset_index()
    fig.add_trace(go.Scatter(
        x=totals["year"],
        y=totals["Valeur"],
        mode="text",
        text=totals["Valeur"].astype(int),
        textposition="top center",
        showlegend=False,
        hoverinfo="skip",
    ))

    fig.update_layout(
        barmode="stack",
        title={"text": title, "x": 0.5},
        legend_title="Type d'incident",
        legend=dict(x=1.02, y=1, xanchor="left", yanchor="top"),
        margin=dict(l=70, r=210, t=70, b=70),
        xaxis=dict(title=x_title, type="category", categoryorder="array", categoryarray=order, showline=True, linecolor="black", ticks="outside"),
        yaxis=dict(title="Nombre d'incidents", range=[0, max(totals["Valeur"].max() * 1.17, 1)], showline=True, linecolor="black", ticks="outside"),
    )
    return finish_figure(fig, 540)


import plotly.graph_objects as go
import pandas as pd

import pandas as pd
import plotly.graph_objects as go

def make_boxplot():
    top_types = interventions["incident_type"].value_counts().head(15).index
    df = interventions[interventions["incident_type"].isin(top_types)].copy()
    df["nombre_unites"] = df["nombre_unites"].fillna(0)

    stats_rows = []
    outlier_rows = []

    for incident_type, s in df.groupby("incident_type")["nombre_unites"]:
        s = s.dropna().sort_values()

        if s.empty:
            continue
        q1 = s.quantile(0.25)
        median = s.quantile(0.50)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        inside_fences = s[(s >= lower_bound) & (s <= upper_bound)]
        lower_fence = inside_fences.min() if not inside_fences.empty else s.min()
        upper_fence = inside_fences.max() if not inside_fences.empty else s.max()

        stats_rows.append({
            "incident_type": incident_type,
            "min": s.min(),
            "lower_fence": lower_fence,
            "q1": q1,
            "median": median,
            "q3": q3,
            "upper_fence": upper_fence,
            "max": s.max(),
        })

        outliers = s[(s < lower_fence) | (s > upper_fence)]

        if not outliers.empty:
            low_extreme = outliers.min()
            high_extreme = outliers.max()

            for value in outliers:
                label = ""
                if value == low_extreme:
                    label = "Outlier extrême bas"
                elif value == high_extreme:
                    label = "Outlier extrême haut"
                outlier_rows.append({
                    "incident_type": incident_type,
                    "nombre_unites": value,
                    "label": label,
                })

    stats_df = pd.DataFrame(stats_rows)
    stats_df = stats_df.sort_values("median", ascending=False)
    order = stats_df["incident_type"].tolist()

    fig = go.Figure()

    for row in stats_df.itertuples(index=False):
        fig.add_trace(
            go.Box(
                orientation="h",
                y0=row.incident_type,
                q1=[row.q1],
                median=[row.median],
                q3=[row.q3],
                lowerfence=[row.lower_fence],
                upperfence=[row.upper_fence],
                boxpoints=False,
                line=dict(color="#b7380d"),
                fillcolor="rgba(183, 56, 13, 0.35)",
                showlegend=False,
                name="",
                customdata=[[
                    row.min,
                    row.lower_fence,
                    row.q1,
                    row.median,
                    row.q3,
                    row.upper_fence,
                    row.max,
                    row.incident_type,
                ]],
                hovertemplate=(
                    "<b>%{customdata[7]}</b><br>"
                    "min: %{customdata[0]:.0f}<br>"
                    "lower fence: %{customdata[1]:.0f}<br>"
                    "q1: %{customdata[2]:.0f}<br>"
                    "median: %{customdata[3]:.0f}<br>"
                    "q3: %{customdata[4]:.0f}<br>"
                    "upper fence: %{customdata[5]:.0f}<br>"
                    "max: %{customdata[6]:.0f}"
                    "<extra></extra>"
                ),
            )
        )

    if outlier_rows:
        outliers_df = pd.DataFrame(outlier_rows)
        for incident_type, g in outliers_df.groupby("incident_type", sort=False):
            fig.add_trace(
                go.Scatter(
                    x=g["nombre_unites"],
                    y=[incident_type] * len(g),
                    mode="markers",
                    marker=dict(
                        color="#b7380d",
                        size=8,
                    ),
                    showlegend=False,
                    name="",
                    customdata=g[["label"]].values,
                    hovertemplate=(
                    "%{x:.0f}<br>"
                    "%{customdata[0]}"
                    "<extra></extra>"
                    ),
    )
)

    fig.update_layout(
        title="Distribution des unités mobilisées par type d'incident",
        title_x=0.5,
        xaxis_title="Nombre d'unités (intensité)",
        yaxis_title="Type d'incident",
        showlegend=False,
        margin=dict(l=215, r=35, t=65, b=65),
        hovermode="y unified",
    )
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=order,
        autorange="reversed"
    )

    return finish_figure(fig, 580)


def make_waffle():
    df = interventions[interventions["zone"] != "Indéterminé"].copy()
    zones = sorted(df["zone"].unique())
    n_cols = 4
    n_rows = int(np.ceil(len(zones) / n_cols))

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=zones,
        horizontal_spacing=0.02,
        vertical_spacing=0.04,
    )

    for category, color in zip(SEVERITY_ORDER, SEVERITY_COLORS):
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=12, color=color, symbol="square"),
            name=category,
            showlegend=True,
        ))

    color_index = {cat: i for i, cat in enumerate(SEVERITY_ORDER)}
    colorscale = [[i / (len(SEVERITY_COLORS) - 1), color] for i, color in enumerate(SEVERITY_COLORS)]

    for i, zone in enumerate(zones):
        part = df[df["zone"] == zone]
        cells = (part["severity"].value_counts(normalize=True).reindex(SEVERITY_ORDER, fill_value=0) * 100).round().astype(int)
        diff = 100 - int(cells.sum())

        if diff:
            cells.loc[cells.idxmax()] += diff

        values = []
        for category in reversed(SEVERITY_ORDER):
            values.extend([category] * max(int(cells.get(category, 0)), 0))
        values = (values + [SEVERITY_ORDER[0]] * 100)[:100]
        matrix = np.array(values).reshape(10, 10)

        fig.add_trace(go.Heatmap(
            z=[[color_index[c] for c in row] for row in matrix],
            text=matrix,
            hovertemplate="%{text}<extra></extra>",
            colorscale=colorscale,
            zmin=0,
            zmax=len(SEVERITY_COLORS) - 1,
            showscale=False,
            xgap=1,
            ygap=1,),
            row=i // n_cols + 1, col=i % n_cols + 1)

    fig.update_layout(
        title_text="Gravité des interventions par arrondissement",
        title_x=0.5,
        height=max(900, 220 * n_rows),
        margin=dict(t=100, b=40, l=40, r=210),
        legend=dict(title="Gravité", x=1.01, y=1),
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=11),
    )
    fig.update_annotations(font_size=10)
    
    for i in range(1, len(zones) + 1):
        axis_name = f"x{i if i > 1 else ''}"
        fig.update_yaxes(
            scaleanchor=axis_name,
            scaleratio=1,
            row=(i - 1) // n_cols + 1,
            col=(i - 1) % n_cols + 1
        )

    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, autorange="reversed")
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)

    return fig


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


app = Dash(__name__)
external_stylesheets = [
    "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap"
]
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title = "Interventions incendie à Montréal"

try:
    interventions = prepare_interventions()
    casernes = prepare_casernes()
    geojson_data = prepare_geojson(load_geojson(LIMITES_GEOJSON_FILE))

    if interventions.empty:
        raise RuntimeError("Aucune intervention valide trouvée pour les années configurées.")

    zone_stats = build_zone_stats()
    if geojson_data is not None:
        zone_stats = add_geojson_join_key(zone_stats, geojson_data)

    total_incidents = len(interventions)
    top_zone = zone_stats.sort_values("Incendies", ascending=False).iloc[0]["zone"]
    top_type = interventions["incident_type"].value_counts().index[0]
    data_error = None

except Exception as exc:
    interventions = pd.DataFrame()
    casernes = pd.DataFrame()
    zone_stats = pd.DataFrame()
    geojson_data = None
    data_error = str(exc)


if data_error:
    app.layout = html.Div([
        html.H1("Erreur de chargement des données"),
        html.P(data_error),
        html.P("Vérifie ta connexion internet au premier lancement. Les fichiers sont ensuite mis en cache dans data/cache."),
    ], className="error-page")

else:
    waffle_height = max(900, 220 * int(np.ceil(len(interventions[interventions["zone"] != "Indéterminé"]["zone"].unique()) / 4)))

    app.layout = html.Div([
        html.Header([
            html.P("INF8808 - Pré-release", className="eyebrow"),
            html.H1("Les interventions incendie à Montréal"),
            html.P("Évolution temporelle, répartition spatiale et couverture des casernes entre 2020 et 2024.", className="lead"),
            html.P("Par Robin Holden, Ali Karaki et Paul Besse", style={"fontStyle": "italic", "fontSize": "14px", "marginBottom": "15px", "color": "var(--muted)"}),
            html.P("Chaque jour, les pompiers du Service de sécurité incendie de Montréal (SIM) répondent à des centaines d’appels." \
            " Mais que se passe-t-il réellement derrière les sirènes ? Cette exploration interactive propose d’analyser les données " \
            "l’historique des interventions entre 2020 et 2024 pour comprendre les risques qui touchent l’agglomération de Montréal" \
            ", l’évolution des interventions au fil des saisons et l’incroyable logistique nécessaire pour protéger plus de deux" \
            " millions de citoyens."),
            html.Div([
                stat_card("Interventions analysées", format_int(total_incidents)),
                stat_card("Secteur le plus touché", top_zone),
                stat_card("Type le plus fréquent", top_type),
                stat_card("Casernes actives", format_int(len(casernes))),
            ], className="stats"),
        ], className="hero reveal"),

        html.Section([
            html.Div([
                html.H2("Quels types d'incendies dominent selon les secteurs ?"),
                html.P("Chaque quartier de Montréal possède sa propre identité, allant des zones denses à d’autres plus verdoyantes"
                ". Cela influence directement la nature des risques. Le diagramme à barres empilées suivant compare les types " \
                "d’incendies les plus fréquents d’un secteur à l’autre. Les interventions les plus rares ont été regroupées dans la " \
                "catégorie « Autres » pour faciliter la lecture.", style={"marginTop": "20px"}),
                html.P([
                    html.B("Comment ça marche : "),
                    "Plus une barre est haute, plus il y a eu d’interventions. Les couleurs permettent de "
                    "voir si un quartier fait face à plus de feux de bâtiments ou de feux extérieurs."
                ]),
            ], className="section-text"),
            graph_card(figure=make_type_bar_by_zone(), height=600),
            html.P("L’arrondissement Ville-Marie domine largement le bilan avec le nombre d’interventions le plus élevé de " \
            "la métropole, suivi par les secteurs denses de Mercier-Hochelaga-Maisonneuve et du Plateau-Mont-Royal. Cela " \
            "s'explique par son statut de centre-ville où la forte densité d'activités et de population multiplie par conséquent " \
            "les risques d’incidents. Les déchets en feu sont la cause majoritaire des interventions dans presque tous les " \
            "arrondissements. Par ailleurs, les secteurs verts comme Ahuntsic-Cartierville et Pierrefonds-Roxboro sont " \
            "particulièrement touchés par les feux de champ.", style={"marginTop": "20px"}),
        ], className="section reveal delay-1"),

        html.Section([
            html.Div([
                html.H2("Carte montrant la répartition des feux (2020-2024)"),
                html.P("La sécurité est également une question de proximité. La carte ci-dessous montre l’emplacement des 64 casernes " \
                "de l’île impliquées dans la lutte contre les incendies et le niveau d’activité " \
                "de chaque zone.", style={"marginTop": "20px"}),
                html.P([
                    html.B("Comment ça marche : "),
                    "Il est possible de basculer entre le nombre total d’incidents et l’indice de risque. Ce dernier ajuste "
                    "les chiffres en fonction du nombre d’habitants, ce qui rend la comparaison équitable entre un petit "
                    "quartier très peuplé et un grand arrondissement moins dense."
                ]),
                dcc.RadioItems(
                    id="map-mode",
                    options=[
                        {"label": " Nombre d'incidents (5 ans)", "value": "Incendies"},
                        {"label": " Indice de risque annuel", "value": "Ratio"},
                    ],
                    value="Incendies",
                    inline=True,
                    className="radio-centered",
                ),
            ], className="section-text centered"),
            html.Div([
                graph_card(graph_id="map-montreal", figure=make_map(), height=610, class_name="map-card"),
                html.Div(id="side-panel", className="side-panel"),
            ], className="map-layout"),
            html.P("La répartition géographique montre une grande concentration des interventions dans les environs " \
            "du centre-ville, comme vu précédemment avec Ville-Marie. Cependant, l'indice de risque annuel apporte " \
            "une information supplémentaire importante. En effet, une fois les données rapportées à la population, des " \
            "arrondissements moins denses comme Montréal-Est et Senneville révèlent une vulnérabilité importante. Cela " \
            "confirme que le risque incendie est à la fois lié à la densité humaine et aux environnements verts, davantage " \
            "sujets aux feux de végétation. La répartition des casernes assure ainsi une présence stratégique pour intervenir " \
            "rapidement sur l'ensemble du territoire.", style={"marginTop": "20px"}),
        ], className="section reveal delay-2"),

        html.Section([
            html.Div([
                html.H2("Évolution et cycles d'intervention"),
                html.P("Le risque incendie n’est pas statique, il suit le rythme de la ville et des saisons. Cette visualisation " \
                "permet de retracer l’historique des interventions par année, par mois ou même par semaine et " \
                "par jour.", style={"marginTop": "20px"}),
                html.P([
                    html.B("Comment ça marche : "),
                    "Les boutons permettent de choisir l’échelle temporelle souhaitée. Les vues « Annuelle » et « Mensuelle » "
                    "présentent des graphiques à barres empilées montrant le volume total d'incidents et leur nature. La vue "
                    "« Hebdomadaire » utilise une carte de chaleur : plus la case est foncée, plus le nombre d'interventions "
                    "est élevé pour cette plage horaire précise."
                ]),
                dcc.RadioItems(
                    id="time-selector",
                    options=[
                        {"label": " Annuelle", "value": "year"},
                        {"label": " Mensuelle", "value": "month"},
                        {"label": " Hebdomadaire", "value": "week"},
                    ],
                    value="year",
                    inline=True,
                    className="radio-centered",
                ),
            ], className="section-text centered"),
            graph_card(graph_id="chrono-chart", figure=make_temporal("year"), height=540),
            html.P("L'évolution temporelle montre une bonne stabilité du volume global d'interventions sur les cinq dernières " \
            "années, malgré un pic notable en 2020 lié à un été rude. Cependant, l'analyse à une maille plus fine révèle une " \
            "dynamique saisonnière et quotidienne très marquée. Le nombre d'incidents double presque lors du passage de l'hiver " \
            "à l'été, principalement à cause de la chaleur et de la sécheresse favorisant les feux extérieurs. Cela confirme " \
            "que le rythme des incidents est étroitement lié à la période de l’année et aux cycles d'activité humaine, qui " \
            "est plus élevée entre 13h et 22h. Cette connaissance de la temporalité permet ainsi au SIM d’accroître sa vigilance " \
            "en fonction des périodes de vulnérabilité accrue.", style={"marginTop": "20px"}),
        ], className="section reveal delay-3"),

        html.Section([
            html.Div([
                html.H2("Combien d'unités sont mobilisées par type d'incident et par arrondissement ?"),
                html.P("Toutes les interventions ne se ressemblent pas. En effet, l’effort déployé change radicalement selon " \
                "que les pompiers font face à une poubelle en feu ou un incendie de bâtiment par exemple. Le diagramme en boîtes " \
                "montre le nombre habituel d’unités déployées pour chaque type d’appel. Le graphique en gaufrier situé en " \
                "dessous représente la répartition du nombre d’unités déployées par quartier.", style={"marginTop": "20px"}),
                html.P([
                    html.B("Comment ça marche : "),
                    "Pour le diagramme en boîtes, chaque ligne verticale dans une boîte représente la mobilisation médiane. "
                    "Plus la boîte est étirée vers la droite, plus les besoins en ressources varient pour ce type d'appel. "
                    "Les points isolés à droite indiquent des interventions d'une ampleur exceptionnelle. Pour le graphique "
                    "en gaufrier, chaque petit carré représente 1% des interventions du secteur. Plus il y a de carrés "
                    "foncés, plus les interventions de grande ampleur ont été fréquentes."
                ]),
            ], className="section-text"),
            graph_card(figure=make_boxplot(), height=580),
            graph_card(figure=make_waffle(), height=waffle_height),
            html.P("L'analyse de la mobilisation montre que la vaste majorité des incidents quotidiens, tels que les feux de " \
            "déchets ou de broussailles, sont maîtrisés avec seulement une à deux unités. Cependant, dès que l'on passe aux " \
            "alertes de niveau supérieur comme les incendies de bâtiments confirmés, l'effort augmente pour atteindre souvent " \
            "plus de 15, voire 60 unités pour les cas les plus critiques.", style={"marginTop": "20px"}),
            html.P("Le portrait par quartier montre que les interventions de routine, « Mineure » ou « Standard », constituent " \
            "la plus grande partie de l'activité partout sur l'île. On note toutefois que des secteurs denses ou " \
            "institutionnels, comme Ville-Marie ou Westmount, affichent une proportion de carrés foncés légèrement plus " \
            "élevée, exigeant une mobilisation massive d'unités spécialisées.", style={"marginTop": "10px"}),
        ], className="section reveal delay-4"),

        html.Footer([
            html.Div([
                html.H2("Protéger, prévenir, progresser"),
                html.P([
                    "Les visualisations proposées illustrent bien que la sécurité incendie à Montréal est un défi d’équilibre. "
                    "Entre la prévention des petits incidents quotidiens et la préparation aux accidents majeurs, le SIM adapte "
                    "ses ressources à la réalité de chaque quartier. Ces cinq années reflètent sa capacité à veiller sur une "
                    "métropole en constante mutation, peu importe l’heure, le quartier ou la saison."
                ], style={"marginTop": "20px", "lineHeight": "1.6"}),
            ], className="section reveal", style={"marginBottom": "25px"}),

            html.Div([
                html.P(html.B("En lien avec le sujet"), style={"marginTop": "10px", "marginBottom": "10px"}),
                html.Ul([
                    html.Li(html.A(
                        "L'incendie de l'église Saint-Paul emporte en fumée une page d'histoire",
                        href="https://www.journaldemontreal.com/2026/02/23/lincendie-de-leglise-saint-paul-apporte-en-fumee-une-page-dhistoire",
                        target="_blank",
                        style={"color": "var(--accent)", "textDecoration": "none"}
                    )),
                    html.Li(html.A(
                        "Un incendie sur le boulevard Saint-Laurent",
                        href="https://www.lapresse.ca/actualites/justice-et-faits-divers/2026-04-06/montreal/un-incendie-sur-le-boulevard-saint-laurent.php",
                        target="_blank",
                        style={"color": "var(--accent)", "textDecoration": "none"}
                    )),
                ], style={"listStyleType": "none", "padding": 0, "lineHeight": "1.8"})
            ], className="section reveal delay-4")
        ], style={"backgroundColor": "transparent", "padding": "0"})
    ])


    @app.callback(Output("map-montreal", "figure"), Input("map-mode", "value"))
    def update_map(mode):
        return make_map(mode)


    @app.callback(Output("side-panel", "children"), Input("map-montreal", "clickData"))
    def update_side_panel(click_data):
        zone = selected_zone(click_data)
        row = zone_stats[zone_stats["zone"] == zone].iloc[0]

        pop_text = "Non disponible" if pd.isna(row["Population"]) else f"{format_int(row['Population'])} habitants"
        ratio_text = "Non disponible" if pd.isna(row["Ratio"]) else f"{row['Ratio']:.2f} interventions / 1k hab."

        return [
            html.H2(zone),
            html.P([html.B("Casernes : "), str(int(row["Nb_Casernes"]))]),
            html.Hr(),
            html.P([html.B("Nombre d'interventions : "), format_int(row["Incendies"])]),
            html.P([html.B("Population moyenne : "), pop_text]),
            html.P([html.B("Indice de risque annuel : "), ratio_text]),
            html.H4("Répartition des 10 types de feux principaux :"),
            dcc.Graph(figure=make_detail_bar(zone), config=GRAPH_CONFIG, style={"height": "305px"}),
        ]


    @app.callback(Output("chrono-chart", "figure"), Input("time-selector", "value"))
    def update_chrono(scale):
        return make_temporal(scale)


if __name__ == "__main__":
    app.run(debug=False)