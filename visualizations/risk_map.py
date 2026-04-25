# IMPORTS ET DÉPENDANCES
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import dcc, html
from utils import empty_figure, RED_STEPS, graph_card, top_type_columns, button_description

# CRÉATION DE LA CARTE CHOROPLÈTHE
def make_map(zone_stats, casernes, geojson_data, mode="Incendies"):
    if geojson_data is None:
        return empty_figure("Carte indisponible", 610)

    # Détermination de la colonne à afficher selon le bouton sélectionné
    df_map = zone_stats.copy()
    value_col = "Incendies" if mode == "Ratio" and df_map["Ratio"].isna().all() else mode

    # Calcul des limites de l'échelle de couleurs
    raw_max = df_map[value_col].max()
    raw_max = 1 if pd.isna(raw_max) or raw_max <= 0 else raw_max
    v_max = int(np.ceil(raw_max / 100) * 100) if value_col == "Incendies" else float(np.ceil(raw_max))
    bins = np.linspace(0, v_max, 6)
    labels = [str(int(v)) for v in bins] if value_col == "Incendies" else [f"{v:.1f}" for v in bins]

    fig = go.Figure()
    
    # Ajout de la couche des polygones (quartiers colorés)
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
            x=1.02, xanchor="left", len=0.66, thickness=18, ticks="outside",
            tickmode="array", tickvals=bins, ticktext=labels,
            outlinecolor="black", outlinewidth=1, title=dict(text=""),
        ),
        name="Nombre d'incidents",
    ))

    # Préparation et ajout des points noirs pour représenter les casernes
    casernes_geojson = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]}}
            for lon, lat in zip(casernes["longitude"], casernes["latitude"])
        ],
    }

    fig.add_trace(go.Scattermapbox(
        lat=[0], lon=[0], mode="markers",
        marker=dict(size=8, color="black"),
        name="Casernes", hoverinfo="skip",
    ))

    # Configuration du fond de carte mapbox et du centrage
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=9.60,
        mapbox_center={"lat": 45.55, "lon": -73.735},
        mapbox_layers=[{
            "sourcetype": "geojson", "source": casernes_geojson,
            "type": "circle", "color": "black", "circle_radius": 3, "below": "",
        }],
        margin=dict(l=0, r=80, t=15, b=0),
        height=610,
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.78)", bordercolor="black", borderwidth=1),
        paper_bgcolor="white",
        uirevision="constant",
    )
    return fig

# GESTION DES CLICS SUR LA CARTE
def selected_zone(zone_stats, click_data):
    # Identifie le quartier cliqué pour actualiser le panneau latéral
    if click_data and click_data.get("points"):
        point = click_data["points"][0]
        if point.get("customdata") is not None:
            return point["customdata"][0]
        if point.get("location") is not None and "join_key" in zone_stats.columns:
            row = zone_stats[zone_stats["join_key"].eq(point["location"])]
            if not row.empty:
                return row.iloc[0]["zone"]

    # Valeur par défaut si aucun clic ou zone invalide
    if "Ville-Marie" in set(zone_stats["zone"]):
        return "Ville-Marie"
    return zone_stats.sort_values("Incendies", ascending=False).iloc[0]["zone"]

# CRÉATION DU GRAPHIQUE LATÉRAL
def make_detail_bar(zone_stats, zone):
    row = zone_stats[zone_stats["zone"] == zone]
    if row.empty:
        return empty_figure("Aucune donnée pour ce secteur.", 300)

    # Extraction des colonnes correspondantes aux types d'incidents
    values = pd.Series({col: row.iloc[0][col] for col in top_type_columns(zone_stats)}).sort_values()
    
    fig = px.bar(
        x=values.values, y=values.index, orientation="h",
        text=values.values.astype(int), labels={"x": "", "y": ""},
    )

    fig.update_traces(
        marker_color="#4a6984", marker_line_color="white", marker_line_width=1,
        textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Nombre: %{x}<extra></extra>",
    )
    fig.update_layout(
        height=305, margin=dict(l=8, r=45, t=10, b=20),
        showlegend=False, paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=10),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee", range=[0, max(values.max() * 1.25, 10)])
    fig.update_yaxes(automargin=True)
    return fig

# STRUCTURE DE LA SECTION HTML
def map_section_layout(zone_stats, casernes, geojson_data):
    return html.Section([
        # En-tête et texte explicatif
        html.Div([
            html.H2("Géographie des interventions et maillage des services de secours"),
            html.P("La sécurité est également une question de proximité. La carte ci-dessous montre l’emplacement des 64 casernes "
            "de l’île impliquées dans la lutte contre les incendies et le niveau d’activité "
            "de chaque zone.", style={"marginTop": "20px"}),
            html.P([
                html.B("Comment ça marche : "),
                "Il est possible de basculer entre le nombre total d’incidents et l’indice de risque. Ce dernier ajuste "
                "les chiffres en fonction du nombre d’habitants, ce qui rend la comparaison équitable entre un petit "
                "quartier très peuplé et un grand arrondissement moins dense."
            ]),

            html.P("Carte montrant la répartition des feux (2020-2024)", style={"text-align": "center", "marginTop": "40px", "marginBottom": "10px"}),

            # Boutons radio pour alterner les modes de la carte
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
        
        # Grille asymétrique contenant la carte et le panneau de détails
        html.Div([
            graph_card(graph_id="map-montreal", figure=make_map(zone_stats, casernes, geojson_data), height=610, class_name="map-card"),
            html.Div(id="side-panel", className="side-panel"),
        ], className="map-layout"),
        
        # Bouton d'accessibilité décrivant la visualisation
        button_description(button_id="viz2",
                           description=[
                                html.Br(), 
                                "Cette carte choroplèthe interactive, intitulée « Carte montrant la répartition des feux "
                                "(2020-2024) », présente les secteurs de l'agglomération de Montréal colorés selon l'intensité des interventions. "
                                "Un dégradé de couleurs allant du blanc au rouge foncé indique, selon le mode choisi, le volume d'incidents ou l'indice "
                                "de risque annuel. Des points noirs superposés sur la carte marquent l'emplacement géographique des casernes de pompiers. "
                                "À droite de la carte, un panneau latéral dynamique affiche les informations du secteur sélectionné. Il affiche le nom du quartier, "
                                "le nombre de casernes actives, le volume total d'interventions sur 5 ans et l'indice de risque annuel (ratio par 1 000 habitants).\n "
                                "En dessous, un graphique à barres horizontales détaille la répartition des types de feux. L'axe vertical énumère les dix catégories "
                                "d'incidents les plus fréquentes comme les feux de bâtiments ou de déchets, tandis que l'axe horizontal indique le nombre d'interventions "
                                "pour chaque catégorie. Chaque barre affiche sa valeur numérique à son extrémité.",
                                html.Hr(style={"marginTop": "20px", "marginBottom": "20px", "borderTop": "1px solid #ccc"}),
                            ]),

        # Paragraphe d'analyse
        html.P("La répartition géographique montre une grande concentration des interventions dans les environs "
        "du centre-ville, comme vu précédemment avec Ville-Marie. Cependant, l'indice de risque annuel apporte "
        "une information supplémentaire importante. En effet, une fois les données rapportées à la population, des "
        "arrondissements moins denses comme Montréal-Est et Senneville révèlent une vulnérabilité importante. Cela "
        "confirme que le risque incendie est à la fois lié à la densité humaine et aux environnements verts, davantage "
        "sujets aux feux de végétation. La répartition des casernes assure ainsi une présence stratégique pour intervenir "
        "rapidement sur l'ensemble du territoire.", style={"marginTop": "20px"}),
    ], className="section reveal delay-2")