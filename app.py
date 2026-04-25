# IMPORTS ET DÉPENDANCES
import pandas as pd
from dash import Dash, Input, Output, dcc, html, MATCH

# Fonctions utilitaires partagées
from utils import load_all_data, format_int, stat_card, GRAPH_CONFIG

# Modules de visualisation spécifiques au projet
from visualizations.neighborhood_analysis import comparison_section
from visualizations.risk_map import map_section_layout, make_map, selected_zone, make_detail_bar
from visualizations.temporal_trends import temporal_section, make_temporal
from visualizations.response_intensity import distribution_section

# CHARGEMENT DES DONNÉES GLOBALES
interventions, casernes, zone_stats, geojson_data, data_error = load_all_data()

# INITIALISATION DE L'APPLICATION
app = Dash(__name__, external_stylesheets=["https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap"])
server = app.server
app.title = "Interventions incendie à Montréal"

# VÉRIFICATION DE L'INTÉGRITÉ DES DONNÉES
if data_error:
    # Affichage d'une page d'erreur de secours si les fichiers CSV/JSON sont manquants
    app.layout = html.Div([
        html.H1("Erreur de chargement des données"),
        html.P(data_error),
        html.P("Vérifiez votre configuration ou la présence des fichiers dans le dossier cleaned_data."),
    ], className="error-page")

else:
    # PRÉPARATION DES CHIFFRES CLÉS
    total_incidents = len(interventions)
    top_zone = zone_stats.sort_values("Incendies", ascending=False).iloc[0]["zone"]
    top_type = interventions["incident_type"].value_counts().index[0]

    # STRUCTURE PRINCIPALE DU DASHBOARD (LAYOUT)
    app.layout = html.Div([
        
        # EN-TÊTE ET TEXTE INTRODUCTIF
        html.Header([
            html.P("INF8808 - FINAL RELEASE", className="eyebrow"),
            html.H1("Les interventions incendie à Montréal"),
            html.P("Évolution temporelle, répartition spatiale et couverture des casernes entre 2020 et 2024.", className="lead"),
            html.P("Par Robin Holden, Ali Karaki et Paul Besse", style={"fontStyle": "italic", "fontSize": "14px", "marginBottom": "15px", "color": "var(--muted)"}),
            html.P("Chaque jour, les pompiers du Service de sécurité incendie de Montréal (SIM) répondent à des centaines d’appels."
            " Mais que se passe-t-il réellement derrière les sirènes ? Cette exploration interactive propose d’analyser les données "
            "l’historique des interventions entre 2020 et 2024 pour comprendre les risques qui touchent l’agglomération de Montréal"
            ", l’évolution des interventions au fil des saisons et l’incroyable logistique nécessaire pour protéger plus de deux"
            " millions de citoyens."),
            
            # Grille des statistiques globales
            html.Div([
                stat_card("Interventions analysées", format_int(total_incidents)),
                stat_card("Secteur le plus touché", top_zone),
                stat_card("Type le plus fréquent", top_type),
                stat_card("Casernes actives", format_int(len(casernes))),
            ], className="stats"),
        ], className="hero reveal"),

        # SECTIONS GRAPHIQUES ET ANALYSES
        comparison_section(interventions),
        map_section_layout(zone_stats, casernes, geojson_data),
        temporal_section(interventions),
        distribution_section(interventions),

        # PIED DE PAGE ET CONCLUSION
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

            # Liens externes et articles connexes
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


    # LOGIQUE INTERACTIVE (CALLBACKS)

    # Mise à jour de la carte principale selon le mode sélectionné (Volume vs Indice)
    @app.callback(Output("map-montreal", "figure"), Input("map-mode", "value"))
    def update_map(mode):
        return make_map(zone_stats, casernes, geojson_data, mode)

    # Actualisation du panneau latéral textuel lors du clic sur un arrondissement
    @app.callback(Output("side-panel", "children"), Input("map-montreal", "clickData"))
    def update_side_panel(click_data):
        zone = selected_zone(zone_stats, click_data)
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
            dcc.Graph(figure=make_detail_bar(zone_stats, zone), config=GRAPH_CONFIG, style={"height": "305px"}),
        ]

    # Changement d'échelle temporelle (Année, Mois, Semaine)
    @app.callback(Output("chrono-chart", "figure"), Input("time-selector", "value"))
    def update_chrono(scale):
        return make_temporal(interventions, scale)
    
    # Gestion du menu déroulant pour les descriptions d'accessibilité
    @app.callback(Output({"type": "toggle-text", "section": MATCH}, "style"),
                  Output({"type": "toggle-button", "section": MATCH}, "children"),
                  Input({"type": "toggle-button", "section": MATCH}, "n_clicks")
    )
    def toggle_details(n_clicks):
        if n_clicks % 2 == 1:
            return {"display": "block"}, " Cacher la description de la visualisation"
        return {"display": "none"}, " Afficher la description de la visualisation"

# LANCEMENT DU SERVEUR
if __name__ == "__main__":
    app.run(debug=False)