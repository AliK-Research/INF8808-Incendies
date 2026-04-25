# IMPORTS ET DÉPENDANCES
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from utils import finish_figure, RED_STEPS, FIRE_COLORS, ordered_type_groups, DAYS, MONTHS, YEARS, graph_card, button_description

# CRÉATION DES GRAPHIQUES TEMPORELS MULTI-ÉCHELLES
def make_temporal(interventions, scale):
    
    # Logique pour la vue hebdomadaire (heatmap)
    if scale == "week":
        heat = interventions.groupby(["day_label", "hour"]).size().reset_index(name="Valeur")
        pivot = heat.pivot(index="day_label", columns="hour", values="Valeur").reindex(DAYS).reindex(columns=range(24)).fillna(0)
        vmax = max(float(pivot.max().max()), 1)
        bins = np.linspace(0, vmax, 6)

        fig = go.Figure(go.Heatmap(
            z=pivot.values, x=[f"{h}h" for h in pivot.columns], y=pivot.index,
            colorscale=RED_STEPS, zmin=0, zmax=vmax, xgap=3, ygap=3,
            colorbar=dict(
                title=dict(text="Nombre d'incidents", side="top"),
                tickmode="array", tickvals=bins, ticktext=[str(int(b)) for b in bins],
                thickness=18, len=0.75, ticks="outside", outlinecolor="black", outlinewidth=1,
            ),
            hovertemplate="Jour: %{y}<br>Heure: %{x}<br>Interventions: %{z}<extra></extra>",
        ))
        
        # Ajustement des axes pour la heatmap
        fig.update_layout(
            title={"text": "Analyse des périodes critiques sur la semaine", "x": 0.5},
            xaxis=dict(title="Heure de la journée", showline=True, linecolor="black", ticks="outside"),
            yaxis=dict(title="Jour de la semaine", autorange="reversed", showline=True, linecolor="black", ticks="outside"),
            margin=dict(l=90, r=95, t=70, b=70),
        )
        return finish_figure(fig, 540)

    # Logique de groupement pour les vues annuelle et mensuelle
    if scale == "year":
        df = interventions.groupby(["year", "type_grouped"]).size().reset_index(name="Valeur")
        title, x_title, order = "Répartition annuelle par type d'incident", "Année", sorted(interventions["year"].unique())
    else:
        df = interventions.groupby(["month", "month_label", "type_grouped"]).size().reset_index(name="Valeur")
        df["Valeur"] = (df["Valeur"] / max(len(YEARS), 1)).round().astype(int)
        df = df.rename(columns={"month_label": "year"})
        title, x_title, order = "Saisonnalité des types d'incidents (Moyenne)", "Mois", MONTHS

    # Construction du graphique à barres empilées
    fig = go.Figure()
    for category in ordered_type_groups(interventions):
        sub = df[df["type_grouped"] == category].copy()
        if scale == "month":
            sub["year"] = pd.Categorical(sub["year"], categories=MONTHS, ordered=True)
            sub = sub.sort_values("year")

        fig.add_trace(go.Bar(
            name=category, x=sub["year"], y=sub["Valeur"],
            marker_color=FIRE_COLORS.get(category),
            hovertemplate="%{x}<br>%{data.name}: %{y}<extra></extra>",
        ))

    # Calcul et affichage du texte global au sommet de chaque barre
    totals = df.groupby("year", observed=False)["Valeur"].sum().reindex(order).dropna().reset_index()
    fig.add_trace(go.Scatter(
        x=totals["year"], y=totals["Valeur"], 
        name="<b>Total de la période</b>",
        mode="text",
        text=totals["Valeur"],
        textposition="top center",
        hovertemplate="Total : %{y}<extra></extra>", 
        showlegend=False
    ))

    # Mise en page finale (légendes et axes)
    fig.update_layout(
        barmode="stack", title={"text": title, "x": 0.5},
        legend_title="Type d'incident", legend=dict(x=1.02, y=1, xanchor="left", yanchor="top"),
        margin=dict(l=70, r=210, t=70, b=70),
        xaxis=dict(title=x_title, type="category", categoryorder="array", categoryarray=order, showline=True, linecolor="black", ticks="outside"),
        yaxis=dict(title="Nombre d'incidents", showline=True, linecolor="black", ticks="outside"),
    )
    return finish_figure(fig, 540)

# STRUCTURE DE LA SECTION HTML
def temporal_section(interventions):
    return html.Section([
        # En-tête et texte explicatif
        html.Div([
            html.H2("Évolution et cycles d'intervention"),
            html.P("Le risque incendie n’est pas statique, il suit le rythme de la ville et des saisons. Cette visualisation "
            "permet de retracer l’historique des interventions par année, par mois ou même par semaine et "
            "par jour.", style={"marginTop": "20px"}),
            html.P([
                html.B("Comment ça marche : "),
                "Les boutons permettent de choisir l’échelle temporelle souhaitée. Les vues « Annuelle » et « Mensuelle » "
                "présentent des graphiques à barres empilées montrant le volume total d'incidents et leur nature. La vue "
                "« Hebdomadaire » utilise une carte de chaleur : plus la case est foncée, plus le nombre d'interventions est "
                "élevé pour cette plage horaire précise."
            ]),
            
            # Boutons radio pour alterner les modes temporels
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
        
        # Intégration du graphique principal (actualisé via callbacks)
        graph_card(graph_id="chrono-chart", figure=make_temporal(interventions, "year"), height=540),
        
        # Bouton d'accessibilité décrivant la visualisation
        button_description(button_id="viz3",
                           description=[
                            html.Br(),"Cette visualisation interactive multi-vues, intitulée « Évolution et cycles d'intervention », permet d'alterner "
                            "entre trois échelles temporelles via des boutons d'options : Annuelle, Mensuelle et Hebdomadaire. ",
                            html.Br(),
                            html.Br(),
                            "•	La vue Annuelle affiche un graphique à barres empilées intitulé « Répartition annuelle par type d'incident ». L'axe horizontal présente "
                            "les années de 2020 à 2024, et l'axe vertical indique le nombre total d'incidents. Chaque barre est segmentée selon les mêmes catégories d'incidents "
                            "et couleurs que la première visualisation et est surmontée du chiffre total pour l'année concernée. ",
                            html.Br(),
                            html.Br(),
                            "•	La vue Mensuelle présente un graphique à barres empilées intitulé « Saisonnalité des types d'incidents (Moyenne) ». L'axe horizontal affiche les mois de janvier à décembre, "
                            "et l'axe vertical montre le nombre moyen d'incidents. La structure de couleurs par type d'incident est identique à la vue annuelle, avec la valeur moyenne sur les 5 ans inscrite au ",
                            "sommet de chaque mois.",
                            html.Br(),
                            html.Br(),
                            "•	La vue Hebdomadaire expose une carte de chaleur (heatmap) intitulée « Analyse des périodes critiques sur la semaine ». L'axe horizontal représente les 24 heures de la journée (de 0h à 23h) "
                            "et l'axe vertical les sept jours de la semaine, de lundi à dimanche. L'intensité des interventions est illustrée par un dégradé de couleurs allant du blanc (0 incident) au rouge foncé (maximum de 214 incidents).",
                            html.Hr(style={"marginTop": "20px", "marginBottom": "20px", "borderTop": "1px solid #ccc"}),
                            ]),

        # Paragraphe d'analyse
        html.P("L'évolution temporelle montre une bonne stabilité du volume global d'interventions sur les cinq dernières "
        "années, malgré un pic notable en 2020 lié à un été rude. Cependant, l'analyse à une maille plus fine révèle une "
        "dynamique saisonnière et quotidienne très marquée. Le nombre d'incidents double presque lors du passage de l'hiver "
        "à l'été, principalement à cause de la chaleur et de la sécheresse favorisant les feux extérieurs. Cela confirme "
        "que le rythme des incidents est étroitement lié à la période de l’année et aux cycles d'activité humaine, qui "
        "est plus élevée entre 13h et 22h. Cette connaissance de la temporalité permet ainsi au SIM d’accroître sa vigilance "
        "en fonction des périodes de vulnérabilité accrue.", style={"marginTop": "20px"}),
    ], className="section reveal delay-3")