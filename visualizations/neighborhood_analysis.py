import plotly.graph_objects as go
import numpy as np
import plotly.express as px
from dash import html
from utils import finish_figure, FIRE_COLORS, graph_card, button_description

def make_type_bar_by_zone(interventions):
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

    if not grouped.empty:
        totals = grouped.groupby("zone")["Nombre"].sum().reindex(zones).fillna(0)
        fig.add_trace(go.Scatter(
            x=totals.index,
            y=totals.values,
            name="<b>Total du secteur</b>",
            mode="text",  # Reste en texte pour le total
            text=totals.values,
            textposition="top center",
            hovertemplate="Total : %{y}<extra></extra>",
            showlegend=False
        ))

    fig.update_layout(
        title_x=0.5,
        barmode="stack",
        legend_title_text="Types d'incidents",
        legend=dict(x=1.02, y=1, xanchor="left", yanchor="top"),
        margin=dict(l=65, r=210, t=65, b=135),
    )

    fig.update_xaxes(tickangle=-45)
    
    return finish_figure(fig, 600)

def comparison_section(interventions):
    return html.Section([
        html.Div([
            html.H2("Quels types d'incendies dominent selon les secteurs ?"),
            html.P("Chaque quartier de Montréal possède sa propre identité, allant des zones denses à d’autres plus verdoyantes"
            ". Cela influence directement la nature des risques. Le diagramme à barres empilées suivant compare les types "
            "d’incendies les plus fréquents d’un secteur à l’autre. Les interventions les plus rares ont été regroupées dans la "
            "catégorie « Autres » pour faciliter la lecture.", style={"marginTop": "20px"}),
            html.P([
                html.B("Comment ça marche : "),
                "Plus une barre est haute, plus il y a eu d’interventions. Les couleurs permettent de voir si un quartier fait "
                "face à plus de feux de bâtiments ou de feux extérieurs."
            ]),
        ], className="section-text"),
        
        graph_card(figure=make_type_bar_by_zone(interventions), height=600),

        button_description(button_id="viz1",
                           description=[
                               html.Br(),
                               "Ce diagramme à barres empilées, intitulé « Incendies par arrondissement et type », " 
                                "présente les arrondissements sur l'axe horizontal et le volume d'incidents sur l'axe vertical. "
                                "Chaque barre se divise en six catégories colorées : déchets (bleu), feux de champ (rouge), "
                                "appels incendie (vert), véhicules extérieurs (violet), bâtiments (orange) et autres (gris). Chaque barre affiche sa"
                                " valeur numérique à son extrémité",
                                html.Hr(style={"marginTop": "20px", "marginBottom": "20px", "borderTop": "1px solid #ccc"}),
                                ]),
        
        html.P("L’arrondissement Ville-Marie domine largement le bilan avec le nombre d’interventions le plus élevé de "
        "la métropole, suivi par les secteurs denses de Mercier-Hochelaga-Maisonneuve et du Plateau-Mont-Royal. Cela "
        "s'explique par son statut de centre-ville où la forte densité d'activités et de population multiplie par conséquent "
        "les risques d’incidents. Les déchets en feu sont la cause majoritaire des interventions dans presque tous les "
        "arrondissements. Par ailleurs, les secteurs verts comme Ahuntsic-Cartierville et Pierrefonds-Roxboro sont "
        "particulièrement touchés par les feux de champ.", style={"marginTop": "20px"}),
    ], className="section reveal delay-1")