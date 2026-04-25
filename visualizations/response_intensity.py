# IMPORTS ET DÉPENDANCES
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import html
from utils import finish_figure, SEVERITY_ORDER, SEVERITY_COLORS, graph_card, button_description

# CRÉATION DU BOXPLOT
def make_boxplot(interventions):
    # Conservation des 15 types d'incidents les plus fréquents
    top_types = interventions["incident_type"].value_counts().head(15).index
    df = interventions[interventions["incident_type"].isin(top_types)].copy()
    df["nombre_unites"] = df["nombre_unites"].fillna(0)

    stats_rows, outlier_rows = [], []

    # Calcul des quartiles, médianes et limites pour chaque boîte
    for incident_type, s in df.groupby("incident_type")["nombre_unites"]:
        s = s.dropna().sort_values()
        if s.empty: continue
        
        q1, median, q3 = s.quantile(0.25), s.quantile(0.50), s.quantile(0.75)
        iqr = q3 - q1
        lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr

        inside_fences = s[(s >= lower_bound) & (s <= upper_bound)]
        lower_fence = inside_fences.min() if not inside_fences.empty else s.min()
        upper_fence = inside_fences.max() if not inside_fences.empty else s.max()

        stats_rows.append({
            "incident_type": incident_type, "min": s.min(), "lower_fence": lower_fence,
            "q1": q1, "median": median, "q3": q3, "upper_fence": upper_fence, "max": s.max(),
        })

        # Isolement des outliers
        outliers = s[(s < lower_fence) | (s > upper_fence)]
        if not outliers.empty:
            low_ex, high_ex = outliers.min(), outliers.max()
            for value in outliers:
                label = "Outlier extrême bas" if value == low_ex else ("Outlier extrême haut" if value == high_ex else "")
                outlier_rows.append({"incident_type": incident_type, "nombre_unites": value, "label": label})

    stats_df = pd.DataFrame(stats_rows).sort_values("median", ascending=False)
    order = stats_df["incident_type"].tolist()

    # Construction du graphique avec les boîtes horizontales
    fig = go.Figure()
    for row in stats_df.itertuples(index=False):
        fig.add_trace(go.Box(
            orientation="h", y0=row.incident_type, q1=[row.q1], median=[row.median], q3=[row.q3],
            lowerfence=[row.lower_fence], upperfence=[row.upper_fence], boxpoints=False,
            line=dict(color="#b7380d"), fillcolor="rgba(183, 56, 13, 0.35)", showlegend=False, name="",
            customdata=[[row.min, row.lower_fence, row.q1, row.median, row.q3, row.upper_fence, row.max, row.incident_type]],
            hovertemplate="<b>%{customdata[7]}</b><br>min: %{customdata[0]:.0f}<br>lower fence: %{customdata[1]:.0f}<br>q1: %{customdata[2]:.0f}<br>median: %{customdata[3]:.0f}<br>q3: %{customdata[4]:.0f}<br>upper fence: %{customdata[5]:.0f}<br>max: %{customdata[6]:.0f}<extra></extra>",
        ))

    # Ajout des outliers par-dessus les boîtes
    if outlier_rows:
        for incident_type, g in pd.DataFrame(outlier_rows).groupby("incident_type", sort=False):
            fig.add_trace(go.Scatter(
                x=g["nombre_unites"], y=[incident_type] * len(g), mode="markers",
                marker=dict(color="#b7380d", size=8), showlegend=False, name="",
                customdata=g[["label"]].values, hovertemplate="%{x:.0f}<br>%{customdata[0]}<extra></extra>",
            ))

    # Mise en page finale et configuration de l'infobulle
    fig.update_layout(
        title="Distribution des unités mobilisées par type d'incident", title_x=0.5,
        xaxis_title="Nombre d'unités (intensité)", yaxis_title="Type d'incident",
        showlegend=False, margin=dict(l=215, r=35, t=65, b=65), hovermode="y unified",
    )
    fig.update_yaxes(categoryorder="array", categoryarray=order, autorange="reversed")
    return finish_figure(fig, 580)


# CRÉATION DES WAFFLE CHARTS
def make_waffle(interventions):
    df = interventions[interventions["zone"] != "Indéterminé"].copy()
    zones = sorted(df["zone"].unique())
    n_cols = 4
    n_rows = int(np.ceil(len(zones) / n_cols))

    # Initialisation de la grille de sous-graphiques
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=zones, horizontal_spacing=0.02, vertical_spacing=0.04)

    # Ajout d'une trace invisible pour générer la légende
    for category, color in zip(SEVERITY_ORDER, SEVERITY_COLORS):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", marker=dict(size=12, color=color, symbol="square"),
            name=category, showlegend=True,
        ))

    color_index = {cat: i for i, cat in enumerate(SEVERITY_ORDER)}
    colorscale = [[i / (len(SEVERITY_COLORS) - 1), color] for i, color in enumerate(SEVERITY_COLORS)]

    # Calcul et remplissage de la martrice 10x10 pour chaque quartier
    for i, zone in enumerate(zones):
        part = df[df["zone"] == zone]
        cells = (part["severity"].value_counts(normalize=True).reindex(SEVERITY_ORDER, fill_value=0) * 100).round().astype(int)
        
        # Ajustement des arrondis pour garantir exactement 100 cases
        diff = 100 - int(cells.sum())
        if diff: cells.loc[cells.idxmax()] += diff

        values = []
        for category in reversed(SEVERITY_ORDER):
            values.extend([category] * max(int(cells.get(category, 0)), 0))
        values = (values + [SEVERITY_ORDER[0]] * 100)[:100]
        matrix = np.array(values).reshape(10, 10)

        # Ajout du gaufrier en tant que heatmap
        fig.add_trace(go.Heatmap(
            z=[[color_index[c] for c in row] for row in matrix], text=matrix,
            hovertemplate="%{text}<extra></extra>", colorscale=colorscale, zmin=0, zmax=len(SEVERITY_COLORS) - 1,
            showscale=False, xgap=1, ygap=1,), row=i // n_cols + 1, col=i % n_cols + 1)

    # Configuration de la grille globale et suppression des axes superflus
    fig.update_layout(
        title_text="Gravité des interventions par arrondissement", title_x=0.5,
        height=max(900, 220 * n_rows), margin=dict(t=100, b=40, l=40, r=210),
        legend=dict(title="Gravité", x=1.01, y=1), template="plotly_white",
        paper_bgcolor="white", plot_bgcolor="white", font=dict(family="Arial, sans-serif", size=11),
    )
    fig.update_annotations(font_size=10)
    
    for i in range(1, len(zones) + 1):
        axis_name = f"x{i if i > 1 else ''}"
        fig.update_yaxes(scaleanchor=axis_name, scaleratio=1, row=(i - 1) // n_cols + 1, col=(i - 1) % n_cols + 1)

    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, autorange="reversed")
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    return fig


# STRUCTURE DE LA SECTION HTML
def distribution_section(interventions):
    # Calcul dynamique de la hauteur nécessaire pour les gaufriers
    waffle_height = max(900, 220 * int(np.ceil(len(interventions[interventions["zone"] != "Indéterminé"]["zone"].unique()) / 4)))
    
    return html.Section([
        # En-tête et texte explicatif
        html.Div([
            html.H2("Combien d'unités sont mobilisées par type d'incident et par arrondissement ?"),
            html.P("Toutes les interventions ne se ressemblent pas. En effet, l’effort déployé change radicalement selon "
            "que les pompiers font face à une poubelle en feu ou un incendie de bâtiment par exemple. Le diagramme en boîtes "
            "montre le nombre habituel d’unités déployées pour chaque type d’appel. Le graphique en gaufrier situé en "
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
        
        # Intégration des graphiques
        graph_card(figure=make_boxplot(interventions), height=580),
        graph_card(figure=make_waffle(interventions), height=waffle_height),
        
        # Bouton d'accessibilité décrivant les visualisations
        button_description(button_id="viz4",
                           description=[
                               html.Br(),
                               "•   Le premier graphique est un diagramme en boîtes (boxplot) horizontal intitulé « Distribution des unités mobilisées "
                                "par type d'incident ». L'axe vertical représente les différents types d'incidents (du plus lourd au plus léger), tandis que l'axe horizontal "
                                "représente le nombre d'unités mobilisées. Pour chaque type d'incident, une boîte illustre la mobilisation médiane et les quartiles, complétée "
                                "par des points isolés vers la droite marquant les interventions d'une ampleur exceptionnelle (outliers).",
                                html.Br(),
                                html.Br(),
                                "•  Le second graphique est un ensemble de 33 graphiques en gaufrier regroupés sous le titre « Gravité des interventions par arrondissement ». "
                                "Chaque gaufrier représente un secteur spécifique et se présente sous la forme d'une grille de 10x10 carrés, où chaque carré équivaut à 1 % des "
                                "interventions du secteur. La répartition est illustrée par cinq niveaux de gravité utilisant un dégradé de couleurs : Mineure <= 1 unité (rose très clair), "
                                "Standard <= 5 unités (rose), Confirmée <= 15 unités (rouge), Majeure <= 30 unités (rouge foncé) et Critique > 30 unités (noir).",
                                html.Hr(style={"marginTop": "20px", "marginBottom": "20px", "borderTop": "1px solid #ccc"}),
                                ]),
 
        # Paragraphes d'analyse
        html.P("L'analyse de la mobilisation montre que la vaste majorité des incidents quotidiens, tels que les feux de "
        "déchets ou de broussailles, sont maîtrisés avec seulement une à deux unités. Cependant, dès que l'on passe aux "
        "alertes de niveau supérieur comme les incendies de bâtiments confirmés, l'effort augmente pour atteindre souvent "
        "plus de 15, voire 60 unités pour les cas les plus critiques.", style={"marginTop": "20px"}),
        html.P("Le portrait par quartier montre que les interventions de routine, « Mineure » ou « Standard », constituent "
        "la plus grande partie de l'activité partout sur l'île. On note toutefois que des secteurs denses ou "
        "institutionnels, comme Ville-Marie ou Westmount, affichent une proportion de carrés foncés légèrement plus "
        "élevée, exigeant une mobilisation massive d'unités spécialisées.", style={"marginTop": "10px"}),
    ], className="section reveal delay-4")