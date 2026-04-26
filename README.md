# Titre du projet

Projet de visualisation de données du cours INF8808. Porte sur des visualisations de la répartition d'incendies et d'interventions sur lîle de Montréal.

## Description

Ce projet analyse les interventions du Service de sécurité incendie de Montréal (SIM) entre 2020 et 2024, à partir de données ouvertes.

L’objectif est de permettre au grand public de comprendre :

- la répartition spatiale des incendies sur l’île de Montréal
- leur évolution temporelle (années, saisons, heures)
- les types d’incidents les plus fréquents
- la relation entre les incendies et la localisation des casernes

Le projet prend la forme d’un tableau de bord interactif permettant d’explorer ces dimensions via plusieurs visualisations complémentaires (carte, bar charts, heatmap, etc.).

## Getting Started

### Dépendances

- Python 3.11
- Dash
- Plotly
- Pandas
- Navigateur web (Chrome recommandé)

### Installation

- Cloner le repository GitHub du projet
- Se placer dans le dossier du projet
- Installer les dépendances avec :
`pip install -r requirements.txt`

### Execution du programme

- Lancer l’application avec :
`python app.py`
- Ouvrir un navigateur web à l’adresse suivante :
`http://127.0.0.1:8050/`

## Auteurs

 Ali Karaki
 (ali.karaki@etud.polymtl.ca)

 Paul Besse
 (paul.besse@etud.polymtl.ca)

 Robin Holden
 (robin.holden@etud.polymtl.ca)

## Historique des versions

* 0.1
    * Beta (15 Avril 2026)
    * Final release (25 Avril 2026)


## Sources

Template readme de:
* [awesome-readme](https://github.com/matiassingers/awesome-readme)

Inspiration, code snippets, etc.
* [Données ouvertes de la Ville de Montréal](https://donnees.montreal.ca/)
* [Institut de la statistique du Québec (ISQ)](https://statistique.quebec.ca/)
* [Documentation Plotly Dash](https://dash.plotly.com/)
* [Documentation Plotly](https://plotly.com/python/)
* [Documentation Pandas](https://pandas.pydata.org/docs/)
* [StackOverflow (divers snippets)](https://stackoverflow.com/)