import pandas as pd
import numpy as np
import re
import json
import unicodedata
from pathlib import Path

# CONFIGURATION INITIALE
INPUT_DIR = Path("data")
OUTPUT_DIR = Path("cleaned_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = list(range(2020, 2025))
MONTHS = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sept", "Oct", "Nov", "Déc"]
DAYS = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
SEVERITY_ORDER = ["Mineure <= 1 unité", "Standard <= 5 unités", "Confirmée <= 15 unités", "Majeure <= 30 unités", "Critique > 30 unités"]

# FONCTIONS UTILITAIRES
# Standardise les chaînes de caractères pour faciliter les jointures exactes entre fichiers
def normalize_key(value):
    value = "" if pd.isna(value) else str(value)
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").upper()
    value = value.replace("SAINTE", "ST").replace("SAINT", "ST")
    value = re.sub(r"\b(LE|LA|LES|DES|DE|DU|D|L|VILLE)\b", "", value)
    return re.sub(r"[^A-Z0-9]", "", value)

def clean_text(series):
    return series.fillna("").astype(str).str.replace("\uFFFD", "-", regex=False).str.strip()

# Génère une signature simplifiée pour comparer les noms de quartiers malgré les écarts de nomenclature
def get_clean_signature(text):
    if pd.isna(text): return ""
    text = str(text).upper()
    text = re.sub(r'\b(LE|LA|LES|DES|DE|DU|AU|AUX|ET|SAINT|SAINTE|VILLE)\b', '', text)
    return re.sub(r'[^A-Z]', '', text)

# Découpe le texte en petits segments (n-grammes) pour le calcul de similarité textuelle
def get_ngrams(text, n=4):
    return {text[i:i+n] for i in range(len(text) - n + 1)}


# TRAITEMENT DES INTERVENTIONS
raw_interventions = pd.read_csv(INPUT_DIR / "interventions.csv", low_memory=False)

# Uniformisation des noms de colonnes pour éviter les erreurs de casse
raw_interventions.columns = [c.upper() for c in raw_interventions.columns]

df_int = raw_interventions.copy()
date_col = "CREATION_DATE_TIME" if "CREATION_DATE_TIME" in df_int.columns else "DATE"
df_int["date"] = pd.to_datetime(df_int[date_col], errors="coerce")
df_int = df_int[df_int["date"].dt.year.isin(YEARS)].copy()

# Filtrage pour ne conserver que les incendies réels (exclusion des fausses alertes, etc.)
groups = clean_text(df_int["DESCRIPTION_GROUPE" if "DESCRIPTION_GROUPE" in df_int.columns else "GROUPE"]).str.upper()
df_int = df_int[groups.isin(["INCENDIE", "AUTREFEU"])].copy()

# Nettoyage et définition de la zone géographique
df_int["NOM_ARROND"] = df_int["NOM_ARROND"].replace('Indéterminé', pd.NA) if "NOM_ARROND" in df_int.columns else pd.NA
df_int["zone"] = df_int["NOM_ARROND"].fillna(df_int["NOM_VILLE"]).str.strip()

df_int["incident_type"] = clean_text(df_int["INCIDENT_TYPE_DESC"])
df_int["caserne"] = pd.to_numeric(df_int["CASERNE"], errors="coerce")
df_int["nombre_unites"] = pd.to_numeric(df_int["NOMBRE_UNITES"], errors="coerce")
df_int["latitude"] = pd.to_numeric(df_int["LATITUDE"], errors="coerce")
df_int["longitude"] = pd.to_numeric(df_int["LONGITUDE"], errors="coerce")

# Création des variables temporelles pour l'analyse chronologique
dt = pd.DatetimeIndex(df_int["date"])
df_int["year"] = dt.year.astype(str)
df_int["month"] = dt.month
df_int["month_label"] = df_int["month"].map(dict(enumerate(MONTHS, start=1)))
df_int["weekday"] = dt.weekday
df_int["day_label"] = df_int["weekday"].map(dict(enumerate(DAYS)))
df_int["hour"] = dt.hour

# Regroupement des types d'incidents (Top 5 + Autres) et catégorisation de la gravité
top5 = df_int["incident_type"].value_counts().head(5).index
df_int["type_grouped"] = np.where(df_int["incident_type"].isin(top5), df_int["incident_type"], "Autres")
df_int["severity"] = pd.cut(df_int["nombre_unites"].fillna(0), bins=[-np.inf, 1, 5, 15, 30, np.inf], labels=SEVERITY_ORDER).astype(str)

df_int = df_int.dropna(subset=["date"])

col_int_finales = ["date", "incident_type", "caserne", "zone", "nombre_unites", "latitude", "longitude", 
                   "year", "month", "month_label", "weekday", "day_label", "hour", "type_grouped", "severity"]
df_int[col_int_finales].to_csv(OUTPUT_DIR / "cleaned_interventions.csv", index=False)
print('Fichier "cleaned_interventions.csv" enregistré dans le dossier "cleaned_data"')


# TRAITEMENT DES CASERNES
raw_casernes = pd.read_csv(INPUT_DIR / "casernes.csv", low_memory=False)
raw_casernes.columns = [c.upper() for c in raw_casernes.columns]

df_cas = raw_casernes.copy()

# Exclusion des casernes inactives (2 et 79)
df_cas["caserne"] = pd.to_numeric(df_cas["CASERNE"], errors="coerce")
casernes_actives = df_int["caserne"].unique()
df_cas = df_cas[df_cas["caserne"].isin(casernes_actives)].copy()

df_cas["ZONE_OFFICIELLE"] = df_cas["ARRONDISSEMENT"].fillna(df_cas["VILLE"]).str.strip()
df_cas["caserne"] = pd.to_numeric(df_cas["CASERNE"], errors="coerce")
df_cas["latitude"] = pd.to_numeric(df_cas["LATITUDE"], errors="coerce")
df_cas["longitude"] = pd.to_numeric(df_cas["LONGITUDE"], errors="coerce")

df_cas = df_cas.dropna(subset=["latitude", "longitude"])
df_cas[["caserne", "latitude", "longitude", "ZONE_OFFICIELLE"]].to_csv(OUTPUT_DIR / "cleaned_casernes.csv", index=False)
print('Fichier "cleaned_casernes.csv" enregistré dans le dossier "cleaned_data"')


# LOGIQUE DE CORRESPONDANCE APPROXIMATIVE
# Associe les noms de zones des interventions avec ceux des casernes en calculant un score de similarité (Jaccard)
incendie_names = df_int['zone'].unique()
caserne_names = df_cas['ZONE_OFFICIELLE'].unique()
mapping_fuzzy = {}

for name_inc in incendie_names:
    sig_inc = get_clean_signature(name_inc)
    chunks_inc = get_ngrams(sig_inc, n=4)
    best_match = None
    max_jaccard = 0
    for name_cas in caserne_names:
        sig_cas = get_clean_signature(name_cas)
        chunks_cas = get_ngrams(sig_cas, n=4)
        intersection = chunks_inc.intersection(chunks_cas)
        union = chunks_inc.union(chunks_cas)
        if not union: continue
        jaccard_score = len(intersection) / len(union)
        if sig_inc == sig_cas and sig_inc != "":
            jaccard_score = 1.0 
        if jaccard_score > max_jaccard:
            max_jaccard = jaccard_score
            best_match = name_cas
    if best_match and max_jaccard > 0.35:
        mapping_fuzzy[name_inc] = best_match.strip()


# AGRÉGATION DES STATISTIQUES PAR ZONE
# Calcul du nombre total d'incendies et détail du top 10 par arrondissement
df_total = df_int.groupby('zone').size().reset_index(name='Incendies')
top_10_types = df_int['incident_type'].value_counts().head(10).index
df_feux_top10 = df_int[df_int['incident_type'].isin(top_10_types)]
df_pivot = df_feux_top10.pivot_table(index='zone', columns='incident_type', aggfunc='size', fill_value=0).reset_index()

stats = pd.merge(df_total, df_pivot, on='zone', how='left').fillna(0)

# Ajout du nombre de casernes par secteur en utilisant le mapping
df_nb_casernes = df_cas.groupby('ZONE_OFFICIELLE').size().reset_index(name='Nb_Casernes')
stats['ZONE_POUR_CASERNE'] = stats['zone'].map(mapping_fuzzy)
stats = pd.merge(stats, df_nb_casernes, left_on='ZONE_POUR_CASERNE', right_on='ZONE_OFFICIELLE', how='left')
stats['Nb_Casernes'] = stats['Nb_Casernes'].fillna(0).astype(int)
stats = stats.drop(columns=['ZONE_OFFICIELLE', 'ZONE_POUR_CASERNE'])


# EXTRAPOLATION DE LA POPULATION
df_pop = pd.read_excel(INPUT_DIR / "population.xlsx")
df_pop['ZONE_POP'] = df_pop['Arrondissement'].fillna(df_pop['Ville']).str.strip()

# Extrapolation basée sur la croissance globale de Montréal pour lisser les données manquantes
mask_total_mtl = (df_pop['Ville'] == 'Montréal') & (df_pop['Arrondissement'].isna())
row_total_mtl = df_pop[mask_total_mtl].iloc[0]
annees = ['2020', '2021', '2022', '2023', '2024']

for annee in annees:
    facteur = row_total_mtl[f'Population {annee}'] / row_total_mtl['Population 2025']
    df_pop[f'Population {annee}'] = df_pop[f'Population {annee}'].fillna(df_pop['Population 2025'] * facteur)

df_pop['Population'] = df_pop[[f'Population {a}' for a in annees]].mean(axis=1)

# Intégration de la population et calcul de l'indice de risque (Ratio)
stats = pd.merge(stats, df_pop[['ZONE_POP', 'Population']], left_on='zone', right_on='ZONE_POP', how='left')
stats['Ratio'] = round(((stats['Incendies'] / 5) / stats['Population']) * 1000, 2)
stats = stats.drop(columns=['ZONE_POP'])


# PRÉPARATION GEOJSON POUR LA CARTE
# S'assure que les clés correspondent parfaitement aux noms du GeoJSON
try:
    with open(INPUT_DIR / "limites-administratives-agglomeration.geojson", encoding="utf-8") as file:
        geojson = json.load(file)
    available_geo = {}
    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        name = props.get("NOM") or props.get("nom") or props.get("LIBELLE") or props.get("MUNICIPALITE")
        if not name:
            name = next((v for v in props.values() if isinstance(v, str)), None)
        key = normalize_key(name)
        available_geo[key] = name

    # Corrections manuelles pour les secteurs présentant des différences d'écriture
    manual_map = {
        normalize_key("Rivière-des-Prairies-Pointe-aux-Trembles"): normalize_key("Rivière-des-Prairies - Pointe-aux-Trembles"),
        normalize_key("Côte-des-Neiges-Notre-Dame-de-Grâce"): normalize_key("Côte-des-Neiges - Notre-Dame-de-Grâce"),
        normalize_key("Villeray-Saint-Michel-Parc-Extension"): normalize_key("Villeray - Saint-Michel - Parc-Extension"),
        normalize_key("Mercier-Hochelaga-Maisonneuve"): normalize_key("Mercier - Hochelaga-Maisonneuve"),
        normalize_key("Rosemont-La Petite-Patrie"): normalize_key("Rosemont - La Petite-Patrie"),
        normalize_key("L'Île-Bizard-Sainte-Geneviève"): normalize_key("L'Île-Bizard - Sainte-Geneviève"),
    }
    stats["join_key"] = stats["zone"].map(normalize_key).replace(manual_map)
    stats["geo_name"] = stats["join_key"].map(available_geo)
except Exception as e:
    # Impression d'une erreur si le GeoJSON manque ou est corrompu
    print(f"Attention, GeoJSON non traité : {e}")

# Sauvegarde du fichier final
stats.to_csv(OUTPUT_DIR / "merged_zone_stats.csv", index=False)
print('Fichier "merged_zone_stats.csv" enregistré dans le dossier "cleaned_data"')