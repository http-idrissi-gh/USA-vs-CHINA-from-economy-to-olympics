"""
Script de visualisation comparative des données USA vs Chine aux JO
Auteurs : Théo JANSSENS - Mohamed IDRISSI GHALMI - Youssouf Diakite
Date : 19/04/2025
"""

# ======================================================================
# 1. Import des librairies
# ======================================================================
import numpy as np
import pandas as pd
from bokeh.plotting import figure, show
from bokeh.models import (ColumnDataSource, HoverTool, Tabs, TabPanel, 
                         WMTSTileSource, NumeralTickFormatter, Div, FactorRange)
from bokeh.layouts import column, row
from bokeh.io import output_file
from bokeh.transform import dodge
from bokeh.palettes import Category10


# ======================================================================
# 2. Chargement des données de base
# ======================================================================
donnees_athletes = pd.read_csv('athlete_events.csv')
condition_compare = donnees_athletes['NOC'].isin(['USA', 'CHN'])
donnees_compare = donnees_athletes[condition_compare].copy()

# ======================================================================
# 3. Visualisation 1: Médailles par sport
# ======================================================================

# --------------------------
# 3.1 Préparation des données
# --------------------------
medailles_par_sport_pays = donnees_compare[donnees_compare['Medal'].notna()].groupby(
    ['Sport', 'NOC', 'Medal']).size().unstack(level=['NOC', 'Medal'], fill_value=0)

medailles_par_sport_pays.columns = ['_'.join(col).strip() for col in medailles_par_sport_pays.columns.values]
medailles_par_sport_pays['Total'] = medailles_par_sport_pays.sum(axis=1)
medailles_par_sport_pays = medailles_par_sport_pays.sort_values('Total', ascending=False).head(10)

sports = medailles_par_sport_pays.index.tolist()
nocs = ['USA', 'CHN']
medals = ['Gold', 'Silver', 'Bronze']
colors = {'Gold': '#FFD700', 'Silver': '#C0C0C0', 'Bronze': '#CD7F32'}

data_bar = {'Sport': sports}
for noc in nocs:
    for medal in medals:
        col_name = f'{noc}_{medal}'
        data_bar[col_name] = medailles_par_sport_pays.get(col_name, 0)

source_medailles = ColumnDataSource(data=data_bar)

# --------------------------
# 3.2 Création du plot
# --------------------------
figure_medailles_par_sport = figure(
    x_range=FactorRange(*sports),
    height=450,
    width=900,
    title="Médailles par Sport: USA vs Chine (Top 10 Sports Combinés)",
    toolbar_location="above",
    tools="pan,wheel_zoom,box_zoom,reset,save",
    x_axis_label="Sport",
    y_axis_label="Nombre de médailles"
)

# Paramètres
base_offsets = {"USA": -0.2, "CHN": 0.2}
medailles_offsets = {"Gold": -0.08, "Silver": 0, "Bronze": 0.08}
width = 0.15

# Générer dynamiquement les barres et hovertools
for pays, base_offset in base_offsets.items():
    for medaille, sub_offset in medailles_offsets.items():
        col = f"{pays}_{medaille}"
        total_offset = base_offset + sub_offset
        legende = f"{pays} - {medaille.capitalize()}"
        color = colors[medaille]

        vbar = figure_medailles_par_sport.vbar(
            x=dodge('Sport', total_offset, range=figure_medailles_par_sport.x_range),
            top=col, width=width, source=source_medailles,
            color=color, legend_label=legende, name=col
        )

        hover = HoverTool(
            renderers=[vbar],
            tooltips=[
                ("Sport", "@Sport"),
                (legende, f"@{col}")
            ]
        )
        figure_medailles_par_sport.add_tools(hover)

figure_medailles_par_sport.x_range.range_padding = 0.1
figure_medailles_par_sport.xgrid.grid_line_color = None
figure_medailles_par_sport.xaxis.major_label_orientation = 1.2
figure_medailles_par_sport.legend.location = "top_right"
figure_medailles_par_sport.legend.orientation = "vertical"
figure_medailles_par_sport.legend.click_policy = "hide"

# ======================================================================
# 4. Visualisation 2: Distribution par âge
# ======================================================================

# --------------------------
# 4.1 Préparation des données
# --------------------------
data_age_sport = donnees_compare[['ID', 'Sport', 'Age', 'NOC']].copy()
data_age_sport['Age'] = pd.to_numeric(data_age_sport['Age'], errors='coerce').dropna().astype(int)

top_sports_age = data_age_sport.groupby('Sport')['ID'].nunique().sort_values(ascending=False).head(10).index.tolist()
data_boxplot_base = data_age_sport[data_age_sport['Sport'].isin(top_sports_age)]

def calculate_boxplot_stats(df_grouped_by_sport):
    q1 = df_grouped_by_sport['Age'].quantile(q=0.25)
    q2 = df_grouped_by_sport['Age'].quantile(q=0.50)
    q3 = df_grouped_by_sport['Age'].quantile(q=0.75)
    iqr = q3 - q1

    lower_whiskers = []
    upper_whiskers = []

    def whisker_bounds(x, q1_val, q3_val, iqr_val):
        lower_bound = q1_val - 1.5 * iqr_val
        upper_bound = q3_val + 1.5 * iqr_val
        valid_lower = x[x >= lower_bound]
        valid_upper = x[x <= upper_bound]
        lower_whisker = valid_lower.min() if not valid_lower.empty else q1_val
        upper_whisker = valid_upper.max() if not valid_upper.empty else q3_val
        return lower_whisker, upper_whisker

    q1 = q1.reindex(top_sports_age)
    q2 = q2.reindex(top_sports_age)
    q3 = q3.reindex(top_sports_age)
    iqr = iqr.reindex(top_sports_age)

    for sport in top_sports_age:
        try:
            ages = df_grouped_by_sport.get_group(sport)['Age']
            q1_val = q1.loc[sport]
            q3_val = q3.loc[sport]
            iqr_val = iqr.loc[sport]

            if pd.notna(q1_val) and pd.notna(q3_val) and pd.notna(iqr_val):
                lw, uw = whisker_bounds(ages, q1_val, q3_val, iqr_val)
                lower_whiskers.append(lw)
                upper_whiskers.append(uw)
            else:
                lower_whiskers.append(np.nan)
                upper_whiskers.append(np.nan)
        except KeyError:
             lower_whiskers.append(np.nan)
             upper_whiskers.append(np.nan)

    stats_df = pd.DataFrame({
        'Sport': top_sports_age,
        'q1': q1.values,
        'q2': q2.values,
        'q3': q3.values,
        'lower': lower_whiskers,
        'upper': upper_whiskers
    }).dropna()

    return stats_df, stats_df['Sport'].tolist()

grouped_usa = data_boxplot_base[data_boxplot_base['NOC'] == 'USA'].groupby('Sport')
grouped_chn = data_boxplot_base[data_boxplot_base['NOC'] == 'CHN'].groupby('Sport')

boxplot_stats_usa, valid_sports_usa = calculate_boxplot_stats(grouped_usa)
boxplot_stats_chn, valid_sports_chn = calculate_boxplot_stats(grouped_chn)

common_valid_sports = sorted(list(set(valid_sports_usa) & set(valid_sports_chn)), key=top_sports_age.index)
boxplot_stats_usa = boxplot_stats_usa[boxplot_stats_usa['Sport'].isin(common_valid_sports)]
boxplot_stats_chn = boxplot_stats_chn[boxplot_stats_chn['Sport'].isin(common_valid_sports)]

source_boxes_usa = ColumnDataSource(boxplot_stats_usa)
source_boxes_chn = ColumnDataSource(boxplot_stats_chn)

# --------------------------
# 4.2 Création des plots
# --------------------------
def create_boxplot_figure(source, title, x_range, color_upper, color_lower):
    p = figure(
        x_range=x_range, 
        height=500,
        width=450, 
        title=title,
        x_axis_label="Sport",
        y_axis_label="Âge",
        toolbar_location=None, 
        tools=""
    )
    # Tiges
    p.segment('Sport', 'upper', 'Sport', 'q3', source=source, line_color="black")
    p.segment('Sport', 'lower', 'Sport', 'q1', source=source, line_color="black")
    # Boîtes
    box_width = 0.7
    p.vbar(x='Sport', width=box_width, top='q3', bottom='q2', source=source, 
           fill_color=color_upper, line_color="black")
    p.vbar(x='Sport', width=box_width, top='q2', bottom='q1', source=source, 
           fill_color=color_lower, line_color="black")
    # Moustaches
    whisker_width = 0.3
    p.rect('Sport', 'lower', whisker_width, 0.01, source=source, 
           fill_color="black", line_color="black")
    p.rect('Sport', 'upper', whisker_width, 0.01, source=source, 
           fill_color="black", line_color="black")
    # Outil Hover
    hover = HoverTool(tooltips=[
        ("Q1", "@q1{(0)}"), 
        ("Médiane", "@q2{(0)}"), 
        ("Q3", "@q3{(0)}"),
        ("Limite Sup.", "@upper{(0)}"), 
        ("Limite Inf.", "@lower{(0)}")
    ])
    p.add_tools(hover)
    # Style
    p.xaxis.major_label_orientation = 1.2
    p.x_range.range_padding = 0.1
    p.yaxis.formatter = NumeralTickFormatter(format="0")
    p.grid.grid_line_alpha = 0.3
    return p

fig_boxplot_usa = create_boxplot_figure(
    source_boxes_usa, 
    f"Distribution Âge Athlètes USA (Top {len(common_valid_sports)} Sports)",
    common_valid_sports, 
    Category10[4][0], 
    Category10[4][1]
)

fig_boxplot_chn = create_boxplot_figure(
    source_boxes_chn, 
    f"Distribution Âge Athlètes CHN (Top {len(common_valid_sports)} Sports)",
    common_valid_sports, 
    Category10[4][2], 
    Category10[4][3]
)

# ======================================================================
# 5. Visualisation 3: Carte des villes hôtes
# ======================================================================

# --------------------------
# 5.1 Préparation des données
# --------------------------
city_coords = {
    'London': {'lat': 51.5074, 'lon': -0.1278}, 'Helsinki': {'lat': 60.1699, 'lon': 24.9384},
    'Melbourne': {'lat': -37.8136, 'lon': 144.9631}, 'Roma': {'lat': 41.9028, 'lon': 12.4964},
    'Tokyo': {'lat': 35.6895, 'lon': 139.6917}, 'Mexico City': {'lat': 19.4326, 'lon': -99.1332},
    'Munich': {'lat': 48.1351, 'lon': 11.5820}, 'Montreal': {'lat': 45.5017, 'lon': -73.5673},
    'Moscow': {'lat': 55.7558, 'lon': 37.6173}, 'Los Angeles': {'lat': 34.0522, 'lon': -118.2437},
    'Seoul': {'lat': 37.5665, 'lon': 126.9780}, 'Barcelona': {'lat': 41.3851, 'lon': 2.1734},
    'Atlanta': {'lat': 33.7490, 'lon': -84.3880}, 'Sydney': {'lat': -33.8688, 'lon': 151.2093},
    'Athina': {'lat': 37.9838, 'lon': 23.7275}, 'Beijing': {'lat': 39.9042, 'lon': 116.4074},
    'Vancouver': {'lat': 49.2827, 'lon': -123.1207}, 'Sochi': {'lat': 43.5853, 'lon': 39.7203},
    'Rio de Janeiro': {'lat': -22.9068, 'lon': -43.1729}, 'Oslo': {'lat': 59.9139, 'lon': 10.7522},
    "Cortina d'Ampezzo": {'lat': 46.5404, 'lon': 12.1357}, 'Squaw Valley': {'lat': 39.1970, 'lon': -120.2357},
    'Innsbruck': {'lat': 47.2692, 'lon': 11.4041}, 'Grenoble': {'lat': 45.1885, 'lon': 5.7245},
    'Sapporo': {'lat': 43.0618, 'lon': 141.3545}, 'Lake Placid': {'lat': 44.2795, 'lon': -73.9799},
    'Sarajevo': {'lat': 43.8563, 'lon': 18.4131}, 'Calgary': {'lat': 51.0447, 'lon': -114.0719},
    'Albertville': {'lat': 45.6760, 'lon': 6.3920}, 'Lillehammer': {'lat': 61.1153, 'lon': 10.4660},
    'Nagano': {'lat': 36.6485, 'lon': 138.1951}, 'Salt Lake City': {'lat': 40.7608, 'lon': -111.8910},
    'Torino': {'lat': 45.0703, 'lon': 7.6869}
}

def project_mercator(lat, lon):
    radius = 6378137.0
    x = radius * np.radians(lon)
    y = radius * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2.0))
    return x, y

def get_lat(city): return city_coords.get(city, {}).get('lat')
def get_lon(city): return city_coords.get(city, {}).get('lon')

donnees_compare['latitude'] = donnees_compare['City'].apply(get_lat)
donnees_compare['longitude'] = donnees_compare['City'].apply(get_lon)
donnees_carte = donnees_compare.dropna(subset=['latitude', 'longitude']).copy()
donnees_carte['x'], donnees_carte['y'] = project_mercator(donnees_carte['latitude'], donnees_carte['longitude'])

agg_ville_pays = donnees_carte.groupby(['City', 'x', 'y', 'NOC']).agg(
    count_participants=('ID', pd.Series.nunique),
    gold_medals=('Medal', lambda s: (s == 'Gold').sum()),
    silver_medals=('Medal', lambda s: (s == 'Silver').sum()),
    bronze_medals=('Medal', lambda s: (s == 'Bronze').sum())
).reset_index()

agg_ville_pivot = agg_ville_pays.pivot_table(
    index=['City', 'x', 'y'],
    columns='NOC',
    values=['count_participants', 'gold_medals', 'silver_medals', 'bronze_medals'],
    fill_value=0
)
agg_ville_pivot.columns = ['_'.join(map(str, col)).strip() for col in agg_ville_pivot.columns.values]
agg_ville_pivot.reset_index(inplace=True)

agg_ville_pivot['total_participants'] = agg_ville_pivot.get('count_participants_USA', 0) + agg_ville_pivot.get('count_participants_CHN', 0)
agg_ville_pivot['size'] = np.log1p(agg_ville_pivot['total_participants']) * 3 + 5
source_carte = ColumnDataSource(agg_ville_pivot)

# --------------------------
# 5.2 Création du plot
# --------------------------
osm_tile_source = WMTSTileSource(
    url='https://tile.openstreetmap.org/{Z}/{X}/{Y}.png',
    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
)

figure_carte = figure(
    height=600, 
    width=900,
    title="Villes Hôtes avec Participation USA & Chine",
    x_axis_type="mercator", 
    y_axis_type="mercator",
    toolbar_location="right", 
    tools="pan,wheel_zoom,box_zoom,reset,save"
)
figure_carte.add_tile(osm_tile_source)

figure_carte.circle(
    x='x', 
    y='y', 
    source=source_carte, 
    size='size',
    fill_color="dodgerblue", 
    fill_alpha=0.6, 
    line_color="black", 
    line_width=0.5
)

hover_carte = HoverTool(
    tooltips="""
        <div>
            <span style="font-size: 14px; font-weight: bold;">@City</span><br>
            <hr style="margin: 2px 0;">
            <b>USA:</b><br>
            Participants (uniques): @{count_participants_USA}{0,0}<br>
            Or: @{gold_medals_USA}{0,0} | Arg: @{silver_medals_USA}{0,0} | Bro: @{bronze_medals_USA}{0,0}<br>
            <hr style="margin: 2px 0;">
            <b>CHN:</b><br>
            Participants (uniques): @{count_participants_CHN}{0,0}<br>
            Or: @{gold_medals_CHN}{0,0} | Arg: @{silver_medals_CHN}{0,0} | Bro: @{bronze_medals_CHN}{0,0}
        </div>
    """
)
figure_carte.add_tools(hover_carte)
figure_carte.xaxis.visible = False
figure_carte.yaxis.visible = False
figure_carte.grid.visible = False

# ======================================================================
# 6. Visualisation 4: Évolution des médailles
# ======================================================================

# --------------------------
# 6.1 Préparation des données
# --------------------------
medailles_par_annee_pays = donnees_compare.groupby(['Year', 'NOC', 'Medal']).size().unstack(level=['NOC', 'Medal'], fill_value=0)
medailles_par_annee_pays.columns = ['_'.join(col).strip() for col in medailles_par_annee_pays.columns.values]
medailles_par_annee_pays = medailles_par_annee_pays.reset_index()
source_evolution = ColumnDataSource(medailles_par_annee_pays)

# --------------------------
# 6.2 Création du plot
# --------------------------
figure_evolution = figure(
    title="Évolution des médailles par année (USA vs Chine)",
    x_axis_label='Année', 
    y_axis_label='Nombre de médailles',
    height=400, 
    width=900,
    tools="pan,wheel_zoom,box_zoom,reset,save",
    tooltips=None
)

# Lignes pour USA
figure_evolution.line(
    x='Year', 
    y='USA_Gold', 
    source=source_evolution, 
    line_color='gold', 
    line_width=2, 
    legend_label='USA - Or', 
    name='USA_Gold'
)
figure_evolution.line(
    x='Year', 
    y='USA_Silver', 
    source=source_evolution, 
    line_color='silver', 
    line_width=2, 
    legend_label='USA - Argent', 
    name='USA_Silver'
)
figure_evolution.line(
    x='Year', 
    y='USA_Bronze', 
    source=source_evolution, 
    line_color='#CD7F32', 
    line_width=2, 
    legend_label='USA - Bronze', 
    name='USA_Bronze'
)

# Setup des hovertools
countries = {
    "USA": {
        "colors": {"Gold": "gold", "Silver": "silver", "Bronze": "peru"},
        "dash": "solid"
    },
    
    "CHN": {
        "colors": {"Gold": "red", "Silver": "green", "Bronze": "blue"},
        "dash": "dashed"
    }
}

for country, conf in countries.items():
    for medal, color in conf["colors"].items():
        col_name = f"{country}_{medal}"
        legend = f"{country} - {medal.capitalize()}"
        line = figure_evolution.line(
            x='Year',
            y=col_name,
            source=source_evolution,
            line_color=color,
            line_width=2,
            line_dash=conf["dash"],
            legend_label=legend,
            name=col_name
        )
        hover = HoverTool(
            renderers=[line],
            tooltips=[
                ("Année", "@Year"),
                (legend, f"@{col_name}")
            ]
        )
        figure_evolution.add_tools(hover)

figure_evolution.legend.location = 'top_left'
figure_evolution.legend.click_policy = "hide"
figure_evolution.legend.orientation = "vertical"


# ======================================================================
# 7. Mise en page finale et commentaires
# ======================================================================

# Titre global
titre_global = Div(
    text="<h1>USA vs CHINA, de l'économie aux JO!</h1>",
    width=800,
    align='center'
)

# Contenu des onglets
texte_presentation = Div(text="""
<h3>Comparaison USA vs Chine aux Jeux Olympiques</h3>
<p>Cette application interactive explore et compare les données des athlètes des États-Unis (USA) et de la Chine (CHN) ayant participé aux Jeux Olympiques (JO).</p>
<p>Les données proviennent du fichier <code>athlete_events.csv</code> (source: Kaggle,
<a href="https://www.kaggle.com/heesoo37/120-years-of-olympic-history-athletes-and-results" target="_blank">
120 years of Olympic history</a>).</p>
<p>L'application est divisée en plusieurs onglets :</p>
<ul>
    <li><b>Médailles par Sport :</b> Compare le nombre de médailles (Or, Argent, Bronze) par sport pour les deux pays (Top 10 sports combinés).</li>
    <li><b>Distribution par Âge :</b> Compare la distribution de l'âge des athlètes par sport (Top 10 sports combinés) via des diagrammes en boîte côte à côte.</li>
    <li><b>Carte des Événements :</b> Montre les villes hôtes où les athlètes des deux pays ont participé. Les infobulles détaillent la participation et les médailles pour chaque pays.</li>
    <li><b>Évolution des Médailles :</b> Suit l'évolution annuelle du nombre de médailles pour les USA et la Chine.</li>
</ul>
<p>Utilisez les outils interactifs (survol, zoom, déplacement) pour explorer les visualisations.</p>
<hr>
""", width=800)

cmnt_tab2 = Div(text="""
<hr>
<h3>Commentaires - Médailles par Sport:</h3>
<p><i>
Ce graphique met en évidence la répartition des médailles remportées par les USA et la Chine 
dans leurs 10 disciplines sportives les plus fructueuses, tous types de médailles confondus. 
On observe que les États-Unis dominent la plupart de ces sports, confirmant leur polyvalence 
et leur supériorité historique dans les compétitions olympiques clés.
</i></p>
""", width=800)


cmnt_tab3 = Div(text="""
<hr>
<h3>Commentaires:</h3>
<p><i>On observe que les athlètes chinois sont globalement plus jeunes que les américains dans plusieurs disciplines, notamment en gymnastique et en natation. Cela peut refléter une stratégie de formation et de sélection plus précoce en Chine. 
    À l'inverse, les États-Unis semblent aligner des athlètes plus expérimentés, en particulier dans des sports comme le tir ou la lutte.</i></p>
""", width=800)

cmnt_tab4 = Div(text="""
<hr>
<h3>Commentaires:</h3>
<p><i>On remarque que les États-Unis dominent systématiquement la Chine en nombre total de médailles, 
quelle que soit la localisation des Jeux Olympiques. Cette supériorité se vérifie aussi bien 
lors des éditions organisées sur leur propre continent (Amérique), que lors des Jeux en Europe, 
et même en Asie, y compris ceux qui se sont tenus en Chine. Cela reflète la régularité 
et la puissance sportive des USA, indépendamment du contexte géographique ou politique.</i></p>
""", width=800)


cmnt_tab5 = Div(text="""
<hr>
<h3>Commentaires:</h3>
<p><i>Ce graphique compare l'évolution du nombre de médailles (or, argent, bronze) remportées par les États-Unis (lignes continues) et la Chine (lignes pointillées) aux Jeux Olympiques au fil des ans. On observe une domination américaine historique, tandis que la Chine montre une progression spectaculaire à partir des années 1980.
Les deux nations se disputent régulièrement le haut du classement des médailles lors des olympiades récentes.</i></p>
""", width=800)

# Création des onglets
tab_presentation = TabPanel(
    child=column(texte_presentation), 
    title="Présentation"
)

tab_medailles = TabPanel(
    child=column(figure_medailles_par_sport, cmnt_tab2), 
    title="Médailles par sport"
)

tab_distribution = TabPanel(
    child=column(row(fig_boxplot_usa, fig_boxplot_chn), cmnt_tab3), 
    title="Distribution par Âge"
)

tab_carte = TabPanel(
    child=column(figure_carte, cmnt_tab4), 
    title="Carte des villes hôtes"
)

tab_evolution = TabPanel(
    child=column(figure_evolution, cmnt_tab5), 
    title="Évolution des Médailles"
)

# Layout final
layout_application = column(
    titre_global,
    Tabs(tabs=[
        tab_presentation, 
        tab_medailles, 
        tab_distribution, 
        tab_carte, 
        tab_evolution
    ])
)

# Enregistrer et afficher
output_file("USA_vs_CHINA_JO_Comparison.html", title="USA vs CHINA, de l'économie aux JO!")
show(layout_application)