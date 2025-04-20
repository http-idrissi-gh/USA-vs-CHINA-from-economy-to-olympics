# USA vs China, from economy to olympics!

## Project Overview

This project analyzes and visualizes historical Olympic Games data to compare the participation and medal performance of athletes from the United States (USA) and China (CHN). It utilizes Python, Pandas for data manipulation, and Bokeh for generating interactive web-based visualizations.

The primary goal is to explore trends and differences between these two major Olympic nations across various dimensions like performance in specific sports, athlete demographics, and historical evolution.

**Authors:**
* Théo JANSSENS
* Mohamed IDRISSI GHALMI
* Youssouf DIAKITE


## Features & Visualizations

The project generates an interactive HTML dashboard containing the following tabs:

1.  **Presentation:** Introduces the project, data source, and the different visualization tabs.
2.  **Medals per Sport:**
    * A grouped bar chart comparing the number of Gold, Silver, and Bronze medals won by the USA and China in the top 10 sports (based on combined total medals for both countries).
    * Hover tooltips provide exact medal counts for each country, medal type, and sport.
3.  **Age Distribution:**
    * Side-by-side box plots comparing the age distribution of athletes from the USA and China across the top 10 sports (based on combined unique participants).
    * Hover tooltips show quartile values (Q1, Median, Q3) and whisker limits for age in each sport and country.
4.  **Host Cities Map:**
    * An interactive map (using OpenStreetMap tiles) showing Olympic host cities where athletes from both the USA and China have participated.
    * Circles represent cities, sized based on the total number of participants from both countries.
    * Hover tooltips detail the unique participant counts and medal tallies (Gold, Silver, Bronze) for both USA and China in that specific city's Games.
5.  **Medal Evolution:**
    * Line charts tracking the total number of Gold, Silver, and Bronze medals won by the USA (solid lines) and China (dashed lines) across different Olympic years.
    * Hover tooltips (using `vline` mode) show the medal counts for both countries for the specific year hovered over.

Each visualization includes interactive tools like pan, zoom, save, and hover inspect provided by Bokeh.

## Data Source

The analysis is based on the "120 years of Olympic history athletes and results" dataset.
* **Source:** Kaggle
* **Link:** [https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results](https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results)
* **File used:** `athlete_events.csv`

## Technologies Used

* Python 3.13
* Pandas
* Bokeh 3.6
* NumPy

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone git@github.com:http-idrissi-gh/USA-vs-CHINA-from-economy-to-olympics.git
    cd USA-vs-CHINA-from-economy-to-olympics
    ```
2.  **Install dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
    

## Usage

To generate the interactive visualization:

1.  Make sure your virtual environment is activated and you are in the project's root directory.
2.  Run the main Python script (assuming you saved the provided code as `olympic_comparison.py`):
    ```bash
    python olympic_comparison.py
    ```

## Output

The script will generate an HTML file named `USA_vs_CHINA_JO_Comparison.html` in the project's root directory.

* Open this `USA_vs_CHINA_JO_Comparison.html` file in your preferred web browser to view and interact with the visualizations.

