# -*- coding: utf-8 -*-
# --------------------------------------------------------------
# BSD ... a block size distribution code and app by Mariella Illeditsch
# App for visualising a block size distribution and fitting a probability function
# --------------------------------------------------------------
#
# Code and App Version 1, Mar 2025
# (c) Mariella Illeditsch, 2025
# mariella.illeditsch@pi-geo.at
# Version 2.1 Mar 2026: Zeile 67: Runden der Volumina entfernt
# Version 2.2 Jun 2026: Adding Rosin-Rammler/Weibull_min
# Version 2.3 Jul 2026: replacing powerlaw by lognorm and adding annuality analysis
#
# --------------------------------------------------------------

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import base64
import io
from io import BytesIO
from scipy import stats
from PIL import Image

# --- Konfiguration und Konstanten ---
# Set page configuration
st.set_page_config(
    page_title="BSD Block Size Distribution by curve fitting",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Füge diesen Stilblock hinzu, um die Schriftgröße von st.info anzupassen
st.markdown("""
    <style>
    /* Reduziert die Schriftgröße des Textes in st.info Boxen */
    div[data-testid="stInfo"] p {
        font-size: smaller;
    }
    /* Stil für die Jährlichkeitsergebnisse */
    .annual_result {
        font-size: 1.1em;
        font-weight: bold;
        color: #004280; /* Streamlit Info-Text-Blau */
        margin-left: 15px;
    }
    .result-label {
        font-weight: normal;
        color: #495057; /* Dunkelgrau */
    }
    </style>
""", unsafe_allow_html=True)

# Pfad zum Logo
LOGO_PATH = "pi-geotechnik-1-RGB-192-30-65.png"

# Beispiel-Datei URLs auf GitHub
EXAMPLE_FILES = {
    "Rauwacke": "https://github.com/pi-geotechnik/BSD/raw/main/blocklist_mils_rauwacke_m3.txt",
    "Orthogneiss": "https://github.com/pi-geotechnik/BSD/raw/main/blocklist_rossatz_orthogneis_m3.txt",
    "Slate": "https://github.com/pi-geotechnik/BSD/raw/main/blocklist_vals_schiefer_m3.txt"
}

# --- Hilfsfunktionen ---

def calculate_mass_in_tonnes(volume_m3, density_kg_m3):
    """Calculates mass in tonnes from volume in m³ and density in kg/m³."""
    return (volume_m3 * density_kg_m3) / 1000

def calculate_cubic_root(v):
    """Calculates the cubic root of a volume (m³) to get block axis (m)."""
    return (v ** (1/3))

def visualize_histograms_m3_and_m(m_values, m3_values):
    """Visualizes histograms for block volumes (m³) and block axes (m)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.hist(m3_values, bins=20, color='lightgreen', edgecolor='black')
    ax1.set_title("Histogram of the block volumes (m³)")
    ax1.set_xlabel("Block volume [m³]")
    ax1.set_ylabel("Frequency")
    
    ax2.hist(m_values, bins=20, color='skyblue', edgecolor='black')
    ax2.set_title("Histogram of the block axes [m]")
    ax2.set_xlabel("Block axis [m]")
    ax2.set_ylabel("Frequency")
    
    plt.tight_layout()
    return fig

def calculate_and_visualize_percentiles(m_axes):
    """
    Calculates percentiles and visualizes the probability density (PDF)
    and cumulative distribution function (CDF) on normal and log scales.
    """
    steps = np.linspace(0.01, 1.00, num=100)
    percentiles_m_axes = np.quantile(m_axes, steps)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))

    ax1.hist(m_axes, density=True, bins='auto', histtype='stepfilled', color='tab:blue', alpha=0.3, label='upload pdf')
    ax2.plot(percentiles_m_axes, steps, lw=2.0, color='tab:blue', alpha=0.7, label='upload cdf')
    ax3.plot(percentiles_m_axes, steps, lw=2.0, color='tab:blue', alpha=0.7, label='upload cdf')

    ax1.set_xlim(left=None, right=None)
    ax1.set_xlabel('Block axis a [m]', fontsize=14)
    ax1.set_ylabel('Probability density f(a)', fontsize=14)
    ax2.set_xlim(left=None, right=None)
    ax2.set_xlabel('Block axis a [m]', fontsize=14)
    ax2.set_ylabel('Cumulative probability F(a)', fontsize=14)
    from matplotlib.ticker import FormatStrFormatter
    ax3.xaxis.set_major_formatter(FormatStrFormatter('%g')) 
    ax3.set_xlabel('Block axis a [m] (log)', fontsize=14)
    ax3.set_ylabel('Cumulative probability F(a)', fontsize=14)
    ax1.legend(loc='best', frameon=False)
    ax2.legend(loc='best', frameon=False)
    ax3.legend(loc='best', frameon=False)
    
    plt.tight_layout()
    return fig
    
def fit_distributions_and_visualize(m_axes, selected_distributions):
    """
    Fits selected probability distributions to the data and visualizes their
    PDFs and CDFs against the uploaded data.
    """
    fig, (ax4, ax5) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    ax4.hist(m_axes, color='tab:blue', density=True, bins='auto', histtype='stepfilled', alpha=0.3, label='upload pdf')
    steps = np.linspace(0.01, 1.00, num=100)
    percentiles_m_axes = np.quantile(m_axes, steps)
    ax5.plot(percentiles_m_axes, steps, lw=8.0, color='tab:blue', alpha=0.3, label='upload cdf')
    
    if 'expon' in selected_distributions:
        loc3, scale3 = stats.expon.fit(m_axes)
        X3 = np.linspace(stats.expon.ppf(0.001, loc=loc3, scale=scale3), stats.expon.ppf(0.999, loc=loc3, scale=scale3), len(m_axes))
        ax4.plot(X3, stats.expon.pdf(X3, loc=loc3, scale=scale3), '#333333', lw=1.0, alpha=0.7, label='expon pdf')
        ax5.plot(X3, stats.expon.cdf(X3, loc=loc3, scale=scale3), '#333333', lw=1.0, alpha=0.7, label='expon cdf')
        st.session_state.loc3, st.session_state.scale3 = loc3, scale3

    if 'genexpon' in selected_distributions:
        a1, b1, c1, loc1, scale1 = stats.genexpon.fit(m_axes)
        X1 = np.linspace(stats.genexpon.ppf(0.001, a1, b1, c1, loc=loc1, scale=scale1), stats.genexpon.ppf(0.999, a1, b1, c1, loc=loc1, scale=scale1), len(m_axes))
        ax4.plot(X1, stats.genexpon.pdf(X1, a1, b1, c1, loc=loc1, scale=scale1), '#800020', lw=1.0, alpha=0.7, label='genexpon pdf')
        ax5.plot(X1, stats.genexpon.cdf(X1, a1, b1, c1, loc=loc1, scale=scale1), '#800020', lw=1.0, alpha=0.7, label='genexpon cdf')
        st.session_state.a1, st.session_state.b1, st.session_state.c1, st.session_state.loc1, st.session_state.scale1 = a1, b1, c1, loc1, scale1
                
    if 'lognorm' in selected_distributions:
        s4, loc4, scale4 = stats.lognorm.fit(m_axes)
        X4 = np.linspace(stats.lognorm.ppf(0.001, s4, loc=loc4, scale=scale4), stats.lognorm.ppf(0.999, s4, loc=loc4, scale=scale4), len(m_axes))
        ax4.plot(X4, stats.lognorm.pdf(X4, s4, loc=loc4, scale=scale4), '#006400', lw=1.0, alpha=0.7, label='lognorm pdf')
        ax5.plot(X4, stats.lognorm.cdf(X4, s4, loc=loc4, scale=scale4), '#006400', lw=1.0, alpha=0.7, label='lognorm cdf')
        st.session_state.s4, st.session_state.loc4, st.session_state.scale4 = s4, loc4, scale4
        
    if 'weibull_min' in selected_distributions:
        c2, loc2, scale2 = stats.weibull_min.fit(m_axes)
        X2 = np.linspace(stats.weibull_min.ppf(0.001, c2, loc=loc2, scale=scale2), stats.weibull_min.ppf(0.999, c2, loc=loc2, scale=scale2), len(m_axes))
        ax4.plot(X2, stats.weibull_min.pdf(X2, c2, loc=loc2, scale=scale2), '#C05A3E', lw=1.0, alpha=0.7, label='weibull_min pdf')
        ax5.plot(X2, stats.weibull_min.cdf(X2, c2, loc=loc2, scale=scale2), '#C05A3E', lw=1.0, alpha=0.7, label='weibull_min cdf')
        st.session_state.c2, st.session_state.loc2, st.session_state.scale2 = c2, loc2, scale2
    
    counts, bins = np.histogram(m_axes, bins='auto', density=True)
    max_y_value = max(counts)

    ax4.legend(loc='best', frameon=False)
    ax4.set_ylim(0, max_y_value * 1.1)
    ax4.set_xlabel('Block axis a [m]', fontsize=12)
    ax4.set_ylabel('Probability density f(a)', fontsize=12)
    ax5.legend(loc='best', frameon=False)
    ax5.set_xscale('log')
    ax5.set_xlabel('Block axis a [m] (log)', fontsize=12)
    ax5.set_ylabel('Cumulative probability F(a)', fontsize=12)
    
    plt.tight_layout()
    return fig
    
def calculate_distribution_percentiles(distribution, percentiles, *params):
    """Calculates percentiles for a given distribution and its parameters."""
    return [distribution.ppf(p / 100, *params) for p in percentiles]

def process_uploaded_data(file_content, unit_type, density_kg_m3=None):
    """Processes uploaded file content, converts units, and calculates m_achsen."""
    try:
        values_list = [float(val.strip().replace(',', '.')) for val in file_content.splitlines() if val.strip()]
        values = sorted([wert for wert in values_list if wert > 0.000])
        
        if unit_type == "Volume in m³":
            volumes_m3 = values
        elif unit_type == "Mass in t (density required)":
            if not density_kg_m3 or density_kg_m3 <= 0:
                st.error("Density must be a positive number for mass input.")
                return None, None, None
            volumes_m3 = [val * 1000 / density_kg_m3 for val in values]
        
        m_axes = [calculate_cubic_root(val) for val in volumes_m3]
        return m_axes, volumes_m3, len(values)
    except (ValueError, TypeError) as e:
        st.error(f"Error processing data: {e}. Please ensure the file contains only valid numbers and uses a dot '.' as a decimal separator.")
        return None, None, None

def clear_all_data():
    """Clears all data-related session state variables."""
    keys_to_clear = [
        'm_achsen', 'volumes_m3', 'uploaded_file_content', 'uploaded_filename',
        'block_count', 'file_source', 'success_message', 'last_error_message',
        'a1', 'b1', 'c1', 'loc1', 'scale1', 'c2', 'loc2', 'scale2', 
        'loc3', 'scale3', 's4', 'loc4', 'scale4',
        'annual_results' # Auch die Jährlichkeits-Ergebnisse löschen
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

# --- Streamlit App Layout ---
try:
    # Bild einlesen und in Base64 umwandeln
    with open(LOGO_PATH, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    # HTML mit Link und eingebettetem Bild erzeugen
    st.markdown(
        f"""
        <a href="https://pi-geo.at/" target="_blank">
            <img src="data:image/png;base64,{encoded_string}" width="300" style="margin-bottom: 15px;">
        </a>
        """,
        unsafe_allow_html=True
    )
except FileNotFoundError:
    st.warning(f"Logo file not found: {LOGO_PATH}")

st.title("Block Size Distribution by Curve Fitting")
st.markdown("*A block distribution code by Mariella ILLEDITSCH*")
with st.expander("ℹ️ About This Project"):
    st.markdown("""
    Version 2.3, July 2026
    * This method was developed during the [doctoral thesis](https://repositum.tuwien.at/handle/20.500.12708/189867) of Mariella ILLEDITSCH at TU Wien (2023). 
    * Corresponding references: [Illeditsch & Preh (2020)](https://onlinelibrary.wiley.com/doi/abs/10.1002/geot.202000021), [Illeditsch & Preh (2024)](https://link.springer.com/article/10.1007/s11069-024-06432-4).
    ### 🧗‍♀️ The Challenge
    As demonstrated in the research above, designing rockfall protection measures based on a single *design block* often results in unreliable trajectories (i.e., kinetic energies, bounce heights, and runout distances). Furthermore, Block Size Distributions (BSDs) derived purely from debris fields are highly subjective. Linking the percentiles of such field-derived BSDs (e.g., the 95th-98th percentile according to ONR 24810) directly to the frequency of rockfall events is scientifically problematic and physically flawed.
    ### 💡 The Solution
    This application provides a more certain, accurate, verifiable, and objective workflow for holistic rockfall hazard assessment — even when based on a limited number of block size measurements (min. 65 blocks are recommended!).
    * **Curve Fitting & Block Lists:** The app visualizes BSDs, fits advanced statistical distribution functions (like genexpon or weibull_min, also known as Rosin-Rammler), and generates robust block lists (intensity-frequency-relations) for rockfall simulations.
    * **Return Period Analysis (Annuality):** It bridges the gap between spatial geometry and time. A fitted distribution natively only provides the *relative* probability of block sizes. By defining a known worst-case anchor event (a specific block size and its estimated return period), the geometrical distribution is translated into temporal return periods. This allows for the calculation of block sizes for specific annualities (e.g., 30, 100, and 300-year events) and of yearly events (of any size).
    ### ⚠️ Practical Application & Expert Judgment
    Fitted statistical distributions range mathematically from dust particles to infinite rock masses, which is not useful for numerical modeling. **Expert opinion is required** to define meaningful boundaries:
    * **Upper Cut-off (Worst-Case):** Based on the annuality analysis, unrealistic extreme events with return periods far beyond the structure's lifespan can be neglected.
    * **Lower Cut-off:** Simulation programs generally struggle to realistically calculate the trajectories of very small blocks. Therefore, a reasonable minimum block size should be defined by the user depending on the simulation tool.
    
    *This application is an open-source project.*
    """)
    
    

# --- Sidebar for user input ---
with st.sidebar:
    st.header("Input Data")
    
    # KORREKTUR 1: Den uploader_key initialisieren
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0
    
    # Session state Initialisierung (unverändert)
    if 'einheit' not in st.session_state: st.session_state.einheit = "Volume in m³"

    selected_unit = st.selectbox("Select the unit of the input data:", ["Volume in m³", "Mass in t (density required)"])

    if st.session_state.einheit != selected_unit:
        clear_all_data() # Löscht alle Daten, inkl. annual_results
        st.session_state.einheit = selected_unit
        st.rerun()

    density_input = None
    if selected_unit == "Mass in t (density required)":
        density_input = st.number_input("Enter the density in kg/m³:", min_value=250, value=2650, step=50)

    st.subheader("Load Example Files")
    for name, url in EXAMPLE_FILES.items():
        if st.button(f"Load sample file '{name}'"):
            # KORREKTUR 2: uploader_key erhöhen, um das Upload-Widget zurückzusetzen
            st.session_state.uploader_key += 1
            clear_all_data() # Alle alten Daten löschen
            
            with st.spinner(f"Loading '{name}'..."):
                try:
                    response = requests.get(url)
                    response.raise_for_status()
                    example_file_content = response.content.decode("utf-8")
                    m_axes, volumes_m3, block_count = process_uploaded_data(example_file_content, selected_unit, density_input)
                    if m_axes is not None:
                        st.session_state.update({
                            'uploaded_file_content': example_file_content, 'uploaded_filename': f"sample_{name}.txt",
                            'm_achsen': m_axes, 'volumes_m3': volumes_m3, 'block_count': block_count,
                            'file_source': 'sample', 'success_message': f"The sample file '{name}' was loaded successfully."
                        })
                    else:
                        st.session_state['last_error_message'] = f"Error processing the sample file '{name}'."
                except requests.exceptions.RequestException as e:
                    st.session_state['last_error_message'] = f"Error loading file from URL: {e}"
            st.rerun()

    st.subheader("Upload Your Own File")
    # KORREKTUR 3: Den uploader mit dem Schlüssel verknüpfen
    uploaded_user_file = st.file_uploader(
        f"Upload your own file with {'m³' if selected_unit == 'Volume in m³' else 't'} values:",
        type=["txt"],
        key=f"uploader_{st.session_state.uploader_key}"
    )
    st.info("Note: please make sure that all numbers in the uploaded text file use the dot ('.') instead of the comma (',') as decimal separator.")

    if uploaded_user_file:
        # Prüfen, ob es eine NEUE Datei ist
        if st.session_state.get('uploaded_filename') != uploaded_user_file.name:
            clear_all_data()
            
            # WICHTIG: Sofort den Namen merken, um die Endlosschleife zu stoppen!
            st.session_state.uploaded_filename = uploaded_user_file.name
            st.session_state.file_source = 'user'
            
            with st.spinner("Processing uploaded file..."):
                file_content = uploaded_user_file.read().decode("utf-8")
                m_axes, volumes_m3, block_count = process_uploaded_data(file_content, selected_unit, density_input)
                
                if m_axes is not None:
                    st.session_state.m_achsen = m_axes
                    st.session_state.volumes_m3 = volumes_m3
                    st.session_state.block_count = block_count
                    st.session_state.uploaded_file_content = file_content
                    st.session_state.success_message = "Your file was processed successfully."
                else:
                    st.session_state.last_error_message = "File could not be processed. Please check format and content."
            
            st.rerun() # Führt nun nur noch EINMAL aus

    # Falls der Nutzer das kleine 'X' im Uploader klickt
    elif st.session_state.get('file_source') == 'user':
        clear_all_data()
        st.info("File removed. Please upload a new file or select a sample file.")
        st.rerun()

    st.subheader("Support this Project")
    st.link_button("Buy me a coffee ☕", url='https://www.buymeacoffee.com//ztilleditsz')
    st.markdown("Thank you for your support! ❤️")

# --- Main Content Area ---
if st.session_state.get('success_message'):
    st.success(st.session_state.get('success_message'))
    st.session_state.success_message = None

if st.session_state.get('last_error_message'):
    st.error(st.session_state.get('last_error_message'))
    st.session_state.last_error_message = None

if not st.session_state.get('m_achsen'):
    st.info("Please select a unit in the left sidebar and load a sample file or upload your own file to get started.")
else:
    # Alle nachfolgenden Berechnungen und Anzeigen
    # ... (Der Rest Ihres Codes für die Hauptseite bleibt unverändert) ...
    m_achsen = st.session_state['m_achsen']

    # Display file information
    if st.session_state.get('uploaded_filename') and st.session_state.get('block_count') is not None:
        st.header("📂 1. Input Data & Diagnostics")
        st.subheader("File Information")
        st.write(f"**Filename:** {st.session_state.uploaded_filename}")
        st.write(f"**Number of blocks:** {st.session_state.block_count}")
        
        # Display file content
        if st.session_state.get('uploaded_file_content'):
            with st.expander(f"Show Contents of {st.session_state.uploaded_filename}"):
                 st.text_area("File content:", st.session_state.uploaded_file_content, height=200, disabled=True)

    st.divider() # <--- Fügt eine feine Trennlinie ein!
    st.header("📊 2. Statistical Analysis & Distribution Fitting")
    st.subheader("Visualization of Probability Distribution")
    fig1 = calculate_and_visualize_percentiles(st.session_state.m_achsen)
    st.pyplot(fig1)

    st.subheader("Fitting Probability Functions")
    selected_dists = ['genexpon', 'weibull_min', 'expon', 'lognorm'] 
    
    fig2 = fit_distributions_and_visualize(st.session_state.m_achsen, selected_dists)
    st.pyplot(fig2)

    st.subheader("Tabular Comparison of Percentiles")
    
    table_view = st.radio("Select table view:", ("Compact", "Full"), horizontal=True)
    percentiles_to_show = [0, 25, 50, 75, 95, 96, 97, 98, 100] if table_view == "Compact" else list(range(0, 95, 5)) + list(range(95, 101))
        
    required_params = ['a1', 'b1', 'c1', 'loc1', 'scale1', 'c2', 'loc2', 'scale2', 'loc3', 'scale3', 's4', 'loc4', 'scale4']
    if all(param in st.session_state for param in required_params):
        upload_perz = np.percentile(m_achsen, percentiles_to_show)
        L1s = np.array(calculate_distribution_percentiles(stats.genexpon, percentiles_to_show, st.session_state.a1, st.session_state.b1, st.session_state.c1, st.session_state.loc1, st.session_state.scale1))
        L2s = np.array(calculate_distribution_percentiles(stats.weibull_min, percentiles_to_show, st.session_state.c2, st.session_state.loc2, st.session_state.scale2))
        L3s = np.array(calculate_distribution_percentiles(stats.expon, percentiles_to_show, st.session_state.loc3, st.session_state.scale3))
        L4s = np.array(calculate_distribution_percentiles(stats.lognorm, percentiles_to_show, st.session_state.s4, st.session_state.loc4, st.session_state.scale4))
        
        df1 = pd.DataFrame({
            "percentile": [str(p) for p in percentiles_to_show],
            "sample [m³]": upload_perz**3, "genexpon [m³]": L1s**3,
            "weibull_min [m³]": L2s**3, "expon [m³]": L3s**3,
            "lognorm [m³]": L4s**3
        })
        st.dataframe(df1.style.hide(axis="index")) # Ihre Original-Darstellung
    else:
        st.info("Fitting parameters not yet available. Results will be shown after data processing.")

    st.divider() # <--- Trennlinie vor dem neuen Kapitel
    st.header("⚙️ 3. Return Period Analysis & Export")
    st.subheader("Return Period Analysis (Annual Exceedance Probability)")
    st.markdown("Calibrate the fitted distribution using a known anchor event. This allows translating the geometric distribution into time-based return periods (annualities).")
    st.markdown("Enter the size and return period of a known event (i.e., the largest observed block in a given timeframe) to calculate the annual rockfall frequency (λ₀) and the corresponding block sizes for other return periods (30, 100, and 300 years).")

    col_anchor1, col_anchor2 = st.columns(2)
    # NEU: Eingabe als Volumen (m³)
    anchor_block_volume = col_anchor1.number_input("Block volume of the anchor event [m³]", min_value=0.001, value=2.00, step=0.1, format="%.3f")
    anchor_return_period = col_anchor2.number_input("Return period of the anchor event [years]", min_value=1, value=50, step=10)
    
    # Im Hintergrund für die Formeln wieder in die Kantenlänge (m) umrechnen
    anchor_block_axis = anchor_block_volume ** (1/3)

    if st.button("Calculate Annualities"):
        results_data = []
        target_periods = [30, 100, 300]
        
        distributions_dict = {
            'genexpon': (stats.genexpon, ['a1', 'b1', 'c1', 'loc1', 'scale1']),
            'weibull_min': (stats.weibull_min, ['c2', 'loc2', 'scale2']),
            'expon': (stats.expon, ['loc3', 'scale3']),
            'lognorm': (stats.lognorm, ['s4', 'loc4', 'scale4'])
        }
        
        for dist_name, (dist_func, param_keys) in distributions_dict.items():
            if all(k in st.session_state for k in param_keys):
                params = tuple(st.session_state[k] for k in param_keys)
                
                try:
                    lambda_anchor = 1 / anchor_return_period
                    exceedance_prob_anchor = 1 - dist_func.cdf(anchor_block_axis, *params)
                    
                    if exceedance_prob_anchor <= 1e-9:
                        # NEU: Spalten heißen jetzt [m³]
                        row = {"Distribution": dist_name, "λ₀ [blocks/year]": "Error", "30-year [m³]": "Error", "100-year [m³]": "Error", "300-year [m³]": "Error"}
                    else:
                        lambda_0 = lambda_anchor / exceedance_prob_anchor
                        
                        row = {"Distribution": dist_name, "λ₀ [blocks/year]": f"{lambda_0:.3f}"}
                        
                        for T_target in target_periods:
                            target_exceedance_prob = (1 / T_target) / lambda_0
                            target_cdf = 1 - np.clip(target_exceedance_prob, 0, 1)
                            calc_val = dist_func.ppf(target_cdf, *params)
                            # NEU: Ergebnis wird für die Tabelle wieder in m³ (hoch 3) umgerechnet
                            row[f"{T_target}-year [m³]"] = f"{(calc_val**3):.3f}"
                            
                    results_data.append(row)
                except Exception as e:
                    row = {"Distribution": dist_name, "λ₀ [blocks/year]": "Error", "30-year [m³]": "Error", "100-year [m³]": "Error", "300-year [m³]": "Error"}
                    results_data.append(row)
        
        if results_data:
            st.session_state.annual_results_df = pd.DataFrame(results_data)

    # Tabelle anzeigen, falls berechnet
    if 'annual_results_df' in st.session_state:
        # NEU: Die angepasste Überschrift
        st.markdown("##### Annual Total Rockfall Frequency (λ₀) & Calibrated Return Periods:")
        st.dataframe(st.session_state.annual_results_df.style.hide(axis="index"))

        
        
# --- ABSCHNITT: Generiere und lade gefilterte Verteilung herunter ---
    st.subheader("Generate and Download Filtered Distribution")
    st.markdown("Select a fitted distribution and define the block axis range to generate a custom block list. You can then choose the output unit for download.")

    # Determine which distributions have fitted parameters and can be selected
    available_dists_for_download = []
    # Check if parameters for genexpon are available
    if 'a1' in st.session_state and 'b1' in st.session_state and 'c1' in st.session_state and 'loc1' in st.session_state and 'scale1' in st.session_state:
        available_dists_for_download.append('genexpon')
    # Check if parameters for weibull_min are available
    if 'c2' in st.session_state and 'loc2' in st.session_state and 'scale2' in st.session_state:
        available_dists_for_download.append('weibull_min')
    # Check if parameters for expon are available
    if 'loc3' in st.session_state and 'scale3' in st.session_state:
        available_dists_for_download.append('expon')
    # Check if parameters for lognorm are available
    if 's4' in st.session_state and 'loc4' in st.session_state and 'scale4' in st.session_state:
        available_dists_for_download.append('lognorm')
    
    selected_download_distribution = None
    if not available_dists_for_download:
        st.info("Please load data and select one distribution in the 'Fitting Probability Functions' section to enable generating a filtered distribution.")
        generation_possible_overall = False # Flag to disable generation controls
    else:
        selected_download_distribution = st.selectbox(
            "Select Distribution for Generation:",
            options=available_dists_for_download,
            key="selected_download_dist"
        )
        generation_possible_overall = True


    # Display controls only if at least one distribution is available for generation
    if generation_possible_overall:
        st.info(f"Generating blocks from the fitted **{selected_download_distribution}** distribution.")

        # 2. Min/Max Block Axis Input (for block axis [m])
        st.markdown("Set the minimum and maximum **block volume** values for the generated distribution (values outside this range will be excluded).")
        
        st.info("""
* **Lower Cut-off:** Define a reasonable minimum block size based on the limitations of your target simulation tool.
* **Upper Cut-off:** Use the return period calibration above to neglect extreme, highly improbable events, depending on your specific protection goals.
        """)
        
        col1, col2 = st.columns(2)
        # NEU: Eingaben sind jetzt in Volumen (m³)
        min_block_volume = col1.number_input("Minimum Block Volume [m³]:", min_value=0.001, value=0.027, step=0.05, format="%.3f")
        max_block_volume = col2.number_input("Maximum Block Volume [m³]:", min_value=min_block_volume + 0.001, value=3.375, step=0.05, format="%.3f")

        # Im Hintergrund für die Filter-Logik wieder in die Kantenlänge (m) umrechnen
        min_block_axis = min_block_volume ** (1/3)
        max_block_axis = max_block_volume ** (1/3)

        if min_block_volume >= max_block_volume:
            st.error("Minimum block volume must be less than maximum block volume.")
            generation_possible_overall = False # Generation deaktivieren, wenn die Spanne ungültig ist

        # Number of samples to generate
        num_samples = st.slider(
            "Number of blocks to generate:", 
            min_value=500, 
            max_value=10000, 
            value=1000, 
            step=500, 
            key="num_samples_slider_download",
            help="Generate a larger number of samples to ensure sufficient blocks after filtering."
        )

        # 3. Output Type Selection
        st.markdown("Choose the output unit for the generated block list (block mass [t] is required for THROW):")
        selected_output_unit = st.radio(
            "Output Unit:",
            ('Block Volume [m³]', 'Block Mass [t]'),
            key="output_unit_selector"
        )

        download_density_kg_m3 = None
        if selected_output_unit == 'Block Mass (t)':
            download_density_kg_m3 = st.number_input(
                "Density for mass calculation (kg/m³):", 
                min_value=250, 
                value=2650, 
                step=50, 
                key="download_density_input"
            )
            if download_density_kg_m3 <= 0:
                st.error("Density must be greater than 0 for mass calculation.")
                generation_possible_overall = False # Disable generation if density is invalid

        # Generation Button
        if generation_possible_overall:
            if st.button("Generate Filtered Block List", key="generate_filtered_button"):
                with st.spinner("Generating and filtering distribution..."):
                    generated_raw_axes = []
                    params_for_rvs = ()
                    
                    # Get parameters based on selected distribution
                    if selected_download_distribution == 'genexpon':
                        params_for_rvs = (st.session_state.a1, st.session_state.b1, st.session_state.c1, 
                                          st.session_state.loc1, st.session_state.scale1)
                        dist_function = stats.genexpon
                    elif selected_download_distribution == 'weibull_min':
                        params_for_rvs = (st.session_state.c2, st.session_state.loc2, st.session_state.scale2)
                        dist_function = stats.weibull_min
                    elif selected_download_distribution == 'expon':
                        params_for_rvs = (st.session_state.loc3, st.session_state.scale3)
                        dist_function = stats.expon
                    elif selected_download_distribution == 'lognorm':
                        params_for_rvs = (st.session_state.s4, st.session_state.loc4, st.session_state.scale4)
                        dist_function = stats.lognorm
                    
                    # Generate more samples to account for filtering
                    # This ensures we have enough data points after filtering for narrow ranges
                    generated_raw_axes = dist_function.rvs(*params_for_rvs, size=num_samples * 10) 
                    
                    # Filter samples based on min/max block axis
                    filtered_samples_m_axis = [s for s in generated_raw_axes if min_block_axis <= s <= max_block_axis]
                    
                    # Take exactly num_samples, or fewer if not enough were generated, and sort
                    final_generated_blocks_m_axis = sorted(filtered_samples_m_axis[:num_samples])
                    
                    if final_generated_blocks_m_axis:
                        # Convert to desired output unit
                        final_output_blocks_converted = []
                        #if selected_output_unit == 'Block Axis (m)':
                        #    final_output_blocks_converted = final_generated_blocks_m_axis
                        #    st.session_state.output_unit_for_download = 'm'
                        if selected_output_unit == 'Block Volume (m³)':
                            final_output_blocks_converted = [x**3 for x in final_generated_blocks_m_axis]
                            st.session_state.output_unit_for_download = 'm3'
                        elif selected_output_unit == 'Block Mass (t)':
                            if download_density_kg_m3 and download_density_kg_m3 > 0:
                                final_output_blocks_converted = [(x**3 * download_density_kg_m3) / 1000 for x in final_generated_blocks_m_axis]
                                st.session_state.output_unit_for_download = 't'
                            else:
                                st.error("Invalid density for mass calculation. Please provide a positive density.")
                                final_output_blocks_converted = [] # Clear if density is invalid
                                st.session_state.output_unit_for_download = None

                        st.session_state.generated_blocks_for_download = final_output_blocks_converted
                        
                        if final_output_blocks_converted:
                            st.success(f"Generated {len(final_output_blocks_converted)} blocks for download in '{selected_output_unit}'.")
                            
                            # Optional: Display a small histogram of the generated blocks (in their final unit)
                            fig_gen, ax_gen = plt.subplots(figsize=(8, 4))
                            ax_gen.hist(final_output_blocks_converted, bins='auto', color='lightcoral', edgecolor='black')
                            ax_gen.set_title(f"Histogram of Generated Blocks ({len(final_output_blocks_converted)} blocks)")
                            ax_gen.set_xlabel(selected_output_unit)
                            ax_gen.set_ylabel("Frequency")
                            st.pyplot(fig_gen)
                            plt.close(fig_gen)
                        else:
                            st.warning("No blocks generated within the specified min/max range after unit conversion. Adjust parameters.")
                            st.session_state.generated_blocks_for_download = []
                            st.session_state.output_unit_for_download = None
                    else:
                        st.warning("No blocks generated within the specified min/max range. Try adjusting the range or increase the 'Number of blocks to generate'.")
                        st.session_state.generated_blocks_for_download = []
                        st.session_state.output_unit_for_download = None
        else: # generation_possible_overall is False (due to min/max error or density error)
            st.info("Adjust input values or ensure selected distribution parameters are available for generation.")


    # 4. Download Button (only display if blocks were successfully generated)
    if 'generated_blocks_for_download' in st.session_state and st.session_state.generated_blocks_for_download and st.session_state.output_unit_for_download:
        
        # Format the blocks as a string, each on a new line
        # Use different precision for volume (m³) vs mass (t)
        if st.session_state.output_unit_for_download == 'm³':  # 'm3'
             output_string = "\n".join(f"{x:.6f}" for x in st.session_state.generated_blocks_for_download) # 6 decimal places for m³
             file_name = f"{selected_download_distribution}_filtered_{st.session_state.output_unit_for_download}.txt"
        else: # 't'
             output_string = "\n".join(f"{x:.6f}" for x in st.session_state.generated_blocks_for_download) # 6 decimal places for t
             file_name = f"{selected_download_distribution}_filtered_{st.session_state.output_unit_for_download}.txt"

        st.download_button(
            label=f"Download Generated Block List ({st.session_state.output_unit_for_download.upper()})",
            data=output_string,
            file_name=file_name,
            mime="text/plain",
            help=f"Download the list of generated block values in {st.session_state.output_unit_for_download.upper()}."
        )