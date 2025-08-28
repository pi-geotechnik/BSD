# -*- coding: utf-8 -*-
# --------------------------------------------------------------
# BSD ... a block size distribution code and app by Mariella Illeditsch
# App for visualising a block size distribution and fitting a probability function
# --------------------------------------------------------------
#
# Code and App version 1, Mar 2025
# (c) Mariella Illeditsch, 2025
# mariella.illeditsch@pi-geo.at
#
# --------------------------------------------------------------

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
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
    </style>
""", unsafe_allow_html=True)

# Pfad zum Logo
LOGO_PATH = "pi-geotechnik-1-RGB-192-30-65.png"

# Beispiel-Datei URLs auf GitHub
EXAMPLE_FILES = {
    #"Lime": "https://github.com/pi-geotechnik/Blockverteilung/raw/main/blocklist_dachsteinkalk_m3.txt", # Auskommentiert, da Zustimmung von Laimer erforderlich
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
    return round(v ** (1/3), 2)

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

    # Histogram of the Probability density
    ax1.hist(m_axes, density=True, bins='auto', histtype='stepfilled', color='tab:blue', alpha=0.3, label='upload pdf')

    # CDF on normal scale
    ax2.plot(percentiles_m_axes, steps, lw=2.0, color='tab:blue', alpha=0.7, label='upload cdf')

    # CDF on Log-scale
    ax3.plot(percentiles_m_axes, steps, lw=2.0, color='tab:blue', alpha=0.7, label='upload cdf')

    # Axis labels
    ax1.set_xlim(left=None, right=None)
    ax1.set_xlabel('Block axis a [m]', fontsize=14)
    ax1.set_ylabel('Probability density f(a)', fontsize=14)

    ax2.set_xlim(left=None, right=None)
    ax2.set_xlabel('Block axis a [m]', fontsize=14)
    ax2.set_ylabel('Cumulative probability F(a)', fontsize=14)

    ax3.set_xscale('log')
    ax3.set_xlabel('Block axis a [m] (log)', fontsize=14)
    ax3.set_ylabel('Cumulative probability F(a)', fontsize=14)

    # Legends
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

    # Histogram of m_axes
    ax4.hist(m_axes, color='tab:blue', density=True, bins='auto', histtype='stepfilled', alpha=0.3,
             label='upload pdf')

    # CDF for m_axes (cumulative distribution)
    steps = np.linspace(0.01, 1.00, num=100)
    percentiles_m_axes = np.quantile(m_axes, steps)
    ax5.plot(percentiles_m_axes, steps, lw=8.0, color='tab:blue', alpha=0.3, label='upload cdf')

    # Cumulative Distributions and CDF Calculations
    if 'expon' in selected_distributions:
        loc3, scale3 = stats.expon.fit(m_axes)
        X3 = np.linspace(stats.expon.ppf(0.001, loc=loc3, scale=scale3),
                         stats.expon.ppf(0.999, loc=loc3, scale=scale3), len(m_axes))
        ax4.plot(X3, stats.expon.pdf(X3, loc=loc3, scale=scale3), '#333333', lw=1.0, alpha=0.7, label='expon pdf')
        ax5.plot(X3, stats.expon.cdf(X3, loc=loc3, scale=scale3), '#333333', lw=1.0, alpha=0.7, label='expon cdf')

        # Save parameters to session_state
        st.session_state.loc3 = loc3
        st.session_state.scale3 = scale3

    if 'genexpon' in selected_distributions:
        a1, b1, c1, loc1, scale1 = stats.genexpon.fit(m_axes)
        # KORRIGIERTE ZEILE: Das np.linspace war unvollständig
        X1 = np.linspace(stats.genexpon.ppf(0.001, a1, b1, c1, loc=loc1, scale=scale1),
                         stats.genexpon.ppf(0.999, a1, b1, c1, loc=loc1, scale=scale1), len(m_axes))
        ax4.plot(X1, stats.genexpon.pdf(X1, a1, b1, c1, loc=loc1, scale=scale1), '#800020', lw=1.0, alpha=0.7, label='genexpon pdf')
        ax5.plot(X1, stats.genexpon.cdf(X1, a1, b1, c1, loc=loc1, scale=scale1), '#800020', lw=1.0, alpha=0.7, label='genexpon cdf')

        # Save parameters to session_state
        st.session_state.a1 = a1
        st.session_state.b1 = b1
        st.session_state.c1 = c1
        st.session_state.loc1 = loc1
        st.session_state.scale1 = scale1

    if 'powerlaw' in selected_distributions:
        a4, loc4, scale4 = stats.powerlaw.fit(m_axes)
        X4 = np.linspace(stats.powerlaw.ppf(0.001, a4, loc=loc4, scale=scale4),
                         stats.powerlaw.ppf(0.999, a4, loc=loc4, scale=scale4), len(m_axes))
        ax4.plot(X4, stats.powerlaw.pdf(X4, a4, loc=loc4, scale=scale4), '#006400', lw=1.0, alpha=0.7, label='powerlaw pdf')
        ax5.plot(X4, stats.powerlaw.cdf(X4, a4, loc=loc4, scale=scale4), '#006400', lw=1.0, alpha=0.7, label='powerlaw cdf')

        # Save parameters to session_state
        st.session_state.a4 = a4
        st.session_state.loc4 = loc4
        st.session_state.scale4 = scale4

    # Calculate histogram (counts and bins)
    counts, bins = np.histogram(m_axes, bins='auto', density=True)
    # Find the maximum value of the histogram
    max_y_value = max(counts)

    # Axes for the plot
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
        # Replace commas with dots for decimal separator and filter out empty lines or non-numeric values
        values_list = [
            float(val.strip().replace(',', '.'))
            for val in file_content.splitlines()
            if val.strip().replace(',', '.', 1).replace('.', '', 1).isdigit() or (val.strip().replace(',', '.', 1).startswith('-') and val.strip().replace(',', '.', 1)[1:].replace('.', '', 1).isdigit())
        ]

        # Filter values to be non-negative
        values = [wert for wert in values_list if wert >= 0.000]

        # Sort values in ascending order
        values.sort()

        if unit_type == "Volume in m³":
            m_axes = [calculate_cubic_root(val) for val in values]
            volumes_m3 = values
        elif unit_type == "Mass in t (density required)":
            if density_kg_m3 is None:
                st.error("Density is required for mass input.")
                return None, None, None
            volumes_m3 = [val * 1000 / density_kg_m3 for val in values]
            m_axes = [calculate_cubic_root(val) for val in volumes_m3]

        return m_axes, volumes_m3, len(values)
    except Exception as e:
        # Fehlermeldung präziser machen
        st.error(f"Error processing data: {e}. Please ensure numbers use a dot '.' as a decimal separator and the file contains valid numerical data.")
        return None, None, None

def clear_all_data():
    """Clears all data-related session state variables."""
    keys_to_clear = [
        'm_achsen', 'volumes_m3', 'uploaded_file_content', 'uploaded_filename',
        'block_count', 'file_source', 'success_message', 'last_error_message',
        'a1', 'b1', 'c1', 'loc1', 'scale1', 'loc3', 'scale3', 'a4', 'loc4', 'scale4'
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

# --- Streamlit App Layout ---

# Header und Info
#st.image(Image.open(LOGO_PATH), caption="https://pi-geo.at/", width=300)
st.title("Block Size Distribution by Curve Fitting")
st.markdown("*A block distribution code by Mariella ILLEDITSCH*")
with st.expander("ℹ️ About This Project"):
    st.markdown("""
    Version 2.2, August 2025

    This method was developed during the [doctoral thesis](https://repositum.tuwien.ac.at/handle/20.500.12708/189867) of Mariella ILLEDITSCH at the TU Wien in the year 2023.

    Corresponding references: [Illeditsch & Preh (2020)](https://onlinelibrary.wiley.com/doi/abs/10.1002/geot.202000021?msockid=0f6932b2fb8d63ee018527a8fa6a62a9), [Illeditsch & Preh (2024)](https://link.springer.com/article/10.1007/s11069-024-06432-4)

    As worked out in the above refereces, designing rock fall protection measures with only one *design block* may result in unreliable trajectories (i.e. kinetic energies, bounce heights and runout).
    Block size distributions (BSDs) derived from debris fields are very subjective.
    Linking the percentiles of such BSDs, that define the size of the design block (95th-98th, according to ONR24810), with the frequency of rockfall events is not meaningful.

    **This application visualizes block size distributions, fits distribution functions to them and provides block lists of the fitted distribution for rockfall simulations (e.g. with THROW).
    It offers the determination of more meaningful BSDs, which is also possible with a limited number of block size measurements.
    This provides more certain, accurate, verifiable, holistic, and objective results for more meaningful rockfall hazard assessment.**

    Fitted distributions often result in infinite block sizes or dust particle sizes, which are not useful for rockfall modelling.
    In the design of rockfall protection measures and hazard analyses, rockfall frequencies (magnitude to frequency relations M/F; Corominas et al. (2018)) and return periods play an important role.
    This requires knowledge of the events on the one hand and the definition of a worst-case scenario on the other.
    **Expert opinion is required!**
    Based on a defined worst-case scenario (with a certain annuality), events/block sizes with higher return periods may be neglected (cut off).
    At the lower end, simulation programs are generally not able to realistically calculate trajectories of very small blocks. Minimum block sizes of 0.025 m³ are recommended.

    This app is an open source project.
    """)

# --- Sidebar for user input ---
with st.sidebar:
    st.header("Data Source")

    # --- Section: Upload Your Own File ---
    st.subheader("Upload Your Own File")

    # Selection of the unit, moved under "Upload Your Own File"
    selected_unit = st.selectbox("Select the unit of the input data:", ["Volume in m³", "Mass in t (density required)"])

    # Initialize session state variables if they don't exist
    if 'einheit' not in st.session_state:
        st.session_state.einheit = "Volume in m³"
    if 'm_achsen' not in st.session_state:
        st.session_state.m_achsen = None
    if 'volumes_m3' not in st.session_state:
        st.session_state.volumes_m3 = None
    if 'uploaded_file_content' not in st.session_state:
        st.session_state.uploaded_file_content = None
    if 'uploaded_filename' not in st.session_state:
        st.session_state.uploaded_filename = None
    if 'block_count' not in st.session_state:
        st.session_state.block_count = None
    if 'file_source' not in st.session_state:
        st.session_state.file_source = None
    if 'success_message' not in st.session_state:
        st.session_state.success_message = None
    if 'last_error_message' not in st.session_state:
        st.session_state.last_error_message = None

    # Check if the unit has changed
    if st.session_state.einheit != selected_unit:
        st.session_state.einheit = selected_unit
        # Clear all data if unit changes, forcing re-upload
        clear_all_data()
        st.rerun()

    # Density input only if mass is selected
    density_input = None
    if selected_unit == "Mass in t (density required)":
        density_input = st.number_input("Enter the density in kg/m³:", min_value=250, value=2650, step=50)
        if density_input <= 0:
            st.error("Density must be equal to or greater than 250 kg/m³.")
            density_input = None

    uploaded_user_file = st.file_uploader(f"Upload your own file with {'m³' if selected_unit == 'Volume in m³' else 't'} values:", type=["txt"])
    st.info("Note: please make sure that all numbers in the uploaded text file use the dot ('.') instead of the comma (',') as decimal separator.")

    # Process user file if uploaded
    if uploaded_user_file is not None:
        if (st.session_state.uploaded_filename != uploaded_user_file.name or
                st.session_state.file_source != 'user'):
            with st.spinner("Processing uploaded file..."):
                clear_all_data()
                file_content = uploaded_user_file.read().decode("utf-8")

                m_axes, volumes_m3, block_count = process_uploaded_data(
                    file_content,
                    selected_unit,
                    density_input
                )

                if m_axes is not None:
                    st.session_state.uploaded_file_content = file_content
                    st.session_state.uploaded_filename = uploaded_user_file.name
                    st.session_state.m_achsen = m_axes
                    st.session_state.volumes_m3 = volumes_m3
                    st.session_state.block_count = block_count
                    st.session_state.file_source = 'user'
                    st.session_state.success_message = "Your file was processed successfully."
                    st.session_state.last_error_message = None
                    st.rerun()
                else:
                    st.session_state.last_error_message = "File could not be processed. Please check the format and decimal separator (use '.' instead of ',') and ensure the file contains valid numerical data."
                    st.session_state.success_message = None
                    st.rerun()

    # Check if user file was removed
    elif (uploaded_user_file is None and
          st.session_state.file_source == 'user' and
          st.session_state.m_achsen is not None):
        clear_all_data()
        st.info("File removed. Please upload a new file or select a sample file.")
        st.rerun()

    # Re-process file after unit change if a file is already loaded
    elif ('uploaded_file_content' in st.session_state and st.session_state.uploaded_file_content is not None and st.session_state.m_achsen is None):
        st.info("File needs to be re-processed due to unit change or initial load.")
        st.session_state.last_error_message = None

        m_axes, volumes_m3, block_count = process_uploaded_data(
            st.session_state.uploaded_file_content,
            selected_unit,
            density_input
        )

        if m_axes is not None:
            st.session_state.m_achsen = m_axes
            st.session_state.volumes_m3 = volumes_m3
            st.session_state.block_count = block_count
            st.rerun()
        else:
            st.session_state.last_error_message = "File could not be re-processed after unit change. Please check format."
            st.rerun()

    # --- Section: Load Example Files ---
    st.subheader("Load Sample File [m³]")
    st.info("Note: please make sure that your own uploaded file (if any) is removed before selecting a sample file.")
    for name, url in EXAMPLE_FILES.items():
        if st.button(f"Load sample file '{name}'"):
            with st.spinner(f"Loading '{name}'..."):
                clear_all_data()

                response = requests.get(url)
                if response.status_code == 200:
                    example_file_content = response.content.decode("utf-8")

                    m_axes, volumes_m3, block_count = process_uploaded_data(
                        example_file_content,
                        selected_unit,
                        density_input
                    )

                    if m_axes is not None:
                        st.session_state.uploaded_file_content = example_file_content
                        st.session_state.uploaded_filename = f"sample_{name}.txt"
                        st.session_state.m_achsen = m_axes
                        st.session_state.volumes_m3 = volumes_m3
                        st.session_state.block_count = block_count
                        st.session_state.file_source = 'sample'
                        st.session_state.success_message = f"The sample file '{name}' was loaded successfully."
                        st.session_state.last_error_message = None
                        st.rerun()
                    else:
                        st.session_state.last_error_message = f"Error processing the sample file '{name}'."
                        st.session_state.success_message = None
                        st.rerun()
                else:
                    st.session_state.last_error_message = f"Error loading the file '{name}'. Status code: {response.status_code}"
                    st.rerun()

    # --- Section: Support This Project ---
    st.markdown("---")
    st.subheader("Support this Project")
    st.write("If you find this application useful, consider supporting its development!")
    DONATION_LINK = 'https://www.buymeacoffee.com//ztilleditsz'
    st.link_button("Buy me a coffee ☕", url=DONATION_LINK)
    st.markdown("Thank you for your support! ❤️")

# --- Main Content Area ---
if st.session_state.m_achsen is not None:
    if st.session_state.success_message:
        st.success(st.session_state.success_message)
        st.session_state.success_message = None

if st.session_state.last_error_message:
    st.error(st.session_state.last_error_message)
    st.session_state.last_error_message = None

if st.session_state.m_achsen is None:
    st.info("Please upload your own file (and select a unit) in the left sidebar or load a sample file to get started.")
else:
    # Display file information
    if st.session_state.uploaded_filename and st.session_state.block_count is not None:
        st.subheader(f"File Information:")
        st.write(f"**Filename:** {st.session_state.uploaded_filename}")
        st.write(f"**Number of blocks:** {st.session_state.block_count}")

        # Display file content
        if st.session_state.uploaded_file_content:
            st.subheader(f"Contents of {st.session_state.uploaded_filename}:")
            st.text_area("File content:", st.session_state.uploaded_file_content, height=200, disabled=True)

    st.subheader("Visualization of Probability Distribution")
    fig1 = calculate_and_visualize_percentiles(st.session_state.m_achsen)
    st.pyplot(fig1)

    st.subheader("Fitting Probability Functions")
    selected_dists = ['genexpon', 'expon', 'powerlaw']

    fig2 = fit_distributions_and_visualize(st.session_state.m_achsen, selected_dists)
    st.pyplot(fig2)

    st.subheader("Tabular Comparison of Percentiles")

    percentiles_to_show = [0, 25, 50, 75, 95, 96, 97, 98, 99, 100]

    required_params = ['a1', 'b1', 'c1', 'loc1', 'scale1', 'loc3', 'scale3', 'a4', 'loc4', 'scale4']
    if all(param in st.session_state for param in required_params):
        try:
            L1s = calculate_distribution_percentiles(stats.genexpon, percentiles_to_show,
                                                    st.session_state.a1, st.session_state.b1, st.session_state.c1,
                                                    st.session_state.loc1, st.session_state.scale1)
            L3s = calculate_distribution_percentiles(stats.expon, percentiles_to_show,
                                                    st.session_state.loc3, st.session_state.scale3)
            L4s = calculate_distribution_percentiles(stats.powerlaw, percentiles_to_show,
                                                    st.session_state.a4, st.session_state.loc4, st.session_state.scale4)

            upload_perz = np.array(np.percentile(st.session_state.m_achsen, percentiles_to_show))
            L1s = np.array(L1s)
            L3s = np.array(L3s)
            L4s = np.array(L4s)

            upload_perz3 = upload_perz**3
            L1s3 = L1s**3
            L3s3 = L3s**3
            L4s3 = L4s**3

            df1 = pd.DataFrame({
                "percentile": [str(p) for p in percentiles_to_show],
                "sample [m³]": upload_perz3,
                "expon [m³]": L3s3,
                "genexpon [m³]": L1s3,
                "powerlaw [m³]": L4s3
            })

            def highlight_rows(s):
                is_bold = ['font-weight: bold' if i in [4, 5, 6, 7] else '' for i in range(len(s))]
                return is_bold

            styled_df = df1.style.apply(highlight_rows, axis=0)
            styled_df = styled_df.hide(axis="index")
            st.dataframe(styled_df)

        except Exception as e:
            st.error(f"Could not calculate percentiles: {e}. Please ensure data is loaded correctly and distributions are fitted.")
    else:
        st.info("Please load data and ensure distributions are fitted to see the percentile table.")

    # --- NEUER ABSCHNITT: Generiere und lade gefilterte Verteilung herunter ---
    st.subheader("Generate and download filtered distribution")
    st.markdown("Select a fitted distribution and define the block axis range to generate a custom block list. You can then choose the output unit for download.")

    available_dists_for_download = []
    if 'a1' in st.session_state and 'b1' in st.session_state and 'c1' in st.session_state and 'loc1' in st.session_state and 'scale1' in st.session_state:
        available_dists_for_download.append('genexpon')
    if 'loc3' in st.session_state and 'scale3' in st.session_state:
        available_dists_for_download.append('expon')
    if 'a4' in st.session_state and 'loc4' in st.session_state and 'scale4' in st.session_state:
        available_dists_for_download.append('powerlaw')

    selected_download_distribution = None
    if not available_dists_for_download:
        st.info("Please load data and select one distribution in the 'Fitting Probability Functions' section to enable generating a filtered distribution.")
        generation_possible_overall = False
    else:
        selected_download_distribution = st.selectbox(
            "Select Distribution for Generation:",
            options=available_dists_for_download,
            key="selected_download_dist"
        )
        generation_possible_overall = True

    if generation_possible_overall:
        st.markdown(f"Generating blocks from the fitted **{selected_download_distribution}** distribution.")

        st.markdown("Set the minimum and maximum **block axis** values for the generated distribution (values outside this range will be excluded).")
        st.info("Note: the default min. value of 30 cm block axis corresponds to a volume of 0.027 m³; the default max. value of 1.5 m block axis corresponds to a volume of 3.375 m³ --- **Expert Opinion required!** ---")
        col_min_max_1, col_min_max_2 = st.columns(2)
        with col_min_max_1:
            min_block_axis = st.number_input(
                "Minimum Block Axis [m]:",
                min_value=0.10,
                value=0.30,
                step=0.05,
                format="%.2f",
                key="min_block_axis_input_download"
            )
        with col_min_max_2:
            max_block_axis = st.number_input(
                "Maximum Block Axis [m]",
                min_value=min_block_axis + 0.05,
                value=1.50,
                step=0.05,
                format="%.2f",
                key="max_block_axis_input_download"
            )

        if min_block_axis >= max_block_axis:
            st.error("Minimum block axis must be less than maximum block axis.")
            generation_possible_overall = False

        num_samples = st.slider(
            "Number of blocks to generate:",
            min_value=500,
            max_value=10000,
            value=1000,
            step=500,
            key="num_samples_slider_download",
            help="Generate a larger number of samples to account for filtering."
        )

        st.markdown("Choose the output unit for the generated block list (block mass (t) is required for THROW):")
        selected_output_unit = st.radio(
            "Output Unit:",
            ('Block Volume (m³)', 'Block Mass (t)'),
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
                generation_possible_overall = False

        if generation_possible_overall:
            if st.button("Generate Filtered Block List", key="generate_filtered_button"):
                with st.spinner("Generating and filtering distribution..."):
                    generated_raw_axes = []
                    params_for_rvs = ()

                    if selected_download_distribution == 'genexpon':
                        params_for_rvs = (st.session_state.a1, st.session_state.b1, st.session_state.c1,
                                          st.session_state.loc1, st.session_state.scale1)
                        dist_function = stats.genexpon
                    elif selected_download_distribution == 'expon':
                        params_for_rvs = (st.session_state.loc3, st.session_state.scale3)
                        dist_function = stats.expon
                    elif selected_download_distribution == 'powerlaw':
                        params_for_rvs = (st.session_state.a4, st.session_state.loc4, st.session_state.scale4)
                        dist_function = stats.powerlaw

                    generated_raw_axes = dist_function.rvs(*params_for_rvs, size=num_samples * 10)

                    filtered_samples_m_axis = [s for s in generated_raw_axes if min_block_axis <= s <= max_block_axis]

                    final_generated_blocks_m_axis = sorted(filtered_samples_m_axis[:num_samples])

                    if final_generated_blocks_m_axis:
                        final_output_blocks_converted = []
                        if selected_output_unit == 'Block Volume (m³)':
                            final_output_blocks_converted = [x**3 for x in final_generated_blocks_m_axis]
                            st.session_state.output_unit_for_download = 'm3'
                        elif selected_output_unit == 'Block Mass (t)':
                            if download_density_kg_m3 and download_density_kg_m3 > 0:
                                final_output_blocks_converted = [(x**3 * download_density_kg_m3) / 1000 for x in final_generated_blocks_m_axis]
                                st.session_state.output_unit_for_download = 't'
                            else:
                                st.error("Invalid density for mass calculation. Please provide a positive density.")
                                final_output_blocks_converted = []
                                st.session_state.output_unit_for_download = None

                        st.session_state.generated_blocks_for_download = final_output_blocks_converted

                        if final_output_blocks_converted:
                            st.success(f"Generated {len(final_output_blocks_converted)} blocks for download in '{selected_output_unit}'.")

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
        else:
            st.info("Adjust input values or ensure selected distribution parameters are available for generation.")

    if 'generated_blocks_for_download' in st.session_state and st.session_state.generated_blocks_for_download and st.session_state.output_unit_for_download:
        if st.session_state.output_unit_for_download == 'm3':
             output_string = "\n".join(f"{x:.6f}" for x in st.session_state.generated_blocks_for_download)
             file_name = f"{selected_download_distribution}_filtered_{st.session_state.output_unit_for_download}.txt"
        else:
             output_string = "\n".join(f"{x:.6f}" for x in st.session_state.generated_blocks_for_download)
             file_name = f"{selected_download_distribution}_filtered_{st.session_state.output_unit_for_download}.txt"

        st.download_button(
            label=f"Download Generated Block List ({st.session_state.output_unit_for_download.upper()})",
            data=output_string,
            file_name=file_name,
            mime="text/plain",
            help=f"Download the list of generated block values in {st.session_state.output_unit_for_download.upper()}."
        )