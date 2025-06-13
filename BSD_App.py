# -*- coding: utf-8 -*-
# --------------------------------------------------------------
# BVG ... a block size distribution code and app by Mariella Illeditsch
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

# Pfad zum Logo
LOGO_PATH = "pi-geotechnik-1-RGB-192-30-65.png"

# Beispiel-Datei URLs auf GitHub
EXAMPLE_FILES = {
    #"Kalk": "https://github.com/pi-geotechnik/Blockverteilung/raw/main/blocklist_dachsteinkalk_m3.txt", # Auskommentiert, da nicht in der Original-App verwendet
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
        
        st.write(f"Number of blocks: {len(values)}")
        
        if unit_type == "Volume in m³":
            m_axes = [calculate_cubic_root(val) for val in values]
            volumes_m3 = values
        elif unit_type == "Mass in t (density required)":
            if density_kg_m3 is None:
                st.error("Density is required for mass input.")
                return None, None
            volumes_m3 = [val * 1000 / density_kg_m3 for val in values]
            m_axes = [calculate_cubic_root(val) for val in volumes_m3]
        
        return m_axes, volumes_m3
    except Exception as e:
        st.error(f"Error processing data: {e}. Please ensure numbers use a dot '.' as a decimal separator.")
        return None, None

# --- Streamlit App Layout ---

# Header and Info
st.image(Image.open(LOGO_PATH), caption="https://pi-geo.at/", width=300)
st.title("Block Size Distribution")
st.markdown("""
    *A block distribution code by Mariella ILLEDITSCH, adapted for Streamlit by Mariella ILLEDITSCH*
    
    Version 1, Mar 2025
    
    This application visualizes block size distributions, fits distribution functions to them and (coming soon) returns blocklists of the fitted distribution for rockfall simulation (with THROW).
""")

# --- Sidebar for user input ---
with st.sidebar:
    st.header("Input Data")

    # Initialize session state variables if they don't exist
    if 'einheit' not in st.session_state:
        st.session_state.einheit = "Volume in m³"
    if 'm_achsen' not in st.session_state:
        st.session_state.m_achsen = None
    if 'volumes_m3' not in st.session_state:
        st.session_state.volumes_m3 = None

    selected_unit = st.selectbox("Select the unit of the input data:", ["Volume in m³", "Mass in t (density required)"])

    # Check if the unit has changed
    if st.session_state.einheit != selected_unit:
        st.session_state.einheit = selected_unit
        # Clear data if unit changes, forcing re-upload
        st.session_state.m_achsen = None
        st.session_state.volumes_m3 = None
        st.session_state.uploaded_file = None # Clear uploaded file
        st.warning("Please upload a block file. Attention: Please make sure that all numbers in the uploaded text file use the dot ('.') instead of the comma (',') as decimal separator!")
        st.rerun() # Rerun to clear plots immediately

    # Density input only if mass is selected
    density_input = None
    if selected_unit == "Mass in t (density required)":
        density_input = st.number_input("Enter the density in kg/m³:", min_value=1, value=2650, step=10)
        if density_input <= 0:
            st.error("Density must be greater than 0.")
            density_input = None # Prevent processing with invalid density

    st.subheader("Load Example Files")
    for name, url in EXAMPLE_FILES.items():
        if st.button(f"Load sample file '{name}'"):
            with st.spinner(f"Loading '{name}'..."):
                response = requests.get(url)
                if response.status_code == 200:
                    example_file_content = response.content
                    # Simulate file upload for consistent processing
                    st.session_state.uploaded_file_content = example_file_content.decode("utf-8")
                    st.session_state.uploaded_filename = f"sample_{name}.txt"
                    
                    # Process the data
                    m_axes, volumes_m3 = process_uploaded_data(
                        st.session_state.uploaded_file_content,
                        selected_unit,
                        density_input
                    )
                    st.session_state.m_achsen = m_axes
                    st.session_state.volumes_m3 = volumes_m3

                    st.success(f"The sample file '{name}' was loaded successfully.")
                    # Force rerun to update main content area with processed data
                    st.rerun()
                else:
                    st.error(f"Error loading the file '{name}'. Status code: {response.status_code}")

    st.subheader("Upload Your Own File")
    uploaded_user_file = st.file_uploader(f"Upload your own file with {'m³' if selected_unit == 'Volume in m³' else 't'} values:", type=["txt"])
    
    if uploaded_user_file is not None:
        with st.spinner("Processing uploaded file..."):
            file_content = uploaded_user_file.read().decode("utf-8")
            st.session_state.uploaded_file_content = file_content
            st.session_state.uploaded_filename = uploaded_user_file.name
            
            m_axes, volumes_m3 = process_uploaded_data(
                file_content,
                selected_unit,
                density_input
            )
            st.session_state.m_achsen = m_axes
            st.session_state.volumes_m3 = volumes_m3
            
            # --- DEBUGGING OUTPUT ---
            st.write(f"DEBUG: m_axes after user upload: {m_axes[:5]}...") # Show first 5 elements
            st.write(f"DEBUG: session_state.m_achsen after user upload: {st.session_state.m_achsen[:5]}...") # Show first 5 elements
            # --- END DEBUGGING OUTPUT ---
            
            if m_axes is not None: # Check if data processing was successful
                st.success("Your file was processed successfully.")
                # Force rerun to update main content area with processed data
                st.rerun()
            else:
                st.error("File could not be processed. Please check the format and decimal separator (use '.' instead of ',').")
                # Do not rerun if processing failed, to keep error message visible
    elif 'uploaded_file_content' in st.session_state and st.session_state.uploaded_file_content is not None and st.session_state.m_achsen is None:
        # If a file was loaded (e.g., example) but then unit changed and m_achsen cleared, re-process if possible
        # This handles the case where unit changes and an example file is already "loaded"
        st.info("File needs to be re-processed due to unit change or initial load.")
        m_axes, volumes_m3 = process_uploaded_data(
            st.session_state.uploaded_file_content,
            selected_unit,
            density_input
        )
        st.session_state.m_achsen = m_axes
        st.session_state.volumes_m3 = volumes_m3
        
        # --- DEBUGGING OUTPUT ---
        st.write(f"DEBUG: m_axes after re-processing: {m_axes[:5]}...") # Show first 5 elements
        st.write(f"DEBUG: session_state.m_achsen after re-processing: {st.session_state.m_achsen[:5]}...") # Show first 5 elements
        # --- END DEBUGGING OUTPUT ---

        if m_axes is not None: # Only rerun if re-processing was successful
            st.rerun() # Added rerun here to update UI after re-processing
            
# --- Main Content Area ---
if st.session_state.m_achsen is None:
    st.info("Please select a unit and load an example file or upload your own file to get started.")
else:
    # Display file content if available
    if 'uploaded_file_content' in st.session_state and st.session_state.uploaded_file_content:
        st.subheader(f"Contents of {st.session_state.uploaded_filename}:")
        st.text_area("File content:", st.session_state.uploaded_file_content, height=200, disabled=True)

    st.subheader("Visualization of Probability Distribution")
    fig1 = calculate_and_visualize_percentiles(st.session_state.m_achsen)
    st.pyplot(fig1)

    st.subheader("Fitting Probability Functions")
    # Allow user to select distributions
    # selected_dists = st.multiselect(
    #     "Select distributions to fit:",
    #     ['genexpon', 'expon', 'powerlaw'],
    #     default=['genexpon', 'expon', 'powerlaw'] # All selected by default
    # )
    # For now, keep all distributions automatically calculated as per original code logic
    selected_dists = ['genexpon', 'expon', 'powerlaw'] 
    
    fig2 = fit_distributions_and_visualize(st.session_state.m_achsen, selected_dists)
    st.pyplot(fig2)

    st.subheader("Tabular Comparison of Percentiles")

    percentiles_to_show = [0, 25, 50, 75, 95, 96, 97, 98, 99, 100]

    # Check if all required parameters for fitting are in session_state
    # This ensures the table is only shown after distributions have been successfully fitted
    required_params = ['a1', 'b1', 'c1', 'loc1', 'scale1', 'loc3', 'scale3', 'a4', 'loc4', 'scale4']
    if all(param in st.session_state for param in required_params):
        try:
            # Calculate percentiles for each distribution
            L1s = calculate_distribution_percentiles(stats.genexpon, percentiles_to_show, 
                                                    st.session_state.a1, st.session_state.b1, st.session_state.c1, 
                                                    st.session_state.loc1, st.session_state.scale1)
            L3s = calculate_distribution_percentiles(stats.expon, percentiles_to_show, 
                                                    st.session_state.loc3, st.session_state.scale3)
            L4s = calculate_distribution_percentiles(stats.powerlaw, percentiles_to_show, 
                                                    st.session_state.a4, st.session_state.loc4, st.session_state.scale4)
            
            # Ensure all percentiles are numpy arrays for easier cubic calculation
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
                "upload [m³]": upload_perz3,
                "expon [m³]": L3s3,
                "genexpon [m³]": L1s3,
                "powerlaw [m³]": L4s3
            })
            
            # CSS-Styling for specific rows (5th to 8th row bold) - using a slightly more robust method
            def highlight_rows(s):
                is_bold = ['font-weight: bold' if i in [4, 5, 6, 7] else '' for i in range(len(s))]
                return is_bold

            styled_df = df1.style.apply(highlight_rows, axis=0) # Apply to columns to check index
            styled_df = styled_df.hide(axis="index") # Remove the index column
            st.dataframe(styled_df)

        except Exception as e:
            st.error(f"Could not calculate percentiles: {e}. Please ensure data is loaded correctly and distributions are fitted.")
    else:
        st.info("Please load data and ensure distributions are fitted to see the percentile table.")