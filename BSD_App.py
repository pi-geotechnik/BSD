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
                         stats.genexpon.ppf(0.999, a1, b1, c1, loc=loc1, scale=scale1), len