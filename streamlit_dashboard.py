"""
Streamlit dashboard for fertiliser affordability analysis
========================================================

This script builds an interactive dashboard for comparing fertiliser
affordability across crops and regions.  It reads data from the
``Monthly Avg Summary`` sheet of a provided Excel workbook and
calculates a variety of analytics including price ratios, z‑scores,
percentile ranks, indexed affordability and programme cost per hectare.

The dashboard offers two modes:

* **Admin mode** – enter a password in the sidebar to unlock the file
  uploader.  Upload a new workbook to refresh the underlying data.  By
  default the script will read ``OSRFW Historic.xlsx`` from the
  working directory.
* **User mode** – choose the crop, region, date range, fertiliser
  products and metric to visualise.  Charts update automatically and
  short descriptions explain each metric in plain language.

To run the app locally, install the required dependencies and start
Streamlit:

```
pip install streamlit pandas altair openpyxl numpy
streamlit run streamlit_dashboard.py
```

Note: for production use you should implement a secure authentication
system and manage uploaded files appropriately.  The example below
uses a simple password check for demonstration purposes only.
"""

import datetime
from typing import List
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt


###############################
# Configuration
###############################

# Title and layout
st.set_page_config(
    page_title="Fertiliser Affordability Dashboard",
    page_icon="🌾",
    layout="wide",
)

# Hard‑coded admin password.  In a real application you would store
# this securely (e.g. in Streamlit Secrets or an environment
# variable).  To change the password, set an ``ADMIN_PASSWORD`` secret
# or edit this default value.
DEFAULT_ADMIN_PASSWORD = "admin123"
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", DEFAULT_ADMIN_PASSWORD)


###############################
# Utility functions
###############################

@st.cache_data
def load_workbook(uploaded_file=None) -> pd.DataFrame:
    """
    Load and preprocess the workbook.

    Parameters
    ----------
    uploaded_file : Any, optional
        The file object provided by Streamlit's uploader.  If ``None``
        is supplied, the function will attempt to read a local file
        named ``OSRFW Historic.xlsx``.  The relevant sheet is
        ``Monthly Avg Summary``.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the cleaned data and calculated
        analytics.
    """
    if uploaded_file is not None:
        data = pd.read_excel(uploaded_file, sheet_name="Monthly Avg Summary")
    else:
        data = pd.read_excel("OSRFW Historic.xlsx", sheet_name="Monthly Avg Summary")

    # Ensure date column is parsed correctly
    data["Date"] = pd.to_datetime(data["Date"])

    # Fertiliser price columns
    fert_cols = [
        "Nitram",
        "Imported Ammonium Nitrate",
        "Granular Urea",
        "MOP",
        "DAP",
        "TSP",
        "Sum Complex",
    ]

    # Compute per‑crop metrics
    analytics = data.copy()

    # Ratio metrics: fertiliser price divided by crop price
    for col in fert_cols:
        analytics[f"{col}_ratio"] = analytics[col] / analytics["Average Price"]

    # Basket ratio: simple average of available ratios
    ratio_columns = [f"{c}_ratio" for c in fert_cols]
    analytics["Basket_ratio"] = analytics[ratio_columns].mean(axis=1, skipna=True)

    # Prepare group‑by object for computing z‑scores, percentiles and indexes
    grouped = analytics.groupby("Commodity")
    def compute_group_metrics(group: pd.DataFrame) -> pd.DataFrame:
        # Sort by date for index calculations
        group = group.sort_values("Date").copy()
        for col in ratio_columns:
            series = group[col]
            # z‑score: (value – mean) / std
            mean = series.mean()
            std = series.std(ddof=0)
            group[f"{col}_z"] = (series - mean) / std
            # percentile rank: fraction of values less than or equal to current
            group[f"{col}_percentile"] = series.rank(pct=True)
            # indexed affordability: first value normalised to 100
            base = series.iloc[0]
            group[f"{col}_index"] = (series / base) * 100 if pd.notnull(base) else np.nan
        # Basket metrics
        b = group["Basket_ratio"]
        b_mean = b.mean()
        b_std = b.std(ddof=0)
        group["Basket_z"] = (b - b_mean) / b_std
        group["Basket_percentile"] = b.rank(pct=True)
        base_b = b.iloc[0]
        group["Basket_index"] = (b / base_b) * 100 if pd.notnull(base_b) else np.nan
        return group
    analytics = grouped.apply(compute_group_metrics)
    # Remove redundant index columns introduced by groupby
    analytics = analytics.reset_index(drop=True)

    # Programme cost: simple nitrogen programme based on RB209 guidelines
    # We assume an N requirement of 220 kg/ha for cereal crops and a
    # yield of 8.5 t/ha.  Fertiliser contents: Nitram and Imported AN
    # contain 34.5% N; granular urea contains 46% N.
    N_REQUIREMENT = 220  # kg/ha
    YIELD = 8.5          # t/ha
    # Compute tonnes of product required per hectare
    nitram_tonnes = N_REQUIREMENT / (0.345 * 1000)
    urea_tonnes = N_REQUIREMENT / (0.46 * 1000)
    # Imported AN is treated the same as Nitram
    imp_an_tonnes = nitram_tonnes
    # Cost per tonne as percentage of crop revenue
    analytics["Nitram_cost_pct"] = (analytics["Nitram"] * nitram_tonnes) / (YIELD * analytics["Average Price"]) * 100
    analytics["Imported_AN_cost_pct"] = (analytics["Imported Ammonium Nitrate"] * imp_an_tonnes) / (YIELD * analytics["Average Price"]) * 100
    analytics["Urea_cost_pct"] = (analytics["Granular Urea"] * urea_tonnes) / (YIELD * analytics["Average Price"]) * 100

    return analytics


@st.cache_data
def get_metric_description(metric_key: str) -> str:
    """Return a human‑readable description for a metric.

    Parameters
    ----------
    metric_key: str
        One of ``ratio``, ``z``, ``percentile``, ``index``, ``cost_pct``,
        ``Basket_ratio``, ``Basket_z``, ``Basket_percentile`` or
        ``Basket_index``.

    Returns
    -------
    str
        Description explaining the meaning of the metric in plain language.
    """
    descriptions = {
        "ratio": "Fertiliser price divided by the crop price on the same date.  Values above 1 indicate the fertiliser is more expensive per tonne than the crop.",
        "z": "Standardised score showing how many standard deviations the current ratio is above or below the historical mean.  Positive values indicate fertiliser is expensive relative to its long‑run average.",
        "percentile": "Percentile rank of the ratio within its historical distribution for the chosen crop.  For example, 0.8 means the ratio is higher than 80% of past observations.",
        "index": "Indexed affordability, normalised to 100 at the earliest date for the selected crop.  Values above 100 indicate fertiliser has become relatively more expensive.",
        "cost_pct": "Estimated programme cost as a percentage of crop revenue, based on a nitrogen requirement of 220 kg/ha and a yield of 8.5 t/ha.",
        "Basket_ratio": "Average of all available fertiliser price ratios for the selected crop and region.",
        "Basket_z": "Z‑score of the basket ratio, measuring how far it deviates from its historical mean for the selected crop.",
        "Basket_percentile": "Percentile rank of the basket ratio within its historical distribution.",
        "Basket_index": "Indexed basket affordability, normalised to 100 at the earliest date.",
    }
    return descriptions.get(metric_key, "")


###############################
# Sidebar: admin and controls
###############################

st.sidebar.header("Settings")

# Admin upload section
admin_expander = st.sidebar.expander("Admin Login", expanded=False)
with admin_expander:
    admin_input = st.text_input("Enter admin password", type="password")
    if admin_input:
        if admin_input == ADMIN_PASSWORD:
            st.success("Admin authenticated")
            uploaded_file = st.file_uploader(
                "Upload updated workbook (Excel)", type=["xlsx"], key="file_uploader"
            )
        else:
            uploaded_file = None
            st.error("Incorrect password")
    else:
        uploaded_file = None
        st.info("Enter password to upload new data")

###############################
# Data loading
###############################
df = None
if uploaded_file:
    # Use the uploaded file
    df = load_workbook(uploaded_file)
else:
    # If running locally, attempt to read a default workbook. Otherwise
    # inform the user that they need to upload a file.
    default_path = Path("OSRFW Historic.xlsx")
    if default_path.exists():
        df = load_workbook()
    else:
        st.info(
            "No workbook uploaded and no default dataset found. Please use the Admin Login to upload your spreadsheet."
        )
        st.stop()


###############################
# Sidebar: user filters
###############################

# Commodity selection
commodities = df["Commodity"].dropna().unique().tolist()
selected_commodity = st.sidebar.selectbox(
    "Commodity", options=sorted(commodities), index=0
)

# Region/home selection
homes = df[df["Commodity"] == selected_commodity]["Home"].dropna().unique().tolist()
selected_home = st.sidebar.selectbox(
    "Region (Home)", options=sorted(homes), index=0
)

# Date range selection
crop_df = df[(df["Commodity"] == selected_commodity) & (df["Home"] == selected_home)]
min_date = crop_df["Date"].min()
max_date = crop_df["Date"].max()
date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date(),
)

# Fertiliser products selection
fertiliser_columns = [
    "Nitram",
    "Imported Ammonium Nitrate",
    "Granular Urea",
    "MOP",
    "DAP",
    "TSP",
    "Sum Complex",
    "Basket",  # special case for basket metrics
]

selected_products = st.sidebar.multiselect(
    "Fertiliser products", options=fertiliser_columns, default=["Nitram"]
)

# Metric selection
metric_options = {
    "Ratio": "ratio",
    "Z‑Score": "z",
    "Percentile": "percentile",
    "Indexed": "index",
    "Programme Cost %": "cost_pct",
}

selected_metric_label = st.sidebar.selectbox(
    "Metric", options=list(metric_options.keys()), index=0
)
selected_metric_key = metric_options[selected_metric_label]

###############################
# Main content
###############################

st.title("📊 Fertiliser Affordability Dashboard")
st.markdown(
    """
    Use the controls in the sidebar to explore how fertiliser prices
    compare to crop values over time.  Select a commodity, region,
    date range, products and a metric to see the results.  Hover
    over the lines for precise values.
    """
)

# Filter data based on user selections
mask = (
    (df["Commodity"] == selected_commodity)
    & (df["Home"] == selected_home)
    & (df["Date"].dt.date >= date_range[0])
    & (df["Date"].dt.date <= date_range[1])
)
plot_data = df[mask].copy()

# Prepare data for plotting
def prepare_long_form(data: pd.DataFrame, products: List[str], metric_key: str) -> pd.DataFrame:
    records = []
    for prod in products:
        if prod == "Basket":
            if metric_key == "cost_pct":
                # Basket cost is not defined; skip
                continue
            col_name = f"Basket_{metric_key}" if metric_key != "ratio" else "Basket_ratio"
            if col_name in data.columns:
                subset = data[["Date", col_name]].rename(columns={col_name: "value"})
                subset["Product"] = "Basket"
                records.append(subset)
        else:
            if metric_key == "cost_pct":
                if prod == "Nitram":
                    col_name = "Nitram_cost_pct"
                elif prod == "Imported Ammonium Nitrate":
                    col_name = "Imported_AN_cost_pct"
                elif prod == "Granular Urea":
                    col_name = "Urea_cost_pct"
                else:
                    # No cost data for this fertiliser
                    continue
            else:
                base_col = f"{prod}_{metric_key}" if metric_key != "ratio" else f"{prod}_ratio"
                col_name = base_col
            if col_name in data.columns:
                subset = data[["Date", col_name]].rename(columns={col_name: "value"})
                subset["Product"] = prod
                records.append(subset)
    if records:
        long_df = pd.concat(records)
        return long_df
    else:
        return pd.DataFrame(columns=["Date", "value", "Product"])

chart_df = prepare_long_form(plot_data, selected_products, selected_metric_key)

###############################
# Plotting
###############################

if not chart_df.empty:
    # Create an interactive line chart
    line_chart = (
        alt.Chart(chart_df)
        .mark_line()
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("value:Q", title=selected_metric_label),
            color=alt.Color("Product:N", legend=alt.Legend(title="Product")),
            tooltip=["Date:T", "Product:N", alt.Tooltip("value:Q", format=".2f")],
        )
        .interactive()
    )
    st.altair_chart(line_chart, use_container_width=True)
else:
    st.warning("No data available for the selected filters and metric.")


###############################
# Metric explanation
###############################

with st.expander("Metric description", expanded=False):
    description = get_metric_description(
        "Basket_index" if selected_products == ["Basket"] and selected_metric_key == "index" else
        "Basket_z" if selected_products == ["Basket"] and selected_metric_key == "z" else
        "Basket_percentile" if selected_products == ["Basket"] and selected_metric_key == "percentile" else
        selected_metric_key
    )
    st.write(description)


###############################
# Data table
###############################

with st.expander("Show data", expanded=False):
    st.dataframe(chart_df.sort_values("Date"), use_container_width=True)
