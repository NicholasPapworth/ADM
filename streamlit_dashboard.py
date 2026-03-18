"""
Streamlit dashboard for fertiliser affordability analysis
========================================================

This script builds an interactive dashboard for comparing fertiliser
affordability across crops and regions. It reads data from the
"Monthly Avg Summary" sheet of a provided Excel workbook.

Key logic:
- Load raw data
- Filter by the user's selected commodity, home, date range
- Build the selected product series
- Calculate viewer-specific metrics only on that filtered series

This ensures that:
if the user selects Feed Wheat + East Anglia + Granular Urea,
then percentile/z-score/index are calculated only versus the
historical East Anglia Granular Urea vs Feed Wheat series that
the user is actually viewing.
"""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt


###############################
# Configuration
###############################

st.set_page_config(
    page_title="Fertiliser Affordability Dashboard",
    page_icon="🌾",
    layout="wide",
)

DEFAULT_ADMIN_PASSWORD = "admin123"
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", DEFAULT_ADMIN_PASSWORD)


###############################
# Utility functions
###############################

@st.cache_data
def load_workbook(uploaded_file=None) -> pd.DataFrame:
    """
    Load and preprocess the workbook.

    Only raw/base metrics are calculated here.
    Viewer-specific metrics (z-score, percentile, index) are calculated
    later after filtering.
    """
    if uploaded_file is not None:
        data = pd.read_excel(uploaded_file, sheet_name="Monthly Avg Summary")
    else:
        data = pd.read_excel("OSRFW Historic.xlsx", sheet_name="Monthly Avg Summary")

    data["Date"] = pd.to_datetime(data["Date"])

    fert_cols = [
        "Nitram",
        "Imported Ammonium Nitrate",
        "Granular Urea",
        "MOP",
        "DAP",
        "TSP",
        "Sum Complex",
    ]

    analytics = data.copy()

    # Base affordability ratios only
    for col in fert_cols:
        analytics[f"{col}_ratio"] = analytics[col] / analytics["Average Price"]

    # Basket ratio only
    ratio_columns = [f"{c}_ratio" for c in fert_cols]
    analytics["Basket_ratio"] = analytics[ratio_columns].mean(axis=1, skipna=True)

    # Programme cost assumptions
    N_REQUIREMENT = 220  # kg/ha
    YIELD = 8.5          # t/ha

    nitram_tonnes = N_REQUIREMENT / (0.345 * 1000)
    urea_tonnes = N_REQUIREMENT / (0.46 * 1000)
    imp_an_tonnes = nitram_tonnes

    analytics["Nitram_cost_pct"] = (
        analytics["Nitram"] * nitram_tonnes
    ) / (YIELD * analytics["Average Price"]) * 100

    analytics["Imported_AN_cost_pct"] = (
        analytics["Imported Ammonium Nitrate"] * imp_an_tonnes
    ) / (YIELD * analytics["Average Price"]) * 100

    analytics["Urea_cost_pct"] = (
        analytics["Granular Urea"] * urea_tonnes
    ) / (YIELD * analytics["Average Price"]) * 100

    return analytics


@st.cache_data
def get_metric_description(metric_key: str) -> str:
    descriptions = {
        "ratio": (
            "Fertiliser price divided by the crop price on the same date. "
            "Values above 1 indicate the fertiliser is more expensive per tonne than the crop."
        ),
        "z": (
            "Standardised score showing how far the currently viewed series sits above or below "
            "its own historical mean. Positive values indicate the ratio is rich relative to its "
            "own history."
        ),
        "percentile": (
            "Percentile rank of the currently viewed series versus its own historical distribution. "
            "For example, 0.8 means the current ratio is higher than 80% of past observations in "
            "that same selected series."
        ),
        "index": (
            "Indexed affordability for the currently viewed series, normalised to 100 at the first "
            "available observation in the selected history."
        ),
        "cost_pct": (
            "Estimated programme cost as a percentage of crop revenue, based on a nitrogen "
            "requirement of 220 kg/ha and a yield of 8.5 t/ha."
        ),
        "Basket_ratio": (
            "Average of all available fertiliser price ratios for the selected crop and region."
        ),
        "Basket_z": (
            "Z-score of the basket ratio versus the history of the same selected basket series."
        ),
        "Basket_percentile": (
            "Percentile rank of the basket ratio versus the history of the same selected basket series."
        ),
        "Basket_index": (
            "Indexed basket affordability, normalised to 100 at the first observation in the selected series."
        ),
    }
    return descriptions.get(metric_key, "")


def prepare_long_form(data: pd.DataFrame, products: List[str], metric_key: str) -> pd.DataFrame:
    """
    Build a long-form dataframe from the filtered dataset.

    For ratio/cost_pct this uses precomputed base columns.
    For z/percentile/index we always start from the filtered ratio series
    and calculate those metrics afterwards.
    """
    records = []

    for prod in products:
        if prod == "Basket":
            if metric_key == "cost_pct":
                continue
            col_name = "Basket_ratio" if metric_key in {"ratio", "z", "percentile", "index"} else "Basket_ratio"
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
                    continue
            else:
                col_name = f"{prod}_ratio"

            if col_name in data.columns:
                subset = data[["Date", col_name]].rename(columns={col_name: "value"})
                subset["Product"] = prod
                records.append(subset)

    if records:
        return pd.concat(records, ignore_index=True)

    return pd.DataFrame(columns=["Date", "value", "Product"])


def compute_viewer_metric(chart_df: pd.DataFrame, metric_key: str) -> pd.DataFrame:
    """
    Compute viewer-specific metrics only on the filtered series being viewed.
    """
    if chart_df.empty:
        return chart_df

    chart_df = chart_df.copy()
    chart_df = chart_df.dropna(subset=["value"])
    chart_df = chart_df.sort_values(["Product", "Date"])

    if metric_key == "ratio":
        return chart_df

    if metric_key == "percentile":
        chart_df["value"] = chart_df.groupby("Product")["value"].rank(pct=True)
        return chart_df

    if metric_key == "z":
        def zscore(s: pd.Series) -> pd.Series:
            std = s.std(ddof=0)
            if pd.isna(std) or std == 0:
                return pd.Series(np.zeros(len(s)), index=s.index)
            return (s - s.mean()) / std

        chart_df["value"] = chart_df.groupby("Product")["value"].transform(zscore)
        return chart_df

    if metric_key == "index":
        def index_series(s: pd.Series) -> pd.Series:
            first_valid = s.iloc[0] if len(s) > 0 else np.nan
            if pd.isna(first_valid) or first_valid == 0:
                return pd.Series(np.nan, index=s.index)
            return (s / first_valid) * 100

        chart_df["value"] = chart_df.groupby("Product")["value"].transform(index_series)
        return chart_df

    if metric_key == "cost_pct":
        return chart_df

    return chart_df


###############################
# Sidebar: admin and controls
###############################

st.sidebar.header("Settings")

admin_expander = st.sidebar.expander("Admin Login", expanded=False)
with admin_expander:
    admin_input = st.text_input("Enter admin password", type="password")
    if admin_input:
        if admin_input == ADMIN_PASSWORD:
            st.success("Admin authenticated")
            uploaded_file = st.file_uploader(
                "Upload updated workbook (Excel)",
                type=["xlsx"],
                key="file_uploader",
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
    df = load_workbook(uploaded_file)
else:
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

commodities = df["Commodity"].dropna().unique().tolist()
selected_commodity = st.sidebar.selectbox(
    "Commodity", options=sorted(commodities), index=0
)

homes = (
    df[df["Commodity"] == selected_commodity]["Home"]
    .dropna()
    .unique()
    .tolist()
)
selected_home = st.sidebar.selectbox(
    "Region (Home)", options=sorted(homes), index=0
)

crop_df = df[
    (df["Commodity"] == selected_commodity)
    & (df["Home"] == selected_home)
]

min_date = crop_df["Date"].min()
max_date = crop_df["Date"].max()

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date(),
)

fertiliser_columns = [
    "Nitram",
    "Imported Ammonium Nitrate",
    "Granular Urea",
    "MOP",
    "DAP",
    "TSP",
    "Sum Complex",
    "Basket",
]

selected_products = st.sidebar.multiselect(
    "Fertiliser products",
    options=fertiliser_columns,
    default=["Nitram"],
)

metric_options = {
    "Ratio": "ratio",
    "Z-Score": "z",
    "Percentile": "percentile",
    "Indexed": "index",
    "Programme Cost %": "cost_pct",
}

selected_metric_label = st.sidebar.selectbox(
    "Metric",
    options=list(metric_options.keys()),
    index=0,
)
selected_metric_key = metric_options[selected_metric_label]


###############################
# Main content
###############################

st.title("Fertiliser Affordability Dashboard")
st.markdown(
    """
    Use the controls in the sidebar to explore how fertiliser prices
    compare to crop values over time. Select a commodity, region,
    date range, products and a metric to see the results. Hover
    over the lines for precise values.
    """
)

mask = (
    (df["Commodity"] == selected_commodity)
    & (df["Home"] == selected_home)
    & (df["Date"].dt.date >= date_range[0])
    & (df["Date"].dt.date <= date_range[1])
)

plot_data = df[mask].copy()

# Build filtered base series first
if selected_metric_key == "cost_pct":
    chart_df = prepare_long_form(plot_data, selected_products, "cost_pct")
    chart_df = chart_df.dropna(subset=["value"])
    chart_df = chart_df.sort_values(["Product", "Date"])
else:
    base_chart_df = prepare_long_form(plot_data, selected_products, "ratio")
    chart_df = compute_viewer_metric(base_chart_df, selected_metric_key)


###############################
# Plotting
###############################

if not chart_df.empty:
    line_chart = (
        alt.Chart(chart_df)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("value:Q", title=selected_metric_label),
            color=alt.Color("Product:N", legend=alt.Legend(title="Product")),
            tooltip=[
                "Date:T",
                "Product:N",
                alt.Tooltip("value:Q", format=".2f"),
            ],
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
        "Basket_index" if selected_products == ["Basket"] and selected_metric_key == "index"
        else "Basket_z" if selected_products == ["Basket"] and selected_metric_key == "z"
        else "Basket_percentile" if selected_products == ["Basket"] and selected_metric_key == "percentile"
        else selected_metric_key
    )
    st.write(description)


###############################
# Data table
###############################

with st.expander("Show data", expanded=False):
    st.dataframe(chart_df.sort_values("Date"), use_container_width=True)
