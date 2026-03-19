"""
Streamlit dashboard for fertiliser affordability analysis
========================================================

This script builds an interactive dashboard for comparing fertiliser
affordability across crops and regions. It reads data from the
"Monthly Avg Summary" sheet of a provided Excel workbook.

Key logic:
- Load raw data
- Filter by the user's selected commodity, home, date range
- Select a forward delivery month
- For each observation date, choose the nearest available delivery
  month-year to that target month
- Build the selected product series
- Calculate viewer-specific metrics only on that filtered series
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

MONTH_ABBR_TO_NUM = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
    "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
    "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
}


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

    ratio_columns = [f"{c}_ratio" for c in fert_cols]
    analytics["Basket_ratio"] = analytics[ratio_columns].mean(axis=1, skipna=True)

    # Crop-specific programme cost assumptions
    crop_assumptions = {
        "Feed Wheat": {
            "n_requirement": 220,   # kg N/ha
            "yield": 8.5,           # t/ha
        },
        "Oilseed Rape": {
            "n_requirement": 180,   # kg N/ha
            "yield": 3.5,           # t/ha
        },
    }

    # Map assumptions onto each row
    analytics["N_REQUIREMENT"] = analytics["Commodity"].map(
        lambda x: crop_assumptions.get(x, {}).get("n_requirement", np.nan)
    )
    analytics["YIELD_ASSUMPTION"] = analytics["Commodity"].map(
        lambda x: crop_assumptions.get(x, {}).get("yield", np.nan)
    )

    # Tonnes of product required per hectare
    analytics["Nitram_tonnes_per_ha"] = analytics["N_REQUIREMENT"] / (0.345 * 1000)
    analytics["Imported_AN_tonnes_per_ha"] = analytics["N_REQUIREMENT"] / (0.345 * 1000)
    analytics["Urea_tonnes_per_ha"] = analytics["N_REQUIREMENT"] / (0.46 * 1000)

    # Revenue per hectare
    analytics["Crop_Revenue_per_ha"] = (
        analytics["YIELD_ASSUMPTION"] * analytics["Average Price"]
    )

    # Nitrogen cost as % of crop revenue
    analytics["Nitram_cost_pct"] = (
        analytics["Nitram"] * analytics["Nitram_tonnes_per_ha"]
    ) / analytics["Crop_Revenue_per_ha"] * 100

    analytics["Imported_AN_cost_pct"] = (
        analytics["Imported Ammonium Nitrate"] * analytics["Imported_AN_tonnes_per_ha"]
    ) / analytics["Crop_Revenue_per_ha"] * 100

    analytics["Urea_cost_pct"] = (
        analytics["Granular Urea"] * analytics["Urea_tonnes_per_ha"]
    ) / analytics["Crop_Revenue_per_ha"] * 100

    # Parse "Delivery Month and Year" like "Nov 2024"
    delivery_parts = analytics["Delivery Month and Year"].astype(str).str.strip().str.extract(
        r"^(?P<month>[A-Za-z]{3})\s+(?P<year>\d{4})$"
    )
    analytics["DeliveryMonthNum"] = delivery_parts["month"].map(MONTH_ABBR_TO_NUM)
    analytics["DeliveryYearNum"] = pd.to_numeric(delivery_parts["year"], errors="coerce")

    # Convert delivery label into a comparable month index
    analytics["DeliveryMonthIndex"] = (
        analytics["DeliveryYearNum"] * 12 + analytics["DeliveryMonthNum"]
    )

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


def build_target_month_index(date_series: pd.Series, target_month_num: int) -> pd.Series:
    """
    For each observation date, build the target month index for the next occurrence
    of the selected month.

    Example target_month_num=11:
    - 2024-05 -> Nov 2024
    - 2024-12 -> Nov 2025
    """
    obs_month = date_series.dt.month
    obs_year = date_series.dt.year
    target_year = np.where(obs_month <= target_month_num, obs_year, obs_year + 1)
    return pd.Series(target_year * 12 + target_month_num, index=date_series.index)


def select_nearest_forward_rows(data: pd.DataFrame, target_month_num: int) -> pd.DataFrame:
    """
    For each observation date, select the nearest available delivery row to the
    target forward month.

    This preserves continuity better than demanding an exact delivery label match.
    """
    if data.empty:
        return data.copy()

    working = data.copy()

    working = working.dropna(subset=["DeliveryMonthIndex"]).copy()
    working["TargetMonthIndex"] = build_target_month_index(working["Date"], target_month_num)

    # Distance to the ideal target
    working["ForwardDistance"] = (working["DeliveryMonthIndex"] - working["TargetMonthIndex"]).abs()

    # Tie-break: prefer future/at-target over earlier if equally close
    working["ForwardBias"] = np.where(
        working["DeliveryMonthIndex"] >= working["TargetMonthIndex"], 0, 1
    )

    working = working.sort_values(
        ["Date", "ForwardDistance", "ForwardBias", "DeliveryMonthIndex"]
    )

    # One best row per observation date / commodity / home
    best = (
        working.groupby(["Date", "Commodity", "Home"], as_index=False)
        .first()
        .copy()
    )

    return best.sort_values("Date")


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
            col_name = "Basket_ratio"
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

delivery_month_map = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}

selected_delivery_month_name = st.sidebar.selectbox(
    "Forward Delivery Month",
    options=list(delivery_month_map.keys()),
    index=10,  # November default
)
selected_delivery_month_num = delivery_month_map[selected_delivery_month_name]

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
    forward delivery month, date range, products and a metric to see the results.
    The forward month logic uses the nearest available quote to the selected
    seasonal target from each observation date.
    """
)

mask = (
    (df["Commodity"] == selected_commodity)
    & (df["Home"] == selected_home)
    & (df["Date"].dt.date >= date_range[0])
    & (df["Date"].dt.date <= date_range[1])
)

plot_data = df[mask].copy()

# Select nearest available forward match to the target seasonal month
plot_data = select_nearest_forward_rows(plot_data, selected_delivery_month_num)

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
    st.warning(
        "No data available for the selected filters."
    )


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
with st.expander("Show data", expanded=False):
    st.dataframe(chart_df.sort_values("Date"), use_container_width=True)
