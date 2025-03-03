import pandas as pd
import geopandas as gpd
import folium
import numpy as np
import streamlit as st
from streamlit_folium import st_folium

# Load data
df_orig = pd.read_csv("../data/outputs/final_dataset.csv")
df_new = pd.read_csv("../data/outputs/final_scaled.csv")

# Merge data
df = df_orig.merge(df_new, how="inner", left_index=True, right_index=True)[
    [
        "CNTY_NM_x",
        "City_x",
        "TRFC_STATN_ID",
        "LATITUDE_x",
        "LONGITUDE_x",
        "AADT_RPT_QTY",
        "AADT_RPT_HIST_05_QTY",
        "5Yr_Growth",
        "Wildfire_Risk_y",
        "RISK_STATE_RANK_Scaled",
        "AADT_RPT_QTY_Scaled",
        "5Yr_Growth_Scaled",
        "Gridlock_Risk_Score",
    ]
]
df.columns = [
    "County",
    "City",
    "TRFC_STATN_ID",
    "LATITUDE",
    "LONGITUDE",
    "AADT_RPT_QTY",
    "AADT_RPT_HIST_05_QTY",
    "5Yr_Growth",
    "Direct_Road_Wildfire_Risk",
    "RISK_STATE_RANK_Scaled",
    "AADT_RPT_QTY_Scaled",
    "5Yr_Growth_Scaled",
    "Gridlock_Risk_Score",
]

# Weights
current_traffic_wt = 0.3
traffic_growth_wt = 0.3
local_risk_wt = 0.2
city_risk_wt = 0.2

# Calculate Gridlock_Risk_Score with weights
df["Gridlock_Risk_Score"] = (
    df["Direct_Road_Wildfire_Risk"] * local_risk_wt
    + df["RISK_STATE_RANK_Scaled"] * city_risk_wt
    + df["AADT_RPT_QTY_Scaled"] * current_traffic_wt
    + df["5Yr_Growth_Scaled"] * traffic_growth_wt
)

# Sidebar filters
st.sidebar.title("Filters")
selected_counties = st.sidebar.multiselect(
    "Select Counties", df["County"].unique(), default=["Travis"]
)
filtered_df_counties = df[df["County"].isin(selected_counties)]
selected_cities = st.sidebar.multiselect(
    "Select Cities",
    filtered_df_counties["City"].unique(),
    default=filtered_df_counties["City"].unique(),
)

# Filter data based on selections
filtered_df = filtered_df_counties[filtered_df_counties["City"].isin(selected_cities)]

# Check if the filtered DataFrame is empty
if filtered_df.empty:
    st.title("Geographic Data Visualization")
    st.subheader("Gridlock KPI")
    st.markdown(
        "<div style='color: red; font-size: 24px; font-weight: bold;'>**Please select a county and city from the filters on the left to view the data.**</div>",
        unsafe_allow_html=True,
    )
else:
    # Apply the conversion to the filtered DataFrame
    filtered_df["Direct_Road_Wildfire_Risk"] = [
        "Yes" if x == 1 else "No" for x in filtered_df["Direct_Road_Wildfire_Risk"]
    ]
    filtered_df["5Yr_Growth"] = (filtered_df["5Yr_Growth"] * 100).map("{:.2f}%".format)

    # Update GeoDataFrame
    gdf = gpd.GeoDataFrame(
        filtered_df,
        geometry=gpd.points_from_xy(filtered_df["LONGITUDE"], filtered_df["LATITUDE"]),
    )


# Calculate bin edges and labels
def std_dev_binning(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    bin_edges = [
        mean - 2 * std_dev,
        mean - std_dev,
        mean,
        mean + std_dev,
        mean + 2 * std_dev,
    ]
    bin_edges = [float("-inf")] + bin_edges + [float("inf")]
    return bin_edges


bin_edges = std_dev_binning(gdf["Gridlock_Risk_Score"])
bin_labels = ["Very Low", "Low", "Average", "High", "Very High", "Extremely High"]

filtered_df["Gridlock_Risk_label"] = pd.cut(
    filtered_df["Gridlock_Risk_Score"], bins=bin_edges, labels=bin_labels
)

gdf = gpd.GeoDataFrame(
    filtered_df[
        [
            "City",
            "TRFC_STATN_ID",
            "LATITUDE",
            "LONGITUDE",
            "AADT_RPT_QTY",
            "AADT_RPT_HIST_05_QTY",
            "5Yr_Growth",
            "Direct_Road_Wildfire_Risk",
            "Gridlock_Risk_Score",
            "Gridlock_Risk_label",
        ]
    ],
    geometry=gpd.points_from_xy(filtered_df["LONGITUDE"], filtered_df["LATITUDE"]),
)

# Load city shapefile and merge with city data
df_cities = (
    df[["City", "RISK_STATE_RANK_Scaled"]].groupby(["City"]).mean().reset_index()
)
city_geo = gpd.read_file(
    "../data/inputs/cb_2023_48_place_500k/cb_2023_48_place_500k.shp"
)
merged_geo_df = city_geo.merge(df_cities, left_on="NAME", right_on="City")

# Filter merged_geo_df based on selected cities
merged_geo_df = merged_geo_df[merged_geo_df["City"].isin(selected_cities)]

# Calculate the bounding box of the selected counties
if not filtered_df.empty:
    bounds = gdf.total_bounds
    map_center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    map_bounds = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]
else:
    map_center = [filtered_df["LATITUDE"].mean(), filtered_df["LONGITUDE"].mean()]
    map_bounds = None

# Create map
m = folium.Map(
    location=[filtered_df["LATITUDE"].mean(), filtered_df["LONGITUDE"].mean()],
    zoom_start=11,
)

# Add city boundaries to the map if there are selected cities
if not merged_geo_df.empty:
    folium.GeoJson(
        merged_geo_df,
        name="City Boundaries",
        style_function=lambda x: {
            "fillColor": "blue",
            "color": "blue",
            "weight": 2,
            "fillOpacity": 0.1,
        },
    ).add_to(m)

if not merged_geo_df.empty:
    m = merged_geo_df[["geometry", "City", "RISK_STATE_RANK_Scaled"]].explore(
        m=m, column="RISK_STATE_RANK_Scaled", cmap="inferno", name="cities"
    )
# Add gridlock risk points to the map
m = gdf.explore(
    m=m,
    column="Gridlock_Risk_label",
    cmap=["purple", "blue", "green", "yellow", "orange", "red"],
    legend=True,
    name="points",
    tooltip=[
        "City",
        "TRFC_STATN_ID",
        "AADT_RPT_QTY",
        "5Yr_Growth",
        "Gridlock_Risk_label",
    ],
)
folium.LayerControl().add_to(m)

# Fit the map to the bounding box
if map_bounds:
    m.fit_bounds(map_bounds)

# Display map in Streamlit
st.title("Geographic Data Visualization")
st.subheader("Gridlock KPI")

# Add a message to instruct users to select a filter
st.markdown(
    "**Please select a county and city from the filters on the left to view the data.**"
)

# Calculate KPIs
top_wildfire_risk_city = (
    filtered_df.loc[filtered_df["Gridlock_Risk_Score"].idxmax(), "City"]
    if not filtered_df.empty
    else "N/A"
)
num_direct_road_wildfire_risks = filtered_df["Direct_Road_Wildfire_Risk"].count()
aggregate_gridlock_risk_score = filtered_df["Gridlock_Risk_Score"].sum()

# Create columns for the KPIs and legend
kpi1, kpi2 = st.columns(2)

# Define the KPI HTML and CSS
kpi_html = """
<div style="background-color: {color}; padding: 12px; border-radius: 15px; text-align: center;">
    <h4 style="color: white;">{label}</h4>
    <p style="color: white; font-size: 20px;">{value}</p>
</div>
"""

# Display KPIs with inferno-colored boxes
kpi1.markdown(
    kpi_html.format(
        color="#000001", label="Top Gridlock Risk City", value=top_wildfire_risk_city
    ),
    unsafe_allow_html=True,
)
kpi2.markdown(
    kpi_html.format(
        color="#2c2d2d",
        label="# of Road Wildfire Risks",
        value=num_direct_road_wildfire_risks,
    ),
    unsafe_allow_html=True,
)

# Define the legend HTML and CSS
legend_html = """
<div style="padding: 10px; background-color: white; border-radius: 5px;">
    <h4>Gridlock Risk Score Legend</h4>
    <div style="display: flex; align-items: center;">
        <div style="width: 20px; height: 20px; background-color: purple; margin-right: 10px;"></div>
        <span>Very Low</span>
    </div>
    <div style="display: flex; align-items: center;">
        <div style="width: 20px; height: 20px; background-color: blue; margin-right: 10px;"></div>
        <span>Low</span>
    </div>
    <div style="display: flex; align-items: center;">
        <div style="width: 20px; height: 20px; background-color: green; margin-right: 10px;"></div>
        <span>Average</span>
    </div>
    <div style="display: flex; align-items: center;">
        <div style="width: 20px; height: 20px; background-color: yellow; margin-right: 10px;"></div>
        <span>High</span>
    </div>
    <div style="display: flex; align-items: center;">
        <div style="width: 20px; height: 20px; background-color: orange; margin-right: 10px;"></div>
        <span>Very High</span>
    </div>
    <div style="display: flex; align-items: center;">
        <div style="width: 20px; height: 20px; background-color: red; margin-right: 10px;"></div>
        <span>Extremely High</span>
    </div>
</div>
"""
st.subheader("Wildfire Risk Map")
# Add the legend to the sidebar
st.sidebar.markdown(legend_html, unsafe_allow_html=True)

# Create a single column for the map to stretch it to the width of the KPIs
map_col = st.columns(1)

# Display the map in the single column
with map_col[0]:
    st_folium(m, width=900, height=500)

# Add a slider to select the number of roads to display
num_roads = st.slider("Number of Roads to Display", min_value=1, max_value=50, value=10)

# Filter the top roads based on Gridlock_Risk_Score using the slider value
top_roads = filtered_df.nlargest(num_roads, "Gridlock_Risk_Score")[
    [
        "County",
        "City",
        "TRFC_STATN_ID",
        "AADT_RPT_QTY",
        "5Yr_Growth",
        "Gridlock_Risk_label",
    ]
]

# Rename columns
top_roads.columns = [
    "County",
    "City",
    "Traffic Station ID",
    "# of Avg. Annual Daily Traffic",
    "5Yr Growth",
    "Gridlock Risk",
]

# Display the top roads DataFrame
st.subheader(f"Top {num_roads} Roads by Gridlock Risk Score")
st.dataframe(top_roads)
# Maximize the length of the DataFrame
st.markdown("<style>.dataframe{max-width: 100%;}</style>", unsafe_allow_html=True)
