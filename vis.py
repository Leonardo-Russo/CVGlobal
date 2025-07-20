import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import box

def convert_areas_dict_to_list(areas_dict):
    """
    Converts a nested dictionary of areas into a list of dictionaries with bounding box coordinates.
    
    Parameters:
    - areas_dict (dict): Dictionary of regions with urban and rural areas.
    
    Returns:
    - list of dict: A list of dictionaries with bounding box coordinates.
    """
    areas_list = []
    for continent, area_types in areas_dict.items():
        for area_type, bounds in area_types.items():
            areas_list.append(bounds)
    return areas_list

def plot_defined_areas(areas, urban_areas_shp_path=None):
    """
    Plots the defined areas on a map along with urban areas if a shapefile is provided.
    
    Parameters:
    - areas (list of dict): List of dictionaries defining min/max lat/lon of areas.
    - urban_areas_shp_path (str, optional): Path to the urban areas shapefile.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot urban areas if shapefile is provided
    if urban_areas_shp_path:
        urban_areas = gpd.read_file(urban_areas_shp_path)
        urban_areas.plot(ax=ax, color="lightgray", alpha=0.5, edgecolor="black", linewidth=0.5, label="Urban Areas")

    # Convert defined areas into GeoDataFrame
    area_polygons = [
        box(area["min_lon"], area["min_lat"], area["max_lon"], area["max_lat"]) for area in areas
    ]
    area_gdf = gpd.GeoDataFrame(geometry=area_polygons)

    # Plot the defined areas
    area_gdf.plot(ax=ax, edgecolor="red", facecolor="none", linewidth=2, label="Defined Areas")

    # Customize plot
    ax.set_title("Defined Areas for Image Collection")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.grid(True)
    plt.savefig("defined_areas.png", dpi=300)


# AREAS = [
#     {"min_lat": 34.0522, "max_lat": 34.1522, "min_lon": -118.2437, "max_lon": -118.1437},  # Los Angeles
#     {"min_lat": 47.6062, "max_lat": 47.7062, "min_lon": -122.3321, "max_lon": -122.2321},  # Seattle
#     {"min_lat": 25.7617, "max_lat": 25.8617, "min_lon": -80.1918, "max_lon": -80.0918},  # Miami
#     {"min_lat": 41.8781, "max_lat": 41.9781, "min_lon": -87.6298, "max_lon": -87.5298},  # Chicago
# ]

AREAS = {
    "North America": {
        "urban": {"min_lat": 40.7128, "max_lat": 40.8128, "min_lon": -74.0060, "max_lon": -73.9060},  # NYC
        "rural": {"min_lat": 36.7783, "max_lat": 36.8783, "min_lon": -119.4179, "max_lon": -119.3179}  # California farmland
    },
    "Europe": {
        "urban": {"min_lat": 48.8566, "max_lat": 48.9566, "min_lon": 2.3522, "max_lon": 2.4522},  # Paris
        "rural": {"min_lat": 46.2276, "max_lat": 46.3276, "min_lon": 2.2137, "max_lon": 2.3137}  # French countryside
    },
    "Asia": {
        "urban": {"min_lat": 35.6895, "max_lat": 35.7895, "min_lon": 139.6917, "max_lon": 139.7917},  # Tokyo
        "rural": {"min_lat": 27.1751, "max_lat": 27.2751, "min_lon": 78.0421, "max_lon": 78.1421}  # Rural India (Agra region)
    },
    "South America": {
        "urban": {"min_lat": -23.5505, "max_lat": -23.4505, "min_lon": -46.6333, "max_lon": -46.5333},  # São Paulo
        "rural": {"min_lat": -14.2350, "max_lat": -14.1350, "min_lon": -51.9253, "max_lon": -51.8253}  # Brazilian rainforest
    },
    "Africa": {
        "urban": {"min_lat": -1.2921, "max_lat": -1.1921, "min_lon": 36.8219, "max_lon": 36.9219},  # Nairobi
        "rural": {"min_lat": -2.1540, "max_lat": -2.0540, "min_lon": 37.3088, "max_lon": 37.4088}  # Kenyan savanna
    }
}

if __name__ == "__main__":
    areas_list = convert_areas_dict_to_list(AREAS)
    plot_defined_areas(areas_list, urban_areas_shp_path="utils/urban_areas/urban_areas_full.shp")