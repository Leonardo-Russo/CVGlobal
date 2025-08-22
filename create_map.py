import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from matplotlib.image import imread

import matplotlib.colors as mcolors              # added
from matplotlib import patheffects as pe         # added

# Example structure still works; code also accepts lists of boxes per region.
AREAS = {
    "North America": {
        "urban": [
            # New York City area
            {"min_lat": 40.7128, "max_lat": 40.8128, "min_lon": -74.0060, "max_lon": -73.9060},
            # Chicago area
            {"min_lat": 41.8781, "max_lat": 41.9781, "min_lon": -87.6298, "max_lon": -87.5298},
            # Los Angeles area
            {"min_lat": 34.0522, "max_lat": 34.1522, "min_lon": -118.2437, "max_lon": -118.1437}
        ],
        "rural": [
            # Central Missouri farmland
            {"min_lat": 38.5000, "max_lat": 39.2000, "min_lon": -92.8000, "max_lon": -91.8000},
            # Kansas agricultural areas
            {"min_lat": 38.8000, "max_lat": 39.5000, "min_lon": -98.5000, "max_lon": -97.5000},
            # Nebraska countryside
            {"min_lat": 41.0000, "max_lat": 41.7000, "min_lon": -97.5000, "max_lon": -96.5000}
        ]
    },
    "Europe": {
        "urban": [
            # Paris area
            {"min_lat": 48.8566, "max_lat": 48.9566, "min_lon": 2.3522, "max_lon": 2.4522},
            # Berlin area
            {"min_lat": 52.5200, "max_lat": 52.6200, "min_lon": 13.4050, "max_lon": 13.5050},
            # Madrid area
            {"min_lat": 40.4168, "max_lat": 40.5168, "min_lon": -3.7038, "max_lon": -3.6038}
        ],
        "rural": [
            # French countryside (Loire Valley)
            {"min_lat": 47.0000, "max_lat": 47.8000, "min_lon": 1.5000, "max_lon": 2.8000},
            # German countryside (Bavaria)
            {"min_lat": 48.5000, "max_lat": 49.2000, "min_lon": 11.0000, "max_lon": 12.5000},
            # Spanish countryside (Castile and León)
            {"min_lat": 41.5000, "max_lat": 42.2000, "min_lon": -5.5000, "max_lon": -4.0000}
        ]
    },
    "Asia": {
        "urban": [
            # Tokyo area
            {"min_lat": 35.6895, "max_lat": 35.7895, "min_lon": 139.6917, "max_lon": 139.7917},
            # Mumbai area
            {"min_lat": 19.0760, "max_lat": 19.1760, "min_lon": 72.8777, "max_lon": 72.9777},
            # Seoul area
            {"min_lat": 37.5665, "max_lat": 37.6665, "min_lon": 126.9780, "max_lon": 127.0780}
        ],
        "rural": [
            # Rural India (Uttar Pradesh agricultural areas)
            {"min_lat": 26.8000, "max_lat": 27.5000, "min_lon": 77.5000, "max_lon": 78.5000},
            # Rural China (Shandong Province)
            {"min_lat": 36.0000, "max_lat": 36.8000, "min_lon": 117.0000, "max_lon": 118.0000},
            # Rural Japan (Hokkaido)
            {"min_lat": 43.0000, "max_lat": 43.8000, "min_lon": 142.0000, "max_lon": 143.0000}
        ]
    },
    "South America": {
        "urban": [
            # São Paulo area
            {"min_lat": -23.5505, "max_lat": -23.4505, "min_lon": -46.6333, "max_lon": -46.5333},
            # Buenos Aires area
            {"min_lat": -34.6118, "max_lat": -34.5118, "min_lon": -58.3960, "max_lon": -58.2960},
            # Bogotá area
            {"min_lat": 4.7110, "max_lat": 4.8110, "min_lon": -74.0721, "max_lon": -73.9721}
        ],
        "rural": [
            # Brazilian countryside (São Paulo state agricultural areas)
            {"min_lat": -23.8000, "max_lat": -22.8000, "min_lon": -47.5000, "max_lon": -46.0000},
            # Argentine Pampas
            {"min_lat": -35.5000, "max_lat": -34.5000, "min_lon": -59.5000, "max_lon": -58.0000},
            # Colombian countryside
            {"min_lat": 5.0000, "max_lat": 6.0000, "min_lon": -74.5000, "max_lon": -73.0000}
        ]
    },
    "Africa": {
        "urban": [
            # Nairobi area
            {"min_lat": -1.2921, "max_lat": -1.1921, "min_lon": 36.8219, "max_lon": 36.9219},
            # Lagos area
            {"min_lat": 6.5244, "max_lat": 6.6244, "min_lon": 3.3792, "max_lon": 3.4792},
            # Cape Town area
            {"min_lat": -33.9249, "max_lat": -33.8249, "min_lon": 18.4241, "max_lon": 18.5241}
        ],
        "rural": [
            # Kenyan countryside
            {"min_lat": -1.8000, "max_lat": -0.8000, "min_lon": 36.3000, "max_lon": 37.3000},
            # Nigerian countryside
            {"min_lat": 7.0000, "max_lat": 8.0000, "min_lon": 3.0000, "max_lon": 4.5000},
            # South African countryside
            {"min_lat": -33.0000, "max_lat": -32.0000, "min_lon": 19.0000, "max_lon": 20.5000}
        ]
    }
}

EARTH_EXTENT = [-180, 180, -90, 90]  # [xmin, xmax, ymin, ymax]

def _iter_boxes_for_continent(regions: dict):
    """
    Yield (region_type, bounds_dict) for all boxes in a continent.
    Each region_type can be a dict (single box) or a list of dicts (multiple boxes).
    """
    for region_type, region_boxes in regions.items():
        if isinstance(region_boxes, dict):
            yield region_type, region_boxes
        else:
            for box in region_boxes:
                yield region_type, box

def _compute_bounds_for_continent(regions: dict, pad_frac: float = 0.10, min_pad: float = 2.0):
    """
    Compute bounding box that includes all region boxes for a continent,
    then expand by padding so all boxes fit comfortably.
    """
    lats_min, lats_max, lons_min, lons_max = [], [], [], []
    for _, b in _iter_boxes_for_continent(regions):
        lats_min.append(b["min_lat"])
        lats_max.append(b["max_lat"])
        lons_min.append(b["min_lon"])
        lons_max.append(b["max_lon"])

    if not lats_min:
        return EARTH_EXTENT[2], EARTH_EXTENT[3], EARTH_EXTENT[0], EARTH_EXTENT[1]

    min_lat, max_lat = float(min(lats_min)), float(max(lats_max))
    min_lon, max_lon = float(min(lons_min)), float(max(lons_max))

    lat_span = max(1e-6, max_lat - min_lat)
    lon_span = max(1e-6, max_lon - min_lon)

    lat_pad = max(min_pad, pad_frac * lat_span)
    lon_pad = max(min_pad, pad_frac * lon_span)

    min_lat_p = max(EARTH_EXTENT[2], min_lat - lat_pad)
    max_lat_p = min(EARTH_EXTENT[3], max_lat + lat_pad)
    min_lon_p = max(EARTH_EXTENT[0], min_lon - lon_pad)
    max_lon_p = min(EARTH_EXTENT[1], max_lon + lon_pad)

    return min_lat_p, max_lat_p, min_lon_p, max_lon_p

def _draw_earth(ax, earth_img_path: str):
    """
    Draw the full equirectangular Earth image with correct extent.
    Keep aspect='equal' to avoid skew/stretching.
    """
    if os.path.isfile(earth_img_path):
        img = imread(earth_img_path)
        ax.imshow(img, extent=EARTH_EXTENT, origin="upper")  # origin upper matches lat decreasing downward
    else:
        # If missing, just set a light background
        ax.set_facecolor("#f0f0f0")

    # Maintain degrees aspect ratio; no skewing when zooming
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(EARTH_EXTENT[0], EARTH_EXTENT[1])
    ax.set_ylim(EARTH_EXTENT[2], EARTH_EXTENT[3])

def _style_axes_off(ax):
    # ax.set_axis_off()
    # Remove margins that can show background canvas
    ax.margins(0)

def create_sampling_regions_map(
    areas,
    output_path="sampling_regions_map.pdf",
    figsize=(14, 8),
    background_image="utils/earth.jpg",
):
    """
    Create a world map showing the sampling regions for the CVGlobal dataset.
    - No axes are shown.
    - Background is not skewed (uses full image + aspect='equal').
    - Supports multiple areas per region type.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    _draw_earth(ax, background_image)

    # Optional: subtle graticules behind boxes
    for x in range(-180, 181, 60):
        ax.axvline(x, color="white", lw=0.6, alpha=0.15, zorder=1)
    for y in range(-90, 91, 30):
        ax.axhline(y, color="white", lw=0.6, alpha=0.15, zorder=1)

    # Colors for region types
    colors = {"urban": "#FF3333", "rural": "#00CC99"}

    # Draw rectangles with light fill + halo for contrast
    for continent, regions in areas.items():
        for region_type, bounds in _iter_boxes_for_continent(regions):
            min_lat = bounds["min_lat"]; max_lat = bounds["max_lat"]
            min_lon = bounds["min_lon"]; max_lon = bounds["max_lon"]
            width = max_lon - min_lon
            height = max_lat - min_lat

            rect = Rectangle(
                (min_lon, min_lat),
                width,
                height,
                facecolor=mcolors.to_rgba(colors.get(region_type, "#FF3333"), 0.18),
                edgecolor=colors.get(region_type, "#FF3333"),
                linewidth=2.5,
                joinstyle="round",
                zorder=3,
            )
            rect.set_path_effects([pe.withStroke(linewidth=4.5, foreground="white")])
            ax.add_patch(rect)

    # Legend inside axes (bottom-left), Urban above Rural
    urban_patch = patches.Patch(
        facecolor=mcolors.to_rgba(colors["urban"], 0.25),
        edgecolor=colors["urban"],
        label="Urban Regions",
    )
    rural_patch = patches.Patch(
        facecolor=mcolors.to_rgba(colors["rural"], 0.25),
        edgecolor=colors["rural"],
        label="Rural Regions",
    )
    ax.legend(
        handles=[urban_patch, rural_patch],
        loc="lower left",
        ncol=1,
        frameon=True,
        fancybox=True,
        edgecolor="#444",
        framealpha=0.9,
        fontsize=12,
        borderpad=0.4,
        labelspacing=0.3,
        handlelength=1.4,
        handletextpad=0.6,
        bbox_to_anchor=(0.02, 0.02),
    )

    ax.set_title(
        "CVGlobal Dataset Sampling Regions",
        fontsize=16,
        fontweight="bold",
        pad=10,
    )

    # Clean frame and ticks
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    _style_axes_off(ax)
    plt.tight_layout()

    # Save PNG and PDF (paper-friendly)
    png_path = output_path.replace(".pdf", ".png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Map saved as {png_path} and {output_path}")
    return fig, ax

def create_detailed_sampling_map(
    areas,
    output_path="detailed_sampling_map.pdf",
    figsize=(15, 10),
    background_image="utils/earth.jpg",
    pad_frac=0.12,
    min_pad=3.0,
):
    """
    Create a detailed map with one subplot per continent.
    - No axes are shown.
    - Background is not skewed (full image with aspect='equal', then set x/y limits).
    - Zoom is computed to include all areas in that continent, with padding.
    - Supports multiple areas per region.
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    colors = {"urban": "#FF3333", "rural": "#00CC99"}

    continents = list(areas.keys())
    n = len(continents)
    # Ensure we have at most 6 slots; remove extras if needed
    for idx in range(len(axes)):
        if idx >= n:
            axes[idx].remove()

    for idx, continent in enumerate(continents):
        ax = axes[idx]

        _draw_earth(ax, background_image)

        # Compute bounds to include all boxes for this continent
        min_lat, max_lat, min_lon, max_lon = _compute_bounds_for_continent(
            areas[continent], pad_frac=pad_frac, min_pad=min_pad
        )
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)
        ax.set_aspect("equal", adjustable="box")

        # Draw filled rectangles with white borders for visibility
        for region_type, bounds in _iter_boxes_for_continent(areas[continent]):
            min_lat_b = bounds["min_lat"]; max_lat_b = bounds["max_lat"]
            min_lon_b = bounds["min_lon"]; max_lon_b = bounds["max_lon"]
            width = max_lon_b - min_lon_b
            height = max_lat_b - min_lat_b

            rect = Rectangle(
                (min_lon_b, min_lat_b),
                width,
                height,
                facecolor=colors.get(region_type, "#FF3333"),
                edgecolor="none",
                linewidth=6,
                alpha=0.8,
            )
            ax.add_patch(rect)

        ax.set_title(
            continent,
            fontsize=13,
            fontweight="bold",
            pad=6,
        )

        _style_axes_off(ax)

    # Figure-level legend
    urban_patch = patches.Patch(color=colors["urban"], label="Urban Regions")
    rural_patch = patches.Patch(color=colors["rural"], label="Rural Regions")
    fig.legend(
        handles=[urban_patch, rural_patch],
        loc="lower center",
        ncol=2,
        frameon=True,
        framealpha=0.95,
        fontsize=12,
        bbox_to_anchor=(0.5, 0.02),
    )

    plt.suptitle(
        "CVGlobal Dataset: Detailed Sampling Regions by Continent",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    # Save PNG and PDF (paper-friendly)
    png_path = output_path.replace(".pdf", ".png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Detailed map saved as {png_path} and {output_path}")
    return fig, axes

if __name__ == "__main__":
    # Create both maps
    out_dir = os.path.join("paper", "images")
    os.makedirs(out_dir, exist_ok=True)

    print("Creating sampling regions map...")
    create_sampling_regions_map(
        AREAS,
        output_path=os.path.join(out_dir, "sampling_regions_map.pdf"),
        background_image=os.path.join("utils", "earth.jpg"),
    )

    print("Creating detailed sampling regions map...")
    create_detailed_sampling_map(
        AREAS,
        output_path=os.path.join(out_dir, "detailed_sampling_map.pdf"),
        background_image=os.path.join("utils", "earth.jpg"),
    )

    plt.show()