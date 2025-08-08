import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np
from matplotlib.image import imread
import os

AREAS = {
    "North America": {
        "urban": {"min_lat": 40.7128, "max_lat": 40.8128, "min_lon": -74.0060, "max_lon": -73.9060},
        "rural": {"min_lat": 36.7783, "max_lat": 36.8783, "min_lon": -119.4179, "max_lon": -119.3179}
    },
    "Europe": {
        "urban": {"min_lat": 48.8566, "max_lat": 48.9566, "min_lon": 2.3522, "max_lon": 2.4522},
        "rural": {"min_lat": 46.2276, "max_lat": 46.3276, "min_lon": 2.2137, "max_lon": 2.3137}
    },
    "Asia": {
        "urban": {"min_lat": 35.6895, "max_lat": 35.7895, "min_lon": 139.6917, "max_lon": 139.7917},
        "rural": {"min_lat": 27.1751, "max_lat": 27.2751, "min_lon": 78.0421, "max_lon": 78.1421}
    },
    "South America": {
        "urban": {"min_lat": -23.5505, "max_lat": -23.4505, "min_lon": -46.6333, "max_lon": -46.5333},
        "rural": {"min_lat": -23.3000, "max_lat": -23.2000, "min_lon": -46.8000, "max_lon": -46.7000}
    },
    "Africa": {
        "urban": {"min_lat": -1.2921, "max_lat": -1.1921, "min_lon": 36.8219, "max_lon": 36.9219},
        "rural": {"min_lat": -3.3000, "max_lat": -3.2000, "min_lon": 36.7000, "max_lon": 36.8000}
    }
}

def create_sampling_regions_map(areas, output_path="sampling_regions_map.pdf", figsize=(14, 8), background_image="utils/earth.jpg"):
    """
    Create a world map showing the sampling regions for the CVGlobal dataset.
    
    Args:
        areas (dict): Dictionary containing continent areas with urban/rural regions
        output_path (str): Path to save the output map
        figsize (tuple): Figure size (width, height)
        background_image (str): Path to the Earth background image
    
    Returns:
        None: Saves the map as a PDF file
    """
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Set up world map projection (simple equirectangular)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    
    # Load and display background Earth image
    try:
        earth_img = imread(background_image)
        ax.imshow(earth_img, extent=[-180, 180, -90, 90], aspect='auto', alpha=0.8)
    except FileNotFoundError:
        print(f"Warning: Background image '{background_image}' not found. Using simple outline instead.")
        # Fallback to simple continent outlines
        world_outline = [
            # North America outline (simplified)
            [(-170, 20), (-60, 20), (-60, 80), (-170, 80), (-170, 20)],
            # South America outline (simplified)  
            [(-85, -60), (-30, -60), (-30, 15), (-85, 15), (-85, -60)],
            # Europe outline (simplified)
            [(-15, 35), (45, 35), (45, 75), (-15, 75), (-15, 35)],
            # Africa outline (simplified)
            [(-20, -40), (55, -40), (55, 40), (-20, 40), (-20, -40)],
            # Asia outline (simplified)
            [(45, -10), (180, -10), (180, 80), (45, 80), (45, -10)]
        ]
        
        # Draw simplified continent outlines
        for outline in world_outline:
            xs, ys = zip(*outline)
            ax.plot(xs, ys, 'k-', alpha=0.3, linewidth=0.5)
    
    # Color schemes for urban and rural regions (with higher contrast for visibility)
    colors = {
        'urban': '#FF3333',     # Bright red for urban
        'rural': '#00CC99'      # Bright teal for rural
    }
    
    # Plot sampling regions as outlined boxes only
    for continent, regions in areas.items():
        for region_type, bounds in regions.items():
            # Extract coordinates
            min_lat = bounds['min_lat']
            max_lat = bounds['max_lat'] 
            min_lon = bounds['min_lon']
            max_lon = bounds['max_lon']
            
            # Create rectangle
            width = max_lon - min_lon
            height = max_lat - min_lat
            
            # Draw rectangle with no fill, just colored outline
            rect = Rectangle(
                (min_lon, min_lat), width, height,
                facecolor='none',
                edgecolor=colors[region_type],
                linewidth=4,
                alpha=1.0
            )
            ax.add_patch(rect)
    
    # Create simple legend with only region types
    urban_patch = patches.Patch(color=colors['urban'], label='Urban Regions')
    rural_patch = patches.Patch(color=colors['rural'], label='Rural Regions')
    
    # Create legend with region types only
    type_legend = ax.legend(handles=[urban_patch, rural_patch], 
                           loc='upper left', fontsize=12, 
                           facecolor='white', edgecolor='black', framealpha=0.95,
                           title='Region Types', title_fontsize=13)
    type_legend.get_frame().set_linewidth(1.5)
    
    # Add subtle gridlines
    ax.grid(True, alpha=0.2, linestyle='--', color='white', linewidth=0.5)
    
    # Set labels and title
    ax.set_xlabel('Longitude (°)', fontsize=12, fontweight='bold', color='black')
    ax.set_ylabel('Latitude (°)', fontsize=12, fontweight='bold', color='black')
    ax.set_title('CVGlobal Dataset Sampling Regions', fontsize=16, fontweight='bold', 
                 pad=20, color='black',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9, edgecolor='black'))
    
    # Add major latitude/longitude lines with better visibility
    major_lats = np.arange(-60, 80, 30)
    major_lons = np.arange(-150, 180, 60)
    
    for lat in major_lats:
        ax.axhline(y=lat, color='white', alpha=0.3, linestyle='-', linewidth=0.8)
    for lon in major_lons:
        ax.axvline(x=lon, color='white', alpha=0.3, linestyle='-', linewidth=0.8)
    
    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=10, colors='black')
    
    # Add a subtle border around the entire plot
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print(f"Map saved as {output_path.replace('.pdf', '.png')}")
    
    return fig, ax

def create_detailed_sampling_map(areas, output_path="detailed_sampling_map.pdf", figsize=(15, 10), background_image="utils/earth.jpg"):
    """
    Create a more detailed map with individual subplot for each continent using Earth background.
    
    Args:
        areas (dict): Dictionary containing continent areas
        output_path (str): Path to save the output map
        figsize (tuple): Figure size
        background_image (str): Path to the Earth background image
    """
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    # Continent zoom regions for detailed view
    continent_bounds = {
        'North America': {'lat': (20, 60), 'lon': (-140, -50)},
        'Europe': {'lat': (35, 70), 'lon': (-15, 35)},
        'Asia': {'lat': (15, 55), 'lon': (65, 155)},
        'South America': {'lat': (-45, 15), 'lon': (-85, -25)},
        'Africa': {'lat': (-25, 25), 'lon': (5, 55)}
    }
    
    colors = {'urban': '#FF3333', 'rural': '#00CC99'}
    
    # Load Earth image once
    earth_img = None
    try:
        earth_img = imread(background_image)
        img_height, img_width = earth_img.shape[:2]
    except FileNotFoundError:
        print(f"Warning: Background image '{background_image}' not found.")
    
    for idx, (continent, regions) in enumerate(areas.items()):
        ax = axes[idx]
        
        # Set continent bounds
        bounds = continent_bounds[continent]
        ax.set_xlim(bounds['lon'])
        ax.set_ylim(bounds['lat'])
        
        # Calculate aspect ratio to maintain proper Earth map scaling
        lat_range = bounds['lat'][1] - bounds['lat'][0]
        lon_range = bounds['lon'][1] - bounds['lon'][0]
        
        # Calculate the aspect ratio based on latitude (Mercator-like projection)
        center_lat = (bounds['lat'][0] + bounds['lat'][1]) / 2
        lat_correction = np.cos(np.radians(center_lat))
        aspect_ratio = (lat_range) / (lon_range * lat_correction)
        
        # Set aspect ratio to maintain proper scaling
        ax.set_aspect(aspect_ratio)
        
        # Add cropped Earth background if available
        if earth_img is not None:
            # Calculate pixel coordinates for cropping
            # Assuming Earth image spans -180 to 180 longitude and -90 to 90 latitude
            lon_min, lon_max = bounds['lon']
            lat_min, lat_max = bounds['lat']
            
            # Convert geographical coordinates to image pixel coordinates
            # Longitude: -180 to 180 maps to 0 to img_width
            x_min = int((lon_min + 180) / 360 * img_width)
            x_max = int((lon_max + 180) / 360 * img_width)
            
            # Latitude: 90 to -90 maps to 0 to img_height (note the flip)
            y_min = int((90 - lat_max) / 180 * img_height)
            y_max = int((90 - lat_min) / 180 * img_height)
            
            # Ensure bounds are within image dimensions
            x_min = max(0, min(x_min, img_width - 1))
            x_max = max(0, min(x_max, img_width - 1))
            y_min = max(0, min(y_min, img_height - 1))
            y_max = max(0, min(y_max, img_height - 1))
            
            # Crop the image
            if x_max > x_min and y_max > y_min:
                cropped_img = earth_img[y_min:y_max, x_min:x_max]
                
                # Display the cropped image with correct geographical extent
                ax.imshow(cropped_img, extent=[lon_min, lon_max, lat_min, lat_max], 
                         aspect='auto', alpha=0.8)
        
        # Plot regions for this continent
        for region_type, region_bounds in regions.items():
            min_lat = region_bounds['min_lat']
            max_lat = region_bounds['max_lat']
            min_lon = region_bounds['min_lon'] 
            max_lon = region_bounds['max_lon']
            
            width = max_lon - min_lon
            height = max_lat - min_lat
            
            # Draw main rectangle
            rect = Rectangle(
                (min_lon, min_lat), width, height,
                facecolor=colors[region_type],
                edgecolor='white',
                linewidth=3,
                alpha=0.9
            )
            ax.add_patch(rect)
            
            # Add black outline for better contrast
            rect_outline = Rectangle(
                (min_lon, min_lat), width, height,
                facecolor='none',
                edgecolor='black',
                linewidth=1.5,
                alpha=1.0
            )
            ax.add_patch(rect_outline)
            
            # Add region label
            center_lat = (min_lat + max_lat) / 2
            center_lon = (min_lon + max_lon) / 2
            ax.annotate(
                region_type.title(),
                xy=(center_lon, center_lat),
                ha='center', va='center',
                fontsize=11, fontweight='bold',
                color='white',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.8, edgecolor='white')
            )
        
        ax.set_title(continent, fontsize=14, fontweight='bold', color='black',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor='black'))
        ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
        ax.tick_params(axis='both', labelsize=9, colors='black')
        
        # Add continent-specific grid lines
        lat_step = 10 if lat_range > 30 else 5
        lon_step = 20 if lon_range > 60 else 10
        
        lat_ticks = np.arange(bounds['lat'][0], bounds['lat'][1] + 1, lat_step)
        lon_ticks = np.arange(bounds['lon'][0], bounds['lon'][1] + 1, lon_step)
        
        for lat in lat_ticks:
            ax.axhline(y=lat, color='white', alpha=0.3, linestyle='-', linewidth=0.5)
        for lon in lon_ticks:
            ax.axvline(x=lon, color='white', alpha=0.3, linestyle='-', linewidth=0.5)
        
        if idx >= 3:  # Bottom row
            ax.set_xlabel('Longitude (°)', fontsize=11, fontweight='bold')
        if idx % 3 == 0:  # Left column
            ax.set_ylabel('Latitude (°)', fontsize=11, fontweight='bold')
        
        # Add border to each subplot
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)
    
    # Remove the extra subplot
    axes[-1].remove()
    
    # Add overall legend
    urban_patch = patches.Patch(color=colors['urban'], label='Urban Regions')
    rural_patch = patches.Patch(color=colors['rural'], label='Rural Regions')
    fig.legend(handles=[urban_patch, rural_patch], 
              loc='center', bbox_to_anchor=(0.83, 0.25), fontsize=14,
              facecolor='white', edgecolor='black', framealpha=0.95)
    
    plt.suptitle('CVGlobal Dataset: Detailed Sampling Regions by Continent', 
                 fontsize=18, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for suptitle
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    
    print(f"Detailed map saved as {output_path.replace('.pdf', '.png')}")
    
    return fig, axes

if __name__ == "__main__":
    # Create both maps
    print("Creating sampling regions map...")
    create_sampling_regions_map(AREAS, output_path=os.path.join("paper", "images", "sampling_regions_map.pdf"))

    print("Creating detailed sampling regions map...")
    create_detailed_sampling_map(AREAS, output_path=os.path.join("paper", "images", "detailed_sampling_map.pdf"))

    plt.show()