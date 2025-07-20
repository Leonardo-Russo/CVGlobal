import os
import random
import requests
import googlemaps
import geopandas as gpd
from shapely.geometry import Point
from io import BytesIO
from PIL import Image
import keys
import argparse


def generate_coordinates(area):
    """
    Generate random latitude and longitufde within the given area.
    """
    lat = random.uniform(area["min_lat"], area["max_lat"])
    lon = random.uniform(area["min_lon"], area["max_lon"])
    return lat, lon

def get_streetview_metadata(lat, lon, api_key):
    """
    Fetches Street View metadata.
    """
    url = f"https://maps.googleapis.com/maps/api/streetview/metadata?location={lat},{lon}&key={api_key}"
    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching Street View metadata: {response.status_code}")
        return None

def is_likely_outdoor(metadata, gmaps):
    """
    Checks if a Street View panorama is likely outdoors.
    """
    if metadata.get("status") != "OK":
        return False

    if "place_id" in metadata:
        try:
            place_details = gmaps.place(metadata["place_id"])
            if 'result' in place_details and 'types' in place_details['result']:
                if any(t in place_details['result']['types'] for t in ['establishment', 'point_of_interest']):
                    return False
        except Exception as e:
            print(f"An error occurred when getting place details: {e}")
            pass

    return True

def is_inside_urban_area(lat, lon, urban_areas):
    """
    Check if the given coordinates are inside any urban area polygon.
    """
    point = Point(lon, lat)  # Shapely uses (lon, lat) format
    return any(urban_areas.contains(point))

def fetch_images(lat, lon, output_dir, gmaps, headings=[0, 90, 180, 270]):
    """
    Fetches satellite and Street View images for a given location.
    """
    location = f"{lat},{lon}"

    # Fetch Street View metadata
    metadata = get_streetview_metadata(lat, lon, keys.API_KEY)

    if metadata is None:
        return 0

    # Check if likely outdoor
    if not is_likely_outdoor(metadata, gmaps):
        print(f"Skipping likely indoor location: {location}")
        return 0

    # Use corrected lat and lon from metadata
    lat = metadata['location']['lat']
    lon = metadata['location']['lng']
    location = f"{lat},{lon}"

    # Download satellite image
    satellite_url = f"https://maps.googleapis.com/maps/api/staticmap?center={location}&zoom=18&size=640x640&maptype=satellite&key={keys.API_KEY}"
    satellite_path = os.path.join(output_dir, f"{lat}_{lon}_satellite.jpg")
    try:
        satellite_response = requests.get(satellite_url, stream=True)
        satellite_response.raise_for_status()
        with open(satellite_path, "wb") as f:
            for chunk in satellite_response.iter_content(chunk_size=8192):
                f.write(chunk)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading satellite image for {location}: {e}")
        return 0

    # Download and stitch Street View images
    streetview_images = []
    for heading in headings:
        streetview_url = f"https://maps.googleapis.com/maps/api/streetview?size=640x640&location={location}&heading={heading}&key={keys.API_KEY}"
        streetview_path = os.path.join(output_dir, f"{lat}_{lon}_streetview_{heading}.jpg")
        try:
            streetview_response = requests.get(streetview_url, stream=True)
            streetview_response.raise_for_status()
            with open(streetview_path, "wb") as f:
                for chunk in streetview_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            streetview_images.append(Image.open(streetview_path))
        except requests.exceptions.RequestException as e:
            print(f"Error downloading Street View image for {location} (heading {heading}): {e}")
            return 0

    # Horizontally stitch Street View images
    if streetview_images:
        widths, heights = zip(*(i.size for i in streetview_images))
        total_width = sum(widths)
        max_height = max(heights)
        stitched_image = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for img in streetview_images:
            stitched_image.paste(img, (x_offset, 0))
            x_offset += img.size[0]

        stitched_image_path = os.path.join(output_dir, f"{lat}_{lon}_streetview_stitched.jpg")
        stitched_image.save(stitched_image_path)

    return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_dir", type=str, default="dataset", help="Output directory")
    parser.add_argument("--num_images", type=int, default=5, help="Number of images to fetch")
    args = parser.parse_args()

    # TODO: change both South America and Africa rural areas as according to the exit code they are all indoor, which is unlikely...

    # Initialize Google Maps client
    gmaps = googlemaps.Client(key=keys.API_KEY)

    # Create output directory
    main_dir = args.main_dir
    os.makedirs(main_dir, exist_ok=True)

    # Broader but still strategic area selection
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

    # Load the Global Urban Areas dataset
    URBAN_AREAS_SHP_PATH = r"utils/urban_areas/urban_areas_full.shp"
    urban_areas = gpd.read_file(URBAN_AREAS_SHP_PATH)

    # Fetch equal number of images for urban and rural regions
    for continent, regions in AREAS.items():
        if continent != "Africa":
            continue
        for region_type, area in regions.items():
            image_count = 0
            while image_count < args.num_images:
                output_dir = os.path.join(main_dir, continent, region_type)
                os.makedirs(output_dir, exist_ok=True)
                
                lat, lon = generate_coordinates(area)
                if is_inside_urban_area(lat, lon, urban_areas) == (region_type == "urban"):
                    print(f"Fetching images for {continent} - {region_type} at {lat}, {lon}")
                    success = fetch_images(lat, lon, output_dir, gmaps)
                    if success:
                        image_count += 1