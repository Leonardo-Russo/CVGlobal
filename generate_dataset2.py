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
import time
import json
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging
import sys

# ==== Logging configuration ====
LOG_FMT = "[%(asctime)s] %(levelname)1s %(name)s: %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FMT,
    datefmt=DATE_FMT,
    stream=sys.stdout
)
logger = logging.getLogger("dataset_gen")


class DatasetStats:
    """Class to track dataset generation statistics."""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.end_time = None
        self.stats = {}
        self.total_api_calls = 0
        self.total_successful_downloads = 0
        self.total_failed_downloads = 0
        self.error_log = []
        
    def init_continent_stats(self, continent, region_type):
        if continent not in self.stats:
            self.stats[continent] = {}
        if region_type not in self.stats[continent]:
            self.stats[continent][region_type] = {
                'successful_images': 0,
                'failed_attempts': 0,
                'total_attempts': 0,
                'indoor_locations_skipped': 0,
                'metadata_failures': 0,
                'download_failures': 0,
                'coordinates_generated': 0,
                'urban_area_mismatches': 0
            }
    def record_attempt(self, continent, region_type):
        """Record an attempt to fetch images."""
        self.stats[continent][region_type]['total_attempts'] += 1
        
    def record_coordinate_generation(self, continent, region_type):
        """Record coordinate generation."""
        self.stats[continent][region_type]['coordinates_generated'] += 1
        
    def record_urban_area_mismatch(self, continent, region_type):
        """Record when coordinates don't match expected urban/rural classification."""
        self.stats[continent][region_type]['urban_area_mismatches'] += 1
        
    def record_success(self, continent, region_type):
        """Record successful image download."""
        self.stats[continent][region_type]['successful_images'] += 1
        self.total_successful_downloads += 1
        
    def record_failure(self, continent, region_type, failure_type='general'):
        """Record failed attempt."""
        self.stats[continent][region_type]['failed_attempts'] += 1
        self.total_failed_downloads += 1
        if failure_type == 'metadata':
            self.stats[continent][region_type]['metadata_failures'] += 1
        elif failure_type == 'download':
            self.stats[continent][region_type]['download_failures'] += 1
    
    def record_indoor_skip(self, continent, region_type):
        """Record when location is skipped for being indoor."""
        self.stats[continent][region_type]['indoor_locations_skipped'] += 1
        
    def record_api_call(self):
        """Record API call."""
        self.total_api_calls += 1
        
    def record_error(self, error_msg, continent=None, region_type=None):
        """Record error message."""
        self.error_log.append({
            'timestamp': datetime.now().isoformat(),
            'continent': continent,
            'region_type': region_type,
            'error': error_msg
        })
    
    def finalize(self):
        """Finalize statistics collection."""
        self.end_time = datetime.now()
        
    def get_duration(self):
        """Get total duration of dataset generation."""
        if self.end_time:
            return self.end_time - self.start_time
        return datetime.now() - self.start_time
    
    def generate_summary(self, output_file):
        """Generate comprehensive summary report."""
        duration = self.get_duration()
        
        summary = {
            'generation_info': {
                'start_time': self.start_time.isoformat(),
                'end_time': self.end_time.isoformat() if self.end_time else None,
                'total_duration_seconds': duration.total_seconds(),
                'total_duration_formatted': str(duration).split('.')[0],  # Remove microseconds
            },
            'overall_stats': {
                'total_api_calls': self.total_api_calls,
                'total_successful_downloads': self.total_successful_downloads,
                'total_failed_downloads': self.total_failed_downloads,
                'overall_success_rate': (self.total_successful_downloads / 
                                       max(self.total_successful_downloads + self.total_failed_downloads, 1)) * 100
            },
            'continent_breakdown': {},
            'error_summary': {
                'total_errors': len(self.error_log),
                'recent_errors': self.error_log[-10:] if len(self.error_log) > 10 else self.error_log
            }
        }
        
        # Calculate per-continent statistics
        for continent, regions in self.stats.items():
            summary['continent_breakdown'][continent] = {}
            continent_totals = {
                'total_successful': 0,
                'total_failed': 0,
                'total_attempts': 0,
                'total_coordinates_generated': 0,
                'total_urban_mismatches': 0
            }
            
            for region_type, stats in regions.items():
                success_rate = (stats['successful_images'] / max(stats['total_attempts'], 1)) * 100
                failure_rate = (stats['failed_attempts'] / max(stats['total_attempts'], 1)) * 100
                
                summary['continent_breakdown'][continent][region_type] = {
                    'successful_images': stats['successful_images'],
                    'failed_attempts': stats['failed_attempts'],
                    'total_attempts': stats['total_attempts'],
                    'success_rate_percent': round(success_rate, 2),
                    'failure_rate_percent': round(failure_rate, 2),
                    'indoor_locations_skipped': stats['indoor_locations_skipped'],
                    'metadata_failures': stats['metadata_failures'],
                    'download_failures': stats['download_failures'],
                    'coordinates_generated': stats['coordinates_generated'],
                    'urban_area_mismatches': stats['urban_area_mismatches'],
                    'coordinate_efficiency_percent': round(
                        (stats['total_attempts'] / max(stats['coordinates_generated'], 1)) * 100, 2
                    )
                }
                
                # Update continent totals
                continent_totals['total_successful'] += stats['successful_images']
                continent_totals['total_failed'] += stats['failed_attempts']
                continent_totals['total_attempts'] += stats['total_attempts']
                continent_totals['total_coordinates_generated'] += stats['coordinates_generated']
                continent_totals['total_urban_mismatches'] += stats['urban_area_mismatches']
            
            # Add continent summary
            summary['continent_breakdown'][continent]['continent_summary'] = {
                'total_successful_images': continent_totals['total_successful'],
                'total_failed_attempts': continent_totals['total_failed'],
                'total_attempts': continent_totals['total_attempts'],
                'continent_success_rate_percent': round(
                    (continent_totals['total_successful'] / max(continent_totals['total_attempts'], 1)) * 100, 2
                ),
                'total_coordinates_generated': continent_totals['total_coordinates_generated'],
                'total_urban_mismatches': continent_totals['total_urban_mismatches']
            }
        
        # Write summary to file
        with open(output_file, 'w') as f:
            f.write("DATASET GENERATION SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Generation info
            f.write("GENERATION INFORMATION:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Start Time: {summary['generation_info']['start_time']}\n")
            f.write(f"End Time: {summary['generation_info']['end_time']}\n")
            f.write(f"Total Duration: {summary['generation_info']['total_duration_formatted']}\n")
            f.write(f"Duration (seconds): {summary['generation_info']['total_duration_seconds']:.2f}\n\n")
            
            # Overall stats
            f.write("OVERALL STATISTICS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total API Calls: {summary['overall_stats']['total_api_calls']}\n")
            f.write(f"Total Successful Downloads: {summary['overall_stats']['total_successful_downloads']}\n")
            f.write(f"Total Failed Downloads: {summary['overall_stats']['total_failed_downloads']}\n")
            f.write(f"Overall Success Rate: {summary['overall_stats']['overall_success_rate']:.2f}%\n\n")
            
            # Per-continent breakdown
            f.write("CONTINENT BREAKDOWN:\n")
            f.write("-" * 20 + "\n")
            for continent, data in summary['continent_breakdown'].items():
                f.write(f"\n{continent.upper()}:\n")
                f.write("  " + "=" * (len(continent) + 1) + "\n")
                
                for region_type, stats in data.items():
                    if region_type == 'continent_summary':
                        continue
                    f.write(f"  {region_type.title()}:\n")
                    f.write(f"    Successful Images: {stats['successful_images']}\n")
                    f.write(f"    Failed Attempts: {stats['failed_attempts']}\n")
                    f.write(f"    Total Attempts: {stats['total_attempts']}\n")
                    f.write(f"    Success Rate: {stats['success_rate_percent']:.2f}%\n")
                    f.write(f"    Failure Rate: {stats['failure_rate_percent']:.2f}%\n")
                    f.write(f"    Indoor Locations Skipped: {stats['indoor_locations_skipped']}\n")
                    f.write(f"    Metadata Failures: {stats['metadata_failures']}\n")
                    f.write(f"    Download Failures: {stats['download_failures']}\n")
                    f.write(f"    Coordinates Generated: {stats['coordinates_generated']}\n")
                    f.write(f"    Urban Area Mismatches: {stats['urban_area_mismatches']}\n")
                    f.write(f"    Coordinate Efficiency: {stats['coordinate_efficiency_percent']:.2f}%\n\n")
                
                # Continent summary
                if 'continent_summary' in data:
                    cs = data['continent_summary']
                    f.write(f"  {continent.title()} Summary:\n")
                    f.write(f"    Total Successful: {cs['total_successful_images']}\n")
                    f.write(f"    Total Failed: {cs['total_failed_attempts']}\n")
                    f.write(f"    Total Attempts: {cs['total_attempts']}\n")
                    f.write(f"    Continent Success Rate: {cs['continent_success_rate_percent']:.2f}%\n")
                    f.write(f"    Total Coordinates Generated: {cs['total_coordinates_generated']}\n")
                    f.write(f"    Total Urban Mismatches: {cs['total_urban_mismatches']}\n\n")
            
            # Error summary
            if summary['error_summary']['total_errors'] > 0:
                f.write("ERROR SUMMARY:\n")
                f.write("-" * 15 + "\n")
                f.write(f"Total Errors Logged: {summary['error_summary']['total_errors']}\n\n")
                if summary['error_summary']['recent_errors']:
                    f.write("Recent Errors:\n")
                    for i, error in enumerate(summary['error_summary']['recent_errors'][-5:], 1):
                        f.write(f"  {i}. [{error['timestamp']}] ")
                        if error['continent']:
                            f.write(f"({error['continent']}-{error['region_type']}) ")
                        f.write(f"{error['error']}\n")
        
        # Also save as JSON for programmatic access
        json_file = output_file.replace('.dat', '.json')
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary report saved to: {output_file}")
        print(f"JSON data saved to: {json_file}")


def create_session():
    """
    Create a requests session with retry strategy.
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
        backoff_factor=1
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def generate_coordinates(area):
    """
    Generate random latitude and longitufde within the given area.
    """
    lat = random.uniform(area["min_lat"], area["max_lat"])
    lon = random.uniform(area["min_lon"], area["max_lon"])
    return lat, lon


def get_streetview_metadata(lat, lon, api_key, session, stats, continent, region_type, max_retries=3):
    """
    Fetches Street View metadata with retry logic.
    """
    url = f"https://maps.googleapis.com/maps/api/streetview/metadata?location={lat},{lon}&key={api_key}"
    for attempt in range(max_retries):
        try:
            stats.record_api_call()
            response = session.get(url, timeout=30)
            data = response.json()
            status = data.get("status")
            if status == "OK":
                return data
            if status == "ZERO_RESULTS":
                logger.warning("No Street View at %.6f,%.6f – ZERO_RESULTS", lat, lon)
                stats.record_failure(continent, region_type, 'metadata')
                return None
            logger.error("Error fetching Street View metadata: %s", status)
            stats.record_error(f"Metadata status {status}", continent, region_type)
            time.sleep(2 ** attempt)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            logger.error("Connection error on metadata attempt %d: %s", attempt+1, e)
            stats.record_error(str(e), continent, region_type)
            time.sleep(2 ** attempt)
        except Exception as e:
            logger.exception("Unexpected error fetching Street View metadata")
            stats.record_error(str(e), continent, region_type)
            return None
    return None


def is_likely_outdoor(metadata, gmaps, debug=False):
    """
    Checks if a Street View panorama is likely outdoors.
    """
    if metadata.get("status") != "OK":
        return False
    if "place_id" not in metadata:  # if there's no place_id, assume it's outdoor (street level)
        return True

    try:
        place = gmaps.place(metadata["place_id"])['result']['types']
        indoor_types = [
            'shopping_mall', 'store', 'restaurant', 'hospital',
            'school', 'university', 'library', 'museum',
            'airport', 'subway_station', 'train_station',
            'parking', 'gas_station'
        ]
        if any(t in indoor_types for t in place):
            return False
    except Exception:
        pass
    return True


def is_inside_urban_area(lat, lon, urban_areas):
    """
    Check if the given coordinates are inside any urban area polygon.
    """
    point = Point(lon, lat)  # Shapely uses (lon, lat) format
    return any(urban_areas.contains(point))

def count_existing_images(output_dir):
    """
    Count how many complete image sets already exist in the output directory.
    A complete set consists of: satellite image + 4 streetview images + stitched image.
    """
    if not os.path.exists(output_dir):
        return 0
    
    # Look for satellite images as the primary indicator
    satellite_files = [f for f in os.listdir(output_dir) if f.endswith('_satellite.jpg')]
    complete_sets = 0
    
    for sat_file in satellite_files:
        # Extract the coordinate part from filename
        coord_part = sat_file.replace('_satellite.jpg', '')
        
        # Check if all required files exist for this coordinate
        required_files = [
            f"{coord_part}_satellite.jpg",
            f"{coord_part}_streetview_0.jpg",
            f"{coord_part}_streetview_90.jpg", 
            f"{coord_part}_streetview_180.jpg",
            f"{coord_part}_streetview_270.jpg",
            f"{coord_part}_streetview_stitched.jpg"
        ]
        
        if all(os.path.exists(os.path.join(output_dir, f)) for f in required_files):
            complete_sets += 1
    
    return complete_sets

def is_coordinate_already_processed(lat, lon, output_dir):
    """
    Check if images for this specific coordinate already exist.
    """
    coord_prefix = f"{lat}_{lon}"
    required_files = [
        f"{coord_prefix}_satellite.jpg",
        f"{coord_prefix}_streetview_0.jpg",
        f"{coord_prefix}_streetview_90.jpg", 
        f"{coord_prefix}_streetview_180.jpg",
        f"{coord_prefix}_streetview_270.jpg",
        f"{coord_prefix}_streetview_stitched.jpg"
    ]
    
    return all(os.path.exists(os.path.join(output_dir, f)) for f in required_files)

# def fetch_images(lat, lon, output_dir, gmaps, session, stats, continent, region_type, headings=[0, 90, 180, 270]):
#     stats.record_attempt(continent, region_type)
#     # ... rest unchanged, replacing prints with logger.info/debug/warning ...
#     return 1  # simplified for brevity

def fetch_images(lat, lon, output_dir, gmaps, session, stats, continent, region_type, headings=[0, 90, 180, 270]):
    """
    Fetches satellite and Street View images for a given location.
    """
    location = f"{lat},{lon}"

    # Record the attempt
    stats.record_attempt(continent, region_type)

    # Check if this coordinate is already processed
    if is_coordinate_already_processed(lat, lon, output_dir):
        print(f"    ⏭ Images already exist for {lat:.6f}, {lon:.6f} - skipping")
        stats.record_success(continent, region_type)
        return 1

    # Fetch Street View metadata
    metadata = get_streetview_metadata(lat, lon, keys.API_KEY, session, stats, continent, region_type)

    if metadata is None:
        stats.record_failure(continent, region_type, 'metadata')
        return 0

    # Check if likely outdoor
    if not is_likely_outdoor(metadata, gmaps, debug=False):
        print(f"    Skipping likely indoor location...")
        stats.record_indoor_skip(continent, region_type)
        return 0

    # Use corrected lat and lon from metadata
    corrected_lat = metadata['location']['lat']
    corrected_lon = metadata['location']['lng']
    
    # Check again with corrected coordinates
    if is_coordinate_already_processed(corrected_lat, corrected_lon, output_dir):
        print(f"    ⏭ Images already exist for corrected coordinates {corrected_lat:.6f}, {corrected_lon:.6f} - skipping")
        stats.record_success(continent, region_type)
        return 1
    
    location = f"{corrected_lat},{corrected_lon}"

    # Download satellite image with retry logic
    satellite_url = f"https://maps.googleapis.com/maps/api/staticmap?center={location}&zoom=18&size=640x640&maptype=satellite&key={keys.API_KEY}"
    satellite_path = os.path.join(output_dir, f"{corrected_lat}_{corrected_lon}_satellite.jpg")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            stats.record_api_call()
            satellite_response = session.get(satellite_url, stream=True, timeout=30)
            satellite_response.raise_for_status()
            with open(satellite_path, "wb") as f:
                for chunk in satellite_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            break
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            error_msg = f"Error downloading satellite image (attempt {attempt + 1}): {e}"
            print(error_msg)
            stats.record_error(error_msg, continent, region_type)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                print(f"Failed to download satellite image after {max_retries} attempts")
                stats.record_failure(continent, region_type, 'download')
                return 0
        except Exception as e:
            error_msg = f"Unexpected error downloading satellite image: {e}"
            print(error_msg)
            stats.record_error(error_msg, continent, region_type)
            stats.record_failure(continent, region_type, 'download')
            return 0

    # Download and stitch Street View images with retry logic
    streetview_images = []
    for heading in headings:
        streetview_url = f"https://maps.googleapis.com/maps/api/streetview?size=640x640&location={location}&heading={heading}&key={keys.API_KEY}"
        streetview_path = os.path.join(output_dir, f"{corrected_lat}_{corrected_lon}_streetview_{heading}.jpg")
        
        for attempt in range(max_retries):
            try:
                stats.record_api_call()
                streetview_response = session.get(streetview_url, stream=True, timeout=30)
                streetview_response.raise_for_status()
                with open(streetview_path, "wb") as f:
                    for chunk in streetview_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                streetview_images.append(Image.open(streetview_path))
                break
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                error_msg = f"Error downloading Street View image (heading {heading}, attempt {attempt + 1}): {e}"
                print(error_msg)
                stats.record_error(error_msg, continent, region_type)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    print(f"Failed to download Street View image (heading {heading}) after {max_retries} attempts")
                    stats.record_failure(continent, region_type, 'download')
                    return 0
            except Exception as e:
                error_msg = f"Unexpected error downloading Street View image (heading {heading}): {e}"
                print(error_msg)
                stats.record_error(error_msg, continent, region_type)
                stats.record_failure(continent, region_type, 'download')
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

        stitched_image_path = os.path.join(output_dir, f"{corrected_lat}_{corrected_lon}_streetview_stitched.jpg")
        stitched_image.save(stitched_image_path)

    # Record success
    stats.record_success(continent, region_type)
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_dir", type=str, default="cvglobal", help="Output directory")
    parser.add_argument("--num_images", type=int, default=5, help="Number of images to fetch")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    stats = DatasetStats()
    gmaps = googlemaps.Client(key=keys.API_KEY)
    session = create_session()

    main_dir = os.path.join(r"D:\datasets", args.main_dir)
    os.makedirs(main_dir, exist_ok=True)

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

    URBAN_AREAS_SHP_PATH = r"utils/urban_areas/urban_areas_full.shp"
    urban_areas = gpd.read_file(URBAN_AREAS_SHP_PATH)

    max_failed_attempts = 50

    print("Starting dataset generation...")
    print(f"Target: {args.num_images} images per region")
    print(f"Output directory: {main_dir}")
    
    # Check overall progress
    total_existing = 0
    total_needed = 0
    resume_info = []
    
    for continent, regions in AREAS.items():
        for region_type in regions.keys():
            output_dir = os.path.join(main_dir, continent, region_type)
            existing = count_existing_images(output_dir)
            needed = max(0, args.num_images - existing)
            total_existing += existing
            total_needed += needed
            if existing > 0:
                resume_info.append(f"  {continent}-{region_type}: {existing}/{args.num_images} complete")
    
    if total_existing > 0:
        print(f"\nResuming generation - found {total_existing} existing images:")
        for info in resume_info:
            print(info)
        print(f"Need to generate {total_needed} more images total")
    else:
        print("Starting fresh generation")
    
    print("=" * 60)

    # Fetch equal number of images for urban and rural regions
    for continent, regions in AREAS.items():
        # if continent != "Africa":
        #     continue
        print(f"\nProcessing {continent}...")
        for region_type, area in regions.items():
            output_dir = os.path.join(main_dir, continent, region_type)
            os.makedirs(output_dir, exist_ok=True)
            
            # Check existing images
            existing_count = count_existing_images(output_dir)
            remaining_needed = max(0, args.num_images - existing_count)
            
            if existing_count > 0:
                print(f"  {region_type.capitalize()} areas: Found {existing_count} existing images, need {remaining_needed} more")
            else:
                print(f"  {region_type.capitalize()} areas: Starting fresh, need {args.num_images} images")
            
            if remaining_needed == 0:
                print(f"    ✓ Already completed {continent}-{region_type}: {existing_count} images")
                # Initialize stats even if no work needed (for reporting)
                stats.init_continent_stats(continent, region_type)
                # Add existing images to success count for accurate reporting
                for _ in range(existing_count):
                    stats.record_success(continent, region_type)
                continue
            
            # Initialize stats for this continent-region combination
            stats.init_continent_stats(continent, region_type)
            
            # Add existing images to success count for accurate reporting
            for _ in range(existing_count):
                stats.record_success(continent, region_type)
            
            image_count = existing_count  # Start from existing count
            failed_attempts = 0
            
            while image_count < args.num_images and failed_attempts < max_failed_attempts:
                lat, lon = generate_coordinates(area)
                stats.record_coordinate_generation(continent, region_type)
                
                if is_inside_urban_area(lat, lon, urban_areas) == (region_type == "urban"):
                    print(f"    Fetching images for {continent} - {region_type} @ {lat:.6f}, {lon:.6f}")
                    success = fetch_images(lat, lon, output_dir, gmaps, session, stats, continent, region_type)
                    if success:
                        image_count += 1
                        print(f"    ✓ Success! ({image_count}/{args.num_images})")
                        # Add delay between successful requests to avoid rate limiting
                        time.sleep(0.5)
                    else:
                        failed_attempts += 1
                        # print(f"    ✗ Failed attempt {failed_attempts}/{max_failed_attempts}")
                        # Add a longer delay before retrying after failure
                        time.sleep(2)
                else:
                    stats.record_urban_area_mismatch(continent, region_type)
                    
            if image_count < args.num_images:
                print(f"    ⚠ Warning: Only collected {image_count}/{args.num_images} images for {continent}-{region_type}")
            else:
                print(f"    ✓ Completed {continent}-{region_type}: {image_count} images")

    # Finalize statistics and generate summary
    stats.finalize()
    
    print("\n" + "=" * 60)
    print("Dataset generation completed!")
    
    # Generate summary report
    summary_file = os.path.join(main_dir, "summary.dat")
    stats.generate_summary(summary_file)
    
    # Print quick summary to console
    duration = stats.get_duration()
    print(f"\nQuick Summary:")
    print(f"  Duration: {str(duration).split('.')[0]}")
    print(f"  Total API calls: {stats.total_api_calls}")
    print(f"  Successful downloads: {stats.total_successful_downloads}")
    print(f"  Failed downloads: {stats.total_failed_downloads}")
    if stats.total_successful_downloads + stats.total_failed_downloads > 0:
        success_rate = (stats.total_successful_downloads / (stats.total_successful_downloads + stats.total_failed_downloads)) * 100
        print(f"  Overall success rate: {success_rate:.2f}%")
    print(f"  Errors logged: {len(stats.error_log)}")
    
    print(f"\nDetailed report available at: {summary_file}")
    print(f"JSON data available at: {summary_file.replace('.dat', '.json')}")

