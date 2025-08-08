import os
import argparse
import json
from datetime import datetime
from collections import defaultdict
import glob

def count_images_in_directory(directory):
    """
    Count different types of images in a directory.
    Returns a dictionary with counts for each image type.
    """
    if not os.path.exists(directory):
        return {
            'satellite': 0,
            'streetview_individual': 0,
            'streetview_stitched': 0,
            'complete_sets': 0
        }
    
    # Count different image types
    satellite_files = glob.glob(os.path.join(directory, "*_satellite.jpg"))
    stitched_files = glob.glob(os.path.join(directory, "*_streetview_stitched.jpg"))
    
    # Count individual streetview images (0, 90, 180, 270 degrees)
    streetview_patterns = ["*_streetview_0.jpg", "*_streetview_90.jpg", 
                          "*_streetview_180.jpg", "*_streetview_270.jpg"]
    streetview_individual = 0
    for pattern in streetview_patterns:
        streetview_individual += len(glob.glob(os.path.join(directory, pattern)))
    
    # Count complete sets (coordinates that have all required files)
    complete_sets = 0
    for sat_file in satellite_files:
        # Extract coordinate prefix
        basename = os.path.basename(sat_file)
        coord_prefix = basename.replace('_satellite.jpg', '')
        
        # Check if all required files exist
        required_files = [
            f"{coord_prefix}_satellite.jpg",
            f"{coord_prefix}_streetview_0.jpg",
            f"{coord_prefix}_streetview_90.jpg",
            f"{coord_prefix}_streetview_180.jpg",
            f"{coord_prefix}_streetview_270.jpg",
            f"{coord_prefix}_streetview_stitched.jpg"
        ]
        
        if all(os.path.exists(os.path.join(directory, f)) for f in required_files):
            complete_sets += 1
    
    return {
        'satellite': len(satellite_files),
        'streetview_individual': streetview_individual,
        'streetview_stitched': len(stitched_files),
        'complete_sets': complete_sets
    }

def get_directory_size(directory):
    """
    Calculate total size of directory in bytes.
    """
    if not os.path.exists(directory):
        return 0
    
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size

def format_size(size_bytes):
    """
    Convert bytes to human readable format.
    """
    if size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024.0 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f} {size_names[i]}"

def load_generation_summary(dataset_dir):
    """
    Load the summary.json file if it exists from dataset generation.
    """
    summary_json = os.path.join(dataset_dir, "summary.json")
    if os.path.exists(summary_json):
        try:
            with open(summary_json, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load {summary_json}: {e}")
    return None

def analyze_dataset(dataset_dir):
    """
    Analyze the dataset directory structure and count images.
    """
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' does not exist!")
        return None
    
    # Expected continents and types based on generate_dataset2.py
    continents = ["North America", "Europe", "Asia", "South America", "Africa"]
    region_types = ["urban", "rural"]
    
    # Initialize results structure
    results = {
        'dataset_dir': dataset_dir,
        'analysis_time': datetime.now().isoformat(),
        'continents': {},
        'totals': {
            'satellite': 0,
            'streetview_individual': 0, 
            'streetview_stitched': 0,
            'complete_sets': 0,
            'total_size_bytes': 0
        }
    }
    
    # Analyze each continent and region type
    for continent in continents:
        results['continents'][continent] = {}
        continent_totals = {
            'satellite': 0,
            'streetview_individual': 0,
            'streetview_stitched': 0,
            'complete_sets': 0,
            'total_size_bytes': 0
        }
        
        for region_type in region_types:
            region_dir = os.path.join(dataset_dir, continent, region_type)
            
            # Count images in this region
            counts = count_images_in_directory(region_dir)
            
            # Get directory size
            dir_size = get_directory_size(region_dir)
            counts['total_size_bytes'] = dir_size
            counts['total_size_formatted'] = format_size(dir_size)
            counts['directory_exists'] = os.path.exists(region_dir)
            
            # Store results
            results['continents'][continent][region_type] = counts
            
            # Update continent totals
            for key in ['satellite', 'streetview_individual', 'streetview_stitched', 'complete_sets', 'total_size_bytes']:
                continent_totals[key] += counts.get(key, 0)
        
        # Store continent totals
        continent_totals['total_size_formatted'] = format_size(continent_totals['total_size_bytes'])
        results['continents'][continent]['continent_totals'] = continent_totals
        
        # Update global totals
        for key in ['satellite', 'streetview_individual', 'streetview_stitched', 'complete_sets', 'total_size_bytes']:
            results['totals'][key] += continent_totals.get(key, 0)
    
    # Format global total size
    results['totals']['total_size_formatted'] = format_size(results['totals']['total_size_bytes'])
    
    # Load generation summary if available
    generation_summary = load_generation_summary(dataset_dir)
    if generation_summary:
        results['generation_summary'] = generation_summary
    
    return results

def print_summary(results):
    """
    Print a formatted summary of the dataset analysis.
    """
    print("=" * 80)
    print("CVGLOBAL DATASET SUMMARY")
    print("=" * 80)
    print(f"Dataset Directory: {results['dataset_dir']}")
    print(f"Analysis Time: {results['analysis_time']}")
    print()
    
    # Print overall totals
    print("OVERALL TOTALS:")
    print("-" * 20)
    totals = results['totals']
    print(f"Complete Image Sets: {totals['complete_sets']}")
    print(f"Satellite Images: {totals['satellite']}")
    print(f"Street View Images (Individual): {totals['streetview_individual']}")
    print(f"Street View Images (Stitched): {totals['streetview_stitched']}")
    print(f"Total Dataset Size: {totals['total_size_formatted']}")
    print()
    
    # Print per-continent breakdown
    print("CONTINENT BREAKDOWN:")
    print("-" * 30)
    
    # Table header
    print(f"{'Continent':<15} {'Type':<6} {'Complete':<8} {'Satellite':<9} {'Individual':<10} {'Stitched':<8} {'Size':<10}")
    print("-" * 80)
    
    for continent, data in results['continents'].items():
        first_row = True
        
        for region_type in ['urban', 'rural']:
            if region_type in data:
                region_data = data[region_type]
                
                # Print continent name only on first row
                continent_name = continent if first_row else ""
                
                print(f"{continent_name:<15} {region_type:<6} {region_data['complete_sets']:<8} "
                      f"{region_data['satellite']:<9} {region_data['streetview_individual']:<10} "
                      f"{region_data['streetview_stitched']:<8} {region_data['total_size_formatted']:<10}")
                
                first_row = False
        
        # Print continent totals
        continent_totals = data.get('continent_totals', {})
        if continent_totals:
            print(f"{'TOTAL':<15} {'      ':<6} {continent_totals['complete_sets']:<8} "
                  f"{continent_totals['satellite']:<9} {continent_totals['streetview_individual']:<10} "
                  f"{continent_totals['streetview_stitched']:<8} {continent_totals['total_size_formatted']:<10}")
            print("-" * 80)
    
    # Print generation information if available
    if 'generation_summary' in results:
        gen_summary = results['generation_summary']
        print("\nGENERATION INFORMATION:")
        print("-" * 25)
        
        if 'generation_info' in gen_summary:
            gen_info = gen_summary['generation_info']
            print(f"Generation Duration: {gen_info.get('total_duration_formatted', 'N/A')}")
            print(f"Start Time: {gen_info.get('start_time', 'N/A')}")
            print(f"End Time: {gen_info.get('end_time', 'N/A')}")
        
        if 'overall_stats' in gen_summary:
            stats = gen_summary['overall_stats']
            print(f"Total API Calls: {stats.get('total_api_calls', 'N/A')}")
            print(f"Success Rate: {stats.get('overall_success_rate', 'N/A'):.2f}%")
    
    print()
    print("=" * 80)

def save_summary_json(results, output_file):
    """
    Save the analysis results to a JSON file.
    """
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Detailed analysis saved to: {output_file}")
    except Exception as e:
        print(f"Error saving JSON summary: {e}")

def main():
    parser = argparse.ArgumentParser(description="Summarize CVGlobal dataset contents")
    parser.add_argument("dataset_dir", default=r'D:\datasets\final', help="Path to the dataset directory")
    parser.add_argument("--output", "-o", default=None, help="Output JSON file for detailed analysis")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only print totals")
    
    args = parser.parse_args()
    
    # Analyze the dataset
    results = analyze_dataset(args.dataset_dir)
    
    if results is None:
        return 1
    
    # Print summary
    if not args.quiet:
        print_summary(results)
    else:
        # Print only totals for quiet mode
        totals = results['totals']
        print(f"Complete Sets: {totals['complete_sets']}")
        print(f"Total Size: {totals['total_size_formatted']}")
    
    # Save detailed analysis if requested
    if args.output:
        output_path = os.path.join(args.dataset_dir, args.output)
        save_summary_json(results, output_path)

    return 0

if __name__ == "__main__":
    exit(main())