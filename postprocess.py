import os
import argparse
import shutil
import json
from collections import defaultdict
import glob
from PIL import Image
import sys

def get_cvglobal_structure():
    """
    Returns the expected CVGlobal dataset structure based on generate_dataset2.py
    """
    continents = ["North America", "Europe", "Asia", "South America", "Africa"]
    region_types = ["urban", "rural"]
    return continents, region_types

def find_image_pairs(cvglobal_dir):
    """
    Find all complete image pairs in the CVGlobal dataset.
    Returns a list of tuples: (satellite_path, streetview_stitched_path, metadata)
    """
    continents, region_types = get_cvglobal_structure()
    image_pairs = []
    
    for continent in continents:
        for region_type in region_types:
            region_dir = os.path.join(cvglobal_dir, continent, region_type)
            
            if not os.path.exists(region_dir):
                print(f"Warning: Directory {region_dir} does not exist, skipping...")
                continue
            
            # Find all satellite images
            satellite_files = glob.glob(os.path.join(region_dir, "*_satellite.jpg"))
            
            for sat_file in satellite_files:
                # Extract coordinate prefix
                basename = os.path.basename(sat_file)
                coord_prefix = basename.replace('_satellite.jpg', '')
                
                # Check for corresponding stitched streetview image
                stitched_file = os.path.join(region_dir, f"{coord_prefix}_streetview_stitched.jpg")
                
                if os.path.exists(stitched_file):
                    # Extract coordinates from filename
                    parts = coord_prefix.split('_')
                    if len(parts) >= 2:
                        try:
                            lat = float(parts[0])
                            lon = float(parts[1])
                            
                            metadata = {
                                'continent': continent,
                                'region_type': region_type,
                                'latitude': lat,
                                'longitude': lon,
                                'original_prefix': coord_prefix
                            }
                            
                            image_pairs.append((sat_file, stitched_file, metadata))
                        except ValueError:
                            print(f"Warning: Could not parse coordinates from {coord_prefix}")
                            continue
                else:
                    print(f"Warning: Missing stitched streetview for {coord_prefix}")
    
    return image_pairs

def generate_unique_id(metadata, id_counter):
    """
    Generate a unique ID for each image pair.
    Format: {continent_code}{region_code}_{counter:06d}
    """
    continent_codes = {
        'North America': 'NA',
        'Europe': 'EU', 
        'Asia': 'AS',
        'South America': 'SA',
        'Africa': 'AF'
    }
    
    region_codes = {
        'urban': 'U',
        'rural': 'R'
    }
    
    continent_code = continent_codes.get(metadata['continent'], 'XX')
    region_code = region_codes.get(metadata['region_type'], 'X')
    
    return f"{continent_code}{region_code}_{id_counter:06d}"

def copy_and_rename_images(image_pairs, output_dir, image_size=None):
    """
    Copy and rename images to CVUSA-compatible structure.
    
    Args:
        image_pairs: List of (satellite_path, streetview_path, metadata) tuples
        output_dir: Output directory for CVUSA-compatible dataset
        image_size: Optional tuple (width, height) to resize images
    """
    
    # Create CVUSA-style directory structure
    sat_dir = os.path.join(output_dir, 'bingmap', '18')  # zoom level 18
    streetview_dir = os.path.join(output_dir, 'streetview', 'panos')
    
    os.makedirs(sat_dir, exist_ok=True)
    os.makedirs(streetview_dir, exist_ok=True)
    
    # Track statistics
    stats = {
        'total_pairs': len(image_pairs),
        'successful_copies': 0,
        'failed_copies': 0,
        'by_continent': defaultdict(int),
        'by_region_type': defaultdict(int)
    }
    
    # Store mapping for metadata
    id_mapping = {}
    
    print(f"Processing {len(image_pairs)} image pairs...")
    
    for i, (sat_path, street_path, metadata) in enumerate(image_pairs):
        try:
            # Generate unique ID
            unique_id = generate_unique_id(metadata, i)
            
            # Define output paths
            sat_output = os.path.join(sat_dir, f"{unique_id}.jpg")
            street_output = os.path.join(streetview_dir, f"{unique_id}.jpg")
            
            # Copy and optionally resize satellite image
            if image_size:
                sat_img = Image.open(sat_path).convert('RGB')
                sat_img = sat_img.resize(image_size, Image.Resampling.LANCZOS)
                sat_img.save(sat_output, 'JPEG', quality=95)
            else:
                shutil.copy2(sat_path, sat_output)
            
            # Copy and optionally resize streetview image
            if image_size:
                street_img = Image.open(street_path).convert('RGB')
                # For streetview, maintain aspect ratio but resize height
                street_width, street_height = street_img.size
                if street_width > street_height:
                    # Panoramic image - resize to maintain aspect ratio
                    new_height = image_size[1]
                    new_width = int((street_width / street_height) * new_height)
                    street_img = street_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                else:
                    street_img = street_img.resize(image_size, Image.Resampling.LANCZOS)
                street_img.save(street_output, 'JPEG', quality=95)
            else:
                shutil.copy2(street_path, street_output)
            
            # Store mapping
            id_mapping[unique_id] = metadata
            
            # Update statistics
            stats['successful_copies'] += 1
            stats['by_continent'][metadata['continent']] += 1
            stats['by_region_type'][metadata['region_type']] += 1
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(image_pairs)} pairs...")
                
        except Exception as e:
            print(f"Error processing pair {i}: {e}")
            stats['failed_copies'] += 1
            continue
    
    return stats, id_mapping

def create_split_files(id_mapping, output_dir, train_ratio=0.8, val_ratio=0.1):
    """
    Create train/val/test split files similar to CVUSA format.
    """
    
    # Group by continent and region type for balanced splits
    grouped_ids = defaultdict(list)
    
    for unique_id, metadata in id_mapping.items():
        key = f"{metadata['continent']}_{metadata['region_type']}"
        grouped_ids[key].append(unique_id)
    
    # Create balanced splits
    train_ids = []
    val_ids = []
    test_ids = []
    
    for group, ids in grouped_ids.items():
        # Shuffle ids within each group
        import random
        random.shuffle(ids)
        
        n_total = len(ids)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train:n_train + n_val])
        test_ids.extend(ids[n_train + n_val:])
    
    # Shuffle final lists
    import random
    random.shuffle(train_ids)
    random.shuffle(val_ids)
    random.shuffle(test_ids)
    
    # Write split files
    splits_dir = os.path.join(output_dir, 'splits')
    os.makedirs(splits_dir, exist_ok=True)
    
    with open(os.path.join(splits_dir, 'train-19zl.csv'), 'w') as f:
        for img_id in train_ids:
            f.write(f"{img_id}.jpg\n")
    
    with open(os.path.join(splits_dir, 'val-19zl.csv'), 'w') as f:
        for img_id in val_ids:
            f.write(f"{img_id}.jpg\n")
    
    with open(os.path.join(splits_dir, 'test-19zl.csv'), 'w') as f:
        for img_id in test_ids:
            f.write(f"{img_id}.jpg\n")
    
    return {
        'train': len(train_ids),
        'val': len(val_ids), 
        'test': len(test_ids)
    }

def create_metadata_file(id_mapping, output_dir):
    """
    Create a comprehensive metadata file for the converted dataset.
    """
    metadata_file = os.path.join(output_dir, 'cvglobal_metadata.json')
    
    # Organize metadata by continent and region
    organized_metadata = {
        'dataset_info': {
            'name': 'CVGlobal',
            'description': 'Global cross-view dataset converted to CVUSA format',
            'total_pairs': len(id_mapping),
            'conversion_date': str(os.path.getmtime(__file__) if '__file__' in globals() else 'unknown')
        },
        'images': id_mapping,
        'statistics': {
            'by_continent': defaultdict(int),
            'by_region_type': defaultdict(int),
            'coordinate_ranges': {}
        }
    }
    
    # Calculate statistics
    lats, lons = [], []
    for metadata in id_mapping.values():
        organized_metadata['statistics']['by_continent'][metadata['continent']] += 1
        organized_metadata['statistics']['by_region_type'][metadata['region_type']] += 1
        lats.append(metadata['latitude'])
        lons.append(metadata['longitude'])
    
    organized_metadata['statistics']['coordinate_ranges'] = {
        'latitude': {'min': min(lats), 'max': max(lats)},
        'longitude': {'min': min(lons), 'max': max(lons)}
    }
    
    # Convert defaultdicts to regular dicts for JSON serialization
    organized_metadata['statistics']['by_continent'] = dict(organized_metadata['statistics']['by_continent'])
    organized_metadata['statistics']['by_region_type'] = dict(organized_metadata['statistics']['by_region_type'])
    
    with open(metadata_file, 'w') as f:
        json.dump(organized_metadata, f, indent=2)
    
    return metadata_file

def print_conversion_summary(stats, split_counts, output_dir):
    """
    Print a summary of the conversion process.
    """
    print("\n" + "="*60)
    print("CVGLOBAL TO CVUSA CONVERSION SUMMARY")
    print("="*60)
    
    print(f"Output Directory: {output_dir}")
    print(f"Total Pairs Processed: {stats['total_pairs']}")
    print(f"Successful Copies: {stats['successful_copies']}")
    print(f"Failed Copies: {stats['failed_copies']}")
    
    if stats['successful_copies'] > 0:
        success_rate = (stats['successful_copies'] / stats['total_pairs']) * 100
        print(f"Success Rate: {success_rate:.1f}%")
    
    print("\nBy Continent:")
    for continent, count in stats['by_continent'].items():
        print(f"  {continent}: {count} pairs")
    
    print("\nBy Region Type:")
    for region_type, count in stats['by_region_type'].items():
        print(f"  {region_type}: {count} pairs")
    
    print("\nDataset Splits:")
    print(f"  Training: {split_counts['train']} images")
    print(f"  Validation: {split_counts['val']} images") 
    print(f"  Test: {split_counts['test']} images")
    
    print("\nCVUSA-Compatible Structure Created:")
    print(f"  📁 {output_dir}/")
    print(f"     ├── 📁 bingmap/18/        (satellite images)")
    print(f"     ├── 📁 streetview/panos/  (street view images)")
    print(f"     ├── 📁 splits/            (train/val/test splits)")
    print(f"     └── 📄 cvglobal_metadata.json")
    
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Convert CVGlobal dataset to CVUSA-compatible format")
    parser.add_argument("input_dir", help="Path to CVGlobal dataset directory")
    parser.add_argument("output_dir", help="Path to output CVUSA-compatible directory")
    parser.add_argument("--resize", nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'),
                       help="Resize images to specified dimensions (e.g., --resize 512 512)")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                       help="Training set ratio (default: 0.8)")
    parser.add_argument("--val-ratio", type=float, default=0.1, 
                       help="Validation set ratio (default: 0.1)")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite output directory if it exists")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist!")
        return 1
    
    # Check output directory
    if os.path.exists(args.output_dir):
        if not args.overwrite:
            print(f"Error: Output directory '{args.output_dir}' already exists!")
            print("Use --overwrite to overwrite existing directory.")
            return 1
        else:
            print(f"Warning: Overwriting existing directory '{args.output_dir}'")
            shutil.rmtree(args.output_dir)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Starting CVGlobal to CVUSA conversion...")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    
    # Find image pairs
    print("\nScanning for image pairs...")
    image_pairs = find_image_pairs(args.input_dir)
    
    if not image_pairs:
        print("Error: No valid image pairs found in the input directory!")
        return 1
    
    print(f"Found {len(image_pairs)} complete image pairs")
    
    # Set image resize parameters
    image_size = None
    if args.resize:
        image_size = tuple(args.resize)
        print(f"Will resize images to {image_size[0]}x{image_size[1]}")
    
    # Copy and rename images
    print("\nCopying and renaming images...")
    stats, id_mapping = copy_and_rename_images(image_pairs, args.output_dir, image_size)
    
    if stats['successful_copies'] == 0:
        print("Error: No images were successfully copied!")
        return 1
    
    # Create train/val/test splits
    print("\nCreating dataset splits...")
    split_counts = create_split_files(id_mapping, args.output_dir, 
                                    args.train_ratio, args.val_ratio)
    
    # Create metadata file
    print("Creating metadata file...")
    metadata_file = create_metadata_file(id_mapping, args.output_dir)
    
    # Print summary
    print_conversion_summary(stats, split_counts, args.output_dir)
    
    print(f"\nConversion completed successfully!")
    print(f"Metadata saved to: {metadata_file}")
    
    return 0

if __name__ == "__main__":
    exit(main())