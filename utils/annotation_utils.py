import json
import numpy as np
import os
from pathlib import Path


def grid_to_annotations(grid_config_path, image_filename, occupancy_list=None):
    """
    Convert grid configuration to annotation format.
    
    Args:
        grid_config_path: Path to parking_grid_config.json from grid.py
        image_filename: Filename of the image (e.g., "frame_0001.jpg")
        occupancy_list: Optional list of boolean occupancy values for each spot
                       If None, all spots default to False (empty)
    
    Returns:
        dict: Annotation data for this image
    """
    with open(grid_config_path, 'r') as f:
        config = json.load(f)
    
    spots = config['spots']
    num_spots = len(spots)
    
    # Get image dimensions from the first spot to normalize
    # Assuming spots are in image coordinates
    img_width = config.get('image_width', 1920)
    img_height = config.get('image_height', 1080)
    
    # Convert spots to normalized coordinates
    rois_list = []
    for spot in spots:
        # Spot is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        normalized_spot = []
        for point in spot:
            x_norm = point[0] / img_width
            y_norm = point[1] / img_height
            normalized_spot.append([x_norm, y_norm])
        rois_list.append(normalized_spot)
    
    # Set occupancy
    if occupancy_list is None:
        occupancy_list = [False] * num_spots
    elif len(occupancy_list) != num_spots:
        raise ValueError(f"occupancy_list length ({len(occupancy_list)}) must match number of spots ({num_spots})")
    
    return {
        'file_name': image_filename,
        'rois': rois_list,
        'occupancy': occupancy_list
    }


def add_grid_to_annotations(annotations_path, grid_config_path, image_filename, 
                           occupancy_list=None, split='train'):
    """
    Add grid annotations to an existing annotations.json file.
    
    Args:
        annotations_path: Path to annotations.json
        grid_config_path: Path to parking_grid_config.json
        image_filename: Filename of the image
        occupancy_list: Optional list of occupancy values
        split: Dataset split ('train', 'valid', or 'test')
    """
    # Load or create annotations
    if Path(annotations_path).exists():
        with open(annotations_path, 'r') as f:
            all_data = json.load(f)
    else:
        all_data = {'train': {'file_names': [], 'rois_list': [], 'occupancy_list': []},
                   'valid': {'file_names': [], 'rois_list': [], 'occupancy_list': []},
                   'test': {'file_names': [], 'rois_list': [], 'occupancy_list': []}}
    
    # Ensure split exists
    if split not in all_data:
        all_data[split] = {'file_names': [], 'rois_list': [], 'occupancy_list': []}
    
    # Convert grid to annotation
    annotation = grid_to_annotations(grid_config_path, image_filename, occupancy_list)
    
    # Add to annotations
    if image_filename in all_data[split]['file_names']:
        # Update existing
        idx = all_data[split]['file_names'].index(image_filename)
        all_data[split]['rois_list'][idx] = annotation['rois']
        all_data[split]['occupancy_list'][idx] = annotation['occupancy']
        print(f"Updated annotations for {image_filename}")
    else:
        # Add new
        all_data[split]['file_names'].append(image_filename)
        all_data[split]['rois_list'].append(annotation['rois'])
        all_data[split]['occupancy_list'].append(annotation['occupancy'])
        print(f"Added annotations for {image_filename}")
    
    # Save
    with open(annotations_path, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"✓ Saved to {annotations_path}")


def annotations_to_grid_config(annotations_path, image_filename, split='train'):
    """
    Extract grid configuration from annotations for a specific image.
    Useful for visualizing or editing existing annotations with the grid tool.
    
    Args:
        annotations_path: Path to annotations.json
        image_filename: Filename of the image
        split: Dataset split
    
    Returns:
        dict: Grid configuration compatible with grid.py
    """
    with open(annotations_path, 'r') as f:
        all_data = json.load(f)
    
    if split not in all_data:
        raise ValueError(f"Split '{split}' not found in annotations")
    
    data = all_data[split]
    
    if image_filename not in data['file_names']:
        raise ValueError(f"Image '{image_filename}' not found in {split} split")
    
    idx = data['file_names'].index(image_filename)
    rois = data['rois_list'][idx]
    occupancy = data['occupancy_list'][idx]
    
    # Assuming standard image dimensions - you may need to adjust
    img_width = 1920
    img_height = 1080
    
    # Convert normalized coordinates back to image coordinates
    spots = []
    for roi in rois:
        spot = []
        for point in roi:
            x = int(point[0] * img_width)
            y = int(point[1] * img_height)
            spot.append([x, y])
        spots.append(spot)
    
    config = {
        'image_path': image_filename,
        'spots': spots,
        'num_spots': len(spots),
        'occupancy': occupancy
    }
    
    return config


def merge_annotations(source_path, target_path, source_split='train', target_split='train'):
    """
    Merge annotations from one file/split into another.
    Useful for combining work from multiple annotators.
    
    Args:
        source_path: Path to source annotations.json
        target_path: Path to target annotations.json
        source_split: Split to merge from
        target_split: Split to merge into
    """
    # Load source
    with open(source_path, 'r') as f:
        source_data = json.load(f)
    
    if source_split not in source_data:
        raise ValueError(f"Split '{source_split}' not found in source")
    
    # Load or create target
    if Path(target_path).exists():
        with open(target_path, 'r') as f:
            target_data = json.load(f)
    else:
        target_data = {'train': {'file_names': [], 'rois_list': [], 'occupancy_list': []},
                      'valid': {'file_names': [], 'rois_list': [], 'occupancy_list': []},
                      'test': {'file_names': [], 'rois_list': [], 'occupancy_list': []}}
    
    # Ensure target split exists
    if target_split not in target_data:
        target_data[target_split] = {'file_names': [], 'rois_list': [], 'occupancy_list': []}
    
    # Merge
    source = source_data[source_split]
    target = target_data[target_split]
    
    added = 0
    updated = 0
    
    for i, fname in enumerate(source['file_names']):
        if fname in target['file_names']:
            # Update existing
            idx = target['file_names'].index(fname)
            target['rois_list'][idx] = source['rois_list'][i]
            target['occupancy_list'][idx] = source['occupancy_list'][i]
            updated += 1
        else:
            # Add new
            target['file_names'].append(fname)
            target['rois_list'].append(source['rois_list'][i])
            target['occupancy_list'].append(source['occupancy_list'][i])
            added += 1
    
    # Save
    with open(target_path, 'w') as f:
        json.dump(target_data, f, indent=2)
    
    print(f"✓ Merged annotations:")
    print(f"  Added: {added} images")
    print(f"  Updated: {updated} images")
    print(f"  Saved to: {target_path}")


def apply_template_to_images(template_file, image_files, output_file='annotations.json', 
                            split='train', default_occupancy=False):
    """
    Apply ROI template to multiple images with default occupancy.
    Useful for bulk initialization before manual occupancy labeling.
    
    Args:
        template_file: Path to roi_template.json
        image_files: List of image file paths
        output_file: Output annotations file
        split: Dataset split
        default_occupancy: Default occupancy value for all spots (default: False)
    """
    # Load template
    with open(template_file, 'r') as f:
        template_data = json.load(f)
    
    template_rois = template_data['rois']
    num_rois = len(template_rois)
    
    # Load or create annotations
    if Path(output_file).exists():
        with open(output_file, 'r') as f:
            all_data = json.load(f)
    else:
        all_data = {'train': {'file_names': [], 'rois_list': [], 'occupancy_list': []},
                   'valid': {'file_names': [], 'rois_list': [], 'occupancy_list': []},
                   'test': {'file_names': [], 'rois_list': [], 'occupancy_list': []}}
    
    if split not in all_data:
        all_data[split] = {'file_names': [], 'rois_list': [], 'occupancy_list': []}
    
    # Apply template to each image
    added = 0
    updated = 0
    
    for img_path in image_files:
        fname = os.path.basename(img_path)
        default_occ = [default_occupancy] * num_rois
        
        if fname in all_data[split]['file_names']:
            idx = all_data[split]['file_names'].index(fname)
            all_data[split]['rois_list'][idx] = template_rois.copy()
            # Don't overwrite existing occupancy
            updated += 1
        else:
            all_data[split]['file_names'].append(fname)
            all_data[split]['rois_list'].append(template_rois.copy())
            all_data[split]['occupancy_list'].append(default_occ)
            added += 1
    
    # Save
    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"✓ Applied template to images:")
    print(f"  Added: {added} images")
    print(f"  Updated: {updated} images")
    print(f"  Each image has {num_rois} ROIs")
    print(f"  Saved to: {output_file}")


def validate_annotations(annotations_path):
    """
    Validate annotation file structure and report statistics.
    
    Args:
        annotations_path: Path to annotations.json
    """
    print("=" * 60)
    print("ANNOTATION VALIDATION")
    print("=" * 60)
    
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    
    for split in ['train', 'valid', 'test']:
        if split not in data:
            print(f"\n⚠ Missing split: {split}")
            continue
        
        split_data = data[split]
        
        print(f"\n{split.upper()} Split:")
        print(f"  Images: {len(split_data['file_names'])}")
        
        # Validate structure
        if len(split_data['file_names']) != len(split_data['rois_list']):
            print(f"  ✗ ERROR: file_names and rois_list lengths don't match!")
        if len(split_data['file_names']) != len(split_data['occupancy_list']):
            print(f"  ✗ ERROR: file_names and occupancy_list lengths don't match!")
        
        # Count ROIs and occupancy
        total_rois = sum(len(rois) for rois in split_data['rois_list'])
        total_occupied = sum(sum(occ) for occ in split_data['occupancy_list'])
        
        print(f"  Total ROIs: {total_rois}")
        print(f"  Occupied: {total_occupied} ({100*total_occupied/total_rois:.1f}%)")
        print(f"  Empty: {total_rois - total_occupied} ({100*(total_rois-total_occupied)/total_rois:.1f}%)")
        
        # Validate coordinates
        for i, rois in enumerate(split_data['rois_list']):
            for j, roi in enumerate(rois):
                if len(roi) != 4:
                    print(f"  ✗ ERROR: Image {i} ROI {j} has {len(roi)} points (expected 4)")
                for point in roi:
                    if len(point) != 2:
                        print(f"  ✗ ERROR: Image {i} ROI {j} has invalid point: {point}")
                    if not (0 <= point[0] <= 1 and 0 <= point[1] <= 1):
                        print(f"  ⚠ WARNING: Image {i} ROI {j} has out-of-bounds point: {point}")
    
    print("\n" + "=" * 60)
    print("✓ Validation complete")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Annotation utilities')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Grid to annotations
    grid_parser = subparsers.add_parser('grid-to-ann', help='Convert grid config to annotations')
    grid_parser.add_argument('grid_config', help='Path to parking_grid_config.json')
    grid_parser.add_argument('image_filename', help='Image filename (e.g., frame_0001.jpg)')
    grid_parser.add_argument('--output', '-o', default='annotations.json', help='Output file')
    grid_parser.add_argument('--split', '-s', default='train', choices=['train', 'valid', 'test'])
    
    # Merge annotations
    merge_parser = subparsers.add_parser('merge', help='Merge annotation files')
    merge_parser.add_argument('source', help='Source annotations.json')
    merge_parser.add_argument('target', help='Target annotations.json')
    merge_parser.add_argument('--source-split', default='train', choices=['train', 'valid', 'test'])
    merge_parser.add_argument('--target-split', default='train', choices=['train', 'valid', 'test'])
    
    # Validate annotations
    validate_parser = subparsers.add_parser('validate', help='Validate annotation file')
    validate_parser.add_argument('annotations', help='Path to annotations.json')
    
    # Apply template
    apply_parser = subparsers.add_parser('apply-template', help='Apply ROI template to images')
    apply_parser.add_argument('template', help='Path to roi_template.json')
    apply_parser.add_argument('image_dir', help='Directory containing images')
    apply_parser.add_argument('--output', '-o', default='annotations.json', help='Output file')
    apply_parser.add_argument('--split', '-s', default='train', choices=['train', 'valid', 'test'])
    
    args = parser.parse_args()
    
    if args.command == 'grid-to-ann':
        add_grid_to_annotations(args.output, args.grid_config, args.image_filename, 
                              occupancy_list=None, split=args.split)
    
    elif args.command == 'merge':
        merge_annotations(args.source, args.target, args.source_split, args.target_split)
    
    elif args.command == 'validate':
        validate_annotations(args.annotations)
    
    elif args.command == 'apply-template':
        import glob
        image_files = glob.glob(os.path.join(args.image_dir, '*.jpg')) + \
                     glob.glob(os.path.join(args.image_dir, '*.JPG')) + \
                     glob.glob(os.path.join(args.image_dir, '*.png'))
        apply_template_to_images(args.template, image_files, args.output, args.split)
    
    else:
        parser.print_help()
