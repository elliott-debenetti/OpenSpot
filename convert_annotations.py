#!/usr/bin/env python3
"""
JSON Annotation Converter with Stratified Train/Valid/Test Splitting

This script converts multiple annotation JSON files into a single JSON file
with train, valid, and test splits. Feature stratification ensures that each
split has comparable proportions of:
  - ROI entries for each spot
  - Occupied/empty entries for each spot
  - Proportional representation from each source file

Supports two modes:
1. Direct file input: Specify annotation JSON files directly
2. Folder mode: Specify a base directory with numbered subfolders

Expected folder structure for folder mode:
    /path/to/data/
      collingwood_1/
        frame_0001.jpg, frame_0002.jpg, ...
        annotations_1.json
      collingwood_2/
        frame_0001.jpg, frame_0002.jpg, ...
        annotations_2.json

Usage:
    # Direct file input
    python convert_annotations.py --input file1.json file2.json --output combined.json
    
    # Folder mode (looks for collingwood_1/annotations_1.json, collingwood_2/annotations_2.json, etc.)
    python convert_annotations.py --base-dir /path/to/data --folder-prefix collingwood --output combined.json
    
    # With custom split ratios
    python convert_annotations.py --base-dir /path/to/data --folder-prefix collingwood --output combined.json --train 0.8 --valid 0.1 --test 0.1
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def load_annotation_file(filepath: str) -> dict:
    """Load a single annotation JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    required_keys = {'file_names', 'rois_list', 'occupancy_list'}
    if not required_keys.issubset(data.keys()):
        missing = required_keys - set(data.keys())
        raise ValueError(f"File {filepath} missing required keys: {missing}")
    
    return data


def extract_frame_number(filename: str) -> int | None:
    """
    Extract frame number from filename like 'frame_0001.jpg'.
    
    Args:
        filename: The filename to parse
        
    Returns:
        The frame number as integer, or None if not found
    """
    match = re.search(r'frame_(\d+)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def offset_frame_filename(filename: str, offset: int) -> str:
    """
    Add offset to frame number in filename.
    
    Args:
        filename: Original filename like 'frame_0001.jpg'
        offset: Number to add to the frame number
        
    Returns:
        New filename with offset applied, e.g., 'frame_0144.jpg'
    """
    def replace_frame_num(match):
        old_num = int(match.group(1))
        new_num = old_num + offset
        # Preserve the original zero-padding width
        width = len(match.group(1))
        return f"frame_{new_num:0{width}d}"
    
    return re.sub(r'frame_(\d+)', replace_frame_num, filename, flags=re.IGNORECASE)


def get_max_frame_number(file_names: list[str]) -> int:
    """
    Get the maximum frame number from a list of filenames.
    
    Args:
        file_names: List of filenames
        
    Returns:
        Maximum frame number found, or 0 if none found
    """
    max_num = 0
    for filename in file_names:
        num = extract_frame_number(filename)
        if num is not None and num > max_num:
            max_num = num
    return max_num


def discover_folders(
    base_dir: str,
    folder_prefix: str,
    annotation_pattern: str = "annotations_{num}.json"
) -> list[tuple[str, str, int]]:
    """
    Discover numbered folders and their annotation files.
    
    Looks for folders like {folder_prefix}_1, {folder_prefix}_2, etc.
    and annotation files inside each folder like annotations_1.json, annotations_2.json, etc.
    
    Args:
        base_dir: Base directory containing the folders
        folder_prefix: Prefix for folder names (e.g., 'collingwood')
        annotation_pattern: Pattern for annotation files with {num} placeholder
        
    Returns:
        List of tuples: (folder_path, annotation_path, folder_number)
        sorted by folder number
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        raise ValueError(f"Base directory does not exist: {base_dir}")
    
    # Find all matching folders
    folder_pattern = re.compile(rf'^{re.escape(folder_prefix)}_(\d+)$')
    
    discovered = []
    for item in base_path.iterdir():
        if item.is_dir():
            match = folder_pattern.match(item.name)
            if match:
                folder_num = int(match.group(1))
                
                # Look for annotation file INSIDE the folder
                annotation_name = annotation_pattern.format(num=folder_num)
                annotation_path = item / annotation_name
                
                if annotation_path.exists():
                    discovered.append((str(item), str(annotation_path), folder_num))
                else:
                    print(f"Warning: No annotation file found in {item.name} "
                          f"(expected: {annotation_name})")
    
    # Sort by folder number
    discovered.sort(key=lambda x: x[2])
    
    if not discovered:
        raise ValueError(f"No matching folders found with prefix '{folder_prefix}' in {base_dir}")
    
    return discovered


def merge_annotation_files(filepaths: list[str], prefix_filenames: bool = True) -> dict:
    """
    Merge multiple annotation files into a single dataset.
    
    Args:
        filepaths: List of paths to annotation JSON files
        prefix_filenames: If True, prefix filenames with source file identifier
                         to avoid conflicts
    
    Returns:
        Merged annotation dictionary
    """
    merged = {
        'file_names': [],
        'rois_list': [],
        'occupancy_list': [],
        'source_file': []  # Track which source file each entry came from
    }
    
    for idx, filepath in enumerate(filepaths):
        data = load_annotation_file(filepath)
        source_name = Path(filepath).stem
        
        for i, filename in enumerate(data['file_names']):
            if prefix_filenames:
                # Prefix with source identifier to ensure uniqueness
                new_filename = f"{source_name}_{filename}"
            else:
                new_filename = filename
            
            merged['file_names'].append(new_filename)
            merged['rois_list'].append(data['rois_list'][i])
            merged['occupancy_list'].append(data['occupancy_list'][i])
            merged['source_file'].append(idx)
    
    return merged


def check_for_duplicate_filenames(folder_annotation_pairs: list[tuple[str, str, int]]) -> tuple[bool, set]:
    """
    Check if there are duplicate filenames across all annotation files.
    
    Args:
        folder_annotation_pairs: List of (folder_path, annotation_path, folder_num) tuples
        
    Returns:
        Tuple of (has_duplicates, set of duplicate filenames)
    """
    all_filenames = []
    seen = set()
    duplicates = set()
    
    for folder_path, annotation_path, folder_num in folder_annotation_pairs:
        data = load_annotation_file(annotation_path)
        for filename in data['file_names']:
            if filename in seen:
                duplicates.add(filename)
            else:
                seen.add(filename)
            all_filenames.append(filename)
    
    return len(duplicates) > 0, duplicates


def rename_files_in_folder(
    folder_path: str,
    annotation_path: str,
    offset: int,
    dry_run: bool = False
) -> dict:
    """
    Rename JPG files in a folder and update the annotation JSON.
    
    Args:
        folder_path: Path to the folder containing images
        annotation_path: Path to the annotation JSON file
        offset: Number to add to frame numbers
        dry_run: If True, only report what would be done without making changes
        
    Returns:
        Updated annotation data with new filenames
    """
    folder = Path(folder_path)
    data = load_annotation_file(annotation_path)
    
    # Build rename mapping
    rename_map = {}  # old_filename -> new_filename
    new_file_names = []
    
    for filename in data['file_names']:
        if offset > 0:
            new_filename = offset_frame_filename(filename, offset)
        else:
            new_filename = filename
        rename_map[filename] = new_filename
        new_file_names.append(new_filename)
    
    if offset > 0:
        if dry_run:
            print(f"    [DRY RUN] Would rename {len(rename_map)} files")
        else:
            # Rename actual files
            # First pass: rename to temporary names to avoid conflicts
            temp_names = {}
            for old_name, new_name in rename_map.items():
                if old_name != new_name:
                    old_path = folder / old_name
                    if old_path.exists():
                        temp_name = f"__temp__{new_name}"
                        temp_path = folder / temp_name
                        old_path.rename(temp_path)
                        temp_names[temp_name] = new_name
            
            # Second pass: rename from temp to final names
            for temp_name, new_name in temp_names.items():
                temp_path = folder / temp_name
                new_path = folder / new_name
                temp_path.rename(new_path)
            
            # Update the annotation JSON file
            data['file_names'] = new_file_names
            with open(annotation_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"    Renamed {len(temp_names)} files and updated annotation JSON")
    
    # Return data with updated filenames
    data['file_names'] = new_file_names
    return data


def merge_annotation_files_with_offset(
    folder_annotation_pairs: list[tuple[str, str, int]],
    folder_prefix: str,
    force_offset: bool = False,
    rename_files: bool = True,
    dry_run: bool = False
) -> dict:
    """
    Merge multiple annotation files with frame number offsetting if needed.
    
    Only applies offsetting if duplicate filenames are detected across files,
    or if force_offset is True. When offsetting is applied, actual JPG files
    are renamed and annotation JSONs are updated.
    
    Args:
        folder_annotation_pairs: List of (folder_path, annotation_path, folder_num) tuples
        folder_prefix: Prefix used for folder names (for display)
        force_offset: If True, always apply offset even without duplicates
        rename_files: If True, rename actual JPG files (not just JSON entries)
        dry_run: If True, report what would be done without making changes
    
    Returns:
        Merged annotation dictionary with offset frame numbers (if needed)
    """
    # First check for duplicates
    has_duplicates, duplicate_names = check_for_duplicate_filenames(folder_annotation_pairs)
    
    apply_offset = has_duplicates or force_offset
    
    if has_duplicates:
        print(f"  Found {len(duplicate_names)} duplicate filename(s) across files - applying offset")
        if len(duplicate_names) <= 5:
            print(f"    Duplicates: {duplicate_names}")
    elif force_offset:
        print(f"  No duplicate filenames found - applying offset anyway (--force-offset)")
    else:
        print(f"  No duplicate filenames found - keeping original filenames")
    
    if apply_offset and rename_files:
        if dry_run:
            print(f"  [DRY RUN] Would rename files and update JSONs")
        else:
            print(f"  Renaming files and updating annotation JSONs...")
    
    merged = {
        'file_names': [],
        'rois_list': [],
        'occupancy_list': [],
        'source_file': [],  # Track which source file each entry came from
        'source_folder': []  # Track the original folder name
    }
    
    cumulative_offset = 0
    
    for idx, (folder_path, annotation_path, folder_num) in enumerate(folder_annotation_pairs):
        folder_name = Path(folder_path).name
        
        # Load annotation to get max frame number before any renaming
        data = load_annotation_file(annotation_path)
        max_frame_in_file = get_max_frame_number(data['file_names'])
        
        if apply_offset:
            print(f"  Processing {folder_name}: {len(data['file_names'])} frames, "
                  f"offset={cumulative_offset}, max_frame={max_frame_in_file}")
            
            if rename_files and cumulative_offset > 0:
                # Rename actual files and update JSON
                data = rename_files_in_folder(
                    folder_path, 
                    annotation_path, 
                    cumulative_offset,
                    dry_run=dry_run
                )
            elif cumulative_offset > 0:
                # Just update filenames in memory (no file renaming)
                data['file_names'] = [
                    offset_frame_filename(fn, cumulative_offset) 
                    for fn in data['file_names']
                ]
        else:
            print(f"  Processing {folder_name}: {len(data['file_names'])} frames")
        
        # Add to merged data
        for i, filename in enumerate(data['file_names']):
            merged['file_names'].append(filename)
            merged['rois_list'].append(data['rois_list'][i])
            merged['occupancy_list'].append(data['occupancy_list'][i])
            merged['source_file'].append(idx)
            merged['source_folder'].append(folder_name)
        
        # Update cumulative offset for next folder (only if applying offsets)
        if apply_offset:
            cumulative_offset += max_frame_in_file
    
    return merged


def compute_stratification_features(data: dict) -> np.ndarray:
    """
    Compute feature vectors for stratification.
    
    For each frame, we create a feature vector that captures:
    1. Number of spots with ROIs
    2. Number of occupied spots
    3. Occupancy pattern encoded as a binary vector
    
    This allows stratified splitting to maintain similar distributions
    of spots and occupancy across train/valid/test sets.
    
    Args:
        data: Merged annotation dictionary
        
    Returns:
        Feature matrix of shape (n_samples, n_features)
    """
    n_samples = len(data['file_names'])
    
    # Determine max number of spots across all frames
    max_spots = max(len(occ) for occ in data['occupancy_list'])
    
    features = []
    for i in range(n_samples):
        occupancy = data['occupancy_list'][i]
        n_spots = len(occupancy)
        n_occupied = sum(occupancy)
        
        # Create feature vector
        # [n_spots, n_occupied, occupancy_ratio, source_file, padded_occupancy...]
        occupancy_padded = list(occupancy) + [False] * (max_spots - len(occupancy))
        
        feature = [
            n_spots,
            n_occupied,
            n_occupied / n_spots if n_spots > 0 else 0,
            data['source_file'][i],
        ] + [int(o) for o in occupancy_padded]
        
        features.append(feature)
    
    return np.array(features)


def create_stratification_bins(features: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """
    Create stratification bins based on feature vectors.
    
    Groups similar samples together for stratified splitting.
    
    Args:
        features: Feature matrix
        n_bins: Number of bins to create for continuous features
        
    Returns:
        Array of bin labels for each sample
    """
    n_samples = features.shape[0]
    
    # Use key features for binning: source_file, n_spots, occupancy_ratio
    source_file = features[:, 3].astype(int)
    n_spots = features[:, 0].astype(int)
    occupancy_ratio = features[:, 2]
    
    # Bin occupancy ratio
    occupancy_bins = np.digitize(
        occupancy_ratio, 
        bins=np.linspace(0, 1, n_bins + 1)[1:-1]
    )
    
    # Create composite bin label
    # Combine source_file, n_spots quantile, and occupancy_ratio bin
    n_spots_unique = np.unique(n_spots)
    n_spots_map = {v: i for i, v in enumerate(n_spots_unique)}
    n_spots_binned = np.array([n_spots_map[s] for s in n_spots])
    
    # Composite stratification key
    max_spots_bins = len(n_spots_unique)
    max_occ_bins = n_bins
    
    strat_labels = (
        source_file * (max_spots_bins * max_occ_bins) +
        n_spots_binned * max_occ_bins +
        occupancy_bins
    )
    
    return strat_labels


def stratified_split(
    data: dict,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int | None = None
) -> tuple[list[int], list[int], list[int]]:
    """
    Perform stratified train/valid/test split.
    
    Stratification is done in two levels:
    1. First by source file (ensures proportional representation from each file)
    2. Then by occupancy characteristics within each source file
    
    Args:
        data: Merged annotation dictionary
        train_ratio: Proportion of data for training
        valid_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_indices, valid_indices, test_indices)
    """
    # Validate ratios
    total = train_ratio + valid_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {total}")
    
    # Set random seed
    rng = np.random.default_rng(random_seed)
    
    n_samples = len(data['file_names'])
    
    # Compute stratification features and bins
    features = compute_stratification_features(data)
    strat_labels = create_stratification_bins(features)
    
    # First, group indices by source file
    source_to_indices = defaultdict(list)
    for idx in range(n_samples):
        source_file = data['source_file'][idx]
        source_to_indices[source_file].append(idx)
    
    train_indices = []
    valid_indices = []
    test_indices = []
    
    # Process each source file separately to ensure proportional representation
    for source_file, source_indices in source_to_indices.items():
        # Within this source file, group by stratification label
        label_to_indices = defaultdict(list)
        for idx in source_indices:
            label = strat_labels[idx]
            label_to_indices[label].append(idx)
        
        source_train = []
        source_valid = []
        source_test = []
        
        # Split each stratum within this source file proportionally
        for label, indices in label_to_indices.items():
            indices = np.array(indices)
            rng.shuffle(indices)
            
            n = len(indices)
            n_train = max(1, int(np.round(n * train_ratio)))
            n_valid = max(0, int(np.round(n * valid_ratio)))
            n_test = n - n_train - n_valid
            
            # Handle edge cases for small strata
            if n <= 2:
                # Put all in train for very small strata
                source_train.extend(indices.tolist())
            else:
                # Ensure at least 1 sample goes to train
                source_train.extend(indices[:n_train].tolist())
                source_valid.extend(indices[n_train:n_train + n_valid].tolist())
                source_test.extend(indices[n_train + n_valid:].tolist())
        
        # Add this source file's splits to the overall splits
        train_indices.extend(source_train)
        valid_indices.extend(source_valid)
        test_indices.extend(source_test)
    
    # Shuffle final indices
    rng.shuffle(train_indices)
    rng.shuffle(valid_indices)
    rng.shuffle(test_indices)
    
    return train_indices, valid_indices, test_indices


def create_split_data(data: dict, indices: list[int], include_source: bool = False) -> dict:
    """
    Create a data dictionary for a specific split.
    
    Args:
        data: Full merged annotation dictionary
        indices: Indices to include in this split
        include_source: Whether to include source_file in output (for stats only)
        
    Returns:
        Dictionary with file_names, rois_list, occupancy_list for the split
    """
    result = {
        'file_names': [data['file_names'][i] for i in indices],
        'rois_list': [data['rois_list'][i] for i in indices],
        'occupancy_list': [data['occupancy_list'][i] for i in indices]
    }
    
    if include_source and 'source_file' in data:
        result['source_file'] = [data['source_file'][i] for i in indices]
    
    return result


def compute_split_statistics(data: dict) -> dict:
    """
    Compute statistics for a data split.
    
    Args:
        data: Split data dictionary
        
    Returns:
        Dictionary of statistics
    """
    n_frames = len(data['file_names'])
    
    if n_frames == 0:
        return {
            'n_frames': 0,
            'n_total_spots': 0,
            'n_occupied': 0,
            'occupancy_rate': 0.0,
            'avg_spots_per_frame': 0.0
        }
    
    n_total_spots = sum(len(occ) for occ in data['occupancy_list'])
    n_occupied = sum(sum(occ) for occ in data['occupancy_list'])
    
    return {
        'n_frames': n_frames,
        'n_total_spots': n_total_spots,
        'n_occupied': n_occupied,
        'occupancy_rate': n_occupied / n_total_spots if n_total_spots > 0 else 0.0,
        'avg_spots_per_frame': n_total_spots / n_frames
    }


def compute_source_file_distribution(data: dict) -> dict:
    """
    Compute distribution of frames by source file.
    
    Args:
        data: Split data dictionary (must include 'source_file' key)
        
    Returns:
        Dictionary mapping source_file to count
    """
    distribution = defaultdict(int)
    if 'source_file' in data:
        for source in data['source_file']:
            distribution[source] += 1
    return dict(distribution)


def print_statistics(
    train_data: dict, 
    valid_data: dict, 
    test_data: dict,
    source_file_names: list[str] | None = None
):
    """Print statistics for all splits."""
    train_stats = compute_split_statistics(train_data)
    valid_stats = compute_split_statistics(valid_data)
    test_stats = compute_split_statistics(test_data)
    
    total_frames = train_stats['n_frames'] + valid_stats['n_frames'] + test_stats['n_frames']
    
    print("\n" + "=" * 60)
    print("SPLIT STATISTICS")
    print("=" * 60)
    
    for name, stats in [('Train', train_stats), ('Valid', valid_stats), ('Test', test_stats)]:
        pct = (stats['n_frames'] / total_frames * 100) if total_frames > 0 else 0
        print(f"\n{name}:")
        print(f"  Frames: {stats['n_frames']} ({pct:.1f}%)")
        print(f"  Total spots: {stats['n_total_spots']}")
        print(f"  Occupied spots: {stats['n_occupied']}")
        print(f"  Occupancy rate: {stats['occupancy_rate']:.3f}")
        print(f"  Avg spots/frame: {stats['avg_spots_per_frame']:.1f}")
    
    print("\n" + "=" * 60)
    
    # Print source file distribution if available
    train_dist = compute_source_file_distribution(train_data)
    valid_dist = compute_source_file_distribution(valid_data)
    test_dist = compute_source_file_distribution(test_data)
    
    if train_dist or valid_dist or test_dist:
        print("\nSOURCE FILE DISTRIBUTION")
        print("=" * 60)
        
        all_sources = sorted(set(train_dist.keys()) | set(valid_dist.keys()) | set(test_dist.keys()))
        
        # Calculate totals per source file
        source_totals = {}
        for source in all_sources:
            source_totals[source] = (
                train_dist.get(source, 0) + 
                valid_dist.get(source, 0) + 
                test_dist.get(source, 0)
            )
        
        print(f"\n{'Source':<20} {'Total':<8} {'Train':<15} {'Valid':<15} {'Test':<15}")
        print("-" * 73)
        
        for source in all_sources:
            total = source_totals[source]
            train_n = train_dist.get(source, 0)
            valid_n = valid_dist.get(source, 0)
            test_n = test_dist.get(source, 0)
            
            train_pct = (train_n / total * 100) if total > 0 else 0
            valid_pct = (valid_n / total * 100) if total > 0 else 0
            test_pct = (test_n / total * 100) if total > 0 else 0
            
            source_name = source_file_names[source] if source_file_names else f"File {source}"
            # Truncate long names
            if len(source_name) > 18:
                source_name = source_name[:15] + "..."
            
            print(f"{source_name:<20} {total:<8} {train_n:>4} ({train_pct:>5.1f}%)   {valid_n:>4} ({valid_pct:>5.1f}%)   {test_n:>4} ({test_pct:>5.1f}%)")
        
        print("=" * 60)


def per_spot_statistics(data: dict) -> dict:
    """
    Compute per-spot occupancy statistics.
    
    Args:
        data: Split data dictionary
        
    Returns:
        Dictionary mapping spot_index to (n_frames_with_spot, n_occupied, occupancy_rate)
    """
    spot_stats = defaultdict(lambda: {'count': 0, 'occupied': 0})
    
    for occupancy in data['occupancy_list']:
        for spot_idx, is_occupied in enumerate(occupancy):
            spot_stats[spot_idx]['count'] += 1
            if is_occupied:
                spot_stats[spot_idx]['occupied'] += 1
    
    return {
        k: {
            **v,
            'occupancy_rate': v['occupied'] / v['count'] if v['count'] > 0 else 0
        }
        for k, v in spot_stats.items()
    }


def print_per_spot_comparison(train_data: dict, valid_data: dict, test_data: dict):
    """Print per-spot occupancy comparison across splits."""
    train_spots = per_spot_statistics(train_data)
    valid_spots = per_spot_statistics(valid_data)
    test_spots = per_spot_statistics(test_data)
    
    all_spots = sorted(set(train_spots.keys()) | set(valid_spots.keys()) | set(test_spots.keys()))
    
    print("\n" + "=" * 60)
    print("PER-SPOT OCCUPANCY RATES")
    print("=" * 60)
    print(f"{'Spot':<6} {'Train':<12} {'Valid':<12} {'Test':<12}")
    print("-" * 42)
    
    # Only show first 10 and last 5 spots if there are many
    if len(all_spots) > 20:
        spots_to_show = list(all_spots[:10]) + ['...'] + list(all_spots[-5:])
    else:
        spots_to_show = all_spots
    
    for spot in spots_to_show:
        if spot == '...':
            print("...")
            continue
        
        train_rate = train_spots.get(spot, {}).get('occupancy_rate', 0)
        valid_rate = valid_spots.get(spot, {}).get('occupancy_rate', 0)
        test_rate = test_spots.get(spot, {}).get('occupancy_rate', 0)
        
        print(f"{spot:<6} {train_rate:<12.3f} {valid_rate:<12.3f} {test_rate:<12.3f}")
    
    print("=" * 60)


def convert_annotations(
    input_files: list[str] | None = None,
    output_file: str = None,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int | None = None,
    prefix_filenames: bool = True,
    verbose: bool = True,
    # Folder mode parameters
    base_dir: str | None = None,
    folder_prefix: str | None = None,
    annotation_pattern: str = "annotations_{num}.json",
    force_offset: bool = False,
    rename_files: bool = True,
    dry_run: bool = False
) -> dict:
    """
    Main conversion function.
    
    Supports two modes:
    1. Direct input: Provide input_files list
    2. Folder mode: Provide base_dir and folder_prefix
    
    Args:
        input_files: List of input JSON file paths (mode 1)
        output_file: Output JSON file path
        train_ratio: Proportion for training set
        valid_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducibility
        prefix_filenames: Whether to prefix filenames (only used in mode 1)
        verbose: Whether to print statistics
        base_dir: Base directory containing numbered folders (mode 2)
        folder_prefix: Prefix for folder names, e.g., 'collingwood' (mode 2)
        annotation_pattern: Pattern for annotation files with {num} placeholder
        force_offset: If True, always apply frame number offset in folder mode
        rename_files: If True, rename actual JPG files when offsetting
        dry_run: If True, show what would be done without making changes
        
    Returns:
        The output dictionary
    """
    # Determine which mode to use
    folder_mode = base_dir is not None and folder_prefix is not None
    direct_mode = input_files is not None and len(input_files) > 0
    
    if not folder_mode and not direct_mode:
        raise ValueError("Must provide either input_files or (base_dir and folder_prefix)")
    
    if folder_mode and direct_mode:
        raise ValueError("Cannot use both input_files and folder mode simultaneously")
    
    if dry_run:
        print("=" * 60)
        print("DRY RUN MODE - No files will be modified")
        print("=" * 60)
    
    if folder_mode:
        # Folder mode with offset
        if verbose:
            print(f"Discovering folders with prefix '{folder_prefix}' in {base_dir}...")
        
        folder_annotation_pairs = discover_folders(base_dir, folder_prefix, annotation_pattern)
        
        if verbose:
            print(f"Found {len(folder_annotation_pairs)} folders:")
            for folder_path, annotation_path, folder_num in folder_annotation_pairs:
                print(f"  - {Path(folder_path).name} -> {Path(annotation_path).name}")
            print(f"\nChecking for duplicate filenames...")
        
        # Store source names for statistics
        source_file_names = [Path(fp).name for fp, ap, fn in folder_annotation_pairs]
        
        # Merge with offset (only if duplicates found or force_offset)
        merged_data = merge_annotation_files_with_offset(
            folder_annotation_pairs, 
            folder_prefix,
            force_offset=force_offset,
            rename_files=rename_files,
            dry_run=dry_run
        )
        
    else:
        # Direct file input mode
        if verbose:
            print(f"Loading {len(input_files)} input files...")
        
        # Store source file names for statistics
        source_file_names = [Path(f).stem for f in input_files]
        
        # Merge all input files
        merged_data = merge_annotation_files(input_files, prefix_filenames=prefix_filenames)
    
    if verbose:
        print(f"\nTotal frames: {len(merged_data['file_names'])}")
        for idx, name in enumerate(source_file_names):
            count = sum(1 for s in merged_data['source_file'] if s == idx)
            print(f"  - {name}: {count} frames")
        
        # Show sample filenames after processing
        print(f"\nSample filenames after processing:")
        for i in [0, len(merged_data['file_names'])//2, -1]:
            if len(merged_data['file_names']) > abs(i):
                print(f"  {merged_data['file_names'][i]}")
    
    if dry_run:
        print("\n[DRY RUN] Skipping file output and splits")
        return {}
    
    # Perform stratified split
    train_idx, valid_idx, test_idx = stratified_split(
        merged_data,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed
    )
    
    # Create split datasets (with source info for stats)
    train_data_with_source = create_split_data(merged_data, train_idx, include_source=True)
    valid_data_with_source = create_split_data(merged_data, valid_idx, include_source=True)
    test_data_with_source = create_split_data(merged_data, test_idx, include_source=True)
    
    # Create output without source_file (for clean output)
    train_data = create_split_data(merged_data, train_idx, include_source=False)
    valid_data = create_split_data(merged_data, valid_idx, include_source=False)
    test_data = create_split_data(merged_data, test_idx, include_source=False)
    
    # Create output structure
    output = {
        'train': train_data,
        'valid': valid_data,
        'test': test_data
    }
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    if verbose:
        print(f"\nOutput saved to: {output_file}")
        print_statistics(
            train_data_with_source, 
            valid_data_with_source, 
            test_data_with_source,
            source_file_names=source_file_names
        )
        print_per_spot_comparison(train_data, valid_data, test_data)
    
    return output


def main():
    parser = argparse.ArgumentParser(
        description='Convert multiple annotation JSONs to train/valid/test split JSON',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Direct file input mode
  %(prog)s --input file1.json file2.json --output combined.json
  %(prog)s --input file1.json file2.json --output combined.json --train 0.8 --valid 0.1 --test 0.1
  %(prog)s --input file1.json file2.json --output combined.json --seed 42 --no-prefix

  # Folder mode (with automatic frame number offsetting)
  %(prog)s --base-dir /path/to/data --folder-prefix collingwood --output combined.json
  %(prog)s --base-dir /path/to/data --folder-prefix collingwood --output combined.json --seed 42
  
  # Folder mode expects structure like:
  #   /path/to/data/
  #     collingwood_1/
  #       frame_0001.jpg, frame_0002.jpg, ...
  #       annotations_1.json
  #     collingwood_2/
  #       frame_0001.jpg, frame_0002.jpg, ...
  #       annotations_2.json
  #
  # Output will have frame numbers offset (if duplicates detected) to prevent overlaps:
  #   collingwood_1: frame_0001.jpg - frame_0143.jpg
  #   collingwood_2: frame_0144.jpg - frame_0286.jpg (offset by 143)
        """
    )
    
    # Input mode group
    input_group = parser.add_argument_group('Input Options (choose one mode)')
    input_group.add_argument(
        '--input', '-i',
        nargs='+',
        help='Input JSON files to merge (direct mode)'
    )
    input_group.add_argument(
        '--base-dir', '-d',
        help='Base directory containing numbered folders (folder mode)'
    )
    input_group.add_argument(
        '--folder-prefix', '-p',
        help='Prefix for folder names, e.g., "collingwood" for collingwood_1, collingwood_2, etc.'
    )
    input_group.add_argument(
        '--annotation-pattern',
        default='annotations_{num}.json',
        help='Pattern for annotation files inside each folder (default: annotations_{num}.json)'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output JSON file path'
    )
    
    # Split ratios
    parser.add_argument(
        '--train',
        type=float,
        default=0.7,
        help='Training set ratio (default: 0.7)'
    )
    
    parser.add_argument(
        '--valid',
        type=float,
        default=0.15,
        help='Validation set ratio (default: 0.15)'
    )
    
    parser.add_argument(
        '--test',
        type=float,
        default=0.15,
        help='Test set ratio (default: 0.15)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--no-prefix',
        action='store_true',
        help='Do not prefix filenames with source file name (only for direct mode)'
    )
    
    parser.add_argument(
        '--force-offset',
        action='store_true',
        help='Always apply frame number offset in folder mode, even without duplicates'
    )
    
    parser.add_argument(
        '--no-rename',
        action='store_true',
        help='Do not rename actual JPG files, only update JSON entries'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making any changes'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress statistics output'
    )
    
    args = parser.parse_args()
    
    # Validate input mode
    direct_mode = args.input is not None
    folder_mode = args.base_dir is not None or args.folder_prefix is not None
    
    if direct_mode and folder_mode:
        parser.error("Cannot use both --input and folder mode (--base-dir/--folder-prefix)")
    
    if not direct_mode and not folder_mode:
        parser.error("Must provide either --input files or --base-dir and --folder-prefix")
    
    if folder_mode:
        if args.base_dir is None:
            parser.error("Folder mode requires --base-dir")
        if args.folder_prefix is None:
            parser.error("Folder mode requires --folder-prefix")
        if not Path(args.base_dir).exists():
            parser.error(f"Base directory not found: {args.base_dir}")
    
    # Validate ratios
    total = args.train + args.valid + args.test
    if not np.isclose(total, 1.0):
        parser.error(f"Ratios must sum to 1.0, got {total:.3f}")
    
    # Validate input files exist (direct mode only)
    if direct_mode:
        for filepath in args.input:
            if not Path(filepath).exists():
                parser.error(f"Input file not found: {filepath}")
    
    # Run conversion
    convert_annotations(
        input_files=args.input if direct_mode else None,
        output_file=args.output,
        train_ratio=args.train,
        valid_ratio=args.valid,
        test_ratio=args.test,
        random_seed=args.seed,
        prefix_filenames=not args.no_prefix,
        verbose=not args.quiet,
        base_dir=args.base_dir if folder_mode else None,
        folder_prefix=args.folder_prefix if folder_mode else None,
        annotation_pattern=args.annotation_pattern,
        force_offset=args.force_offset,
        rename_files=not args.no_rename,
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    main()