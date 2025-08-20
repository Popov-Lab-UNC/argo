#!/usr/bin/env python3
"""
Script to create a filelist.txt with absolute paths to PDBQT files.
This script processes files efficiently without storing all paths in memory.
"""

import os
import argparse
from pathlib import Path


def process_pdbqt_files(directory_path, output_file, num_splits=4):
    """
    Process PDBQT files in the given directory and write to output files split evenly.
    
    Args:
        directory_path (str): Path to directory containing PDBQT files
        output_file (str): Base name for output text files
        num_splits (int): Number of files to split into (default: 4)
    """
    directory = Path(directory_path).resolve()
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if not directory.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")
    
    print(f"Processing PDBQT files in: {directory}")
    print(f"Output will be split into {num_splits} files")
    
    # First, collect all PDBQT files (we need to do this to split evenly)
    pdbqt_files = list(directory.glob("*.pdbqt"))
    total_files = len(pdbqt_files)
    
    if total_files == 0:
        print("No PDBQT files found in the directory")
        return
    
    print(f"Found {total_files} PDBQT files")
    
    # Calculate files per split
    files_per_split = total_files // num_splits
    remainder = total_files % num_splits
    
    print(f"Files per split: {files_per_split} (with {remainder} extra files distributed)")
    
    # Create output files
    output_files = []
    for i in range(num_splits):
        # Add .txt extension if not present
        if not output_file.endswith('.txt'):
            base_name = output_file
        else:
            base_name = output_file[:-4]
        
        split_filename = f"{base_name}_part{i+1:02d}.txt"
        output_files.append(open(split_filename, 'w'))
        print(f"Created: {split_filename}")
    
    # Distribute files across splits
    file_index = 0
    for split_idx in range(num_splits):
        # Calculate how many files this split should get
        if split_idx < remainder:
            files_this_split = files_per_split + 1
        else:
            files_this_split = files_per_split
        
        print(f"Writing {files_this_split} files to part {split_idx + 1}")
        
        # Write files to this split
        for _ in range(files_this_split):
            if file_index >= total_files:
                break
                
            pdbqt_file = pdbqt_files[file_index]
            if pdbqt_file.is_file():
                # Get absolute path
                abs_path = pdbqt_file.resolve()
                # Get base name without extension
                base_name = pdbqt_file.stem
                
                # Write to current split file
                output_files[split_idx].write(f"{abs_path}\n")
                output_files[split_idx].write(f"{base_name}\n")
            
            file_index += 1
    
    # Close all output files
    for f in output_files:
        f.close()
    
    print(f"Successfully processed {total_files} PDBQT files")
    print(f"Split into {num_splits} files:")
    for i in range(num_splits):
        if not output_file.endswith('.txt'):
            base_name = output_file
        else:
            base_name = output_file[:-4]
        print(f"  - {base_name}_part{i+1:02d}.txt")


def main():
    parser = argparse.ArgumentParser(
        description="Create a filelist.txt with absolute paths to PDBQT files, split into multiple parts"
    )
    parser.add_argument(
        "directory",
        help="Directory containing PDBQT files"
    )
    parser.add_argument(
        "-o", "--output",
        default="filelist",
        help="Base name for output files (default: filelist)"
    )
    parser.add_argument(
        "-n", "--num-splits",
        type=int,
        default=4,
        help="Number of files to split into (default: 4)"
    )
    
    args = parser.parse_args()
    
    try:
        process_pdbqt_files(args.directory, args.output, args.num_splits)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
