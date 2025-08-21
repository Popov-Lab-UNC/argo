#!/usr/bin/env python3
"""
Chunked ADGPU Docking Runner
Processes large filelists in smaller chunks to avoid memory issues.
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path
import shutil

def read_filelist_chunk(filelist_path, chunk_size=1000, start_line=0):
    """
    Read a chunk of lines from a filelist.
    
    Args:
        filelist_path: Path to the filelist
        chunk_size: Number of lines to read
        start_line: Starting line number (0-indexed)
    
    Returns:
        List of lines in the chunk
    """
    lines = []
    with open(filelist_path, 'r') as f:
        # Skip to start_line
        for i in range(start_line):
            f.readline()
        
        # Read chunk_size lines
        for i in range(chunk_size):
            line = f.readline().strip()
            if not line:  # End of file
                break
            lines.append(line)
    
    return lines

def count_filelist_lines(filelist_path):
    """Count total lines in a filelist."""
    with open(filelist_path, 'r') as f:
        return sum(1 for _ in f)

def create_temp_filelist(lines, output_path, maps_fld_path=None):
    """
    Create a temporary filelist with the given lines.
    
    Args:
        lines: List of file paths
        output_path: Where to save the temp filelist
        maps_fld_path: Optional maps.fld path to prepend
    """
    with open(output_path, 'w') as f:
        if maps_fld_path:
            f.write(f"{maps_fld_path}\n")
        for line in lines:
            f.write(f"{line}\n")

def run_adgpu_chunk(filelist_path, output_dir):
    """
    Run ADGPU directly on a chunk of ligands.
    
    Args:
        filelist_path: Path to the filelist for this chunk
        output_dir: Output directory for this chunk
    
    Returns:
        True if successful, False if failed
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Set OpenMP environment variables for better performance
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = '16'
        env['OMP_PROC_BIND'] = 'true'
        env['OMP_PLACES'] = 'cores'

        # Run ADGPU directly
        cmd = [
            "adgpu", 
            "-C", "1",
            "--xmloutput", "0", 
            "--dlgoutput", "1", 
            "--filelist", filelist_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        if result.returncode == 0:
            print(f"ADGPU completed successfully in {output_dir}")
            return True
        else:
            print(f"ADGPU failed in {output_dir}")
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Exception running ADGPU in {output_dir}: {e}")
        return False

def process_filelist_in_chunks(filelist_path, maps_fld_path, output_dir, 
                               chunk_size=500):
    """
    Process a large filelist in chunks, running ADGPU directly on each chunk.
    
    Args:
        filelist_path: Path to the master filelist
        maps_fld_path: Path to the maps.fld file
        chunk_size: Number of ligands per chunk (default 500 for 1000 lines)
        output_dir: Output directory for all chunks
    """
    total_lines = count_filelist_lines(filelist_path)
    print(f"Processing {total_lines} lines in chunks of {chunk_size} ligands ({chunk_size * 2} lines each)")
    
    # Create the main output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Single temp filelist that gets rewritten each time
    temp_filelist = os.path.join(output_dir, "temp_filelist.txt")
    
    chunk_num = 0
    chunk_stats = []  # list of tuples: (chunk_num, num_ligands, elapsed_seconds)
    start_line = 0
    
    while start_line < total_lines:
        # Read chunk (each ligand has 2 lines, so read chunk_size * 2 lines)
        lines = read_filelist_chunk(filelist_path, chunk_size * 2, start_line)
        if not lines:
            break
        
        chunk_num += 1
        
        print(f"\nProcessing chunk {chunk_num}: lines {start_line+1}-{start_line+len(lines)} ({len(lines)//2} ligands)")
        
        # Rewrite the temp filelist with new chunk
        create_temp_filelist(lines, temp_filelist, maps_fld_path)
        
        # Run ADGPU directly in the output directory, with timing
        num_ligands = len(lines) // 2
        t0 = time.perf_counter()
        success = run_adgpu_chunk(temp_filelist, output_dir)
        dt = time.perf_counter() - t0
        avg = (dt / num_ligands) if num_ligands else 0.0
        print(f"Chunk {chunk_num} timing: {dt:.2f}s total for {num_ligands} ligands | {avg:.4f}s/ligand")
        chunk_stats.append((chunk_num, num_ligands, dt))
        
        if success:
            print(f"Chunk {chunk_num} completed successfully")
        else:
            print(f"Chunk {chunk_num} failed, stopping")
            break
        
        start_line += chunk_size * 2
    
    print(f"\nCompleted processing {chunk_num} chunks")
    # Print timing summary
    if chunk_stats:
        total_time = sum(dt for _, _, dt in chunk_stats)
        total_ligs = sum(n for _, n, _ in chunk_stats)
        overall_avg = (total_time / total_ligs) if total_ligs else 0.0
        print("\nChunk timing summary:")
        for cnum, n, dt in chunk_stats:
            per = (dt / n) if n else 0.0
            print(f"  Chunk {cnum:>3}: {n:>6} ligands | {dt:>8.2f}s | {per:>8.4f}s/ligand")
        print(f"\nTotals: {total_ligs} ligands | {total_time:.2f}s | overall avg {overall_avg:.4f}s/ligand")

    print(f"All output files are in: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Process large filelists in chunks for ADGPU")
    parser.add_argument("--filelist", required=True, help="Path to the master filelist")
    parser.add_argument("--maps-fld", required=True, help="Path to the maps.fld file")
    parser.add_argument("--output-dir", required=True,
                       help="Output directory for all chunks")
    parser.add_argument("--chunk-size", type=int, default=500, 
                       help="Number of ligands per chunk (default: 500)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.filelist):
        print(f"Error: Filelist {args.filelist} does not exist")
        sys.exit(1)
    
    if not os.path.exists(args.maps_fld):
        print(f"Error: Maps file {args.maps_fld} does not exist")
        sys.exit(1)
    
    # Process the filelist
    process_filelist_in_chunks(
        args.filelist,
        args.maps_fld,
        args.output_dir,
        args.chunk_size
    )

if __name__ == "__main__":
    main()
