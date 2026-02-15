#!/usr/bin/env python3
"""
Generate evaluation_metrics.log from raw evaluation run directories.

This script processes episode directories from evaluation runs and computes
standardized metrics using the metrics_calculator module, outputting in the
same format as the current evaluation logging infrastructure.
"""

import os
import sys
import re
from pathlib import Path
from utils.metrics_calculator import (
    compute_episode_metrics_from_logs,
    ensure_metrics_log,
    append_metrics_log,
)


def parse_dirname(dirname: str) -> dict:
    """
    Parse model directory name to extract metadata.
    
    Expected format: model_ep{episode}_ts{timestep}_seed{seed}_att{attention}
    Example: model_ep14_ts256_seed42_att0
    
    Returns dict with keys: episode, timestep, seed, attention
    """
    pattern = r"model_ep(\d+)_ts(\d+)_seed(\d+)_att(\d+)"
    match = re.match(pattern, dirname)
    if not match:
        raise ValueError(f"Cannot parse directory name: {dirname}")
    
    return {
        "episode": int(match.group(1)),
        "timestep": int(match.group(2)),
        "seed": int(match.group(3)),
        "attention": int(match.group(4)),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python calc_eval_metrics_from_runs.py <evaluation_runs_dir> [output_file]")
        print("Example: python calc_eval_metrics_from_runs.py eval_results/noop-1_deterministic/evaluation_runs")
        sys.exit(1)
    
    eval_runs_dir = Path(sys.argv[1])
    
    # Default output file in the parent directory of evaluation_runs
    if len(sys.argv) >= 3:
        output_file = Path(sys.argv[2])
    else:
        output_file = eval_runs_dir.parent / "evaluation_metrics.log"
    
    if not eval_runs_dir.exists():
        print(f"Error: Directory not found: {eval_runs_dir}")
        sys.exit(1)
    
    print(f"Processing evaluation runs from: {eval_runs_dir}")
    print(f"Output file: {output_file}")
    
    # Ensure the output file has the proper header
    ensure_metrics_log(str(output_file))
    
    # Get all model directories
    model_dirs = sorted([d for d in eval_runs_dir.iterdir() if d.is_dir()])
    
    if not model_dirs:
        print("Error: No model directories found!")
        sys.exit(1)
    
    print(f"Found {len(model_dirs)} model directories to process")
    
    processed = 0
    errors = 0
    
    for model_dir in model_dirs:
        try:
            # Parse metadata from directory name
            metadata = parse_dirname(model_dir.name)
            
            # Find episode subdirectory (usually episode_0000)
            episode_dirs = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")]
            
            if not episode_dirs:
                print(f"Warning: No episode directory in {model_dir.name}, skipping")
                errors += 1
                continue
            
            # Use first episode directory (should only be one)
            episode_dir = episode_dirs[0]
            
            # Compute metrics from the logs
            # We use "deterministic" as policy name to match the evaluation type
            metrics = compute_episode_metrics_from_logs(
                episode_dir=str(episode_dir),
                episode_info={},
                policy="deterministic",
                seed=metadata["seed"],
                num_robots=5,  # Number of taxis in the simulation
            )
            
            # Set the timestep field from the directory name
            metrics.ts = metadata["timestep"]
            
            # Append to the output log
            append_metrics_log(str(output_file), metrics)
            
            processed += 1
            
            if processed % 50 == 0:
                print(f"Processed {processed}/{len(model_dirs)} runs...")
        
        except Exception as e:
            print(f"Error processing {model_dir.name}: {e}")
            errors += 1
            continue
    
    print(f"\nComplete!")
    print(f"Successfully processed: {processed}")
    print(f"Errors: {errors}")
    print(f"Output written to: {output_file}")


if __name__ == "__main__":
    main()
