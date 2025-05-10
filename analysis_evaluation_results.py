#!/usr/bin/env python3

import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

def parse_experiment_folder_name(folder_name: str) -> Optional[Dict[str, Any]]:
    """
    Parses folder name like "combine_[version]_[alpha]_[beta]"
    Returns a dictionary with version, alpha, beta or None if parsing fails.
    Example: "combine_v3_1.0_0.5" -> {"version": "v3", "alpha": 1.0, "beta": 0.5}
    """
    match = re.match(r"combine_(v[34])_([0-9\.]+)_([0-9\.]+)", folder_name)
    if match:
        version_str = match.group(1)
        try:
            alpha_float = float(match.group(2))
            beta_float = float(match.group(3))
            return {"version": version_str, "alpha": alpha_float, "beta": beta_float}
        except ValueError:
            print(f"Warning: Could not parse alpha/beta as float in folder name: {folder_name}")
            return None
    return None

def read_summary_header(summary_txt_path: Path) -> Dict[str, Any]:
    """
    Parses the header section of a summary.txt file.
    """
    header_data = {}
    if not summary_txt_path.exists():
        return header_data

    try:
        with open(summary_txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("--- Aggregated Metrics ---"): # Header section ends here
                    break
                
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Specific parsing for known header fields
                    if key == "Test Samples (P1 source)" or key == "Test Samples (P2 source)":
                        try: value = int(value)
                        except ValueError: pass
                    elif key == "SF Reference Ks for SSD/Delta":
                        try: value = json.loads(value) # It's a list string
                        except json.JSONDecodeError: 
                            print(f"Warning: Could not parse SF Reference Ks in summary: {value}")
                    elif value.lower() == 'n/a':
                        value = None
                        
                    header_data[key] = value
    except Exception as e:
        print(f"Error reading or parsing header from {summary_txt_path}: {e}")
    return header_data

def read_single_experiment_results(exp_folder_path: Path) -> Optional[Dict[str, Any]]:
    """
    Reads evaluation results from a single experiment folder.
    Prioritizes result.json for metrics and args.
    Complements with header info from summary.txt.
    """
    print(f"Processing folder: {exp_folder_path.name}...")
    folder_metadata = parse_experiment_folder_name(exp_folder_path.name)
    if not folder_metadata:
        print(f"Skipping folder with unparsable name: {exp_folder_path.name}")
        return None

    experiment_data = {
        "folder_name": exp_folder_path.name,
        "path": str(exp_folder_path),
        **folder_metadata, # Adds "version", "alpha", "beta"
        "args": None,
        "aggregated_metrics": None,
        "accuracy_counts_per_task": None,
        "summary_header_info": None
    }

    # Read result.json (primary source for metrics and args)
    result_json_file = exp_folder_path / "result.json"
    if result_json_file.exists():
        try:
            with open(result_json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            experiment_data["args"] = data.get("args")
            experiment_data["aggregated_metrics"] = data.get("aggregated_metrics")
            experiment_data["accuracy_counts_per_task"] = data.get("accuracy_counts_per_task")
            print(f"  Successfully parsed result.json")
        except json.JSONDecodeError:
            print(f"  Error: Could not decode result.json in {exp_folder_path.name}. Check for JSON errors or numpy types if not converted.")
        except Exception as e:
            print(f"  Error reading result.json in {exp_folder_path.name}: {e}")
    else:
        print(f"  Warning: result.json not found in {exp_folder_path.name}.")

    # Read header from summary.txt
    summary_txt_file = exp_folder_path / "summary.txt"
    if summary_txt_file.exists():
        experiment_data["summary_header_info"] = read_summary_header(summary_txt_file)
        if experiment_data["summary_header_info"]:
             print(f"  Successfully parsed header from summary.txt")
    else:
        print(f"  Warning: summary.txt not found in {exp_folder_path.name}.")
        
    return experiment_data

def load_all_experiment_results(base_results_path_str: str) -> List[Dict[str, Any]]:
    """
    Walks the base_results_path, finds all experiment folders matching the pattern,
    and reads their results.
    """
    base_path = Path(base_results_path_str)
    if not base_path.is_dir():
        print(f"Error: Base path '{base_results_path_str}' is not a valid directory.")
        return []

    all_results_data = []
    print(f"Scanning for experiment folders under '{base_path}'...")

    for item in base_path.iterdir():
        if item.is_dir() and item.name.startswith("combine_"):
            experiment_data = read_single_experiment_results(item)
            if experiment_data:
                all_results_data.append(experiment_data)
    
    print(f"\nFinished scanning. Loaded data for {len(all_results_data)} experiment(s).")
    return all_results_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and organize evaluation results from multiple experiment runs.")
    parser.add_argument("base_results_directory", 
                        help="The base directory containing subfolders for each experiment run (e.g., 'combine_v3_1.0_1.0').")
    
    args = parser.parse_args()

    # Load all results
    all_experiments = load_all_experiment_results(args.base_results_directory)

    if all_experiments:
        print("\n--- Summary of Loaded Experiments ---")
        for i, exp_data in enumerate(all_experiments):
            print(f"\nRun {i+1}:")
            print(f"  Folder: {exp_data['folder_name']}")
            print(f"  Version: {exp_data['version']}, Alpha: {exp_data['alpha']}, Beta: {exp_data['beta']}")
            if exp_data.get("summary_header_info"):
                print(f"  Summary Model: {exp_data['summary_header_info'].get('Eval Summary - Model', 'N/A')}")
                print(f"  Summary SF Ks: {exp_data['summary_header_info'].get('SF Reference Ks for SSD/Delta', 'N/A')}")
            
            metrics_loaded = "Yes" if exp_data.get("aggregated_metrics") else "No"
            counts_loaded = "Yes" if exp_data.get("accuracy_counts_per_task") else "No"
            args_loaded = "Yes" if exp_data.get("args") else "No"

            print(f"  Args Loaded: {args_loaded}")
            print(f"  Aggregated Metrics Loaded: {metrics_loaded}")
            print(f"  Accuracy Counts Loaded: {counts_loaded}")

            if exp_data.get("aggregated_metrics"):
                # Print a sample metric if available
                sample_metric_key = "average_ssd_cp_vs_sf_top1" # Example key
                if sample_metric_key in exp_data["aggregated_metrics"]:
                    print(f"    Sample Metric ({sample_metric_key}): {exp_data['aggregated_metrics'][sample_metric_key]}")
                elif "average_ssd_cp" in exp_data["aggregated_metrics"]: # Fallback to older key name
                     print(f"    Sample Metric (average_ssd_cp): {exp_data['aggregated_metrics']['average_ssd_cp']}")


        # Now 'all_experiments' list contains dictionaries for each run.
        # This is the data structure we'll use for generating tables/plots next.
        print("\n\nAll results loaded into the 'all_experiments' list.")
        # For a quick peek at the first experiment's aggregated metrics (if loaded):
        # if all_experiments and all_experiments[0].get("aggregated_metrics"):
        #     print("\nAggregated metrics for the first experiment:")
        #     for k, v in all_experiments[0]["aggregated_metrics"].items():
        #         print(f"      {k}: {v}")
    else:
        print("No experiment data was loaded.")