#!/usr/bin/env python3

import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

# --- Functions for Loading Experiment Results ---

def parse_experiment_folder_name(folder_name: str) -> Optional[Dict[str, Any]]:
    """
    Parses folder name like "combine_[version]_[alpha]_[beta]".
    Version must be 'v3' or 'v4'. Alpha and beta are floats/integers.
    Identifies "combine_v3_0.0_0.0" as a user-defined baseline.
    """
    try:
        name_list = folder_name.split('_')
        result = {
            "version": name_list[1],
            "alpha": float(name_list[2]),
            "beta": float(name_list[3])
        }
        return result
    except:
        return None

def read_summary_header(summary_txt_path: Path) -> Dict[str, Any]:
    header_data = {}
    if not summary_txt_path.exists(): return header_data
    try:
        with open(summary_txt_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("--- Aggregated Metrics ---"): break 
                if line.startswith("---") and line_num > 0: break
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip(); value = value.strip()
                    if key == "SF Reference Ks for SSD/Delta":
                        try: value = json.loads(value)
                        except json.JSONDecodeError: pass
                    elif value.isdigit(): value = int(value)
                    elif value.lower() == 'n/a': value = None
                    try: 
                        float_val = float(value)
                        value = int(float_val) if float_val.is_integer() else float_val
                    except (ValueError, TypeError): pass
                    header_data[key] = value
    except Exception as e: print(f"Error parsing header from {summary_txt_path}: {e}")
    return header_data

def read_single_experiment_results(exp_folder_path: Path) -> Optional[Dict[str, Any]]:
    print(f"Processing folder: {exp_folder_path.name}...")
    folder_metadata = parse_experiment_folder_name(exp_folder_path.name)
    if not folder_metadata:
        print(f"  Skipping folder '{exp_folder_path.name}' (doesn't match expected 'combine_vX_A_B' pattern).")
        return None

    experiment_data = {
        "folder_name": exp_folder_path.name, "path": str(exp_folder_path), **folder_metadata,
        "args": None, "aggregated_metrics": None, "accuracy_counts_per_task": None,
        "summary_header_info": None
    }
    result_json_file = exp_folder_path / "result.json"
    if result_json_file.exists():
        try:
            with open(result_json_file, 'r', encoding='utf-8') as f: data = json.load(f)
            experiment_data["args"] = data.get("args")
            experiment_data["aggregated_metrics"] = data.get("aggregated_metrics")
            experiment_data["accuracy_counts_per_task"] = data.get("accuracy_counts_per_task")
            print(f"  Successfully parsed result.json")
        except Exception as e: print(f"  Error reading result.json in {exp_folder_path.name}: {e}")
    else: print(f"  Warning: result.json not found in {exp_folder_path.name}.")
    summary_txt_file = exp_folder_path / "summary.txt"
    if summary_txt_file.exists():
        experiment_data["summary_header_info"] = read_summary_header(summary_txt_file)
    return experiment_data

def load_all_experiment_results(base_results_path_str: str) -> List[Dict[str, Any]]:
    base_path = Path(base_results_path_str)
    if not base_path.is_dir():
        print(f"Error: Base path '{base_results_path_str}' is not a valid directory."); return []
    all_results_data = []
    print(f"Scanning for experiment folders under '{base_path}'...")
    for item in sorted(base_path.iterdir()): 
        if item.is_dir():
            experiment_data = read_single_experiment_results(item)
            if experiment_data: all_results_data.append(experiment_data)
    print(f"\nFinished scanning. Loaded data for {len(all_results_data)} experiment(s).")
    return all_results_data

# --- Functions for Generating LaTeX Tables ---
def find_experiment_data_by_criteria(all_experiments: List[Dict[str, Any]], 
                                     criteria: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for exp in all_experiments:
        match = True
        for key, value_crit in criteria.items():
            exp_value = exp.get(key)
            if isinstance(exp_value, float) and isinstance(value_crit, float):
                if not np.isclose(exp_value, value_crit): match = False; break
            elif exp_value != value_crit:
                match = False; break
        if match: return exp
    print(f"Warning: No experiment found matching criteria: {criteria}")
    return None

def get_metric_value(exp_data: Optional[Dict[str, Any]], 
                       metric_key: str, 
                       default_value: Any = "N/A") -> Any:
    if exp_data and exp_data.get("aggregated_metrics"):
        value = exp_data["aggregated_metrics"].get(metric_key, default_value)
        if isinstance(value, (np.integer, np.int64)): return int(value)
        if isinstance(value, (np.floating, np.float64)): return float(value)
        if value is None or value == "N/A": return "N/A" 
        return value
    return default_value

def generate_latex_table(
    all_experiments: List[Dict[str, Any]],
    row_definitions: List[Dict[str, Any]], 
    flat_column_configs: List[Dict[str, Any]], 
    table_caption: str,
    table_label: str,
    column_group_headers_str: Optional[str] = None, 
    transpose: bool = False,
    highlight_best: bool = True
) -> str:
    
    metric_display_names = [col["display_name"] for col in flat_column_configs]
    
    if transpose:
        model_display_names = [row_def["display_name"] for row_def in row_definitions]
        transposed_table_data = []
        for col_idx, col_config in enumerate(flat_column_configs):
            metric_row = [col_config["display_name"]] 
            raw_values_for_metric = []
            for row_def in row_definitions:
                exp_data = find_experiment_data_by_criteria(all_experiments, row_def["criteria"])
                value = get_metric_value(exp_data, col_config["metric_key"])
                if isinstance(value, (int, float)): raw_values_for_metric.append(value)
                elif value == "N/A": raw_values_for_metric.append(float('-inf') if col_config.get("higher_is_better", True) else float('inf'))
                else: raw_values_for_metric.append(str(value)) 
                if isinstance(value, float): metric_row.append(f"{value:{col_config.get('format_spec', '.4f')}}")
                else: metric_row.append(str(value))
            transposed_table_data.append(metric_row)
            if highlight_best and raw_values_for_metric:
                valid_numeric_values = [v for v in raw_values_for_metric if isinstance(v, (int, float)) and not (np.isinf(v) or np.isnan(v))]
                if valid_numeric_values:
                    best_val = max(valid_numeric_values) if col_config.get("higher_is_better", True) else min(valid_numeric_values)
                    for model_idx_in_row in range(len(raw_values_for_metric)):
                        current_raw_val = raw_values_for_metric[model_idx_in_row]
                        if isinstance(current_raw_val, (int, float)) and np.isclose(current_raw_val, best_val):
                            transposed_table_data[-1][model_idx_in_row + 1] = f"\\textbf{{{transposed_table_data[-1][model_idx_in_row + 1]}}}"
        df = pd.DataFrame(transposed_table_data, columns=["Metric"] + model_display_names)
        col_format = "l" + "r" * len(model_display_names)
        latex_caption = f"{table_caption} (Transposed)"
        latex_label = f"{table_label}_transposed"
        main_header_row_str = " & ".join(df.columns.to_series().str.replace('_', '\\_').tolist()) + " \\\\"
    else: 
        table_data_for_df = []
        raw_col_values_for_highlighting = [[] for _ in flat_column_configs]
        for row_idx, row_def in enumerate(row_definitions):
            exp_data = find_experiment_data_by_criteria(all_experiments, row_def["criteria"])
            display_row = [row_def["display_name"]]
            for col_idx, col_config in enumerate(flat_column_configs):
                value = get_metric_value(exp_data, col_config["metric_key"])
                if isinstance(value, (int, float)): raw_col_values_for_highlighting[col_idx].append(value)
                elif value == "N/A": raw_col_values_for_highlighting[col_idx].append(float('-inf') if col_config.get("higher_is_better", True) else float('inf'))
                else: raw_col_values_for_highlighting[col_idx].append(str(value))
                if isinstance(value, float): display_row.append(f"{value:{col_config.get('format_spec', '.4f')}}")
                else: display_row.append(str(value))
            table_data_for_df.append(display_row)
        if highlight_best and table_data_for_df:
            for col_idx, col_config in enumerate(flat_column_configs):
                valid_numeric_values = [val for val in raw_col_values_for_highlighting[col_idx] if isinstance(val, (int, float)) and not (np.isinf(val) or np.isnan(val))]
                if not valid_numeric_values: continue
                best_val_in_col = max(valid_numeric_values) if col_config.get("higher_is_better", True) else min(valid_numeric_values)
                for row_idx_in_data in range(len(table_data_for_df)):
                    current_raw_val = raw_col_values_for_highlighting[col_idx][row_idx_in_data]
                    if isinstance(current_raw_val, (int,float)) and np.isclose(current_raw_val, best_val_in_col):
                        table_data_for_df[row_idx_in_data][col_idx + 1] = f"\\textbf{{{table_data_for_df[row_idx_in_data][col_idx + 1]}}}"
        df = pd.DataFrame(table_data_for_df, columns=["Model/Configuration"] + metric_display_names)
        col_format = "l" + "r" * len(metric_display_names)
        latex_caption = table_caption
        latex_label = table_label
        main_header_row_str = " & ".join(df.columns.to_series().str.replace('_', '\\_').tolist()) + " \\\\"
    
    latex_output = ["\\begin{table}[!htbp]", "\\centering", f"\\caption{{{latex_caption}}}", f"\\label{{{latex_label}}}",
                   f"\\resizebox{{\\textwidth}}{{!}}{{%", f"\\begin{{tabular}}{{{col_format}}}", "\\toprule"]
    if not transpose and column_group_headers_str:
        latex_output.append(column_group_headers_str)
    latex_output.append(main_header_row_str)
    latex_output.append("\\midrule")
    for _, row_data in df.iterrows():
        row_str_list = []
        for item_idx, item_val in enumerate(row_data):
            item_str = str(item_val)
            # Escape _ only if not already part of a LaTeX command
            # This is a basic check; more robust LaTeX escaping might be needed for arbitrary text
            if '\\' not in item_str and '{' not in item_str : 
                 item_str = item_str.replace('_', '\\_')
            row_str_list.append(item_str)
        latex_output.append(" & ".join(row_str_list) + " \\\\")
    latex_output.extend(["\\bottomrule", "\\end{tabular}%", "} %end resizebox", "\\end{table}"])
    return "\n".join(latex_output)

# --- Configuration for Metrics and Table Columns ---
COLUMN_GROUPS_STORE = {
    "P1: Rule & Move Pred.": [
        {"metric_key": "predict_move_sf_in_top1_accuracy", "display_name": "SF T1 Acc.", "higher_is_better": True, "format_spec": ".3f"},
        {"metric_key": "predict_move_em_accuracy", "display_name": "EM Acc.", "higher_is_better": True, "format_spec": ".3f"},
        {"metric_key": "list_legal_moves_f1_avg", "display_name": "Legal F1", "higher_is_better": True, "format_spec": ".3f"},
    ],
    "P1: Move Quality": [
        {"metric_key": "average_ssd_cp_vs_sf_top1", "display_name": "SSD SF-T1", "higher_is_better": False, "format_spec": ".1f"},
        {"metric_key": "avg_delta_llm_vs_gt_cp", "display_name": "$\\Delta_{\\text{LLM-GT}}$", "higher_is_better": True, "format_spec": ".1f"}, 
        {"metric_key": "llm_better_than_gt_rate", "display_name": "LLM > GT Rate", "higher_is_better": True, "format_spec": ".3f"},
    ],
    "P2: Explanation Quality": [
        {"metric_key": "bert_score_f1_overall", "display_name": "BERT-F1", "higher_is_better": True, "format_spec": ".3f"},
        {"metric_key": "rouge_l_f1_overall", "display_name": "ROUGE-L", "higher_is_better": True, "format_spec": ".3f"},
        {"metric_key": "avg_norm_edit_distance_overall", "display_name": "EditDist", "higher_is_better": False, "format_spec": ".3f"},
        {"metric_key": "distinct_2_overall", "display_name": "Distinct-2", "higher_is_better": True, "format_spec": ".3f"},
    ]
}

# --- Wrapper Function to Create Custom LaTeX Table ---
def create_custom_latex_table(
    all_experiments_data: List[Dict[str, Any]],
    row_model_specs: Dict[str, Dict[str, Any]], 
    selected_column_group_names: List[str],
    table_caption: str,
    table_label: str,
    transpose_table: bool = False,
    highlight_best: bool = True
) -> str:
    row_definitions = [{"display_name": name, "criteria": crit} for name, crit in row_model_specs.items()]
    flat_column_configs = []
    column_group_config_for_latex_multicolumns = {} 
    for group_name in selected_column_group_names:
        if group_name in COLUMN_GROUPS_STORE:
            metrics_in_group = COLUMN_GROUPS_STORE[group_name]
            flat_column_configs.extend(metrics_in_group)
            if not transpose_table: # Grouped headers only for non-transposed
                 column_group_config_for_latex_multicolumns[group_name] = [m["display_name"] for m in metrics_in_group]
        else: print(f"Warning: Column group '{group_name}' not found.")
    if not flat_column_configs: return f"% No columns selected for table: {table_label}"

    multicolumn_header_str = None
    if not transpose_table and column_group_config_for_latex_multicolumns:
        group_header_parts = [""] 
        cmidrule_parts = []
        current_latex_col_idx = 2 
        for group_name in selected_column_group_names: 
            if group_name in column_group_config_for_latex_multicolumns:
                # sub_col_display_names = column_group_config_for_latex_multicolumns[group_name] # Not needed directly
                num_sub_cols = len(COLUMN_GROUPS_STORE[group_name]) # Get actual count from original definition
                if num_sub_cols > 0:
                    # Corrected line:
                    safe_group_name = group_name.replace('&', '\\&')
                    group_header_parts.append(f"\\multicolumn{{{num_sub_cols}}}{{c}}{{{safe_group_name}}}")
                    cmidrule_parts.append(f"\\cmidrule(lr){{{current_latex_col_idx}-{current_latex_col_idx + num_sub_cols - 1}}}")
                    current_latex_col_idx += num_sub_cols
        multicolumn_header_str = " & ".join(group_header_parts) + " \\\\\n" + " ".join(cmidrule_parts)

    return generate_latex_table(
        all_experiments_data, row_definitions, flat_column_configs,
        table_caption, table_label,
        column_group_headers_str=multicolumn_header_str, # Pass the generated string
        transpose=transpose_table, highlight_best=highlight_best
    )

# --- Placeholder for Figure Generation ---
def generate_custom_figure( # (Same placeholder function as before)
    all_experiments_data: List[Dict[str, Any]],
    model_specs_for_plot: Dict[str, Dict[str, Any]],
    x_axis_key: str, 
    y_axis_metric_keys: List[str], 
    plot_type: str = 'line', 
    figure_title: str = "Figure Title",
    figure_label: str = "fig:custom_figure",
    output_filename: str = "custom_figure.pdf",
    hue_key: Optional[str] = None 
):
    print(f"\n--- Figure Generation Placeholder: {figure_title} ---")
    print(f"  Output intended for: {output_filename}")
    print(f"  This function would process 'all_experiments_data', extract data for plotting,")
    print(f"  and use matplotlib/seaborn to create and save the plot.")
    pass

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load evaluation results and generate LaTeX tables/figures.")
    parser.add_argument("base_results_directory", 
                        help="The base directory containing experiment subfolders.")
    parser.add_argument("-o", "--output_folder", type=str, default=".",
                        help="Directory to save the generated LaTeX table and figure files (default: current directory).")
    
    args = parser.parse_args()

    output_dir = Path(args.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generated files will be saved to: {output_dir.resolve()}")

    all_experiments = load_all_experiment_results(args.base_results_directory)
    
    if all_experiments:
        print("\n" + "="*30 + " Example: LaTeX for Main Results Table " + "="*30)
        main_results_rows = {
            "TinyLlama-1.1B (Baseline)": {"version": "v3", "alpha": 0.0, "beta": 0.0},
            "v3 ($\mathcal{A}_R$ only)": {"version": "v3", "alpha": 1.0, "beta": 0.0},
            "v3 ($\mathcal{A}_C$ focus)": {"version": "v3", "alpha": 0.0, "beta": 1.0},
            "v3 (Combined $\\alpha=1, \\beta=1$)": {"version": "v3", "alpha": 1.0, "beta": 1.0}
        }
        main_results_col_groups = ["P1: Rule & Move Pred.", "P1: Move Quality", "P2: Explanation Quality"]
        latex_table1 = create_custom_latex_table(
            all_experiments_data=all_experiments, row_model_specs=main_results_rows,
            selected_column_group_names=main_results_col_groups,
            table_caption="Main Results: Performance of Key v3 Pipeline Configurations.",
            table_label="tab:main_results", transpose_table=False
        )
        print(latex_table1)
        table1_path = output_dir / "table_main_results.tex"; 
        with open(table1_path, "w", encoding='utf-8') as f: f.write(latex_table1)
        print(f"Main results table saved to {table1_path}")
        
        # generate_custom_figure(all_experiments)
    else:
        print("No data loaded, cannot generate tables/figures.")