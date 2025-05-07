import json
import os
import glob
import argparse

def ensure_schema(input_folder, output_folder_base):
    """
    Reads all JSONL files in input_folder, ensures specific keys exist,
    and writes to new files in output_folder_base.
    """
    required_keys = ["task_id", "input", "teacher_raw_output", "output"]

    for subdir in ["train", "val", "test"]: # Assuming you have these subdirectories
        current_input_subdir = os.path.join(input_folder, subdir)
        current_output_subdir = os.path.join(output_folder_base, subdir)
        
        if not os.path.isdir(current_input_subdir):
            print(f"Input subdirectory not found: {current_input_subdir}. Skipping.")
            continue

        os.makedirs(current_output_subdir, exist_ok=True)
        
        print(f"Processing files in: {current_input_subdir}")
        for filepath_in in glob.glob(os.path.join(current_input_subdir, "*.jsonl")):
            filename = os.path.basename(filepath_in)
            filepath_out = os.path.join(current_output_subdir, filename)
            print(f"  Processing {filename} -> {filepath_out}")
            
            corrected_lines = 0
            total_lines = 0
            try:
                with open(filepath_in, "r", encoding="utf-8") as f_in, \
                     open(filepath_out, "w", encoding="utf-8") as f_out:
                    for line in f_in:
                        total_lines += 1
                        try:
                            data = json.loads(line)
                            made_correction = False
                            if "teacher_raw_output" not in data:
                                # If 'teacher_raw_output' is missing, use the 'output' field as a placeholder
                                # Or use None if 'output' is not a suitable raw placeholder
                                data["teacher_raw_output"] = data.get("output", None) 
                                corrected_lines += 1
                                made_correction = True
                            
                            # Ensure all required keys are present, adding None if missing
                            for key in required_keys:
                                if key not in data:
                                    data[key] = None # Add missing key with None
                                    if not made_correction and key == "teacher_raw_output": # Count if this was the key added
                                        corrected_lines +=1
                                    print(f"    Added missing key '{key}' to a record in {filename}")

                            f_out.write(json.dumps(data) + "\n")
                        except json.JSONDecodeError:
                            print(f"    Skipping invalid JSON line in {filename}: {line.strip()}")
                            f_out.write(line) # Write invalid line as is to new file
                if corrected_lines > 0:
                    print(f"  Corrected/added 'teacher_raw_output' or other keys in {corrected_lines}/{total_lines} lines for {filename}.")
            except Exception as e:
                print(f"  Error processing file {filename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensure consistent schema in Phase 2 JSONL data files.")
    parser.add_argument("--input_folder_base", required=True, help="Base folder containing train/val/test subdirs of Phase 2 data (e.g., ./training_data/phase2).")
    parser.add_argument("--output_folder_base", required=True, help="Base folder to save corrected train/val/test subdirs (e.g., ./training_data/phase2_corrected).")
    args = parser.parse_args()

    ensure_schema(args.input_folder_base, args.output_folder_base)
    print("Schema consistency check and correction attempt complete.")
    print(f"Corrected files are in: {args.output_folder_base}. Use this path for your training script's train/val/test folders.")