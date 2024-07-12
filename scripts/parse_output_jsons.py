import json
import glob
import csv
from pathlib import Path

def create_unified_csv(base_dir):
    delta_weights_arr = ["vqa_rad", "ckpt2"]
    wt_precision_arr = ["fp32", "int8", "int4"]

    unified_data = []

    for delta_weights in delta_weights_arr:
        for wt_precision in wt_precision_arr:
            eval_summary_file = str(Path(base_dir) / f"eval_summary_eval_results_{delta_weights}_{wt_precision}.jsonl")
            perf_averages_file = str(Path(base_dir) / f"perf_averages_{delta_weights}_{wt_precision}.json")

            if not Path(eval_summary_file).exists():
                print(f"Evaluation summary file not found: {eval_summary_file}")
                continue

            if not Path(perf_averages_file).exists():
                print(f"Performance averages file not found: {perf_averages_file}")
                continue

            try:
                with open(eval_summary_file, 'r') as esf:
                    eval_summary = json.load(esf)
                with open(perf_averages_file, 'r') as pf:
                    perf_averages = json.load(pf)

                combined_entry = {}

                # Add all items from perf_averages
                for key, value in perf_averages.items():
                    combined_entry[key] = value

                # Add items from eval_summary, flattening nested dictionaries
                for key, value in eval_summary.items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            combined_entry[f"{key}_{subkey}"] = subvalue
                    else:
                        combined_entry[key] = value

                unified_data.append(combined_entry)
            
            except Exception as e:
                print(f"Error processing files for {delta_weights} and {wt_precision}: {e}")

    output_file = Path(base_dir) / "unified_summary.csv"
    with open(output_file, 'w', newline='') as csvfile:
        if unified_data:
            fieldnames = unified_data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for data in unified_data:
                writer.writerow(data)

    print(f"Unified summary CSV written to {output_file}")

# Usage
base_dir = "/home/sdp/ravi/ov-llavamed/scripts/"
create_unified_csv(base_dir)