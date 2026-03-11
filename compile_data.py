import os
import glob
import json
import csv
import argparse

def compile_results(input_dir, output_file):
    print(f"Searching for tracking JSONs in: {input_dir}")
    json_files = glob.glob(os.path.join(input_dir, "*_tracking.json"))
    
    if not json_files:
        print("No JSON files found! Make sure you point to the correct output directory.")
        return

    # Prepare the header for the CSV
    headers = ["Video_Name", "Chamber_1", "Chamber_2", "Chamber_3", "Chamber_4", "Chamber_5", "Chamber_6"]
    
    rows = []
    
    for j_file in sorted(json_files):
        filename = os.path.basename(j_file)
        # Strip "_tracking.json" to get the original video name back
        video_name = filename.replace("_tracking.json", "")
        
        with open(j_file, 'r') as f:
            data = json.load(f)
            
        summary = data.get("Summary", {})
        
        # Build the row dictionary
        row_data = {"Video_Name": video_name}
        for i in range(1, 7):
            c_name = f"Chamber_{i}"
            # Fetch distance, leave blank if chamber was excluded
            row_data[c_name] = summary.get(c_name, "") 
            
        rows.append(row_data)

    # Write to CSV
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Successfully compiled {len(rows)} videos into {output_file}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile tracking summaries into a CSV for Excel.")
    parser.add_argument("-i", "--input", required=True, help="Directory containing the *_tracking.json files")
    parser.add_argument("-o", "--output", default="compiled_results.csv", help="Name of the output CSV file")
    
    args = parser.parse_args()
    compile_results(args.input, args.output)