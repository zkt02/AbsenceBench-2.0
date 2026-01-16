import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

def analyze_ceiling(file_path):
    # Validate source file existence
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    print(f"Analyzing: {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Support multiple JSON schema variations for nested detailed results
    results = []
    if 'google' in data:
        results = data['google'].get('gemini-2.5-flash', {}).get('detailed_results', [])
    elif 'detailed_results' in data:
        results = data['detailed_results']
    
    if not results:
        print(f"Error: No valid results found in {file_path}.")
        return

    gt_counts, resp_counts = [], []
    for item in results:
        # Calculate ground-truth omission size (TP + FN)
        gt_counts.append(item.get('tp', 0) + item.get('fn', 0))
        
        # Extract and count non-empty lines from the model's generated response
        resp = item.get('model_response')
        text = ""
        if isinstance(resp, list):
            text = resp[0] if (len(resp) > 0 and resp[0] is not None) else ""
        else:
            text = str(resp) if resp is not None else ""
        resp_counts.append(len([l for l in text.split('\n') if l.strip()]))

    gt_arr, resp_arr = np.array(gt_counts), np.array(resp_counts)
    
    # Compute descriptive statistics for output capacity audit
    p95 = np.percentile(resp_arr, 95)
    median_val = np.median(resp_arr)

    print("="*60)
    print(f"CAPACITY AUDIT REPORT: {os.path.basename(file_path)}")
    print("="*60)
    print(f"Median Output: {median_val:.1f} lines")
    print(f"95th Percentile Ceiling (P95): {p95:.1f} lines")
    print(f"Average Requested Lines (GT Avg): {np.mean(gt_arr):.1f} lines")
    print("="*60)

    # Generate a zoomed scatter plot for capacity distribution analysis
    plt.figure(figsize=(10, 6))
    plt.scatter(gt_arr, resp_arr, alpha=0.4, s=25, color='royalblue', label='Samples')
    plt.plot([0, 850], [0, 850], 'r--', alpha=0.6, label='Ideal Recall (y=x)')
    
    # Establish a baseline for the 95% production ceiling
    plt.axhline(y=p95, color='green', linewidth=2, label=f'95% Output Ceiling ({p95:.1f} lines)')
    
    # Constrain axes to highlight clustering and truncation near the physical limit
    plt.ylim(0, 300) 
    plt.xlim(0, 850)
    
    plt.title(f"Zoomed Capacity Ceiling: {os.path.basename(file_path)}")
    plt.xlabel("Requested Lines (Ground Truth)")
    plt.ylabel("Produced Lines (Actual)")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.5)
    
    # Synchronize output filename with source file identifier
    output_name = os.path.basename(file_path).replace('.json', '_zoom.png')
    plt.savefig(output_name)
    plt.close()
    print(f"Capacity evidence plot generated: {output_name}")

if __name__ == '__main__':
    # CLI argument parsing for experimental auditing
    parser = argparse.ArgumentParser(description="Audit model output capacity and generate zoomed distribution plots.")
    parser.add_argument('file', help='Path to the JSON evaluation results.')
    args = parser.parse_args()
    
    analyze_ceiling(args.file)