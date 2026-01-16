import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

def analyze_file(file_path):
    """
    Analyzes physical output limits and truncation patterns by comparing 
    the number of requested omitted lines against the actual model production.
    """
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Handle various nested JSON structures to ensure cross-benchmark compatibility
    results = []
    if 'google' in data:
        results = data['google'].get('gemini-2.5-flash', {}).get('detailed_results', [])
    elif 'detailed_results' in data:
        results = data['detailed_results']
    
    if not results:
        print("Error: Could not find detailed_results in JSON.")
        return

    gt_counts = []
    resp_counts = []
    
    for item in results:
        # Calculate the number of ground-truth omitted lines (GT)
        gt_list = item.get('unidentified_lines', [])
        gt_count = len(gt_list)
        if gt_count == 0:
            # Fallback to sum of retrieved and missed lines
            gt_count = item.get('tp', 0) + item.get('fn', 0)
        
        # Parse and count non-empty lines in the model response
        resp = item.get('model_response', "")
        if resp is None:
            resp_text = ""
        elif isinstance(resp, list):
            resp_text = resp[0] if len(resp) > 0 and resp[0] is not None else ""
        else:
            resp_text = str(resp)
        
        # Isolate individual action/step lines
        resp_lines = [l for l in resp_text.split('\n') if l.strip()]
        resp_count = len(resp_lines)
        
        gt_counts.append(gt_count)
        resp_counts.append(resp_count)

    gt_arr = np.array(gt_counts)
    resp_arr = np.array(resp_counts)
    
    # Quantitative evidence for the high-omission regime (GT > 50 lines)
    large_mask = gt_arr > 50
    large_gt = gt_arr[large_mask]
    large_resp = resp_arr[large_mask]
    
    print("="*60)
    print(f"AUDIT REPORT: {os.path.basename(file_path)}")
    print("="*60)
    print(f"Total Samples: {len(results)}")
    print(f"Average Lines Requested (GT): {np.mean(gt_arr):.2f}")
    print(f"Average Lines Produced (Model): {np.mean(resp_arr):.2f}")
    
    if len(large_gt) > 0:
        # Diagnostic: Severe truncation occurs when production is < 50% of the target
        trunc_ratio = np.sum(large_resp < large_gt * 0.5) / len(large_gt)
        print(f"Severe Truncation Rate (>50 lines): {trunc_ratio:.2%}")
    
    # Qualitative Audit: Representative samples for case-level analysis
    for target_id in [0, 1]:
        found = [i for i, item in enumerate(results) if item.get('id') == target_id]
        if found:
            idx = found[0]
            print(f"Case ID {target_id}: Requested {gt_counts[idx]} | Produced {resp_counts[idx]}")

    # Generate a diagnostic scatter plot to visualize retrieval parity (y=x line)
    plt.figure(figsize=(10, 6))
    plt.scatter(gt_arr, resp_arr, alpha=0.3, color='blue', label='Individual Samples')
    
    # Plot the Ideal Recall baseline where model production matches omissions
    plt.plot([0, max(gt_arr) if len(gt_arr)>0 else 100], 
             [0, max(gt_arr) if len(gt_arr)>0 else 100], 'r--', label='Ideal (y=x)')
    
    plt.title(f"Output Truncation Evidence - {os.path.basename(file_path)}")
    plt.xlabel('Requested Lines (Ground Truth)')
    plt.ylabel('Lines Actually Produced by Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Synchronize visualization filename with source data
    plot_name = os.path.basename(file_path).replace('.json', '_truncation.png')
    plt.savefig(plot_name)
    plt.close()
    print(f"Evidence plot saved as: {plot_name}")
    print("="*60)

if __name__ == '__main__':
    # CLI interface for automated output auditing
    parser = argparse.ArgumentParser(description="Audit LLM output capacity and truncation rates.")
    parser.add_argument('file', help='Path to the JSON evaluation results.')
    args = parser.parse_args()
    analyze_file(args.file)