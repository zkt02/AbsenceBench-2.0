import json
import os
import argparse
import difflib
from collections import Counter

def normalize_text(text_data):
    """
    Performs text canonicalization by collapsing multiple spaces, removing 
    newlines, and converting to lowercase to isolate semantic content.
    """
    if isinstance(text_data, list):
        text = " ".join(text_data)
    else:
        text = str(text_data)
    return " ".join(text.replace('\n', ' ').split()).lower()

def find_results_list(data):
    """
    Recursively traverses nested data structures to locate the list containing 
    experimental results (identified by the presence of the 'micro_f1' key).
    """
    if isinstance(data, list):
        # Check if the first element contains the primary evaluation metric
        if len(data) > 0 and isinstance(data[0], dict) and 'micro_f1' in data[0]:
            return data
        for item in data:
            res = find_results_list(item)
            if res: return res
    elif isinstance(data, dict):
        # Prioritize standard keys used in AbsenceBench evaluation schemas
        for key in ['detailed_results', 'results', 'data', 'samples']:
            if key in data and isinstance(data[key], list):
                res = find_results_list(data[key])
                if res: return res
        # Fallback to exhaustive search through all dictionary values
        for v in data.values():
            res = find_results_list(v)
            if res: return res
    return None

def load_data(file_path):
    """
    Implements a robust loader supporting JSONL, standard JSON lists, 
    and nested JSON dictionary formats.
    """
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if not content: return None
        try:
            raw_data = json.loads(content)
            actual_list = find_results_list(raw_data)
            if actual_list:
                print(f"Debug: Successfully extracted list with {len(actual_list)} samples.")
                return actual_list
            else:
                print("Debug: JSON detected but no results list found. Wrapping in list.")
                return [raw_data] if isinstance(raw_data, dict) else []
        except json.JSONDecodeError:
            # Handle JSONL (Line-delimited JSON) format
            f.seek(0)
            lines = [json.loads(l) for l in f if l.strip()]
            print(f"Debug: Detected JSONL with {len(lines)} lines.")
            return lines

def analyze_hidden_success(file_path):
    """
    Conducts a post-hoc audit of evaluation failures (F1=0) to distinguish between 
    genuine retrieval failure and strict formatting/exact-match penalties.
    """
    results = load_data(file_path)
    if not results:
        print(f"Error: Unable to load data from {file_path}")
        return

    categories = Counter()
    total_samples = len(results)
    failed_samples = 0
    # Store representative cases for qualitative error taxonomy analysis
    cat_details = { 
        "Verbatim (Format Only)": [], 
        "Minor Paraphrasing": [], 
        "Structural Difference": [], 
        "True Failure": [] 
    }

    for data in results:
        # Target samples where strict Exact Match failed despite existing ground truth
        if data.get('micro_f1', 1.0) == 0 and data.get('unidentified_lines'):
            failed_samples += 1
            gt_norm = normalize_text(data['unidentified_lines'])
            resp_norm = normalize_text(data['model_response'])
            
            # Calculate semantic similarity using the Gestalt Pattern Matching algorithm
            similarity = difflib.SequenceMatcher(None, gt_norm, resp_norm).ratio()
            
            # Apply heuristic categorization based on similarity thresholds
            if similarity == 1.0:
                cat = "Verbatim (Format Only)"
            elif similarity >= 0.95:
                cat = "Minor Paraphrasing"
            elif similarity >= 0.75:
                cat = "Structural Difference"
            else:
                cat = "True Failure"
            
            categories[cat] += 1
            # Log IDs and thinking budgets to support the "hard evidence" requirement
            if len(cat_details[cat]) < 3:
                cat_details[cat].append({
                    'id': data.get('id'),
                    'sim': similarity,
                    'tokens': data.get('thinking_token', 0)
                })

    # Output diagnostic report following academic reporting standards
    print(f"\n" + "="*60)
    print(f"ABSENCE BENCH 2.0 - ACADEMIC ERROR CATEGORIZATION")
    print(f"="*60)
    print(f"File: {os.path.basename(file_path)}")
    print(f"Total: {total_samples} | Failed Samples (F1=0): {failed_samples}")
    print("-" * 60)
    print(f"{'Category':<30} | {'Count':<10} | {'Ratio':<10}")
    print("-" * 60)
    
    for cat in ["Verbatim (Format Only)", "Minor Paraphrasing", "Structural Difference", "True Failure"]:
        count = categories[cat]
        ratio = (count / failed_samples * 100) if failed_samples > 0 else 0
        print(f"{cat:<30} | {count:<10} | {ratio:>6.2f}%")
    
    print("-" * 60)
    print("REPRESENTATIVE CASE SAMPLES (FOR ANALYSIS):")
    for cat, items in cat_details.items():
        if items:
            ids = ", ".join([str(i['id']) for i in items])
            avg_tokens = sum(i['tokens'] for i in items) / len(items)
            print(f"* {cat}: IDs [{ids}] (Avg Thinking Tokens: {avg_tokens:.0f})")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detailed Error Decomposition for Absence Bench')
    parser.add_argument('--file', required=True, help='Path to result file')
    args = parser.parse_args()
    analyze_hidden_success(args.file)