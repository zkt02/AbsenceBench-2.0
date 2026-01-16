import json
import os
import argparse
import difflib
from collections import Counter, defaultdict

def normalize_text(text_data):
    """
    Performs text canonicalization by collapsing whitespace, removing newlines, 
    and applying case folding to ensure semantic-only comparisons.
    """
    if isinstance(text_data, list):
        if len(text_data) >= 1 and isinstance(text_data[0], str):
            text = text_data[0]
        else:
            text = " ".join([str(i) for i in text_data])
    else:
        text = str(text_data)
    # Remove structural formatting to isolate textual content
    return " ".join(text.replace('\n', ' ').split()).lower()

def extract_detailed_results(data):
    """
    Recursively traverses nested JSON structures to locate and extract 
    the list of experimental result objects.
    """
    if isinstance(data, list): return data
    if isinstance(data, dict):
        if 'detailed_results' in data: return data['detailed_results']
        # Handle specific schema variations for Gemini/Google outputs
        for key in ['google', 'gemini-2.5-flash']:
            if key in data: return extract_detailed_results(data[key])
        for v in data.values():
            res = extract_detailed_results(v)
            if res: return res
    return []

def analyze_poetry_summary(file_path):
    """
    Analyzes model performance in the Poetry domain by calculating semantic similarity
    to diagnose failures caused by strict line-level Exact Match (EM) constraints.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        results = extract_detailed_results(json.load(f))

    if not results: return

    stats = defaultdict(list)
    categories = Counter()
    total_samples = len(results)

    for item in results:
        # Standardize both ground truth and model output for fuzzy matching
        gt_norm = normalize_text(item.get('unidentified_lines', []))
        resp_norm = normalize_text(item.get('model_response', ""))
        
        # Calculate semantic recall using Gestalt Pattern Matching (SequenceMatcher)
        similarity = difflib.SequenceMatcher(None, gt_norm, resp_norm).ratio()
        f1 = item.get('micro_f1', 0)

        # Heuristic classification of failure modes based on similarity thresholds
        if similarity > 0.9: 
            cat = "Format Penalty (Hidden Success)"
        elif similarity > 0.4: 
            cat = "Partial Retrieval"
        elif similarity > 0.05: 
            cat = "Severe Information Loss"
        else: 
            cat = "Complete Retrieval Failure"

        categories[cat] += 1
        stats[cat].append({'f1': f1, 'sim': similarity})

    # Generate academic performance summary
    print(f"\n" + "="*70)
    print(f"POETRY DOMAIN ANALYSIS SUMMARY (Total Samples: {total_samples})")
    print("="*70)
    print(f"{'Category':<32} | {'Count':<8} | {'Ratio':<10} | {'Avg Recall':<10}")
    print("-" * 70)

    # Report metrics stratified by failure category
    for cat in ["Format Penalty (Hidden Success)", "Partial Retrieval", "Severe Information Loss", "Complete Retrieval Failure"]:
        count = categories[cat]
        ratio = (count / total_samples) * 100
        avg_sim = sum(i['sim'] for i in stats[cat]) / count if count > 0 else 0
        print(f"{cat:<32} | {count:<8} | {ratio:>8.2f}% | {avg_sim:>10.2%}")

    print("-" * 70)
    # Compare strict Micro-F1 against semantic-aware recall to highlight EM sensitivity
    overall_f1 = sum(item.get('micro_f1', 0) for item in results) / total_samples
    overall_recall = sum(difflib.SequenceMatcher(None, normalize_text(i.get('unidentified_lines', [])), normalize_text(i.get('model_response', ""))).ratio() for i in results) / total_samples
    
    print(f"OVERALL PERFORMANCE:  Avg F1: {overall_f1:.4f}  |  Avg Semantic Recall: {overall_recall:.2%}")
    print("="*70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnostics for semantic recall vs. verbatim fidelity.")
    parser.add_argument('--file', required=True, help="Path to the JSON results file.")
    args = parser.parse_args()
    analyze_poetry_summary(args.file)