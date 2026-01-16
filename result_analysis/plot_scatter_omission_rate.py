import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import numpy as np

def estimate_tokens(text):
    """
    Approximates token count using a standard heuristic (4 characters per token) 
    common in large language model context window analysis.
    """
    if not text: return 0
    return len(str(text)) // 4

def load_raw_context_lengths(raw_file_path):
    """
    Parses the source dataset to extract and canonicalize the baseline 
    context lengths for each recipe ID.
    """
    id_to_len = {}
    if not raw_file_path or not os.path.exists(raw_file_path):
        print(f"Error: {raw_file_path} not found.")
        return id_to_len
    
    with open(raw_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                # Map sample IDs to their corresponding original text lengths
                rid = item.get('id')
                content = item.get('original_context')
                if rid is not None and content:
                    # Canonicalize IDs as strings to mitigate type mismatch during lookup
                    id_to_len[str(rid)] = estimate_tokens(str(content))
            except: continue
    print(f">>> [Diagnostic] Loaded {len(id_to_len)} unique IDs from raw data.")
    return id_to_len

def find_results_list(data):
    """
    Implements a recursive search strategy to navigate nested JSON outputs 
    and locate the primary evaluation sample list.
    """
    if isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], dict): return data
        for item in data:
            res = find_results_list(item)
            if res: return res
    elif isinstance(data, dict):
        # Scan for common keys used in AbsenceBench schema versions
        for key in ['detailed_results', 'results', 'data']:
            if key in data and isinstance(data[key], list): return data[key]
        for v in data.values():
            res = find_results_list(v)
            if res: return res
    return None

def extract_scatter_data(file_path, id_to_len):
    """
    Cross-references evaluation results with source metadata to synthesize 
    a dataset for correlation analysis.
    """
    extracted = []
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        try:
            raw_data = json.loads(content)
            results = find_results_list(raw_data)
        except:
            # Fallback for JSONL (newline-delimited) formats
            f.seek(0)
            results = [json.loads(line) for line in f if line.strip()]

    if not results: return []

    matched_count = 0
    for r in results:
        # Align evaluation records with original lengths via string-based ID matching
        sid = str(r.get('id', ""))
        context_len = id_to_len.get(sid)
        
        if context_len is not None:
            matched_count += 1
        else:
            # Degraded estimation fallback if direct ID mapping fails
            context_len = estimate_tokens(r.get('original_text', "")) or 0

        # Calculate the omission rate as a normalized independent variable
        tp, fn = r.get('tp', 0), r.get('fn', 0)
        omitted = tp + fn
        total_lines = omitted + 10 # Baseline assumption for omission density calculation
        rate = omitted / total_lines if total_lines > 0 else 0
        
        extracted.append({
            'Context Length': context_len,
            'F1': r.get('micro_f1', 0),
            'Rate': rate
        })
    
    print(f">>> [Diagnostic] Successfully matched {matched_count}/{len(results)} records with raw lengths.")
    return extracted

def plot_scatter(df, output_file, title):
    """
    Generates a multi-dimensional scatter plot featuring a regression trendline 
    to analyze performance decay relative to context length and omission density.
    """
    if df.empty: return
    plt.figure(figsize=(10, 6))
    sns.set_style("ticks")
    
    # Threshold for color-map normalization (maximum expected omission density)
    MAX_RATE = 0.2857142857142857

    # Layer 1: Global trend analysis via dashed linear regression
    sns.regplot(data=df, x='Context Length', y='F1', scatter=False, 
                line_kws={'color': 'black', 'linestyle': '--', 'alpha': 0.5})

    # Layer 2: Multi-dimensional scatter (Hue mapped to omission rate)
    sc = plt.scatter(df['Context Length'], df['F1'], c=df['Rate'], 
                     cmap='coolwarm', vmin=0, vmax=MAX_RATE, 
                     s=60, alpha=0.8, edgecolors='white', linewidth=0.5)

    # Layer 3: Scalar colorbar for omission rate density
    cbar = plt.colorbar(sc)
    cbar.set_label('Omission Rate', fontsize=12)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Total Context Length (Input Tokens)', fontsize=12)
    plt.ylabel('Micro-F1 Score', fontsize=12)
    plt.ylim(-0.05, 1.05)
    
    # Apply academic publication formatting
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f">>> Success! Plot saved to {output_file}")

def main():
    """Entry point for experimental data visualization."""
    parser = argparse.ArgumentParser(description='Multi-dimensional performance analyzer for AbsenceBench.')
    parser.add_argument('--file', required=True, help='Path to evaluation results')
    parser.add_argument('--raw', required=True, help='Path to raw source context data')
    parser.add_argument('--title', required=True, help='Chart title')
    parser.add_argument('--output', required=True, help='Output image path')
    args = parser.parse_args()

    # Pre-process raw lengths for cross-referencing
    id_to_len = load_raw_context_lengths(args.raw)
    data = extract_scatter_data(args.file, id_to_len)
    plot_scatter(pd.DataFrame(data), args.output, args.title)

if __name__ == "__main__":
    main()