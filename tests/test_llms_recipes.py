import json
import argparse
import random
import time
import os
import concurrent.futures
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any

# Dependency check for the local LLM provider module
try:
    from llm_providers import LLMProvider
except ImportError:
    print("Error: llm_providers.py not found. Please ensure it is in the current directory.")

def load_recipes(jsonl_file: str, sample_size: int = None) -> List[Dict[str, Any]]:
    """Loads processed recipe data from a JSONL file with optional sampling."""
    recipes = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            recipes.append(json.loads(line))
    
    if sample_size and sample_size < len(recipes):
        return random.sample(recipes, sample_size)
    return recipes

def evaluate_response(response_list: list, recipe_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates model retrieval performance.
    
    Determines recall and precision by checking if omitted ground-truth steps 
    are present in the model's generated response.
    """
    original_lines = recipe_data["original_context"].split('\n')
    omitted_indices = recipe_data["omitted_index"]

    if not response_list or response_list[0] is None:
        response = ""
        thinking_tokens = 0
    else:
        response = response_list[0]
        thinking_tokens = response_list[1]

    # Strip reasoning/thought tags to isolate the final answer
    if "</think>" in response:
        response = response[response.index("</think>")+8:]
    
    results = {
        "tp": 0, "fp": 0, "fn": 0,
        "identified_lines": [],
        "unidentified_lines": [],
        "wrongly_identified_lines": []
    }

    # Perform line-by-line retrieval detection via substring matching
    for idx, line in enumerate(original_lines):
        clean_line = line.strip().lower()
        if not clean_line: continue
        
        # Check if the current original line appears in the response
        found_in_response = clean_line in response.lower()

        if found_in_response:
            if idx in omitted_indices:
                results["tp"] += 1
                results["identified_lines"].append(line)
            else:
                results["fp"] += 1
                results["wrongly_identified_lines"].append(line)
        else:
            if idx in omitted_indices:
                results["fn"] += 1
                results["unidentified_lines"].append(line)

    # Compute Micro F1 Score
    try:
        results["micro_f1"] = 2 * results["tp"] / (2 * results["tp"] + results["fp"] + results["fn"])
    except ZeroDivisionError:
        results["micro_f1"] = 0.0

    results["thinking_token"] = thinking_tokens
    return results

def process_recipe(recipe: Dict[str, Any], recipe_idx: int, model_provider: str, 
                   model_name: str, system_prompt: str, thinking: bool) -> Dict[str, Any]:
    """Handles a single recipe omission test case including prompt construction and inference."""
    
    # Construct the user message following the AbsenceBench protocol for recipes
    user_message = f"""The following is a complete recipe titled "{recipe['recipe_title']}".

ORIGINAL RECIPE:
{recipe['original_context']}

Now, I will provide a modified version of this recipe where some steps have been removed:

MODIFIED VERSION:
{recipe['modified_context']}

Task: Identify exactly which steps are missing from the MODIFIED VERSION compared to the ORIGINAL RECIPE. 
List only the missing steps, one per line, nothing else."""
    
    try:
        provider = LLMProvider.get_provider(model_provider)
        response = provider.get_response(system_prompt, user_message, model_name, thinking)
        evaluation = evaluate_response(response, recipe)
        evaluation["id"] = recipe.get("id", recipe_idx)
        evaluation["recipe_title"] = recipe['recipe_title']
        evaluation["model_response"] = response[0] if response else ""
        return evaluation
    except Exception as e:
        print(f"Error on recipe {recipe_idx}: {str(e)}")
        return None

def test_model(recipes: List[Dict[str, Any]], model_provider: str, model_name: str, 
               batch_size: int = 5, thinking: bool = False) -> Dict[str, Any]:
    """Execution loop for testing models with batch processing and result aggregation."""
    results = []
    total = len(recipes)
    
    system_prompt = """You are an expert chef and editor. 
You will be given an original recipe and a version with missing steps. 
Your task is to precisely identify and list the missing steps. 
Output only the text of the missing steps, nothing else."""

    for batch_start in tqdm(range(0, total, batch_size)):
        batch_end = min(batch_start + batch_size, total)
        current_batch = recipes[batch_start:batch_end]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = [executor.submit(process_recipe, r, batch_start+i, model_provider, 
                                      model_name, system_prompt, thinking) 
                       for i, r in enumerate(current_batch)]
            
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res: results.append(res)
        
        time.sleep(1) # Rate-limiting delay to avoid API throttling
    
    avg_f1 = sum(r["micro_f1"] for r in results) / len(results) if results else 0
    return {
        "model_info": f"{model_provider}:{model_name}",
        "total_tested": len(results),
        "average_micro_f1": avg_f1,
        "detailed_results": results
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluation script for LLM Recipe Step Omission Recall.')
    
    # Input/Output configuration
    parser.add_argument('--input_file', type=str, default='data/recipe.jsonl',
                      help='Path to the processed recipes JSONL file')
    
    parser.add_argument('--sample_size', type=int,
                      help='Number of recipes to sample for testing')
    parser.add_argument('--provider_models', type=str, nargs='+', default=['google:gemini-2.5-flash'],
                      help='Model endpoint pairs (e.g., provider:model_name)')
    parser.add_argument('--output', type=str, default='results_recipes.json',
                      help='Destination path for evaluation results')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument("--thinking", action='store_true', help='Enable reasoning/thinking mode if supported')
    
    args = parser.parse_args()
    recipes = load_recipes(args.input_file, args.sample_size)
    
    final_output = {"test_date": time.strftime("%Y-%m-%d %H:%M:%S"), "results": {}}
    
    for pair in args.provider_models:
        provider, model = pair.split(':')
        print(f"--- Testing {model} ---")
        results = test_model(recipes, provider, model, args.batch_size, args.thinking)
        final_output["results"][f"{provider}:{model}"] = results
        print(f"Avg Micro F1: {results['average_micro_f1']:.2%}")

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()