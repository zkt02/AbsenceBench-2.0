# AbsenceBench 2.0: Replication & Extension (cLab Trial Project)

**Author:** Zekai Tong (University of Chicago)  
**Full Report:** [Read the technical report (PDF)](./Zekai_Tong-Trail_Project_Report-AbsenceBench.pdf)  
**Context:** Trial Project for Conceptualization Lab (Project 5)  
**Target Model:** Gemini-2.5-Flash (Baseline vs. Thinking Mode)  

This project replicates the core findings of the original AbsenceBench benchmark and extends the evaluation to a new domain, **Recipes**, focusing on the trade-off between "semantic reconstruction" and "verbatim fidelity" in reasoning-enhanced Large Language Models (LLMs).

---

## 1. Key Decision Points & Technical Summary
In alignment with the trial requirements, this project emphasizes several strategic research decisions:

* **Decision 1: Introducing the Recipes Domain.** I introduced Recipes as a low-constraint procedural domain to test if thinking mode improves instruction following while risking verbatim fidelity under strict line-level exact match.
* **Decision 2: Mechanism Attribution.** I treated thinking mode as a controlled intervention rather than comparing disparate models, prioritizing mechanism attribution and error analysis under a fixed evaluation pipeline.
* **Decision 3: Auditing the Poetry Discrepancy.** To investigate the large replication gap in the Poetry domain, I audited the results across three axes: output scaling limits (scale penalty), strict exact-match sensitivity (EM penalty), and sample-split statistics.
* **Decision 4: Composition Re-weighting Analysis.** I implemented a counterfactual re-weighting analysis to isolate dataset composition effects by adjusting zero-omission prevalence, providing a clearer interpretation of cross-domain performance.

---

## 2. Project Structure (Extension & Analysis Tools)
In addition to the original AbsenceBench framework, this repository includes the following modules for in-depth analysis and new domain evaluation:

### New Domain Evaluation
* `tests/test_llms_recipes.py`: Evaluation script specifically for the Recipes domain, processing 500 samples following the original benchmark protocol.

### Result Analysis Suite (`result_analysis/`)
* `analyze_poetry_gap.py` / `poetry_discrepancy.py`: Diagnostic tools for quantitative audits of the Poetry replication gap.
* `poetry_ceiling_zoom.py`: Visualizer for the "output capacity ceiling" (corresponding to Figure 2 in the report), illustrating the physical bottlenecks encountered during long-sequence retrieval.
* `find_hidden_success.py`: Semantic fidelity auditor that performs qualitative error categorization to identify "hidden success" cases—recoveries penalized by strict EM due to minor formatting or punctuation differences.
* `plot_scatter_omission_rate.py`: Replicates Figure 5 of the original study to analyze the sensitivity of F1 scores to omission rates and identify bimodal polarization in the Recipes domain.

---

## 3. Core Experimental Findings

* **Output Capacity Bottleneck:** In long-text retrieval tasks like Poetry, the model faces a persistent "glass ceiling" near the 100-line output range. This bottleneck persists regardless of thinking mode and mechanically limits achievable recall on large omission sets.
* **Semantic Reconstruction vs. Verbatim Fidelity:** Reasoning-enhanced modes tend toward logical reconstruction (e.g., listification or splitting compound instructions). While this improves instructional utility, it frequently results in zero-credit errors under strict line-level exact-match scoring.
* **Instruction Following Gains:** The significant performance surge in the Recipes domain is primarily attributed to superior instruction following in "zero-omission" scenarios—where the thinking mode successfully enforces silence instead of producing conversational explanations.

---

## 4. Running Diagnostics
To generate the key evidence visualizations used in the report, execute the following commands after producing raw result files:

```bash
# Generate the output capacity ceiling analysis (Figure 2)
python result_analysis/poetry_ceiling_zoom.py results/poetry_results.json

# Generate the omission-rate sensitivity scatter plot (Figure 3/5)
python result_analysis/plot_scatter_omission_rate.py \
    --file results/recipes_results.json \
    --raw data/recipe.jsonl \
    --title "Omission Rate Sensitivity (Recipes)" \
    --output results/recipes_scatter.png