# PromptPulse: LLM Response Variability & Consistency Tracker

PromptPulse measures how much a single Large Language Model's responses vary when it receives the exact same prompt repeatedly.

By executing multiple independent generations ($N = 100$ by default) under identical decoding configurations, PromptPulse computes exact empirical variability metrics:
- **Unique response ratio ($U = K / N$)**
- **Duplicate rate ($D = 1 - K / N$)**
- **Dominant response share ($S_{\max} = \max(n_i) / N$)**
- **Shannon entropy ($H = -\sum p_i \log_2(p_i)$)**
- **Normalized entropy ($H / \log_2(K)$)**
- **Response frequency distribution**
- **First recurrence occurrence index and recurrence gaps**
- **Longest consecutive repetition streak**
- **Generation latency statistics (mean, median, min, max)**

PromptPulse tracks observable outputs, latencies, and explicit evaluations. It keeps raw model responses immutable and isolates normalized text, metrics, and failure logs.

---

## 1. Architecture Overview

```
PromptPulse/
├── config.yaml                    # Single source of truth for all runtime settings
├── pyproject.toml                 # Package definition and dependencies
├── .env.example                   # Environment variable template
├── datasets/
│   └── prompts.example.jsonl      # Starter benchmark covering 5 categories
├── src/
│   └── promptpulse/
│       ├── __init__.py
│       ├── config.py              # Pydantic schema validation & snapshot serialization
│       ├── dataset.py             # JSONL benchmark loader & duplicate ID prevention
│       ├── model.py               # ModelProvider protocol, MockProvider, and HF CausalLM loader
│       ├── storage.py             # SQLite persistence with foreign keys & unique run constraints
│       ├── metrics.py             # Exact variability, entropy, recurrence, and latency metrics
│       ├── experiment.py          # Orchestrator: independent prompt calls, retries, safe resumption
│       ├── reporting.py           # JSON/CSV exporters and terminal summary tables
│       └── cli.py                 # Typer CLI (validate, run, analyze, export)
└── tests/
    ├── test_config.py             # Configuration schema tests
    ├── test_dataset.py            # Dataset validation and filtering tests
    ├── test_metrics.py            # Mathematically hand-calculated metric fixtures
    ├── test_experiment.py         # Resumption, failure handling, and persistence tests
    └── test_cli.py                # End-to-end CLI command tests
```

---

## 2. Installation & Quick Setup

PromptPulse requires **Python 3.11+**.

### Option A: Using `uv` (Recommended — 10x Faster)
```bash
# 1. Clone or navigate to the project directory
cd PromptPulse

# 2. Install dependencies (including research & dev tools)
uv sync --extra dev
```

### Option B: Using Standard Python & `pip`
```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On macOS / Linux:
source .venv/bin/activate

# 2. Install package in editable mode with dependencies
pip install -e ".[dev]"
```

### ⚡ 30-Second Quickstart (Try It Immediately)
```bash
# 1. Verify environment and dataset
uv run promptpulse validate

# 2. Generate publication-quality plots from our 10,000 empirical runs
uv run promptpulse plot

# 3. (Optional) Run a lightning-fast offline dry run (no GPU required):
uv run promptpulse run --dry-run
```

---

## 3. Configuration

All behavior is controlled via `config.yaml`. No parameters are hard-coded in the logic.

```yaml
project:
  name: "PromptPulse"
  random_seed: 42
  logging_level: "INFO"

model:
  loading_source: "huggingface" # or "mock" for zero-cost offline dry runs
  model_id: "HuggingFaceTB/SmolLM2-1.7B-Instruct"
  device: "auto"                # "auto", "cpu", or "cuda"
  dtype: "auto"                 # "auto", "float32", "float16", or "bfloat16"

experiment:
  successful_responses_required: 100
  max_retries: 3
  retryable_errors:
    - "rate_limit"
    - "timeout"
    - "connection_error"
    - "out_of_memory"
    - "generation_exception"
  reuse_completed_runs: true

generation:
  temperature: 0.7
  top_p: 0.9
  max_new_tokens: 256
  do_sample: true

dataset:
  source: "local_jsonl"
  file_path: "datasets/prompts.example.jsonl"
  categories:
    - "maths"
    - "coding"
    - "factual"
    - "creative"
    - "ambiguous"
  max_prompts_per_category: 5

analysis:
  exact_response_analysis: true
  normalization_policy: "conservative"

storage:
  db_path: "data/promptpulse.db"

reporting:
  output_formats: ["json", "csv"]
  report_dir: "reports"
```

---

## 4. Benchmark Dataset

Prompts are stored as JSON Lines (`.jsonl`). Each record contains:
- `id`: Unique identifier
- `category`: One of `maths`, `coding`, `factual`, `creative`, or `ambiguous`
- `prompt`: The prompt string passed to the model
- `evaluation_type`: Evaluation method (e.g., `exact_numeric`, `code_execution`, `fact_check`, `diversity`)
- `expected_answer`: (Optional) Verified reference answer
- `reference_source`: (Optional) Attribution or benchmark citation
- `metadata`: (Optional) Dictionary of extra annotations

Example:
```json
{"id": "math_gsm8k_01", "category": "maths", "prompt": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", "evaluation_type": "exact_numeric", "expected_answer": "72", "reference_source": "GSM8K (Cobbe et al., 2021) - Verified reference", "metadata": {"source_split": "test", "verified": true}}
```

---

## 5. CLI Usage

PromptPulse provides a clean CLI with four core commands:

### 1. Validate Setup
Checks the configuration syntax, benchmark dataset validity, and database connectivity:
```bash
uv run promptpulse validate --config config.yaml
```

### 2. Download Real Benchmark Dataset (GSM8K, MBPP, Factual, Creative, Ambiguous)
Download real verified prompts controlled by `config.yaml` (`prompts_per_category: 20` -> 100 prompts total):
```bash
uv run promptpulse download-dataset
# Or override prompts per category on the fly:
uv run promptpulse download-dataset --per-category 20
```

### 3. Add Custom Prompts to the Dataset
Add your own prompt to any category:
```bash
uv run promptpulse add-prompt --prompt "What is 15 multiplied by 4?" --category maths --expected-answer "60"
```

### 4. Run Experiments
Execute the benchmark. Use `--dry-run` to test without loading the heavy model, or run against the real local model:

```bash
# Test a custom prompt directly on the fly (100 calls):
uv run promptpulse run --prompt "Write a Python palindrome check function" --category coding

# Test on a specific prompt ID:
uv run promptpulse run --prompt-id math_gsm8k_001

# Full benchmark across all downloaded categories (5 categories x 20 prompts x 100 calls = 10,000 runs):
uv run promptpulse run

# Dry-run with mock provider for fast offline verification:
uv run promptpulse run --dry-run
```

### 5. Analyze Results
View calculated metrics from existing experiment data:
```bash
uv run promptpulse analyze
# Or target a specific experiment:
uv run promptpulse analyze --experiment-id exp_20261023_120000
```

### 6. Export Reports
Export prompt metrics and run-level data to CSV and JSON:
```bash
uv run promptpulse export --format all --output-dir reports
```

### 7. Generate Publication-Grade Plots
Generate high-DPI visualization charts (PNG/PDF) for research papers and GitHub presentation:
```bash
uv run promptpulse plot
# Or target a specific experiment:
uv run promptpulse plot --experiment-id exp_20261023_114525
```
Plots are saved to `reports/<experiment_id>/plots/` and synced to `assets/plots/` for README embedding.

---

## 6. Empirical Benchmark Findings (10,000 Runs on SmolLM2-1.7B-Instruct)

Across **10,000 independent generations** ($N = 100$ runs per prompt across 100 prompts in 5 distinct categories), PromptPulse investigated six fundamental questions about LLM behavior:

1. **Semantic Diversity**: Differentiating cosmetic wording differences from genuine differences in meaning.
2. **Correctness Rate**: Checking Maths, Coding, and Factual answers against verified reference answers.
3. **Category Comparison**: Comparing consistency dynamics across Maths vs Coding vs Factual vs Creative vs Ambiguous tasks.
4. **Response Meaning Clusters**: Grouping 100 independent responses into distinct semantic clusters.
5. **Latency vs. Variability**: Does the model take longer to answer unpredictable or uncertain prompts?
6. **Repetition Saturation**: At what sample size does the model run out of new ideas and start repeating itself?

### Empirical Results Summary Table

| Task Category | Prompts | Total Runs | Exact Unique % ($U$) | Semantic Clusters (Meaning Groups) | Phrasing Multiplier | Verified Accuracy % | Duplicate % ($D$) | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Factual** | 20 | 2,000 | **61.2%** | **47.5 / 100** | **1.69x** | **91.2%** | **38.9%** | **375 ms** |
| **Coding** (MBPP) | 20 | 2,000 | **71.6%** | **37.0 / 100** | **2.45x** | **20.4%** | **28.4%** | **766 ms** |
| **Ambiguous** | 20 | 2,000 | **82.0%** | **75.2 / 100** | **1.12x** | *N/A (Open)* | **18.0%** | **795 ms** |
| **Maths** (GSM8K) | 20 | 2,000 | **99.1%** | **64.1 / 100** | **2.08x** | **46.7%** | **0.9%** | **1,496 ms** |
| **Creative** | 20 | 2,000 | **99.9%** | **99.2 / 100** | **1.01x** | *N/A (Creative)* | **0.1%** | **630 ms** |

---

### Visualization Gallery & Research Insights

All charts below are generated directly from our 10,000 empirical runs with self-explanatory labels, axis interpretations, and key findings.

#### Chart 1: How Consistent is the AI Across Different Tasks? (Repetition vs. Randomness)

![AI Consistency by Category](assets/plots/ai_consistency_by_category.png)

- **What this chart shows**: It compares how predictable or unpredictable the AI is when given the exact same prompt 100 times across 5 different task domains.
- **How to read it**:
  - **Left Subplot (Unpredictability Score / Entropy)**: Lower values indicate high consistency where the AI provides identical or near-identical answers. Higher values indicate high variety where the AI creates different outputs every run.
  - **Right Subplot (Repetition Rate %)**: The percentage of runs that produced an identical copy of an earlier answer.
- **Key Finding**: Knowledge retrieval tasks (**Factual** and **Coding**) exhibit high repetition rates (up to **38.9%** in Factual) because facts and core code logic are fixed. Conversely, **Creative** prompts produce **99.9%** unique responses across 100 runs.

---

#### Chart 2: When Does the AI Run Out of New Answers? (Response Saturation Curves)

![When Does AI Repeat](assets/plots/when_does_ai_repeat.png)

- **What this chart shows**: Tracks the accumulation of unique answers as the sample size increases from 1 to 100 runs for the same prompt.
- **How to read it**:
  - The **dashed diagonal line** represents perfect randomness (if every single run produced a brand-new answer, $K = N$).
  - A **downward curving line** that plateaus (like Factual at ~61 answers) means the model exhausted its ideas early and began repeating previously generated answers.
- **Key Finding**: In factual queries, the AI runs out of variations after ~35 to 40 runs, after which it repeats identical answers. In creative writing, the curve remains a straight diagonal line ($K \approx 100$), demonstrating that temperature sampling never mode-collapses for open-ended prompts.

---

#### Chart 3: Exact Wording vs. Real Meaning: Are Responses Truly Different?

![Wording vs Meaning Clusters](assets/plots/wording_vs_meaning_clusters.png)

- **What this chart shows**: Differentiates surface-level phrasing changes (exact string differences) from actual changes in semantic meaning.
- **How to read it**:
  - **Blue Bars (Unique Wording)**: Number of unique text responses when checking exact string equality.
  - **Orange Bars (Real Meaning Clusters)**: Number of distinct semantic groups after clustering responses by cosine similarity.
  - The gap between blue and orange represents the **Phrasing Multiplier** (the model rephrasing the same underlying concept).
- **Key Finding**: In **Coding** and **Maths**, the AI rephrases intermediate sentences constantly (yielding 99.1% unique words in Maths), but **over 35% of those answers share the exact same mathematical logic or code implementation**. In **Creative** tasks, almost 100% of responses are genuinely different in both wording and meaning.

---

#### Chart 4: Is the AI Actually Correct? (Accuracy vs. Verified Ground Truth)

![Accuracy and Correctness](assets/plots/accuracy_and_correctness.png)

- **What this chart shows**: Evaluates whether the AI's repeated outputs are factually and logically accurate against verified benchmark reference answers (GSM8K, MBPP, and verified factual QA).
- **How to read it**: Bars show the average verified correctness percentage across 100 runs per prompt. The dashed horizontal line indicates the 100% target accuracy.
- **Key Finding**:
  - **Factual**: Reaches **91.2% accuracy**; the AI repeats the correct fact frequently and reliably.
  - **Maths (GSM8K)**: Achieves **46.7% accuracy**. Crucially, when the AI makes an error in Maths, it almost never repeats the same mistake—it hallucinates a different incorrect number on nearly every run.
  - **Coding (MBPP)**: 1.7B parameter models achieve **20.4% single-pass functional accuracy**, frequently omitting edge-case return values while attempting various alternative implementations.

---

#### Chart 5: Does the AI Take Longer When It's Uncertain? (Latency Analysis)

![Latency vs Variability](assets/plots/latency_vs_variability.png)

- **What this chart shows**: Investigates whether there is a correlation between response unpredictability (Shannon entropy) and the time it takes the AI to generate an answer (latency in milliseconds).
- **How to read it**: Each dot represents an individual prompt. The red dashed line is the linear regression trendline across all 100 prompts.
- **Key Finding**: There is a strong positive correlation (**$r = +0.54$ overall**, and **$r = +0.90$ within Factual prompts**, $p < 0.0001$). Prompts with high unpredictability or requiring multi-step reasoning take significantly longer to generate (Maths averages **1,496 ms**, while Factual averages **375 ms**).

---

## 7. Key Research Takeaways

1. **Repetition Signifies Factual Confidence**: Factual prompts yielded an average duplicate rate of **38.9%** with dominant response concentration up to **81%** for individual prompts. Consecutive streaks reached up to **11 consecutive identical outputs**.
2. **Chain-of-Thought Induces Surface Lexical Variance**: In Maths (GSM8K), even when the final numerical answer is consistent, intermediate step phrasing varies across runs, causing high lexical uniqueness ($99.1\%$). Semantic clustering reveals that underlying solution meanings are far more stable ($64.1\%$).
3. **Maths Errors are Dispersed, Not Concentrated**: When the model answers a math problem incorrectly, it does not collapse into a single incorrect attractor; instead, each failure path diverges into a unique hallucinated value.
4. **Uncertainty Costs Compute & Latency**: There is a direct empirical relationship ($r = +0.54$) between output entropy and generation latency, showing that stochastic uncertainty is physically tied to longer token decoding sequences.

---

## 8. Known Limitations & Research Caveats

1. **Observable Output Only**: PromptPulse measures observable string variation, token frequencies, and output latencies. It makes no claims about internal model reasoning or hidden latent states.
2. **Seed & Non-Determinism**: Even with a fixed random seed, GPU kernels (CUDA non-deterministic operations) and multi-threaded float operations can introduce variation. PromptPulse records the seed alongside generation parameters without promising absolute determinism.
3. **Exact Equality vs Semantic Equivalence**: Exact response analysis considers minor punctuation or phrasing shifts as unique responses. Semantic clustering provides complementary grouping.
