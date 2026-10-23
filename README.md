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

### Visualization Gallery: Plain-English & Non-Technical Guides

*Each chart below is built directly from our 10,000 empirical runs and explained with everyday real-world examples so that anyone—even without a mathematical or machine learning background—can immediately understand what is happening.*

---

#### Chart 1: How Consistent is the AI Across Different Tasks? (Repetition vs. Randomness)

![AI Consistency by Category](assets/plots/ai_consistency_by_category.png)

> **💡 The Big Intuitive Idea**: If you ask an AI the exact same question 100 times in a row, does it act like a strict calculator (giving the exact same answer every time), or does it act like an imaginative poet (giving a brand-new answer every time)?

- **What the Numbers Mean (Out of 100 Runs)**:
  - **Left Graph (Randomness / Variety)**: Think of this as the AI's *"Wildness Score"*. A low score means the AI stays grounded and gives the same predictable answer. A high score means the AI's imagination runs wild.
  - **Right Graph (Repetition Rate %)**: Out of 100 times, how often did the AI produce an exact, word-for-word identical copy of an earlier answer?
- **Real-World Non-Technical Example**:
  - **Factual (38.9% Repetition)**: If you ask *"What is the capital of France?"* 100 times, you *want* high repetition ("Paris"). It would be a disaster if it gave 100 different capital cities!
  - **Creative (0.1% Repetition)**: If you ask *"Write a bedtime story about a baby dragon"*, you *want* fresh ideas every time, not the exact same story repeated over and over.
- **Key Takeaway**: The AI knows when to hold back and when to be creative. For factual tasks, it repeats itself nearly 40% of the time; for creative writing, 99.9% of responses are completely fresh.

---

#### Chart 2: When Does the AI Run Out of New Answers? (Response Saturation Curves)

![When Does AI Repeat](assets/plots/when_does_ai_repeat.png)

> **💡 The Big Intuitive Idea**: Imagine asking a friend the same question 100 times. At first, they might phrase their answer in a few different ways, but eventually, they run out of new ways to say it and start repeating themselves. At what point does the AI run out of new things to say?

- **What the Numbers Mean**:
  - **Bottom Axis (X-axis, 1 to 100)**: How many times we repeated the prompt to the AI.
  - **Side Axis (Y-axis, 0 to 100)**: How many unique/different answers we have collected so far.
  - **The Dashed Diagonal Line**: The theoretical *"Infinite Idea Line"*. If an AI generated a brand-new answer on every single attempt without ever repeating, it would follow this 45-degree straight line ($K = N$).
- **Real-World Non-Technical Example**:
  - **Factual Curve Bends Early (~35 to 40 runs)**: Ask *"Who invented the telephone?"* 100 times. By attempt 35, the AI has said *"Alexander Graham Bell"* in every possible sentence phrasing it knows. After that, the curve flattens out (saturates) because it simply repeats its previous answers.
  - **Creative Line Stays Perfectly Straight**: Ask *"Write a sci-fi poem"*. Even on attempt 100, the AI is still inventing brand-new rhymes and plots ($K \approx 100$). It never runs out of creative variations!
- **Key Takeaway**: Fact-based knowledge has a natural ceiling (it saturates around run 35), while open-ended creativity scales endlessly.

---

#### Chart 3: Exact Wording vs. Real Meaning: Are Responses Truly Different?

![Wording vs Meaning Clusters](assets/plots/wording_vs_meaning_clusters.png)

> **💡 The Big Intuitive Idea**: Does changing a few words mean you actually gave a new answer?  
> Consider these two sentences:  
> 1. *"The cat sat on the mat."*  
> 2. *"On the mat, the cat was sitting."*  
> Word-for-word, a computer sees 2 different sentences. But in real life, any human knows both sentences mean the exact same thing! This chart exposes that exact illusion.

- **What the Bars Represent (Calculated per 100 Runs)**:
  - 🟦 **Blue Bar (Unique Wording)**: Computer counted word-by-word differences. If even a single comma or word changed, it was counted as a "new" response.
  - 🟧 **Orange Bar (Real Meaning Groups)**: AI grouped responses by their *actual solution meaning*. If two responses said the same thing in different words, they were grouped into 1 single meaning cluster.
  - **The Gap Between Blue and Orange ("The Rephrasing Multiplier")**: Shows where the AI is simply rephrasing the same old answer using different English words!
- **Real-World Non-Technical Examples**:
  - **Coding (MBPP): Blue 71.6 vs Orange 37.0 (The Biggest Illusion!)**: Out of 100 runs, the code looked different 72 times on the surface. But when looking at the underlying logic, the AI only had **37 real coding solutions**! The other 35 runs were just the same logic with a renamed variable (e.g. `x` changed to `num`).
  - **Maths (GSM8K): Blue 99.1 vs Orange 64.1**: In word problems, the AI talks step-by-step (*"First Natalia had 48...", "Natalia starts with 48..."*). While 99 out of 100 answers looked different in wording, **over 35 runs shared the exact same mathematical solution**!
  - **Creative: Blue 100.0 vs Orange 99.2**: Here, there is zero illusion. Almost every run produced both unique words AND a uniquely different story.
- **Key Takeaway**: Don't judge an AI by its vocabulary alone! In technical tasks (Coding & Maths), 35% to 50% of "unique" answers are just cosmetic rephrasings of the same core idea.

---

#### Chart 4: Is the AI Actually Correct? (Accuracy vs. Verified Ground Truth)

![Accuracy and Correctness](assets/plots/accuracy_and_correctness.png)

> **💡 The Big Intuitive Idea**: An AI can sound very confident and repeat itself consistently—but is it actually telling the truth? This chart compares the AI's repeated answers against verified answer keys to test its real-world accuracy.

- **What the Numbers Mean**:
  - **Y-axis (0% to 100%)**: What percentage of the AI's 100 answers were factually and mathematically correct when checked against official benchmark answer keys (GSM8K, MBPP, Factual QA).
  - **Dashed Line at 100%**: The target for a perfect AI (getting it right every single time).
- **The Non-Technical Breakdown**:
  - **Factual (91.2% Accuracy)**: When asked factual questions, the AI gets it right 9 out of 10 times. Its repetition is reliable and trustworthy.
  - **Maths (46.7% Accuracy)**: In grade-school word math, the 1.7B parameter AI gets it right less than half the time (47 out of 100).
  - **Coding (20.4% Accuracy)**: Generating bug-free code on the very first try without being able to run tests is hard for a small 1.7B model—it succeeds about 1 in 5 times.
- **The Critical "Hallucination" Finding**:
  - In Maths, when the AI makes a mistake, **it almost never repeats the same mistake twice**. Instead, each time it fails, it hallucinates a brand-new, completely random incorrect number!
- **Key Takeaway**: High repetition in factual tasks equals high accuracy, but in math and logic, small models hallucinate diverse, unpredictable errors.

---

#### Chart 5: Does the AI Take Longer When It's Uncertain? (Latency Analysis)

![Latency vs Variability](assets/plots/latency_vs_variability.png)

> **💡 The Big Intuitive Idea**: When you ask a human a simple question (*"What is 2 + 2?"*), they answer in half a second. But when you ask a tricky puzzle or a confusing question, they pause, ponder, talk more, and take much longer. Does an AI do the exact same thing?

- **What the Numbers Mean**:
  - **Bottom Axis (X-axis, Unpredictability Score)**: Left side = straightforward questions; Right side = highly uncertain, open-ended questions.
  - **Side Axis (Y-axis, Time Taken in Milliseconds)**: $1,000\text{ ms} = 1\text{ second}$.
  - **Each Dot**: Represents one of the 100 benchmark prompts tested.
  - **The Red Upward Trendline**: Confirms a strong positive correlation ($r = +0.54$ overall, and $r = +0.90$ in factual tasks).
- **Real-World Non-Technical Example**:
  - **Simple Facts**: Take only **375 ms** (about 1/3 of a second) because the model knows the answer immediately and writes a short phrase.
  - **Multi-Step Math**: Takes **1,496 ms** (about 1.5 seconds)—**4 times longer**! The model has to "think out loud" through multiple sentences to reach an answer.
- **Key Takeaway**: Yes! When an AI is uncertain or faced with a complex reasoning problem, it takes significantly more time and computing power to generate an answer.

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
