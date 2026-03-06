# Eleusis LLM Benchmark

A benchmark for evaluating Large Language Models on **inductive reasoning and pattern discovery**, using an adaptation of Robert Abbott's card game [Eleusis](https://en.wikipedia.org/wiki/Eleusis_(card_game)) (1956).

A secret rule determines which cards are accepted. The model must discover the rule by playing cards and observing the outcomes, mimicking the process of scientific hypothesis testing.

Full results on the dedicated [Eleusis space on Hugging Face](https://huggingface.co/spaces/huggingface/eleusis-benchmark)

<p align="center">
  <img src="results.png" alt="Eleusis benchmark results" style="width: 70%; max-width: 900px;">
</p>


## Quick Start

```bash
# Install dependencies (requires uv: https://docs.astral.sh/uv/)
uv sync

# Set up API keys (only the providers you need)
cp .env.example .env
# Edit .env with your keys

# Run evaluation with a model
uv run python scripts/evaluate_single.py --model "kimi-k2"

# Run parallel evaluation across models
./scripts/run_parallel_eval.sh eval_models.txt
```

## How It Works

Each evaluation round plays out as follows:

1. A **secret rule** is loaded (a Python function that accepts or rejects cards based on their properties and the sequence of previously played cards)
2. The model receives a hand of 12 cards and sees a starter card on the table
3. Each turn, the model **plays a card** and observes if it's accepted (mainline) or rejected (sideline)
4. The model can **guess the rule** at any point, at the cost of a penalty for wrong guesses
5. The round ends when the model guesses correctly or reaches the turn limit

**Scoring:** `max_turns - turn_used - (penalty × wrong_guesses)` for a correct guess. Higher is better, score is floored at 0.

## Running Evaluations

### Single Model


| Argument | Description |
|----------|-------------|
| `--model MODEL` | Model key from `models.yaml` (required unless `--resume`) |
| `--num-rules N` | Number of distinct rules to test (default: all) |
| `--rule-index N` | Starting rule index |
| `--max-turns N` | Max turns per round (default: 30) |
| `--tag TAG` | Tag appended to output folder name |
| `--resume PATH` | Resume from checkpoint folder |
| `--config FILE` | Config file path (default: `config.yaml`) |

### Multiple Models in Parallel

```bash
# Provide a file listing model keys (one per line)
./scripts/run_parallel_eval.sh eval_models.txt

# With a custom config
./scripts/run_parallel_eval.sh eval_models.txt custom_config.yaml
```

The models file contains one model key per line (lines starting with `#` are ignored).

### Analyzing Results

```bash
uv run python scripts/analyze_results.py results/<folder>
```

Generates charts and tables saved in the input folder: basic metrics comparison, complexity analysis, per-model reports, token usage, and more.

## Configuration

### `models.yaml` — Model Definitions

Each entry defines a model with its provider and provider-specific settings:

```yaml
# Closed-source providers
claude-opus-4.5:
  provider: anthropic
  model_id: claude-opus-4-5-20251101
  reasoning_budget: 16000

gpt-5.2-medium:
  provider: openai
  model_id: gpt-5.2
  reasoning_effort: medium    # none|minimal|low|medium|high|xhigh

gemini-3-pro-preview-high:
  provider: google
  model_id: gemini-3-pro-preview
  thinking_level: high        # low|high

grok-4:
  provider: xai
  model_id: grok-4

# Open models via HuggingFace Inference Providers
deepseek-r1:
  provider: huggingface
  model_id: deepseek-ai/DeepSeek-R1
  hf_provider: together
  reasoning_format: think_tags  # think_tags|separate_field
```

Supported providers: `anthropic`, `openai`, `google`, `xai`, `huggingface`.

### `config.yaml` — Game Settings

```yaml
game:
  num_rules: 0              # 0 = use entire rule library
  num_rounds_per_rule: 3
  max_turns: 30
  hand_size: 12
  wrong_guess_penalty: 2
  seed: 42

rules:
  library_path: "rules.json"
  selection: "sequential"

llm:
  max_tokens: 16384
  max_llm_retries: 3
  temperature: 0.7
```

### Environment Variables

Create a `.env` file with the API keys for the providers you use:

```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
XAI_API_KEY=...
HF_TOKEN=...
```

## Rule Library

Rules are written in natural language in `rules.txt` and compiled to executable Python via an LLM:

```bash
uv run python scripts/generate_rule_library.py --input rules.txt --output rules.json
```

Each compiled rule is a Python function body with access to `card` (current card) and `mainline` (list of previously accepted cards). Card properties: `card.rank` (1–13), `card.color` (`"red"` or `"black"`), `card.suit.suit_name` (`"hearts"`, `"diamonds"`, `"clubs"`, `"spades"`).

The game uses a double deck (104 cards).

## Taking the test

Install dependencies with uv.
```shell
uv sync
```
Then choose a model for the rule_compiler in `config.yaml`. A free solution is to create a Gemini API key.
```yaml
rule_compiler:
  provider: google
  model_id: gemini-3.1-flash-lite-preview
```
Then you can run the script.
```shell
uv run python scripts/evaluate_human.py
```
This starts an interactive session for playing Eleusis in the same conditions as LLMs.
You can quit the session at any time, and restart the script to resume the game from the last **completed** round. This will incrementally fill `results/260121_78_rounds/solo_evaluation_human/results.json`.

After that you will be able to generate the plots with your average score besides LLMs' scores.
```shell
uv run python scripts/analyze_results.py results/260121_78_rounds
```

## Project Structure

```
src/eleusis/
  game/
    cards.py          Card, Deck, Hand (double 52-card deck)
    state.py          GameState, PlayerState, Mainline, Sideline
    engine.py         Rule, GameEngine, scoring
    validator.py      RuleValidator, RuleFactory, simulation-based equivalence
    metrics.py        Rule complexity (cyclomatic, AST node count)
  llm/
    base.py           BaseLLMClient interface
    anthropic.py      Anthropic (extended thinking)
    openai_client.py  OpenAI (reasoning effort)
    google.py         Google (thinking levels)
    xai.py            xAI
    huggingface.py    HuggingFace Inference Providers
  prompts/            Prompt templates (action, rule compilation, game rules)
  analysis/           Result analysis and visualization
  player.py           LLMScientist — main player logic
  runner.py           Round orchestration

scripts/
  evaluate_human.py          Human evaluation
  evaluate_single.py         Single-model evaluation
  run_parallel_eval.sh       Parallel multi-model evaluation
  analyze_results.py         Post-hoc analysis and charts
  generate_rule_library.py   Compile rules.txt → rules.json
```


## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
