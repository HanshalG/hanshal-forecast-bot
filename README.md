## Forecast Bot

Automated forecaster for Metaculus AI tournament questions. Runs locally or on GitHub Actions.


## Forecast Examples

| Question | Bot Forecast | Metaculus Community Prediction | Date |
|----------|-------------|---------------------|------|
| Will housing prices in Madrid rise by more than 18% in 2025? | 28% Yes | 26% Yes | Sep 25 |
| Will US imports from Brazil in November 2025 exceed those of November 2024? | 34% Yes | 30% Yes | Sep 25 |
| Will China invade Taiwan in 2025? | 6% Yes | 1% Yes | Sep 26 |

### How it works
- **Outside view**: Generates historical-research questions (via an LLM), retrieves answers via Exa, filters low-signal results, and prompts an LLM to synthesize an outside-view analysis.
- **Inside view**: Pulls recent news via AskNews and targeted questions (via an LLM) with Exa answers, filters low-signal results, combines with the outside view, and prompts an LLM to synthesize an inside-view analysis with a final probability prediction.

- **Techniques used in prompts**:
  - Superforecasting-style process: base-rate first, Fermi decomposition
  - Reference class analysis (outside view)
  - Source quality screening; separate facts from opinions; prefer identifiable experts
  - Evidence weighting: strong / moderate / weak
  - Timeframe sensitivity: consider halved/doubled horizons
  - Calibration and odds awareness (e.g., 90% = 9:1, 99% = 99:1)
  - Checklist discipline: paraphrase+criteria, base-rate anchoring, consistency check, top evidence, blind-spot, status quo
  - Consider expectations of experts and markets

- **Sampling & aggregation**:
  - Multiple independent runs per question
  - Binary: median of sampled probabilities (median sampling)
  - Numeric/Discrete: element-wise median across sampled CDFs
  - Multiple-choice: trimmed linear opinion pool across runs

### Quick start (local)
1) Create the Poetry virtual environment with Python 3.11 and install dependencies
```bash
poetry env use 3.11
poetry install
```
2) Set environment variables
```bash
export METACULUS_TOKEN=...            # required
export OPENROUTER_API_KEY=...         # required
export ASKNEWS_CLIENT_ID=...          # required
export ASKNEWS_SECRET=...             # required
export EXA_API_KEY=...                # required
```
3) Run the bot
```bash
poetry run python main.py --mode example_questions
```

Enable Supabase logging for a run:
```bash
poetry run python main.py --mode example_questions --log-to-supabase
```

If your local environment has drifted because of manual `venv` usage, reset it:
```bash
rm -rf .venv
poetry env use 3.11
poetry install
```

### Running Agent Modules (New)
To run the standalone agent scripts for Outside and Inside View generation:
```bash
# Outside View Agent
poetry run python -m src.outside_view

# Inside View Agent
poetry run python -m src.inside_view
```

### Logs
Forecast runs are saved under `logs/`:
- `logs/forecasts/by_question/<question_id>.jsonl` and `.txt`
- `logs/forecasts/by_run/<run_id>.jsonl`
- `logs/forecasts/all_forecasts.jsonl` and `.txt`
- `logs/forecasts/all_events.jsonl` (runtime + forecast events)
- `logs/runtime/runtime.jsonl` (runtime events)

Useful logging env vars:
- `LOG_ENABLE` (default: `true`)
- `LOG_DIR` (default: `logs`)
- `LOG_LEVEL` (`DEBUG`/`INFO`/`WARNING`/`ERROR`, default: `INFO`)
- `LOG_CONSOLE` (default: `true`)
- `LOG_CONSOLE_JSON` (default: `false`)
- `LOG_GLOBAL_STREAM` (default: `true`)
- `LOG_PLAINTEXT` (default: `true`)
- `LOG_INCLUDE_COMMENT` (default: `true`)
- `SUPABASE_LOG_ENABLE` (default: `true`)
- `SUPABASE_URL` (Supabase project URL; required for remote logging)
- `SUPABASE_KEY` (service-role key or insert-capable key; required for remote logging)
- `SUPABASE_FORECAST_TABLE` (default: `forecast_events`)
- `SUPABASE_TIMEOUT_S` (default: `5`)

### GitHub Actions (CI) for Metaculus Tournaments
- Workflows:
  - `.github/workflows/run_bot_on_tournament_cj.yaml`
  - `.github/workflows/run_bot_on_minibench.yaml`
- Triggered via manual dispatch (`workflow_dispatch`).
- Set required secrets in repository settings: `METACULUS_TOKEN`, `OPENROUTER_API_KEY`, `ASKNEWS_CLIENT_ID`, `ASKNEWS_SECRET`, `EXA_API_KEY`.
- The workflow uploads `logs/` as an artifact for each run.

### Credit
https://github.com/Metaculus/metac-bot-template - For bot template to start
https://github.com/Panshul42/Forecasting_Bot_Q2 - For prompt templates
