## Forecast Bot

Automated forecaster for Metaculus AI tournament questions. Runs locally or on GitHub Actions.


## Forecast Examples

| Question | Bot Forecast | Date | Forecast Comment |
|----------|--------------|------|------------------|
| Will any political party or coalition acquire a supermajority in the 2026 Hungarian parliamentary elections? | 56% | Feb 20 | - The Hungarian system with 199 seats and a winner-takes-most structure makes a mid-to-high 40s percent national vote capable of delivering 133+ seats, enabling a 2/3 majority.<br>- Across the 2014, 2018, and 2022 elections, the same rules produced supermajorities, supporting a historically recurrent baseline for such an outcome.<br>- The external/base-rate view thus treats a 2/3 majority as plausible given structural incentives, even when national vote shares are not overwhelming.<br>- February 2026 polls reportedly show a credible challenger lead (Tisza) that, if real and sustained, would weaken the incumbent’s path to a 2/3 majority.<br>- However, a plurality for the challenger does not guarantee 133 seats; achieving a 2/3 requires an unusually efficient distribution of votes across districts or strong cross-constituency coordination.<br>- Turnout dynamics, district geography, and potential opposition vote-splitting can produce seat outcomes that diverge from headline national shares.<br>- Incumbent advantages in organization, state/media presence, and resources could cushion momentum shifts and keep the door open to a 2/3 outcome.<br>- The blended view therefore points to a meaningful chance that no party or coalition reaches 133 seats, even as structural factors keep the possibility alive.<br>- The resolution will depend on official election results; if no election occurs before 2027-01-01, the question will be annulled. |
| Will CDU win the most seats in the Baden-Württemberg Landtag election? | 72% | Feb 19 | - The final seat allocation in Baden-Württemberg is determined by the party vote due to the mixed-member proportional system with overhang and compensatory seats, so the party vote largely drives who has the most seats.<br>- Historically CDU has often been the largest party in Baden-Württemberg, but Greens have held the top seat count in three of the last elections, making the long-run base rate less favorable to CDU.<br>- Late-campaign polling places the CDU ahead of the Greens in the party vote by roughly six percentage points, a gap that commonly yields a seat lead under the state’s rules.<br>- Because final seats track the party vote and overhang/compensation introduce some variance, a six-point lead in the vote is typically sufficient for CDU to have the most seats, though not guaranteed.<br>- Several potential headwinds could erode the CDU seat lead: Greens’ high candidate popularity, possible split-ticket voting, enfranchising 16–17-year-olds, and incumbent retirements affecting local contests.<br>- Despite these caveats, the strongest near-term evidence supports CDU finishing with the most seats, with uncertainty mainly arising from multi-party dynamics and small shifts in the final days.<br>- A tie for the most seats remains possible, in which case the outcome would resolve as “No” under the stated rules.<br>- Overall stance: CDU is more likely than not to win the most seats, but the result is not certain given ongoing campaign dynamics. |

### How it works
- **Question input**
  - Tournament modes (`spring-aib-2026`, `minibench`) pull open questions from Metaculus.
  - `manual` mode loads questions from JSON or JSONL.
- **Per-run forecasting pipeline**
  - Generate **outside view** analysis from historical/reference-class evidence.
  - Generate **inside view** analysis from recent news and targeted research.
  - Optionally append prediction market context (`--get-prediction-market`).
  - Run final forecast generation to produce probability/CDF/option distribution + rationale.
- **Aggregation across runs (`--num-runs`)**
  - Binary: median probability.
  - Numeric/Discrete: median CDF (then monotonicity enforcement).
  - Multiple-choice: trimmed linear opinion pool.
- **Output + submission**
  - Always logs forecast events locally.
  - Submits prediction/comment to Metaculus only when `--submit` is enabled (not supported in `manual` mode).

### Quick start (local)
1) Install dependencies (Python 3.11)
```bash
poetry env use 3.11
poetry install
```
2) Configure environment
```bash
cp .env.template .env
# fill required keys in .env:
# METACULUS_TOKEN, OPENROUTER_API_KEY, EXA_API_KEY, ASKNEWS_CLIENT_ID, ASKNEWS_SECRET
```
3) Run locally
```bash
# quick sanity run (built-in example question)
poetry run python main.py --mode example_questions
```

Run a tournament mode:
```bash
poetry run python main.py --mode spring-aib-2026 --num-runs 3 --skip-prev
```

Run Minibench:
```bash
poetry run python main.py --mode minibench --num-runs 3 --skip-prev
```

Run manual questions:
```bash
poetry run python main.py --mode manual --questions-file manual_questions.example.json --num-runs 3
```

Submit to Metaculus + include market context + enable Supabase logging:
```bash
poetry run python main.py --mode spring-aib-2026 --submit --num-runs 5 --skip-prev --get-prediction-market --log-to-supabase
```

### Manual Question File Format
- The file can be JSON array or JSON object with `questions` array.
- Required common fields: `title`, `type`.
- Supported `type` values: `binary`, `multiple_choice`, `numeric`, `discrete`.
- Optional common fields: `id`, `post_id`, `description`, `resolution_criteria`, `fine_print`, `url`.
- For `multiple_choice`, include `options` (2+ entries).
- For `numeric`/`discrete`, include:
  - `scaling.range_min`, `scaling.range_max`, `scaling.zero_point` (nullable)
  - `open_upper_bound`, `open_lower_bound`
  - `unit` (optional but recommended)
  - `scaling.inbound_outcome_count` required for `discrete`
- Example file: `manual_questions.example.json`

### Logs
When `LOG_ENABLE=true`, forecast/runtime events are written under `logs/`:
- `logs/forecasts/by_question/<question_id>.jsonl`
- `logs/forecasts/by_question/<question_id>.txt` (if `LOG_PLAINTEXT=true`)
- `logs/forecasts/by_run/<run_id>.jsonl`
- `logs/forecasts/all_forecasts.jsonl`
- `logs/forecasts/all_forecasts.txt` (if `LOG_PLAINTEXT=true`)
- `logs/forecasts/all_events.jsonl` (forecast + runtime event stream when `LOG_GLOBAL_STREAM=true`)
- `logs/runtime/runtime.jsonl`

Useful logging env vars:
- `LOG_ENABLE` (default: `true`)
- `LOG_DIR` (default: `logs`)
- `LOG_LEVEL` (`DEBUG`/`INFO`/`WARNING`/`ERROR`, default: `INFO`)
- `LOG_CONSOLE` (default: `true`)
- `LOG_CONSOLE_JSON` (default: `false`)
- `LOG_GLOBAL_STREAM` (default: `true`)
- `LOG_PLAINTEXT` (default: `true`)
- `LOG_INCLUDE_COMMENT` (default: `true`)
- `LOG_MESSAGE_PREVIEW_CHARS` (default: `300`)
- `SUPABASE_LOG_ENABLE` (default: `true`)
- `SUPABASE_URL` (Supabase project URL; required for remote logging)
- `SUPABASE_KEY` (service-role key or insert-capable key; required for remote logging)
- `SUPABASE_FORECAST_TABLE` (default: `forecast_events`)
- `SUPABASE_TIMEOUT_S` (default: `5`)
- `SUPABASE_DEDUP_SUBMISSION_EVENTS` (default: `true`)

### GitHub Actions (CI) for Metaculus Tournaments
- Workflows:
  - `.github/workflows/run_bot_on_tournament_cj.yaml`  
    Runs: `python main.py --mode spring-aib-2026 --submit --num-runs 5 --skip-prev --get-prediction-market --log-to-supabase`
  - `.github/workflows/run_bot_on_minibench.yaml`  
    Runs: `python main.py --mode minibench --submit --num-runs 5 --skip-prev --get-prediction-market --log-to-supabase`
- Trigger: manual only (`workflow_dispatch`).
- Required repository secrets:
  - `METACULUS_TOKEN`
  - `OPENROUTER_API_KEY`
  - `ASKNEWS_CLIENT_ID`
  - `ASKNEWS_SECRET`
  - `EXA_API_KEY`
  - `SUPABASE_URL`
  - `SUPABASE_KEY`
- Both workflows set `RUN_ID=github-run-${{ github.run_number }}` and upload `logs/` as an artifact (retention: 30 days).

### Credit
https://github.com/Metaculus/metac-bot-template - For bot template to start
https://github.com/Panshul42/Forecasting_Bot_Q2 - For prompt templates
