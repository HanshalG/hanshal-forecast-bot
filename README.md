## Forecast Bot

Automated forecaster for Metaculus AI tournament questions. Runs locally or on GitHub Actions.

### Quick start (local)
1) Install dependencies
```bash
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
poetry run python main.py --mode fall-aib-2025
```

### Logs
Forecast runs are saved under `logs/` (including per-question JSONL and text files).

### GitHub Actions (CI)
- Workflow: `.github/workflows/run_bot_on_tournament.yaml`
- Triggers every 20 minutes and via manual dispatch.
- Set required secrets in repository settings: `METACULUS_TOKEN`, `OPENROUTER_API_KEY`, `ASKNEWS_CLIENT_ID`, `ASKNEWS_SECRET`, `EXA_API_KEY`.
- The workflow uploads `logs/` as an artifact for each run.

### Tests
```bash
poetry run pytest
```


