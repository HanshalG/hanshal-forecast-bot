from src.eval.reporting import write_eval_reports
from src.eval.types import CalibrationBin, PredictionRecord, StrategySummary


def test_write_eval_reports_creates_required_artifacts(tmp_path):
    predictions = [
        PredictionRecord(
            strategy_id="s1",
            post_id=1,
            question_id=10,
            title="Q",
            as_of_time="2024-01-01T00:00:00Z",
            probability_yes=0.7,
            label=1,
            resolution_raw="yes",
            brier=0.09,
            log_loss=0.356675,
            runtime_seconds=1.23,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.002,
            num_runs=1,
            outside_view_enabled=True,
            inside_view_enabled=True,
            prediction_market_enabled=False,
            final_forecast_use_agent=True,
            error=None,
        )
    ]
    summaries = [
        StrategySummary(
            strategy_id="s1",
            n=1,
            n_success=1,
            n_failed=0,
            mean_brier=0.09,
            mean_log_loss=0.356675,
            ece=0.3,
            mean_cost_usd=0.002,
            mean_runtime_seconds=1.23,
        )
    ]
    calibration = {
        "s1": [
            CalibrationBin(
                strategy_id="s1",
                bin_index=0,
                bin_start=0.0,
                bin_end=0.1,
                count=0,
                avg_pred=0.0,
                avg_outcome=0.0,
                abs_gap=0.0,
            )
        ]
    }

    paths = write_eval_reports(
        output_dir=str(tmp_path),
        predictions=predictions,
        strategy_summaries=summaries,
        calibration_by_strategy=calibration,
        run_config_snapshot={"hello": "world"},
    )

    assert (tmp_path / "predictions.jsonl").exists()
    assert (tmp_path / "strategy_summary.csv").exists()
    assert (tmp_path / "leaderboard.md").exists()
    assert (tmp_path / "run_config_snapshot.json").exists()
    assert (tmp_path / "calibration_s1.csv").exists()
    assert "predictions_jsonl" in paths
