import asyncio
from typing import List, Tuple, Optional

# Reuse existing orchestration without altering current behavior
from src.forecast import forecast_questions
from src.metaculus_utils import get_open_question_ids_from_tournament


class ForecastRunner:
    """
    High-level runner for executing forecasts over one or more Metaculus questions.

    This thin wrapper preserves current behavior by delegating to
    `src.forecast.forecast_questions` and optionally setting the tournament
    context used for logging. It provides both async and sync entry points.

    Parameters
    ----------
    tournament_id: Optional[str]
        Logical identifier for the tournament/run context. If provided, it will
        be passed to the underlying forecasting module for logging context.
    """

    def __init__(self, *, tournament_id: Optional[str] = None) -> None:
        self.tournament_id = tournament_id

    async def run(
        self,
        open_question_id_post_id: List[Tuple[int, int]],
        *,
        submit_prediction: bool = False,
        num_runs_per_question: int = 1,
        skip_previously_forecasted_questions: bool = False,
    ) -> None:
        """
        Execute forecasts for the provided (question_id, post_id) pairs.
        """
        # Maintain compatibility with current logging scheme
        if self.tournament_id is not None:
            try:
                import src.forecast as _forecast_module
                _forecast_module.TOURNAMENT_ID = self.tournament_id
            except Exception:
                pass

        await forecast_questions(
            open_question_id_post_id,
            submit_prediction,
            num_runs_per_question,
            skip_previously_forecasted_questions,
        )

    def run_sync(
        self,
        open_question_id_post_id: List[Tuple[int, int]],
        *,
        submit_prediction: bool = False,
        num_runs_per_question: int = 1,
        skip_previously_forecasted_questions: bool = False,
    ) -> None:
        """
        Synchronous wrapper around `run`.
        """
        asyncio.run(
            self.run(
                open_question_id_post_id,
                submit_prediction=submit_prediction,
                num_runs_per_question=num_runs_per_question,
                skip_previously_forecasted_questions=skip_previously_forecasted_questions,
            )
        )

    async def run_for_tournament(
        self,
        tournament_id: Optional[str] = None,
        *,
        submit_prediction: bool = False,
        num_runs_per_question: int = 1,
        skip_previously_forecasted_questions: bool = False,
    ) -> None:
        """
        Convenience method: fetch open questions for a tournament and run.
        If `tournament_id` argument is omitted, uses the instance's `tournament_id`.
        """
        tid = tournament_id or self.tournament_id
        if tid is None:
            raise ValueError("tournament_id must be provided either to the instance or to this call")

        pairs = get_open_question_ids_from_tournament(tid)
        # Ensure logging context reflects the tournament used for fetching
        self.tournament_id = tid
        await self.run(
            pairs,
            submit_prediction=submit_prediction,
            num_runs_per_question=num_runs_per_question,
            skip_previously_forecasted_questions=skip_previously_forecasted_questions,
        )


