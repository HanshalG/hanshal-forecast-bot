import asyncio
import os
from dotenv import load_dotenv

# Force low limits for testing BEFORE importing agent_infrastructure
# (Note: In the actual implementation, these are module-level constants, 
# so we might need to patch them if we want to change them dynamically for tests,
# or just rely on the defaults set in the file if we can't patch easily.
# Let's try to patch them after import.)

load_dotenv()

from src import agent_infrastructure
from src.forecast import get_binary_prediction

# Patch limits to be very low for testing
agent_infrastructure.RUN_TOOL_LIMIT = 2
agent_infrastructure.THREAD_TOOL_LIMIT = 3
print(f"Testing with RUN_TOOL_LIMIT={agent_infrastructure.RUN_TOOL_LIMIT}, THREAD_TOOL_LIMIT={agent_infrastructure.THREAD_TOOL_LIMIT}")

async def test_tool_limits():
    question_details = {
        "title": "Will SpaceX successfully land a human on Mars by 2030?",
        "type": "binary",
        "resolution_criteria": "Successful landing means...",
        "fine_print": "Must be a human mission."
    }
    
    try:
        prob, rationale = await get_binary_prediction(
            question_details,
            num_runs=1,
            max_outside_searches=10, # These args control the loop in forecast.py, but agent_infrastructure limits apply per run
            max_inside_searches=10
        )
        print(f"\nPrediction: {prob}")
        print(f"Rationale: {rationale[:100]}...")
    except Exception as e:
        print(f"Test failed with error: {e}")

if __name__ == "__main__":
    asyncio.run(test_tool_limits())
