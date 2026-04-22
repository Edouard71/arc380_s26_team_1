import json
import os
from pathlib import Path
from typing import Any, Dict

from openai import OpenAI

# -------- Paths --------
BASE_DIR = Path(__file__).resolve().parent

PROMPT_PATH = BASE_DIR / "prompt.md"
SCHEMA_PATH = BASE_DIR / "tower_plan_schema.json"

# -------- Helpers --------
def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# -------- Core Function --------
def generate_tower_plan(
    tower_description: str,
    available_blocks: int,
    tower_center: list[float],
    workspace: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """
    Calls GPT-5.4 to generate a tower plan JSON.
    """

    print("Accessing Gpt-5.4 Model\n")
    print(f"Tower Description: {tower_description}\n")
    print(f"Available Blocks: {available_blocks}\n")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Load files
    system_prompt = load_text(PROMPT_PATH)
    schema = load_json(SCHEMA_PATH)

    # Build runtime input
    runtime_input = {
        "tower_description": tower_description,
        "available_blocks": available_blocks,
        "tower_center": tower_center,
    }

    if workspace:
        runtime_input["workspace"] = workspace

    # Convert to string for model
    input_str = json.dumps(runtime_input, indent=2)

    # Call model
    print("Calling Gpt-5.4 Model For Tower Construction Plan")
    response = client.responses.create(
        model="gpt-5.4",
        reasoning={"effort": "high"},
        instructions=system_prompt,
        input=input_str,
        text={
            "format": {
                "type": "json_schema",
                "name": "tower_plan",
                "schema": schema,
                "strict": True
            }
        }
    )

    print("GPT-5.4 Response", response)

    # Parse JSON output
    try:
        result = json.loads(response.output_text)
    except Exception as e:
        raise RuntimeError(f"Failed to parse model output: {e}\n{response.output_text}")

    return result


# -------- Optional Validation (lightweight) --------
def validate_plan(plan: Dict[str, Any]) -> None:
    """
    Minimal sanity checks (you can expand this later).
    """
    assert "blocks" in plan, "Missing blocks field"
    assert isinstance(plan["blocks"], list), "Blocks must be a list"

    for block in plan["blocks"]:
        assert len(block["goal_position"]) == 3
        assert len(block["goal_quaternion_wxyz"]) == 4


# -------- Example Run --------
if __name__ == "__main__":
    workspace = {
        "x_min": 0.20,
        "x_max": 0.55,
        "y_min": 0.10,
        "y_max": 0.35,
        "z_min": 0.02,
        "z_max": 0.15,
    }

    plan = generate_tower_plan(
        tower_description=(
            "Build a 1-level square tower with 3 blocks per level. "
            "Alternate each level's rotation such that it creates a triangle"
        ),
        available_blocks=12,
        tower_center=[0.419, 0.221, 0.021],
        workspace=workspace,
    )

    validate_plan(plan)

    print(json.dumps(plan, indent=2))