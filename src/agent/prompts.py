"""Agent prompts (sourced from the versioned prompt registry)."""
from src.prompts import get_prompt


def agent_system() -> str:
    return get_prompt("agent_system")
