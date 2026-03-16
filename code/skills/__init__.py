"""Skills Externalized Knowledge System

Skills is the core implementation of "knowledge externalization", allowing the model to load domain knowledge on demand without requiring fine-tuning.

Features:
- Progressive Disclosure: Only metadata is loaded at startup; full content is loaded on demand.
- Cache Friendly: Injected as a tool_result, does not modify the system_prompt.
- Human Editable: SKILL.md files, supporting version control.
- Token Savings: Expected to save 85% of tokens (in a scenario with 20 skills).

Usage Example:
    >>> from hello_agents.skills import SkillLoader
    >>> loader = SkillLoader(skills_dir=Path("skills"))
    >>> # Get all skill descriptions (for the system prompt)
    >>> descriptions = loader.get_descriptions()
    >>> # Load full skill on demand
    >>> skill = loader.get_skill("pdf")
    >>> print(skill.body)
"""

from .loader import SkillLoader, Skill

__all__ = [
    "SkillLoader",
    "Skill",
]