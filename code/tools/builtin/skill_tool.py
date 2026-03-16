"""Skill Tool

Allows the Agent to load domain knowledge on demand.

Features:
- Progressive disclosure: Load complete skills only when needed
- Cache friendly: Injected as tool_result, does not modify system_prompt
- Resource hints: Automatically lists available scripts, documents, examples, etc.
- Parameter replacement: Supports $ARGUMENTS placeholder

Usage example:
    >>> from hello_agents.skills import SkillLoader
    >>> from hello_agents.tools.builtin.skill_tool import SkillTool
    >>> loader = SkillLoader(skills_dir=Path("skills"))
    >>> tool = SkillTool(skill_loader=loader)
    >>> # Agent call
    >>> response = tool.run({"skill": "pdf"})
"""

from typing import Dict, Any, List
from ..base import Tool, ToolParameter
from ...skills.loader import SkillLoader
from ..response import ToolResponse
from ..errors import ToolErrorCode


class SkillTool(Tool):
    """
    Skill Tool

    Allows the model to load domain knowledge on demand.
    """

    def __init__(self, skill_loader: SkillLoader):
        """Initialize the skill tool

        Args:
            skill_loader: Skill loader instance
        """
        # Generate dynamic descriptions
        descriptions = skill_loader.get_descriptions()

        super().__init__(
            name="Skill",
            description=f"""Load skills to acquire professional knowledge.

Available skills:
{descriptions}

When to use:
- Use immediately when the task clearly matches a skill description
- Before starting domain-specific work
- When professional knowledge not possessed by the model is needed

Note: After loading a skill, please strictly follow the skill instructions to complete the user task.""",
            expandable=False
        )
        self.skill_loader = skill_loader

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="skill",
                type="string",
                description="The name of the skill to load",
                required=True
            ),
            ToolParameter(
                name="args",
                type="string",
                description="Optional parameters, will replace the $ARGUMENTS placeholder in SKILL.md",
                required=False,
                default=""
            )
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        """Execute skill loading

        Args:
            parameters: Parameter dictionary containing skill and optional args

        Returns:
            ToolResponse: Response containing complete skill content
        """
        skill_name = parameters.get("skill", "")
        args = parameters.get("args", "")

        if not skill_name:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="Skill name must be specified",
                context={"params_input": parameters}
            )

        try:
            # Load skills on demand
            skill = self.skill_loader.get_skill(skill_name)

            if not skill:
                available = ", ".join(self.skill_loader.list_skills())
                return ToolResponse.error(
                    code=ToolErrorCode.NOT_FOUND,
                    message=f"Skill '{skill_name}' does not exist. Available skills: {available}",
                    context={"params_input": parameters, "available_skills": self.skill_loader.list_skills()}
                )

            # Replace $ARGUMENTS placeholder
            content = skill.body.replace("$ARGUMENTS", args)

            # List available resources
            resources_hint = self._get_resources_hint(skill)

            # Construct complete skill content (cache-friendly injection method)
            full_content = f"""<skill-loaded name="{skill_name}">
{content}
{resources_hint}
</skill-loaded>

✅ Skill loaded: {skill.name}
📝 Description: {skill.description}

Please strictly follow the above skill instructions to complete the user task."""

            return ToolResponse.success(
                text=full_content,
                data={
                    "name": skill.name,
                    "description": skill.description,
                    "loaded": True,
                    "token_estimate": len(full_content),
                    "has_resources": bool(resources_hint)
                }
            )

        except Exception as e:
            return ToolResponse.error(
                code=ToolErrorCode.INTERNAL_ERROR,
                message=f"Failed to load skill: {str(e)}",
                context={"params_input": parameters, "error": str(e)}
            )

    def _get_resources_hint(self, skill) -> str:
        """Generate resource hint text

        Args:
            skill: Skill object

        Returns:
            Formatted resource hint text
        """
        resources = []

        for folder, label in [
            ("scripts", "Scripts"),
            ("references", "References"),
            ("assets", "Assets"),
            ("examples", "Examples")
        ]:
            folder_path = skill.dir / folder
            if folder_path.exists():
                files = list(folder_path.glob("*"))
                if files:
                    file_list = ", ".join(f.name for f in files[:5])  # Show at most 5
                    if len(files) > 5:
                        file_list += f" and {len(files)} files in total"
                    resources.append(f"  - {label}: {file_list}")

        if not resources:
            return ""

        return "\n\n**Available resources**: \n" + "\n".join(resources)