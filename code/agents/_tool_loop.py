"""Shared OpenAI-style function-calling loop (重要-9 / 重要-10).

The Simple / Reflection / Plan-Solve agents previously each carried a nearly
identical ``invoke_with_tools`` loop (parse ``choices[0].message.tool_calls`` →
append assistant message → run each tool → append tool result → repeat →
fallback ``invoke``). This module hosts that logic once so all callers stay in
sync and PlanSolve's Executor no longer spins up a full ``SimpleAgent`` per step
just to reuse it.

The loop consumes the OpenAI-compatible response shape. Anthropic/Gemini
adapters now return that shape via ``normalize_*_tool_response`` so this path is
provider-agnostic.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional


def build_tool_schemas(tool_registry: Any) -> List[Dict[str, Any]]:
    """Build OpenAI function-calling schemas from a ToolRegistry.

    Standalone twin of ``Agent._build_tool_schemas`` so non-Agent helpers (the
    Plan-Solve Executor) can build schemas without instantiating an Agent.
    """
    if not tool_registry:
        return []

    def _map_type(param_type: str) -> str:
        normalized = (param_type or "").lower()
        if normalized in {"string", "number", "integer", "boolean", "array", "object"}:
            return normalized
        return "string"

    schemas: List[Dict[str, Any]] = []
    for tool in tool_registry.get_all_tools():
        properties: Dict[str, Any] = {}
        required: List[str] = []
        try:
            parameters = tool.get_parameters()
        except Exception:
            parameters = []
        for param in parameters:
            properties[param.name] = {
                "type": _map_type(param.type),
                "description": param.description or "",
            }
            if getattr(param, "default", None) is not None:
                properties[param.name]["default"] = param.default
            if getattr(param, "required", True):
                required.append(param.name)
        function: Dict[str, Any] = {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": {"type": "object", "properties": properties},
        }
        if required:
            function["parameters"]["required"] = required
        schemas.append({"type": "function", "function": function})

    for name, info in getattr(tool_registry, "_functions", {}).items():
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": info.get("description", ""),
                    "parameters": {
                        "type": "object",
                        "properties": {"input": {"type": "string", "description": "Input text"}},
                        "required": ["input"],
                    },
                },
            }
        )
    return schemas


def execute_tool_via_registry(tool_registry: Any, name: str, arguments: Any) -> str:
    """Run a tool through the registry and return its observation text."""
    payload = arguments
    if tool_registry.get_function(name) and isinstance(arguments, dict):
        payload = arguments.get("input", "")
    response = tool_registry.execute_tool(name, payload)
    return getattr(response, "text", str(response))


def run_tool_calling_loop(
    *,
    llm: Any,
    tool_schemas: List[Dict[str, Any]],
    messages: List[Dict[str, Any]],
    execute_tool: Callable[[str, Dict[str, Any]], str],
    max_iterations: int = 3,
    tool_choice: Any = "auto",
    on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    **llm_kwargs: Any,
) -> str:
    """Drive an OpenAI-style function-calling loop.

    Args:
        llm: object exposing ``invoke_with_tools`` and ``invoke``.
        tool_schemas: OpenAI function schemas (may be empty).
        messages: chat messages; mutated in place with assistant/tool turns.
        execute_tool: callable ``(name, args_dict) -> observation_text``.
        max_iterations: max tool-using rounds before a plain fallback answer.
        tool_choice: OpenAI tool_choice policy.
        on_event: optional ``(event_name, payload)`` sink for observability.

    Returns:
        The model's final text answer.
    """
    def _emit(name: str, payload: Dict[str, Any]) -> None:
        if on_event:
            try:
                on_event(name, payload)
            except Exception:
                pass

    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        try:
            response = llm.invoke_with_tools(
                messages=messages,
                tools=tool_schemas,
                tool_choice=tool_choice,
                **llm_kwargs,
            )
        except Exception as exc:
            _emit("llm_error", {"error": str(exc), "iteration": iteration})
            break

        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            return message.content or ""

        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )

        for tool_call in tool_calls:
            name = tool_call.function.name
            call_id = tool_call.id
            _emit("tool_call", {"tool_name": name, "tool_call_id": call_id})
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as exc:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": f"Error: Invalid argument format - {exc}",
                    }
                )
                continue

            result = execute_tool(name, arguments)
            _emit("tool_result", {"tool_name": name, "tool_call_id": call_id, "result": result})
            messages.append({"role": "tool", "tool_call_id": call_id, "content": result})

    # Exhausted the tool budget (or the model errored): ask once for a plain
    # answer so the caller still gets a best-effort response.
    try:
        final = llm.invoke(messages, **llm_kwargs)
        return final.content if hasattr(final, "content") else str(final)
    except Exception:
        return ""
