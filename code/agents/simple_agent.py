from typing import Optional, Iterator, TYPE_CHECKING, Dict, AsyncGenerator
import json

from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..core.message import Message
from ..core.streaming import StreamEvent, StreamEventType
from ..core.lifecycle import LifecycleHook
from ._tool_loop import run_tool_calling_loop

if TYPE_CHECKING:
    from ..tools.registry import ToolRegistry

class SimpleAgent(Agent):
    """Simple conversational Agent, supporting optional tool calling

    Features:
    - Pure conversation mode (no tools)
    - Function Calling tool invocation (optional)
    - Automatic multi-turn tool invocation
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        tool_registry: Optional['ToolRegistry'] = None,
        enable_tool_calling: bool = True,
        max_tool_iterations: int = 3
    ):
        """
        Initialize SimpleAgent

        Args:
            name: Agent name
            llm: LLM instance
            system_prompt: System prompt words
            config: Configuration object
            tool_registry: Tool registry (optional, if provided, tool calling is enabled)
            enable_tool_calling: Whether to enable tool calling (only effective when tool_registry is provided)
            max_tool_iterations: Maximum number of tool calling iterations
        """
        # Pass tool_registry to the base class
        super().__init__(
            name,
            llm,
            system_prompt,
            config,
            tool_registry=tool_registry
        )
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None
        self.max_tool_iterations = max_tool_iterations

    def run(self, input_text: str, **kwargs) -> str:
        """
        Run SimpleAgent (based on Function Calling)

        Args:
            input_text: User input
            **kwargs: Additional parameters

        Returns:
            Final response
        """
        from datetime import datetime

        session_start_time = datetime.now()
        self._init_trace()

        # Build the message list
        messages = self._build_messages(input_text)

        # Log user message
        if self.trace_logger:
            self.trace_logger.log_event(
                "message_written",
                {"role": "user", "content": input_text}
            )

        # If tool calling is not enabled, return the LLM response directly
        if not self.enable_tool_calling or not self.tool_registry:
            llm_response = self.llm.invoke(messages, **kwargs)
            response_text = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)

            # Save to history
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(response_text, "assistant"))

            self._finalize_trace(
                "success",
                duration=(datetime.now() - session_start_time).total_seconds(),
                final_answer=response_text,
            )

            return response_text

        # Enable tool calling mode (重要-9: shared function-calling loop)
        tool_schemas = self._build_tool_schemas()
        steps_holder = {"n": 0}

        def _on_event(name, payload):
            if name == "tool_call":
                steps_holder["n"] += 1
            if not self.trace_logger:
                return
            if name == "tool_call":
                self.trace_logger.log_event("tool_call", payload)
            elif name == "tool_result":
                self.trace_logger.log_event("tool_result", payload)
            elif name == "llm_error":
                self.trace_logger.log_event(
                    "error",
                    {"error_type": "LLM_ERROR", "message": payload.get("error", "")},
                )

        final_response = run_tool_calling_loop(
            llm=self.llm,
            tool_schemas=tool_schemas,
            messages=messages,
            execute_tool=self._execute_tool_call,
            max_iterations=self.max_tool_iterations,
            on_event=_on_event,
            **kwargs,
        )
        if not final_response:
            final_response = "Sorry, I cannot answer this question."

        # Save to history
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_response, "assistant"))

        self._finalize_trace(
            "success",
            duration=(datetime.now() - session_start_time).total_seconds(),
            total_steps=steps_holder["n"],
            final_answer=final_response,
        )

        return final_response

    def add_tool(self, tool, auto_expand: bool = True) -> None:
        """
        Add a tool to the Agent (convenience method)

        Args:
            tool: Tool object
            auto_expand: Whether to automatically expand expandable tools (default is True)

        If the tool is expandable (expandable=True), it will automatically expand into multiple independent tools.
        """
        if not self.tool_registry:
            from ..tools.registry import ToolRegistry
            self.tool_registry = ToolRegistry(config=self.config)
            self.enable_tool_calling = True

        # Use ToolRegistry's register_tool method directly
        # ToolRegistry will automatically handle tool expansion
        self.tool_registry.register_tool(tool, auto_expand=auto_expand)

    def remove_tool(self, tool_name: str) -> bool:
        """Remove a tool (convenience method)"""
        if self.tool_registry:
            existed = self.tool_registry.get_tool(tool_name) is not None
            self.tool_registry.unregister(tool_name)
            return existed
        return False

    def list_tools(self) -> list:
        """List all available tools"""
        if self.tool_registry:
            return self.tool_registry.list_tools()
        return []

    def has_tools(self) -> bool:
        """Check if there are available tools"""
        return self.enable_tool_calling and self.tool_registry is not None

    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        """
        Run the Agent in streaming mode

        Args:
            input_text: User input
            **kwargs: Additional parameters

        Yields:
            Agent response chunks
        """
        from datetime import datetime

        session_start_time = datetime.now()
        self._init_trace()

        messages = self._build_messages(input_text)
        if self.trace_logger:
            self.trace_logger.log_event("message_written", {"role": "user", "content": input_text})

        # Stream call to LLM
        full_response = ""
        try:
            for chunk in self.llm.stream_invoke(messages, **kwargs):
                full_response += chunk
                yield chunk
        finally:
            # Save the complete conversation to history
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(full_response, "assistant"))

            self._finalize_trace(
                "success",
                duration=(datetime.now() - session_start_time).total_seconds(),
                final_answer=full_response,
            )

    async def arun_stream(
        self,
        input_text: str,
        on_start: LifecycleHook = None,
        on_finish: LifecycleHook = None,
        on_error: LifecycleHook = None,
        **kwargs
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        True streaming execution of SimpleAgent

        Returns each text chunk of the LLM output in real-time

        Args:
            input_text: User input
            on_start: Start hook
            on_finish: Completion hook
            on_error: Error hook
            **kwargs: Additional parameters

        Yields:
            StreamEvent: Streaming events
        """
        from datetime import datetime

        session_start_time = datetime.now()
        self._init_trace()

        # Send start event
        yield StreamEvent.create(
            StreamEventType.AGENT_START,
            self.name,
            input_text=input_text
        )

        try:
            messages = self._build_messages(input_text)

            # LLM stream call
            full_response = ""
            async for chunk in self.llm.astream_invoke(messages, **kwargs):
                full_response += chunk

                # Send LLM output chunk
                yield StreamEvent.create(
                    StreamEventType.LLM_CHUNK,
                    self.name,
                    chunk=chunk
                )

            # Send finish event
            yield StreamEvent.create(
                StreamEventType.AGENT_FINISH,
                self.name,
                result=full_response
            )

            # Save to history
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(full_response, "assistant"))

            self._finalize_trace(
                "success",
                duration=(datetime.now() - session_start_time).total_seconds(),
                final_answer=full_response,
            )

        except Exception as e:
            self._finalize_trace(
                "error",
                duration=(datetime.now() - session_start_time).total_seconds(),
                error_type=type(e).__name__,
                message=str(e),
            )

            # Send error event
            yield StreamEvent.create(
                StreamEventType.ERROR,
                self.name,
                error=str(e),
                error_type=type(e).__name__
            )
            raise
