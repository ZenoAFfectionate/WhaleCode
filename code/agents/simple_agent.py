from typing import Optional, Iterator, TYPE_CHECKING, List, Dict, AsyncGenerator
import json

from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..core.message import Message
from ..core.streaming import StreamEvent, StreamEventType
from ..core.lifecycle import LifecycleHook

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
        from ..observability import TraceLogger

        session_start_time = datetime.now()

        # Create a new TraceLogger for each run (avoids issues with closed files during multi-turn dialogues)
        trace_logger = None
        if self.config.trace_enabled:
            trace_logger = TraceLogger(
                output_dir=self.config.trace_dir,
                sanitize=self.config.trace_sanitize,
                html_include_raw_response=self.config.trace_html_include_raw_response
            )
            trace_logger.log_event(
                "session_start",
                {
                    "agent_name": self.name,
                    "agent_type": self.__class__.__name__,
                }
            )

        # Build the message list
        messages = self._build_messages(input_text)

        # Log user message
        if trace_logger:
            trace_logger.log_event(
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

            if trace_logger:
                duration = (datetime.now() - session_start_time).total_seconds()
                trace_logger.log_event(
                    "session_end",
                    {
                        "duration": duration,
                        "final_answer": response_text,
                        "status": "success",
                        "usage": llm_response.usage if hasattr(llm_response, 'usage') else {},
                        "latency_ms": llm_response.latency_ms if hasattr(llm_response, 'latency_ms') else 0
                    }
                )
                trace_logger.finalize()

            return response_text

        # Enable tool calling mode
        tool_schemas = self._build_tool_schemas()

        current_iteration = 0
        final_response = ""

        while current_iteration < self.max_tool_iterations:
            current_iteration += 1

            # Call LLM (Function Calling)
            try:
                response = self.llm.invoke_with_tools(
                    messages=messages,
                    tools=tool_schemas,
                    tool_choice="auto",
                    **kwargs
                )
            except Exception as e:
                print(f"❌ LLM call failed: {e}")
                if trace_logger:
                    trace_logger.log_event(
                        "error",
                        {"error_type": "LLM_ERROR", "message": str(e)},
                        step=current_iteration
                    )
                break

            # Get the response message
            response_message = response.choices[0].message

            # Log model output
            if trace_logger:
                usage = response.usage
                trace_logger.log_event(
                    "model_output",
                    {
                        "content": response_message.content,
                        "tool_calls": len(response_message.tool_calls) if response_message.tool_calls else 0,
                        "usage": {
                            "prompt_tokens": usage.prompt_tokens if usage else 0,
                            "completion_tokens": usage.completion_tokens if usage else 0,
                            "total_tokens": usage.total_tokens if usage else 0
                        }
                    },
                    step=current_iteration
                )

            # Process tool calls
            tool_calls = response_message.tool_calls
            if not tool_calls:
                # No tool calls, return the text response directly
                final_response = response_message.content or "Sorry, I cannot answer this question."
                break

            # Add the assistant message to history
            messages.append({
                "role": "assistant",
                "content": response_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in tool_calls
                ]
            })

            # Execute all tool calls
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                tool_call_id = tool_call.id

                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as e:
                    print(f"❌ Tool argument parsing failed: {e}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": f"Error: Incorrect argument format - {str(e)}"
                    })
                    continue

                # Log tool call
                if trace_logger:
                    trace_logger.log_event(
                        "tool_call",
                        {
                            "tool_name": tool_name,
                            "tool_call_id": tool_call_id,
                            "args": arguments
                        },
                        step=current_iteration
                    )

                # Execute tool (reuse base class method)
                result = self._execute_tool_call(tool_name, arguments)

                # Log tool result
                if trace_logger:
                    trace_logger.log_event(
                        "tool_result",
                        {
                            "tool_name": tool_name,
                            "tool_call_id": tool_call_id,
                            "result": result
                        },
                        step=current_iteration
                    )

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result
                })

        # If the maximum number of iterations is exceeded, get the last answer
        if current_iteration >= self.max_tool_iterations and not final_response:
            llm_response = self.llm.invoke(messages, **kwargs)
            final_response = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)

        # Save to history
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_response, "assistant"))

        if trace_logger:
            duration = (datetime.now() - session_start_time).total_seconds()
            trace_logger.log_event(
                "session_end",
                {
                    "duration": duration,
                    "total_steps": current_iteration,
                    "final_answer": final_response,
                    "status": "success"
                }
            )
            trace_logger.finalize()

        return final_response

    def _build_messages(self, input_text: str) -> List[Dict[str, str]]:
        """Build a model-compatible transcript from persisted history."""
        return self.history_manager.build_llm_messages(
            system_prompt=self.system_prompt,
            latest_user_input=input_text,
        )

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
            self.tool_registry = ToolRegistry()
            self.enable_tool_calling = True

        # Use ToolRegistry's register_tool method directly
        # ToolRegistry will automatically handle tool expansion
        self.tool_registry.register_tool(tool, auto_expand=auto_expand)

    def remove_tool(self, tool_name: str) -> bool:
        """Remove a tool (convenience method)"""
        if self.tool_registry:
            return self.tool_registry.unregister_tool(tool_name)
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
        messages = self._build_messages(input_text)
        
        # Stream call to LLM
        full_response = ""
        for chunk in self.llm.stream_invoke(messages, **kwargs):
            full_response += chunk
            yield chunk
        
        # Save the complete conversation to history
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(full_response, "assistant"))

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

        except Exception as e:
            # Send error event
            yield StreamEvent.create(
                StreamEventType.ERROR,
                self.name,
                error=str(e),
                error_type=type(e).__name__
            )
            raise
