"""Reflection Agent Implementation - An agent for self-reflection and iterative optimization"""

from typing import Optional, List, Dict, Any, TYPE_CHECKING, AsyncGenerator
import json
from datetime import datetime

from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..core.message import Message
from ..core.streaming import StreamEvent, StreamEventType
from ..core.lifecycle import LifecycleHook

if TYPE_CHECKING:
    from ..tools.registry import ToolRegistry

class Memory:
    """
    A simple short-term memory module for storing the agent's actions and reflection trajectory.
    """
    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def add_record(self, record_type: str, content: str):
        """Add a new record to memory"""
        self.records.append({"type": record_type, "content": content})
        print(f"📝 Memory updated, added a new '{record_type}' record.")

    def get_trajectory(self) -> str:
        """Format all memory records into a coherent string text"""
        trajectory = ""
        for record in self.records:
            if record['type'] == 'execution':
                trajectory += f"--- Previous Attempt (Code) ---\n{record['content']}\n\n"
            elif record['type'] == 'reflection':
                trajectory += f"--- Reviewer Feedback ---\n{record['content']}\n\n"
        return trajectory.strip()

    def get_last_execution(self) -> str:
        """Get the result of the most recent execution"""
        for record in reversed(self.records):
            if record['type'] == 'execution':
                return record['content']
        return ""

class ReflectionAgent(Agent):
    """
    Reflection Agent - An agent for self-reflection and iterative optimization

    This Agent can:
    1. Execute the initial task
    2. Self-reflect on the results
    3. Optimize based on the reflection
    4. Iteratively improve until satisfactory
    5. Support tool calling (optional)

    Particularly suitable for tasks requiring iterative optimization like code generation, document writing, analytical reports, etc.

    Uses standard Function Calling format, defining roles and behaviors through the system_prompt.
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_iterations: int = 3,
        tool_registry: Optional['ToolRegistry'] = None,
        enable_tool_calling: bool = True,
        max_tool_iterations: int = 3
    ):
        """
        Initialize ReflectionAgent

        Args:
            name: Agent name
            llm: LLM instance
            system_prompt: System prompt (defines role and reflection strategy)
            config: Configuration object
            max_iterations: Maximum number of iterations
            tool_registry: Tool registry (optional)
            enable_tool_calling: Whether to enable tool calling
            max_tool_iterations: Maximum tool calling iterations
        """
        # Default system_prompt
        default_system_prompt = """You are an AI assistant with self-reflection capabilities. Your workflow is:
1. First, attempt to complete the user's task.
2. Then, reflect on your answer to identify potential issues or room for improvement.
3. Optimize your answer based on the reflection results.
4. If the answer is already excellent, reply "no need for improvement" during reflection.

Please always maintain critical thinking and strive for higher quality output."""

        # Pass tool_registry to base class
        super().__init__(
            name,
            llm,
            system_prompt or default_system_prompt,
            config,
            tool_registry=tool_registry
        )
        self.max_iterations = max_iterations
        self.memory = Memory()
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None
        self.max_tool_iterations = max_tool_iterations

    def run(self, input_text: str, **kwargs) -> str:
        """
        Run the Reflection Agent

        Args:
            input_text: Task description
            **kwargs: Additional parameters

        Returns:
            The final optimized result
        """
        print(f"\n🤖 {self.name} started processing task: {input_text}")

        # Reset memory
        self.memory = Memory()

        # 1. Initial execution
        print("\n--- Making initial attempt ---")
        initial_result = self._execute_task(input_text, **kwargs)
        self.memory.add_record("execution", initial_result)

        # 2. Iteration loop: reflection and optimization
        for i in range(self.max_iterations):
            print(f"\n--- Iteration {i+1}/{self.max_iterations} ---")

            # a. Reflection
            print("\n-> Reflecting...")
            last_result = self.memory.get_last_execution()
            feedback = self._reflect_on_result(input_text, last_result, **kwargs)
            self.memory.add_record("reflection", feedback)

            # b. Check if stopping is needed
            if "no need for improvement" in feedback.lower():
                print("\n✅ Reflection indicates no need for improvement, task completed.")
                break

            # c. Optimization
            print("\n-> Optimizing...")
            refined_result = self._refine_result(input_text, last_result, feedback, **kwargs)
            self.memory.add_record("execution", refined_result)

        final_result = self.memory.get_last_execution()
        print(f"\n--- Task Completed ---\nFinal Result:\n{final_result}")

        # Save to history
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_result, "assistant"))

        return final_result

    def _execute_task(self, task: str, **kwargs) -> str:
        """Execute the initial task"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Please complete the following task:\n\n{task}"}
        ]
        return self._get_llm_response(messages, **kwargs)

    def _reflect_on_result(self, task: str, result: str, **kwargs) -> str:
        """Reflect on the result"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""Please carefully review the following answer and identify potential issues or areas for improvement:

# Original Task:
{task}

# Current Answer:
{result}

Please analyze the quality of this answer, point out shortcomings, and provide specific suggestions for improvement.
If the answer is already excellent, please reply "no need for improvement"."""}
        ]
        return self._get_llm_response(messages, **kwargs)

    def _refine_result(self, task: str, last_attempt: str, feedback: str, **kwargs) -> str:
        """Optimize the result based on feedback"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""Please improve your answer based on the feedback:

# Original Task:
{task}

# Previous Answer:
{last_attempt}

# Feedback:
{feedback}

Please provide an improved answer."""}
        ]
        return self._get_llm_response(messages, **kwargs)

    def _get_llm_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Call the LLM and get the full response (supports Function Calling)

        Args:
            messages: List of messages
            **kwargs: Additional parameters

        Returns:
            LLM response text
        """
        # If tool calling is not enabled, return directly
        if not self.enable_tool_calling or not self.tool_registry:
            llm_response = self.llm.invoke(messages, **kwargs)
            return llm_response.content if hasattr(llm_response, 'content') else str(llm_response)

        # Enable tool calling mode
        tool_schemas = self._build_tool_schemas()
        current_iteration = 0

        while current_iteration < self.max_tool_iterations:
            current_iteration += 1

            try:
                response = self.llm.invoke_with_tools(
                    messages=messages,
                    tools=tool_schemas,
                    tool_choice="auto",
                    **kwargs
                )
            except Exception as e:
                print(f"❌ LLM call failed: {e}")
                break

            response_message = response.choices[0].message

            # Process tool calls
            tool_calls = response_message.tool_calls
            if not tool_calls:
                # No tool calls, return text response
                return response_message.content or ""

            # Add assistant message to history
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

                # Execute tool (reuse base class method)
                result = self._execute_tool_call(tool_name, arguments)

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result
                })

        # If max iterations exceeded, get the last answer
        if current_iteration >= self.max_tool_iterations:
            llm_response = self.llm.invoke(messages, **kwargs)
            return llm_response.content if hasattr(llm_response, 'content') else str(llm_response)

        return ""

    async def arun_stream(
        self,
        input_text: str,
        on_start: LifecycleHook = None,
        on_finish: LifecycleHook = None,
        on_error: LifecycleHook = None,
        **kwargs
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        True streaming execution of ReflectionAgent

        Returns in real-time:
        - LLM output during the initial execution phase
        - Reflection output during the critique phase
        - LLM output during the optimization phase

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
            # Phase 1: Initial execution
            yield StreamEvent.create(
                StreamEventType.STEP_START,
                self.name,
                phase="initial_execution",
                description="Generate initial answer"
            )

            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})

            for msg in self._history:
                messages.append({"role": msg.role, "content": msg.content})

            messages.append({"role": "user", "content": input_text})

            # Stream initial answer
            initial_response = ""
            async for chunk in self.llm.astream_invoke(messages, **kwargs):
                initial_response += chunk
                yield StreamEvent.create(
                    StreamEventType.LLM_CHUNK,
                    self.name,
                    chunk=chunk,
                    phase="execution"
                )

            yield StreamEvent.create(
                StreamEventType.STEP_FINISH,
                self.name,
                phase="initial_execution",
                result=initial_response
            )

            # Phase 2: Reflection and optimization loop
            current_response = initial_response

            for iteration in range(self.max_iterations):
                # Reflection phase
                yield StreamEvent.create(
                    StreamEventType.STEP_START,
                    self.name,
                    phase="reflection",
                    iteration=iteration + 1,
                    description=f"Reflection {iteration + 1}"
                )

                reflection_prompt = self._build_reflection_prompt(input_text, current_response)
                reflection_messages = [{"role": "user", "content": reflection_prompt}]

                reflection = ""
                async for chunk in self.llm.astream_invoke(reflection_messages, **kwargs):
                    reflection += chunk
                    yield StreamEvent.create(
                        StreamEventType.THINKING,
                        self.name,
                        chunk=chunk,
                        phase="reflection",
                        iteration=iteration + 1
                    )

                yield StreamEvent.create(
                    StreamEventType.STEP_FINISH,
                    self.name,
                    phase="reflection",
                    iteration=iteration + 1,
                    reflection=reflection
                )

                # Optimization phase
                yield StreamEvent.create(
                    StreamEventType.STEP_START,
                    self.name,
                    phase="refinement",
                    iteration=iteration + 1,
                    description=f"Optimization {iteration + 1}"
                )

                refinement_prompt = self._build_refinement_prompt(
                    input_text,
                    current_response,
                    reflection
                )
                refinement_messages = [{"role": "user", "content": refinement_prompt}]

                refined_response = ""
                async for chunk in self.llm.astream_invoke(refinement_messages, **kwargs):
                    refined_response += chunk
                    yield StreamEvent.create(
                        StreamEventType.LLM_CHUNK,
                        self.name,
                        chunk=chunk,
                        phase="refinement",
                        iteration=iteration + 1
                    )

                yield StreamEvent.create(
                    StreamEventType.STEP_FINISH,
                    self.name,
                    phase="refinement",
                    iteration=iteration + 1,
                    result=refined_response
                )

                current_response = refined_response

            # Send finish event
            yield StreamEvent.create(
                StreamEventType.AGENT_FINISH,
                self.name,
                result=current_response,
                total_iterations=self.max_iterations
            )

            # Save to history
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(current_response, "assistant"))

        except Exception as e:
            # Send error event
            yield StreamEvent.create(
                StreamEventType.ERROR,
                self.name,
                error=str(e),
                error_type=type(e).__name__
            )
            raise
