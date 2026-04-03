import json
from typing import Optional, List, Dict, TYPE_CHECKING, Any, AsyncGenerator

from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..core.message import Message
from ..core.streaming import StreamEvent, StreamEventType
from ..core.lifecycle import LifecycleHook

if TYPE_CHECKING:
    from ..tools.registry import ToolRegistry


class Planner:
    """Planner - Responsible for breaking down complex problems into simple steps (using Function Calling)"""

    def __init__(self, llm_client: HelloAgentsLLM, system_prompt: Optional[str] = None):
        self.llm_client = llm_client
        self.system_prompt = system_prompt or """You are a top-tier AI planning expert. Your task is to break down complex user problems into an action plan consisting of multiple simple steps.
Please ensure each step in the plan is an independent, executable subtask, and strictly arranged in logical order."""

    def plan(self, question: str, **kwargs) -> List[str]:
        """
        Generate execution plan (using Function Calling)

        Args:
            question: The problem to solve
            **kwargs: LLM invocation parameters

        Returns:
            List of steps
        """
        print("--- Generating plan ---")

        # Define plan generation tool
        plan_tool = {
            "type": "function",
            "function": {
                "name": "generate_plan",
                "description": "Generate a step-by-step plan to solve the problem",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "steps": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "A sequentially ordered list of execution steps"
                        }
                    },
                    "required": ["steps"]
                }
            }
        }

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Please generate a detailed execution plan for the following problem:\n\n{question}"}
        ]

        try:
            response = self.llm_client.invoke_with_tools(
                messages=messages,
                tools=[plan_tool],
                tool_choice={"type": "function", "function": {"name": "generate_plan"}},
                **kwargs
            )

            response_message = response.choices[0].message

            # Extract tool call results
            if response_message.tool_calls:
                tool_call = response_message.tool_calls[0]
                arguments = json.loads(tool_call.function.arguments)
                plan = arguments.get("steps", [])

                print(f"✅ Plan generated:")
                for i, step in enumerate(plan, 1):
                    print(f"  {i}. {step}")

                return plan
            else:
                print("❌ Model did not return a plan tool call")
                return []

        except Exception as e:
            print(f"❌ Error occurred while generating plan: {e}")
            return []

class Executor:
    """Executor - Responsible for executing step-by-step according to the plan (supports Function Calling)"""

    def __init__(
        self,
        llm_client: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        tool_registry: Optional['ToolRegistry'] = None,
        enable_tool_calling: bool = True,
        max_tool_iterations: int = 3
    ):
        self.llm_client = llm_client
        self.system_prompt = system_prompt or """You are a top-tier AI execution expert. Your task is to strictly follow the given plan and solve the problem step by step.
Please focus on solving the current step and output the final answer for that step."""
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None
        self.max_tool_iterations = max_tool_iterations

    def execute(self, question: str, plan: List[str], **kwargs) -> str:
        """
        Execute tasks according to the plan (supports Function Calling)

        Args:
            question: Original problem
            plan: Execution plan
            **kwargs: LLM invocation parameters

        Returns:
            Final answer
        """
        history = []
        final_answer = ""

        print("\n--- Executing plan ---")
        for i, step in enumerate(plan, 1):
            print(f"\n-> Executing step {i}/{len(plan)}: {step}")

            # Build context message
            context = f"""# Original Problem:
{question}

# Complete Plan:
{self._format_plan(plan)}

# Historical Steps and Results:
{self._format_history(history) if history else "None"}

# Current Step:
{step}

Please execute the current step and provide the result."""

            # Execute a single step (supports tool calling)
            response_text = self._execute_step(context, **kwargs)

            history.append({"step": step, "result": response_text})
            final_answer = response_text
            print(f"✅ Step {i} completed, result: {final_answer}")

        return final_answer

    def _format_plan(self, plan: List[str]) -> str:
        """Format plan list"""
        return "\n".join([f"{i}. {step}" for i, step in enumerate(plan, 1)])

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        """Format history records"""
        return "\n\n".join([f"Step {i}: {h['step']}\nResult: {h['result']}"
                           for i, h in enumerate(history, 1)])

    def _execute_step(self, context: str, **kwargs) -> str:
        """
        Execute a single step (supports Function Calling)

        Args:
            context: Context information
            **kwargs: Additional parameters

        Returns:
            Step execution result
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": context}
        ]

        # If tool calling is not enabled, return directly
        if not self.enable_tool_calling or not self.tool_registry:
            llm_response = self.llm_client.invoke(messages, **kwargs)
            return llm_response.content if hasattr(llm_response, 'content') else str(llm_response)

        # Enable tool calling mode
        from .simple_agent import SimpleAgent
        # Temporarily create a SimpleAgent instance to reuse tool calling logic
        temp_agent = SimpleAgent(
            name="temp_executor",
            llm=self.llm_client,
            tool_registry=self.tool_registry
        )
        tool_schemas = temp_agent._build_tool_schemas()

        current_iteration = 0

        while current_iteration < self.max_tool_iterations:
            current_iteration += 1

            try:
                response = self.llm_client.invoke_with_tools(
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
                result = temp_agent._execute_tool_call(tool_name, arguments)

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result
                })

        # If max iterations exceeded, get the last answer
        if current_iteration >= self.max_tool_iterations:
            llm_response = self.llm_client.invoke(messages, **kwargs)
            return llm_response.content if hasattr(llm_response, 'content') else str(llm_response)

        return ""

class PlanSolveAgent(Agent):
    """
    Plan and Solve Agent - An agent for decomposition planning and step-by-step execution

    This Agent can:
    1. Break down complex problems into simple steps (using Function Calling)
    2. Execute step-by-step according to the plan
    3. Maintain execution history and context
    4. Derive the final answer
    5. Support tool calling (optional)

    Particularly suitable for tasks like multi-step reasoning, math problems, complex analysis, etc.
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        planner_prompt: Optional[str] = None,
        executor_prompt: Optional[str] = None,
        tool_registry: Optional['ToolRegistry'] = None,
        enable_tool_calling: bool = True,
        max_tool_iterations: int = 3
    ):
        """
        Initialize PlanSolveAgent

        Args:
            name: Agent name
            llm: LLM instance
            system_prompt: System prompt (Agent level)
            config: Configuration object
            planner_prompt: Planner's system prompt (optional)
            executor_prompt: Executor's system prompt (optional)
            tool_registry: Tool registry (optional)
            enable_tool_calling: Whether to enable tool calling
            max_tool_iterations: Maximum tool calling iterations
        """
        # Pass tool_registry to base class
        super().__init__(
            name,
            llm,
            system_prompt,
            config,
            tool_registry=tool_registry
        )

        self.planner = Planner(self.llm, planner_prompt)
        self.executor = Executor(
            self.llm,
            executor_prompt,
            tool_registry=tool_registry,
            enable_tool_calling=enable_tool_calling,
            max_tool_iterations=max_tool_iterations
        )
    
    def run(self, input_text: str, **kwargs) -> str:
        """
        Run Plan and Solve Agent
        
        Args:
            input_text: The problem to solve
            **kwargs: Additional parameters
            
        Returns:
            Final answer
        """
        print(f"\n🤖 {self.name} started processing the problem: {input_text}")
        
        # 1. Generate plan
        plan = self.planner.plan(input_text, **kwargs)
        if not plan:
            final_answer = "Unable to generate a valid action plan, task terminated."
            print(f"\n--- Task Terminated ---\n{final_answer}")
            
            # Save to history
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(final_answer, "assistant"))
            
            return final_answer
        
        # 2. Execute plan
        final_answer = self.executor.execute(input_text, plan, **kwargs)
        print(f"\n--- Task Completed ---\nFinal Answer: {final_answer}")
        
        # Save to history
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))

        return final_answer

    async def arun_stream(
        self,
        input_text: str,
        on_start: LifecycleHook = None,
        on_finish: LifecycleHook = None,
        on_error: LifecycleHook = None,
        **kwargs
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        True streaming execution of PlanAgent

        Returns in real-time:
        - Plan generation during the planning phase
        - Each step's output during the execution phase

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
            # Phase 1: Planning
            yield StreamEvent.create(
                StreamEventType.STEP_START,
                self.name,
                phase="planning",
                description="Generate execution plan"
            )

            print(f"\n🤖 {self.name} started processing the problem: {input_text}")

            # Generate plan (synchronous method, kept for now)
            plan = self.planner.plan(input_text, **kwargs)

            if not plan:
                error_msg = "Unable to generate a valid action plan, task terminated."

                yield StreamEvent.create(
                    StreamEventType.ERROR,
                    self.name,
                    error=error_msg,
                    phase="planning"
                )

                yield StreamEvent.create(
                    StreamEventType.AGENT_FINISH,
                    self.name,
                    result=error_msg
                )

                self.add_message(Message(input_text, "user"))
                self.add_message(Message(error_msg, "assistant"))
                return

            yield StreamEvent.create(
                StreamEventType.STEP_FINISH,
                self.name,
                phase="planning",
                plan=plan,
                total_steps=len(plan)
            )

            # Phase 2: Execute plan
            step_results = []

            for i, step_description in enumerate(plan):
                step_num = i + 1

                # Step start
                yield StreamEvent.create(
                    StreamEventType.STEP_START,
                    self.name,
                    phase="execution",
                    step=step_num,
                    total_steps=len(plan),
                    description=step_description
                )

                print(f"\n--- Step {step_num}/{len(plan)} ---")
                print(f"📋 {step_description}")

                # Build execution prompt
                context = "\n".join([
                    f"Step {j+1}: {plan[j]} -> {step_results[j]}"
                    for j in range(len(step_results))
                ])

                prompt = f"""Original Problem: {input_text}

Complete Plan:
{chr(10).join([f"{j+1}. {s}" for j, s in enumerate(plan)])}

Completed Steps:
{context if context else "None"}

Current Step: {step_description}

Please execute the current step and provide the result."""

                messages = [{"role": "user", "content": prompt}]

                # Stream execute step
                step_result = ""
                async for chunk in self.llm.astream_invoke(messages, **kwargs):
                    step_result += chunk

                    yield StreamEvent.create(
                        StreamEventType.LLM_CHUNK,
                        self.name,
                        chunk=chunk,
                        phase="execution",
                        step=step_num
                    )

                    print(chunk, end="", flush=True)

                print()  # Newline

                step_results.append(step_result)

                # Step finish
                yield StreamEvent.create(
                    StreamEventType.STEP_FINISH,
                    self.name,
                    phase="execution",
                    step=step_num,
                    result=step_result
                )

            # Generate final answer
            yield StreamEvent.create(
                StreamEventType.STEP_START,
                self.name,
                phase="final_answer",
                description="Generate final answer"
            )

            final_prompt = f"""Original Problem: {input_text}

Execution Plan and Results:
{chr(10).join([f"{i+1}. {plan[i]} -> {step_results[i]}" for i in range(len(plan))])}

Please provide the final answer to the original problem based on the execution results of the steps above."""

            final_messages = [{"role": "user", "content": final_prompt}]

            final_answer = ""
            async for chunk in self.llm.astream_invoke(final_messages, **kwargs):
                final_answer += chunk

                yield StreamEvent.create(
                    StreamEventType.LLM_CHUNK,
                    self.name,
                    chunk=chunk,
                    phase="final_answer"
                )

            # Send finish event
            yield StreamEvent.create(
                StreamEventType.AGENT_FINISH,
                self.name,
                result=final_answer,
                total_steps=len(plan)
            )

            print(f"\n--- Task Completed ---\nFinal Answer: {final_answer}")

            # Save to history
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(final_answer, "assistant"))

        except Exception as e:
            # Send error event
            yield StreamEvent.create(
                StreamEventType.ERROR,
                self.name,
                error=str(e),
                error_type=type(e).__name__
            )
            raise
