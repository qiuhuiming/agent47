"""
AGENT ARCHITECTURE DIAGRAM
=========================

┌─────────────────────────────────────────────────────────────────────────────┐
│                              AGENT SYSTEM FLOW                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐
│   USER      │
│   TASK      │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            AGENT.RUN(task)                                   │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                        INITIALIZATION                               │     │
│  │  • Create AgentState(task)                                        │     │
│  │  • Set iteration_count = 0                                        │     │
│  │  • Initialize empty lists: thoughts, actions, observations        │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                          REACT LOOP                                │     │
│  │  ┌──────────────────────────────────────────────────────────┐    │     │
│  │  │  WHILE iteration_count < max_iterations:                 │    │     │
│  │  │                                                           │    │     │
│  │  │    1. THINK PHASE                                       │    │     │
│  │  │    ┌─────────────────────────────────────┐              │    │     │
│  │  │    │  _think(state)                      │              │    │     │
│  │  │    │  • Build prompt with history        │              │    │     │
│  │  │    │  • Call LLM.complete()              │              │    │     │
│  │  │    │  • Return thought string            │              │    │     │
│  │  │    └─────────────────┬───────────────────┘              │    │     │
│  │  │                      │                                   │    │     │
│  │  │                      ▼                                   │    │     │
│  │  │    ┌─────────────────────────────────────┐              │    │     │
│  │  │    │  CHECK: "Final Answer:" in thought? │              │    │     │
│  │  │    └────────┬──────────────┬─────────────┘              │    │     │
│  │  │             │ YES          │ NO                          │    │     │
│  │  │             ▼              ▼                             │    │     │
│  │  │    ┌──────────────┐   2. ACT PHASE                     │    │     │
│  │  │    │ Extract      │   ┌─────────────────────────────┐  │    │     │
│  │  │    │ final answer │   │  _decide_action(state)      │  │    │     │
│  │  │    │ BREAK LOOP   │   │  • Use last thought         │  │    │     │
│  │  │    └──────────────┘   │  • Ask LLM for JSON action  │  │    │     │
│  │  │                       │  • Parse: {tool, input}     │  │    │     │
│  │  │                       └─────────────┬───────────────┘  │    │     │
│  │  │                                     │                   │    │     │
│  │  │                                     ▼                   │    │     │
│  │  │                       3. OBSERVE PHASE                  │    │     │
│  │  │                       ┌─────────────────────────────┐  │    │     │
│  │  │                       │  _execute_action(action)    │  │    │     │
│  │  │                       │  • Validate tool exists     │  │    │     │
│  │  │                       │  • Execute tool.execute()   │  │    │     │
│  │  │                       │  • Return result/error      │  │    │     │
│  │  │                       └─────────────┬───────────────┘  │    │     │
│  │  │                                     │                   │    │     │
│  │  │                                     ▼                   │    │     │
│  │  │                       ┌─────────────────────────────┐  │    │     │
│  │  │                       │  UPDATE STATE               │  │    │     │
│  │  │                       │  • Append thought           │  │    │     │
│  │  │                       │  • Append action            │  │    │     │
│  │  │                       │  • Append observation       │  │    │     │
│  │  │                       │  • Increment iteration      │  │    │     │
│  │  │                       └─────────────────────────────┘  │    │     │
│  │  │                                     │                   │    │     │
│  │  │                                     └──── LOOP ─────────┘    │     │
│  │  └──────────────────────────────────────────────────────────┘    │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                         FINALIZATION                               │     │
│  │  • If no final_answer: set default message                       │     │
│  │  • Log completion stats                                          │     │
│  │  • Return final_answer                                           │     │
│  └────────────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────┐
│   RESULT    │
└─────────────┘

ERROR HANDLING FLOW
==================
                    ┌─────────────────┐
                    │ Any Exception   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Log Error       │
                    │ Return Error    │
                    │ Message         │
                    └─────────────────┘

DATA STRUCTURES
==============

AgentState:
┌──────────────────────────────┐
│ • task: str                  │
│ • thoughts: List[str]        │
│ • actions: List[Dict]        │
│ • observations: List[str]    │
│ • final_answer: Optional[str]│
│ • iteration_count: int       │
└──────────────────────────────┘

Agent:
┌──────────────────────────────┐
│ • llm: LLMClient             │
│ • tools: Dict[str, Tool]     │
│ • max_iterations: int        │
└──────────────────────────────┘
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog

from src.llm_client import LLMClient
from src.tools.base import Tool, ToolResult

logger = structlog.get_logger(__name__)


@dataclass
class AgentState:
    """Track agent execution state"""

    task: str
    thoughts: List[str] = None
    actions: List[Dict[str, Any]] = None
    observations: List[str] = None
    final_answer: Optional[str] = None
    iteration_count: int = 0

    def __post_init__(self):
        self.thoughts = self.thoughts or []
        self.actions = self.actions or []
        self.observations = self.observations or []


class Agent:
    """Production-ready ReAct agent"""

    # Class constants for prefixes
    FINAL_ANSWER_PREFIX = "Final Answer:"

    def __init__(
        self, llm_client: LLMClient, tools: List[Tool], max_iterations: int = 10
    ):
        self.llm = llm_client
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations
        logger.info("Agent initialized", tools=list(self.tools.keys()))

    def run(self, task: str) -> str:
        """Execute task using ReAct loop"""
        logger.info("Starting task", task=task)
        state = AgentState(task=task)

        try:
            while state.iteration_count < self.max_iterations:
                state.iteration_count += 1

                # Think
                thought = self._think(state)
                state.thoughts.append(thought)

                # Check if we have final answer
                if self.FINAL_ANSWER_PREFIX in thought:
                    # Extract everything after "Final Answer:"
                    final_answer_idx = thought.find(self.FINAL_ANSWER_PREFIX)
                    state.final_answer = thought[final_answer_idx + len(self.FINAL_ANSWER_PREFIX):].strip()
                    break

                # Act
                action = self._decide_action(state)
                state.actions.append(action)

                # Observe
                observation = self._execute_action(action)
                state.observations.append(observation)

            if not state.final_answer:
                state.final_answer = (
                    "I couldn't complete the task within the iteration limit."
                )

            logger.info(
                "Task completed",
                iterations=state.iteration_count,
                answer_preview=state.final_answer[:100],
            )
            return state.final_answer

        except Exception as e:
            logger.error("Agent error", error=str(e), exc_info=True)
            return f"An error occurred: {str(e)}"

    def _think(self, state: AgentState) -> str:
        """Generate next thought based on current state"""
        prompt = self._build_prompt(state)
        system_prompt = self._get_system_prompt()
        
        # Add detailed thinking phase guidance
        prompt += f"""

═══════════════════════════════════════════════════════════════════
THINKING PHASE - REFLECTION QUESTIONS
═══════════════════════════════════════════════════════════════════

Before deciding your next action, consider these questions:

1. **Progress Assessment**:
   - What specific progress have I made toward solving the task?
   - What key information have I gathered so far?
   - How close am I to having a complete answer?

2. **Next Step Planning**:
   - What information am I still missing?
   - Which tool would be most helpful right now?
   - Is there a more efficient approach I should consider?

3. **Solution Readiness**:
   - Do I have all the necessary information to answer the original question?
   - Have I completed all required calculations or lookups?
   - Is my answer complete and accurate?

4. **Quality Check**:
   - Does my current approach make logical sense?
   - Am I avoiding unnecessary or redundant actions?
   - Have I validated my results where possible?

Remember: If you have obtained the final result and completed all necessary steps, begin your response with "{self.FINAL_ANSWER_PREFIX}" followed by the complete solution.

What should I do next?"""

        logger.debug("Calling LLM for thinking phase", 
                    prompt_length=len(prompt),
                    system_prompt_length=len(system_prompt),
                    iteration=state.iteration_count)
        
        response = self.llm.complete(
            prompt,
            stop_sequences=None,  # Don't use stop sequences
            system_prompt=system_prompt,
        )

        logger.debug("LLM raw response", 
                    response=repr(response),
                    response_length=len(response) if response else 0,
                    is_none=response is None,
                    is_empty=response == "" if response is not None else False)
        
        thought = response.strip() if response else ""
        logger.debug("Generated thought", 
                    thought=repr(thought),
                    thought_length=len(thought))
        return thought

    def _decide_action(self, state: AgentState) -> Dict[str, Any]:
        """Decide which action to take"""
        # Get the last thought and extract action
        last_thought = state.thoughts[-1]

        # System prompt for action phase
        action_system_prompt = """You are in the ACT phase of the ReAct loop. Based on your recent thinking, you need to decide which tool to use.

You must respond with ONLY a JSON object that specifies which tool to use and what input to provide.

## Required Format:
{"tool": "<tool_name>", "input": "<input_for_the_tool>"}

## Valid Examples:
{"tool": "calculator", "input": "25 * 4 + 10"}
{"tool": "calculator", "input": "1200 * 0.25"}
{"tool": "web_search", "input": "current USD to EUR exchange rate"}
{"tool": "file_reader", "input": "/path/to/file.txt"}

## Common Mistakes to AVOID:
❌ {"tool": "calculator", "input": 25 * 4}  // Input must be a string
❌ {"tool": "Calculator", "input": "25*4"}  // Tool names are case-sensitive
❌ {"tool": "calc", "input": "25*4"}  // Use exact tool names
❌ {"action": "calculator", "value": "25*4"}  // Wrong field names
❌ {"tool": "calculator", "input": "25*4", "reason": "need to calculate"}  // No extra fields
❌ Let me use the calculator: {"tool": "calculator", "input": "25*4"}  // No text before/after JSON

## Validation Checklist:
✓ Tool name exactly matches one from the available tools list
✓ Input is a properly quoted string
✓ JSON uses double quotes (") not single quotes (')
✓ No trailing commas
✓ No extra fields beyond "tool" and "input"
✓ No explanatory text - ONLY the JSON object

## Tool-Specific Input Formats:
- calculator: Mathematical expressions as strings (e.g., "5 + 3", "10 * 2.5", "sqrt(16)")
- web_search: Search queries as strings (e.g., "Python datetime examples")
- file operations: File paths as strings (e.g., "/home/user/data.txt")

Remember: Output ONLY the JSON object, nothing else."""

        # User prompt with context
        action_prompt = f"""Based on this thought: "{last_thought}"

Available tools:
{self._format_tools_list()}

Decide which tool to use:"""

        response = self.llm.complete(action_prompt, system_prompt=action_system_prompt)

        try:
            action = json.loads(response)
            logger.debug("Parsed action", action=action)
            return action
        except json.JSONDecodeError:
            logger.warning("Failed to parse action", response=response)
            return {"tool": "none", "input": ""}

    def _execute_action(self, action: Dict[str, Any]) -> str:
        """Execute the chosen action"""
        tool_name = action.get("tool")
        tool_input = action.get("input", "")

        if tool_name not in self.tools:
            return f"Error: Unknown tool '{tool_name}'"

        try:
            tool = self.tools[tool_name]
            result = tool.execute(tool_input)
            logger.info("Tool executed", tool=tool_name, success=result.success)
            return result.output
        except Exception as e:
            logger.error("Tool execution failed", tool=tool_name, error=str(e))
            return f"Error executing {tool_name}: {str(e)}"

    def _build_prompt(self, state: AgentState) -> str:
        """Build prompt for next thought"""
        # Calculate progress
        tools_used = [action.get("tool", "unknown") for action in state.actions]
        progress_percent = min((state.iteration_count / self.max_iterations) * 100, 100)
        
        # Build progress header
        prompt = f"""═══════════════════════════════════════════════════════════════════
TASK EXECUTION STATUS
═══════════════════════════════════════════════════════════════════
Iteration: {state.iteration_count} of {self.max_iterations} | Progress: {progress_percent:.0f}% | Tools Used: {tools_used if tools_used else ['none yet']}
═══════════════════════════════════════════════════════════════════

Task: {state.task}

Available Tools:
{self._format_tools_list()}

═══════════════════════════════════════════════════════════════════
EXECUTION HISTORY
═══════════════════════════════════════════════════════════════════
"""

        # Add execution history with better formatting
        if not state.thoughts:
            prompt += "No actions taken yet.\n"
        else:
            for i, (thought, action, observation) in enumerate(
                zip(state.thoughts, state.actions, state.observations)
            ):
                prompt += f"\n▶ Step {i+1}:"
                prompt += f"\n  Thought: {thought}"
                prompt += f"\n  Action: {json.dumps(action) if isinstance(action, dict) else action}"
                prompt += f"\n  Result: {observation}\n"

        prompt += """
═══════════════════════════════════════════════════════════════════
NEXT STEP GUIDANCE
═══════════════════════════════════════════════════════════════════
"""

        # Add context-aware guidance based on state
        if len(state.thoughts) == 0:
            prompt += f"""
This is your first step. Follow the thinking template:
1. Analyze what the task is asking for
2. Identify what information or calculations are needed
3. Determine which tool can help you start
4. Plan your approach step by step

Thought {state.iteration_count}: Let me analyze this task. """
        else:
            # Get last observation for context
            last_observation = state.observations[-1] if state.observations else ""
            
            # Provide specific guidance based on last observation
            if "error" in last_observation.lower():
                prompt += f"""
The last action resulted in an error. Consider:
- What caused the error?
- How can you fix or work around it?
- Is there an alternative approach?

Thought {state.iteration_count}: The previous action failed with an error. """
            elif "success" in last_observation.lower() or last_observation:
                prompt += f"""
The last action was successful. Now consider:
- What information did you gain?
- Are you closer to solving the task?
- What's the next logical step?
- Do you have enough information for the final answer?

Thought {state.iteration_count}: Based on the previous result, """
            else:
                prompt += f"""
Continue with your analysis:
- Review what you've learned so far
- Identify what's still needed
- Choose the next appropriate action

Thought {state.iteration_count}: Continuing from the previous step, """

        # Add reminder about final answer if close to limit
        if state.iteration_count >= self.max_iterations - 2:
            prompt += f"""

⚠️ APPROACHING ITERATION LIMIT ({state.iteration_count}/{self.max_iterations}) ⚠️
Consider whether you have enough information to provide a final answer.
"""

        return prompt

    def _format_tools_list(self) -> str:
        """Format tools with descriptions for the prompt"""
        tool_descriptions = []
        for name, tool in self.tools.items():
            tool_descriptions.append(f"- {name}: {tool.description}")
        return "\n".join(tool_descriptions)

    def _get_system_prompt(self) -> str:
        """Get the system prompt with ReAct framework instructions"""
        return f"""You are an AI agent using the ReAct (Reason + Act) framework to solve tasks systematically.

## How You Work:
You operate in a loop of three phases:
1. **THINK**: Analyze the current situation, reason about what needs to be done, and plan your approach
2. **ACT**: Decide which tool to use based on your thinking
3. **OBSERVE**: Review the results and incorporate them into your next thought

## Structured Thinking Template:
When thinking, follow this structure:
1. **Current Status**: What do I know so far? What information have I gathered?
2. **Goal Analysis**: What am I trying to achieve? What's the specific question to answer?
3. **Gap Assessment**: What information am I missing? What calculations need to be done?
4. **Next Steps**: What's the logical next action? Which tool will help me progress?
5. **Progress Check**: Am I closer to the solution? Do I have enough to answer?

## Chain-of-Thought Examples:
Good reasoning: "I need to calculate 25% of 1200. This requires multiplication: 1200 * 0.25. I'll use the calculator tool."
Poor reasoning: "I'll just try the calculator with some numbers."

Good reasoning: "The previous calculation gave me the base amount. Now I need to add the 15% tax to get the total."
Poor reasoning: "Let me calculate something else now."

## Tool Selection Decision Tree:
- Need to perform math? → Use calculator
- Need external information? → Use appropriate search/lookup tool
- Need to process data? → Use the relevant data processing tool
- Already have all information? → Provide final answer

## Self-Correction Patterns:
- If a tool returns an error: Analyze why it failed and adjust your input
- If a result seems wrong: Double-check your approach and recalculate
- If you're stuck: Step back and reconsider your strategy
- If you're repeating actions: You might already have the answer

## Final Answer Guidelines:
- When you have obtained all necessary information and calculated the final result, you MUST begin your response with "{self.FINAL_ANSWER_PREFIX}" followed by the complete solution
- The final answer should directly address the original task/question
- Include all relevant details and calculations that led to your conclusion
- Be clear and concise in your final answer

## Important Reminders:
- Each thought should show clear reasoning and progress
- Don't continue if you already have the answer
- Always validate your results make sense in context
- If uncertain, think through the problem again before acting
"""
