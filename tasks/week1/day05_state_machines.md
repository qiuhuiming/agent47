# Day 5: State Machines for Agents

## 🎯 Objectives
1. Implement agent loops as proper state machines
2. Build state persistence and recovery mechanisms
3. Create visual state debugging and monitoring
4. Design complex multi-agent state coordination
5. Implement rollback and state history tracking

## 📋 Prerequisites
- Completed Days 1-4
- Understanding of state machine concepts
- Basic knowledge of event-driven programming
- Familiarity with asyncio

## 🌄 Morning Tasks (60-75 minutes)

### Task 1: State Machine Foundation (25 min)
Create `src/state/core.py`:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Callable, TypeVar
from enum import Enum, auto
from datetime import datetime
import asyncio
import structlog

logger = structlog.get_logger()

T = TypeVar('T')

class StateType(Enum):
    """Types of states in the system"""
    INITIAL = auto()
    PROCESSING = auto()
    WAITING = auto()
    ERROR = auto()
    FINAL = auto()
    CHECKPOINT = auto()

@dataclass
class StateTransition:
    """Record of a state transition"""
    from_state: str
    to_state: str
    trigger: str
    timestamp: datetime
    data: Optional[Dict[str, Any]] = None
    success: bool = True
    error: Optional[str] = None

@dataclass
class StateContext:
    """Context carried through state transitions"""
    data: Dict[str, Any] = field(default_factory=dict)
    history: List[StateTransition] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from context"""
        return self.data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set value in context"""
        self.data[key] = value
    
    def update(self, **kwargs) -> None:
        """Update multiple values"""
        self.data.update(kwargs)
    
    def add_transition(self, transition: StateTransition) -> None:
        """Add transition to history"""
        self.history.append(transition)
    
    def get_last_transition(self) -> Optional[StateTransition]:
        """Get most recent transition"""
        return self.history[-1] if self.history else None

class State(ABC):
    """Base class for states"""
    
    def __init__(
        self,
        name: str,
        state_type: StateType = StateType.PROCESSING
    ):
        self.name = name
        self.state_type = state_type
        self.entry_callbacks: List[Callable] = []
        self.exit_callbacks: List[Callable] = []
        
    @abstractmethod
    async def execute(self, context: StateContext) -> Optional[str]:
        """
        Execute state logic and return next state name.
        Return None to stay in current state.
        """
        pass
    
    async def on_enter(self, context: StateContext) -> None:
        """Called when entering the state"""
        logger.info(f"Entering state: {self.name}")
        
        for callback in self.entry_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(context)
                else:
                    callback(context)
            except Exception as e:
                logger.error(f"Entry callback error in {self.name}: {e}")
    
    async def on_exit(self, context: StateContext) -> None:
        """Called when exiting the state"""
        logger.info(f"Exiting state: {self.name}")
        
        for callback in self.exit_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(context)
                else:
                    callback(context)
            except Exception as e:
                logger.error(f"Exit callback error in {self.name}: {e}")
    
    def add_entry_callback(self, callback: Callable) -> None:
        """Add callback for state entry"""
        self.entry_callbacks.append(callback)
    
    def add_exit_callback(self, callback: Callable) -> None:
        """Add callback for state exit"""
        self.exit_callbacks.append(callback)

class StateMachine:
    """Generic state machine implementation"""
    
    def __init__(self, name: str):
        self.name = name
        self.states: Dict[str, State] = {}
        self.transitions: Dict[str, Dict[str, str]] = {}
        self.current_state: Optional[str] = None
        self.initial_state: Optional[str] = None
        self.final_states: Set[str] = set()
        self.context: StateContext = StateContext()
        self.running = False
        self.transition_callbacks: List[Callable] = []
        
    def add_state(
        self,
        state: State,
        initial: bool = False,
        final: bool = False
    ) -> None:
        """Add a state to the machine"""
        self.states[state.name] = state
        
        if initial:
            if self.initial_state:
                raise ValueError(f"Initial state already set: {self.initial_state}")
            self.initial_state = state.name
            
        if final:
            self.final_states.add(state.name)
            
        # Initialize transitions for this state
        if state.name not in self.transitions:
            self.transitions[state.name] = {}
            
        logger.info(f"Added state: {state.name} (initial={initial}, final={final})")
    
    def add_transition(
        self,
        from_state: str,
        to_state: str,
        trigger: str
    ) -> None:
        """Add a transition between states"""
        if from_state not in self.states:
            raise ValueError(f"Unknown state: {from_state}")
        if to_state not in self.states:
            raise ValueError(f"Unknown state: {to_state}")
            
        if from_state not in self.transitions:
            self.transitions[from_state] = {}
            
        self.transitions[from_state][trigger] = to_state
        logger.info(f"Added transition: {from_state} --{trigger}--> {to_state}")
    
    def add_transition_callback(self, callback: Callable) -> None:
        """Add callback for state transitions"""
        self.transition_callbacks.append(callback)
    
    async def start(self, context: Optional[StateContext] = None) -> Any:
        """Start the state machine"""
        if not self.initial_state:
            raise ValueError("No initial state defined")
            
        if context:
            self.context = context
            
        self.current_state = self.initial_state
        self.running = True
        
        logger.info(f"Starting state machine: {self.name}")
        
        # Enter initial state
        current = self.states[self.current_state]
        await current.on_enter(self.context)
        
        # Run state machine
        while self.running and self.current_state not in self.final_states:
            try:
                next_state = await self._execute_current_state()
                
                if next_state and next_state != self.current_state:
                    await self._transition_to(next_state)
                    
            except Exception as e:
                logger.error(f"State machine error in {self.current_state}: {e}")
                
                # Try to transition to error state if available
                if "error" in self.states:
                    await self._transition_to("error")
                else:
                    self.running = False
                    raise
        
        logger.info(f"State machine completed: {self.name}")
        return self.context
    
    async def stop(self) -> None:
        """Stop the state machine"""
        self.running = False
        
        if self.current_state:
            current = self.states[self.current_state]
            await current.on_exit(self.context)
    
    async def _execute_current_state(self) -> Optional[str]:
        """Execute current state and return next state"""
        current = self.states[self.current_state]
        return await current.execute(self.context)
    
    async def _transition_to(self, next_state: str) -> None:
        """Transition to next state"""
        if next_state not in self.states:
            raise ValueError(f"Unknown state: {next_state}")
            
        # Exit current state
        current = self.states[self.current_state]
        await current.on_exit(self.context)
        
        # Record transition
        transition = StateTransition(
            from_state=self.current_state,
            to_state=next_state,
            trigger="automatic",
            timestamp=datetime.now()
        )
        self.context.add_transition(transition)
        
        # Notify callbacks
        for callback in self.transition_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(transition, self.context)
                else:
                    callback(transition, self.context)
            except Exception as e:
                logger.error(f"Transition callback error: {e}")
        
        # Enter next state
        self.current_state = next_state
        next = self.states[next_state]
        await next.on_enter(self.context)
    
    def get_possible_transitions(self) -> List[str]:
        """Get possible transitions from current state"""
        if not self.current_state:
            return []
            
        return list(self.transitions.get(self.current_state, {}).keys())
    
    def can_transition_to(self, state: str) -> bool:
        """Check if transition to state is possible"""
        if not self.current_state:
            return False
            
        return state in self.transitions.get(self.current_state, {}).values()
    
    def visualize(self) -> str:
        """Generate visualization of state machine"""
        lines = [f"State Machine: {self.name}"]
        lines.append("=" * 40)
        
        for state_name, state in self.states.items():
            prefix = ""
            if state_name == self.initial_state:
                prefix += "[INITIAL] "
            if state_name in self.final_states:
                prefix += "[FINAL] "
            if state_name == self.current_state:
                prefix += ">>> "
                
            lines.append(f"{prefix}{state_name}")
            
            # Show transitions from this state
            if state_name in self.transitions:
                for trigger, target in self.transitions[state_name].items():
                    lines.append(f"  --{trigger}--> {target}")
        
        return "\n".join(lines)
```

### Task 2: Agent State Machine Implementation (20 min)
Create `src/state/agent_states.py`:

```python
from typing import Optional, List, Dict, Any
import asyncio
import structlog

from src.state.core import State, StateType, StateContext, StateMachine

logger = structlog.get_logger()

class AgentState(State):
    """Base class for agent states"""
    
    def __init__(self, name: str, agent=None, **kwargs):
        super().__init__(name, **kwargs)
        self.agent = agent

class InitialState(AgentState):
    """Initial state - parse and understand task"""
    
    def __init__(self, agent=None):
        super().__init__("initial", agent, state_type=StateType.INITIAL)
        
    async def execute(self, context: StateContext) -> Optional[str]:
        """Parse task and prepare for execution"""
        task = context.get("task")
        
        if not task:
            context.set("error", "No task provided")
            return "error"
        
        logger.info(f"Processing task: {task}")
        
        # Analyze task complexity
        task_analysis = await self._analyze_task(task)
        context.set("task_analysis", task_analysis)
        
        # Determine next state based on analysis
        if task_analysis["requires_planning"]:
            return "planning"
        elif task_analysis["requires_clarification"]:
            return "clarification"
        else:
            return "thinking"
    
    async def _analyze_task(self, task: str) -> Dict[str, Any]:
        """Analyze task to determine approach"""
        # Simplified analysis
        return {
            "complexity": "simple" if len(task) < 50 else "complex",
            "requires_planning": "plan" in task.lower() or "steps" in task.lower(),
            "requires_clarification": "?" in task and len(task.split()) < 5,
            "estimated_steps": 3 if len(task) < 50 else 5
        }

class PlanningState(AgentState):
    """Planning state - decompose task into steps"""
    
    def __init__(self, agent=None):
        super().__init__("planning", agent)
        
    async def execute(self, context: StateContext) -> Optional[str]:
        """Create execution plan"""
        task = context.get("task")
        
        if self.agent:
            # Use LLM to create plan
            plan_prompt = f"""
Create a step-by-step plan for this task: {task}

Return a numbered list of specific actions.
"""
            plan = await self.agent.llm.complete(plan_prompt)
            
            # Parse plan into steps
            steps = self._parse_plan(plan)
            context.set("plan", steps)
            context.set("current_step", 0)
            
            logger.info(f"Created plan with {len(steps)} steps")
        else:
            # Fallback planning
            context.set("plan", ["Think about the problem", "Execute solution"])
            context.set("current_step", 0)
        
        return "thinking"
    
    def _parse_plan(self, plan_text: str) -> List[str]:
        """Parse plan text into steps"""
        lines = plan_text.strip().split('\n')
        steps = []
        
        for line in lines:
            # Remove numbering and clean
            line = line.strip()
            if line and any(char.isalpha() for char in line):
                # Remove common numbering patterns
                import re
                cleaned = re.sub(r'^[\d\.\-\*\s]+', '', line)
                if cleaned:
                    steps.append(cleaned)
        
        return steps if steps else ["Execute task"]

class ThinkingState(AgentState):
    """Thinking state - reason about current step"""
    
    def __init__(self, agent=None):
        super().__init__("thinking", agent)
        
    async def execute(self, context: StateContext) -> Optional[str]:
        """Generate thoughts about current step"""
        task = context.get("task")
        plan = context.get("plan", [])
        current_step = context.get("current_step", 0)
        history = context.get("history", [])
        
        # Build thinking prompt
        if plan and current_step < len(plan):
            current_goal = plan[current_step]
        else:
            current_goal = task
        
        thinking_prompt = self._build_thinking_prompt(
            task, current_goal, history
        )
        
        if self.agent:
            thought = await self.agent.llm.complete(thinking_prompt)
            context.set("last_thought", thought)
            
            # Determine next action
            if "final answer" in thought.lower():
                context.set("final_answer", self._extract_final_answer(thought))
                return "finalizing"
            elif any(tool in thought.lower() for tool in ["calculate", "search", "read"]):
                return "acting"
            else:
                return "reflecting"
        else:
            # Fallback
            context.set("last_thought", "I need to solve this task")
            return "acting"
    
    def _build_thinking_prompt(
        self,
        task: str,
        current_goal: str,
        history: List[Dict]
    ) -> str:
        """Build prompt for thinking"""
        prompt = f"Task: {task}\n"
        prompt += f"Current goal: {current_goal}\n\n"
        
        if history:
            prompt += "Previous actions:\n"
            for item in history[-3:]:  # Last 3 items
                if "thought" in item:
                    prompt += f"Thought: {item['thought']}\n"
                if "action" in item:
                    prompt += f"Action: {item['action']}\n"
                if "result" in item:
                    prompt += f"Result: {item['result']}\n"
            prompt += "\n"
        
        prompt += "What should I do next? If you have the final answer, start with 'Final Answer:'"
        
        return prompt
    
    def _extract_final_answer(self, thought: str) -> str:
        """Extract final answer from thought"""
        import re
        match = re.search(r'final answer[:\s]+(.*)', thought, re.IGNORECASE)
        return match.group(1) if match else thought

class ActingState(AgentState):
    """Acting state - execute tools and actions"""
    
    def __init__(self, agent=None):
        super().__init__("acting", agent)
        
    async def execute(self, context: StateContext) -> Optional[str]:
        """Execute action based on thought"""
        thought = context.get("last_thought", "")
        
        if self.agent and self.agent.tools:
            # Parse action from thought
            action = await self._parse_action(thought)
            
            if action:
                # Execute tool
                tool_name = action.get("tool")
                tool_input = action.get("input")
                
                if tool_name in self.agent.tools:
                    result = await self.agent.tools[tool_name].execute(tool_input)
                    
                    # Store result
                    context.set("last_action", action)
                    context.set("last_result", result)
                    
                    # Add to history
                    history = context.get("history", [])
                    history.append({
                        "thought": thought,
                        "action": action,
                        "result": result.dict() if hasattr(result, 'dict') else str(result)
                    })
                    context.set("history", history)
                    
                    return "observing"
                else:
                    context.set("error", f"Unknown tool: {tool_name}")
                    return "error"
            else:
                # No clear action, need clarification
                return "clarification"
        else:
            # No tools available
            return "reflecting"
    
    async def _parse_action(self, thought: str) -> Optional[Dict[str, Any]]:
        """Parse action from thought"""
        # Try to extract tool call
        import re
        
        # Pattern: tool_name(input) or tool_name: input
        patterns = [
            r'(\w+)\((.*?)\)',
            r'(\w+):\s*(.*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, thought)
            if match:
                return {
                    "tool": match.group(1).lower(),
                    "input": match.group(2).strip()
                }
        
        return None

class ObservingState(AgentState):
    """Observing state - process action results"""
    
    def __init__(self, agent=None):
        super().__init__("observing", agent)
        
    async def execute(self, context: StateContext) -> Optional[str]:
        """Process observation from action"""
        result = context.get("last_result")
        plan = context.get("plan", [])
        current_step = context.get("current_step", 0)
        
        if result:
            # Check if result completes current step
            if hasattr(result, 'success') and result.success:
                # Move to next step if in plan
                if plan and current_step < len(plan) - 1:
                    context.set("current_step", current_step + 1)
                    return "thinking"
                else:
                    # Check if we have enough to provide final answer
                    return "reflecting"
            else:
                # Action failed, need to retry or adjust
                return "error_recovery"
        
        return "thinking"

class ReflectingState(AgentState):
    """Reflecting state - evaluate progress"""
    
    def __init__(self, agent=None):
        super().__init__("reflecting", agent)
        
    async def execute(self, context: StateContext) -> Optional[str]:
        """Reflect on progress and determine next steps"""
        history = context.get("history", [])
        task = context.get("task")
        
        if self.agent:
            # Use LLM to reflect
            reflection_prompt = self._build_reflection_prompt(task, history)
            reflection = await self.agent.llm.complete(reflection_prompt)
            
            context.set("last_reflection", reflection)
            
            # Determine next state
            if "final answer" in reflection.lower():
                context.set("final_answer", self._extract_final_answer(reflection))
                return "finalizing"
            elif "need more information" in reflection.lower():
                return "thinking"
            else:
                return "thinking"
        else:
            # Simple reflection
            if len(history) > 5:
                return "finalizing"
            else:
                return "thinking"
    
    def _build_reflection_prompt(self, task: str, history: List[Dict]) -> str:
        """Build reflection prompt"""
        prompt = f"Task: {task}\n\n"
        prompt += "Progress so far:\n"
        
        for item in history:
            if "thought" in item:
                prompt += f"- Thought: {item['thought'][:100]}...\n"
            if "result" in item:
                prompt += f"- Result: {str(item['result'])[:100]}...\n"
        
        prompt += "\nReflect on the progress. Do we have enough information to provide a final answer?"
        
        return prompt
    
    def _extract_final_answer(self, text: str) -> str:
        """Extract final answer from text"""
        import re
        match = re.search(r'final answer[:\s]+(.*)', text, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else text

class ErrorState(AgentState):
    """Error state - handle errors"""
    
    def __init__(self, agent=None):
        super().__init__("error", agent, state_type=StateType.ERROR)
        
    async def execute(self, context: StateContext) -> Optional[str]:
        """Handle error and determine recovery"""
        error = context.get("error", "Unknown error")
        error_count = context.get("error_count", 0) + 1
        context.set("error_count", error_count)
        
        logger.error(f"Agent error ({error_count}): {error}")
        
        if error_count > 3:
            # Too many errors, give up
            context.set("final_answer", f"I encountered too many errors: {error}")
            return "finalizing"
        
        # Try to recover
        return "error_recovery"

class ErrorRecoveryState(AgentState):
    """Error recovery state - attempt to recover from errors"""
    
    def __init__(self, agent=None):
        super().__init__("error_recovery", agent)
        
    async def execute(self, context: StateContext) -> Optional[str]:
        """Attempt to recover from error"""
        error = context.get("error", "")
        last_action = context.get("last_action")
        
        # Simple recovery strategies
        if "rate limit" in error.lower():
            # Wait and retry
            await asyncio.sleep(2)
            return "acting"
        elif "not found" in error.lower():
            # Try different approach
            return "thinking"
        else:
            # General recovery - go back to thinking
            return "thinking"

class FinalizingState(AgentState):
    """Finalizing state - prepare final response"""
    
    def __init__(self, agent=None):
        super().__init__("finalizing", agent, state_type=StateType.FINAL)
        
    async def execute(self, context: StateContext) -> Optional[str]:
        """Prepare final response"""
        final_answer = context.get("final_answer")
        
        if not final_answer:
            # Build final answer from history
            history = context.get("history", [])
            if history:
                final_answer = self._build_final_answer(history)
            else:
                final_answer = "I couldn't complete the task."
        
        context.set("final_answer", final_answer)
        logger.info(f"Final answer: {final_answer[:100]}...")
        
        return None  # End state
    
    def _build_final_answer(self, history: List[Dict]) -> str:
        """Build final answer from history"""
        if not history:
            return "No results available."
        
        # Get last result
        for item in reversed(history):
            if "result" in item:
                result = item["result"]
                if isinstance(result, dict) and "output" in result:
                    return result["output"]
                else:
                    return str(result)
        
        return "Task completed but no clear result found."

def create_agent_state_machine(agent=None) -> StateMachine:
    """Create a complete agent state machine"""
    sm = StateMachine("agent")
    
    # Create states
    states = [
        InitialState(agent),
        PlanningState(agent),
        ThinkingState(agent),
        ActingState(agent),
        ObservingState(agent),
        ReflectingState(agent),
        ErrorState(agent),
        ErrorRecoveryState(agent),
        FinalizingState(agent),
    ]
    
    # Add states
    for state in states:
        sm.add_state(
            state,
            initial=(state.name == "initial"),
            final=(state.name == "finalizing")
        )
    
    # Add transitions (automatic based on state execution)
    # Each state returns the next state name
    
    return sm
```

### Task 3: State Persistence (20 min)
Create `src/state/persistence.py`:

```python
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Type
from datetime import datetime
import asyncio
import aiofiles
import structlog

from src.state.core import StateContext, StateTransition, StateMachine

logger = structlog.get_logger()

class StatePersistence:
    """Handle state persistence and recovery"""
    
    def __init__(self, storage_dir: str = ".state_storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
    async def save_state(
        self,
        machine_id: str,
        state_machine: StateMachine
    ) -> None:
        """Save state machine state to disk"""
        state_data = {
            "machine_id": machine_id,
            "machine_name": state_machine.name,
            "current_state": state_machine.current_state,
            "context": self._serialize_context(state_machine.context),
            "timestamp": datetime.now().isoformat(),
            "running": state_machine.running,
        }
        
        filepath = self.storage_dir / f"{machine_id}.json"
        
        try:
            async with aiofiles.open(filepath, 'w') as f:
                await f.write(json.dumps(state_data, indent=2))
                
            logger.info(f"Saved state for {machine_id}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            raise
    
    async def load_state(
        self,
        machine_id: str,
        state_machine: StateMachine
    ) -> bool:
        """Load state machine state from disk"""
        filepath = self.storage_dir / f"{machine_id}.json"
        
        if not filepath.exists():
            logger.warning(f"No saved state found for {machine_id}")
            return False
        
        try:
            async with aiofiles.open(filepath, 'r') as f:
                content = await f.read()
                state_data = json.loads(content)
            
            # Restore state
            state_machine.current_state = state_data["current_state"]
            state_machine.context = self._deserialize_context(state_data["context"])
            state_machine.running = state_data["running"]
            
            logger.info(f"Loaded state for {machine_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False
    
    async def save_checkpoint(
        self,
        machine_id: str,
        checkpoint_name: str,
        state_machine: StateMachine
    ) -> None:
        """Save a named checkpoint"""
        checkpoint_dir = self.storage_dir / "checkpoints" / machine_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = checkpoint_dir / f"{checkpoint_name}.pkl"
        
        checkpoint_data = {
            "machine_state": {
                "current_state": state_machine.current_state,
                "context": state_machine.context,
                "running": state_machine.running,
            },
            "metadata": {
                "checkpoint_name": checkpoint_name,
                "timestamp": datetime.now().isoformat(),
                "machine_id": machine_id,
            }
        }
        
        try:
            async with aiofiles.open(filepath, 'wb') as f:
                await f.write(pickle.dumps(checkpoint_data))
                
            logger.info(f"Saved checkpoint {checkpoint_name} for {machine_id}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    async def load_checkpoint(
        self,
        machine_id: str,
        checkpoint_name: str,
        state_machine: StateMachine
    ) -> bool:
        """Load a named checkpoint"""
        filepath = self.storage_dir / "checkpoints" / machine_id / f"{checkpoint_name}.pkl"
        
        if not filepath.exists():
            logger.warning(f"Checkpoint {checkpoint_name} not found for {machine_id}")
            return False
        
        try:
            async with aiofiles.open(filepath, 'rb') as f:
                content = await f.read()
                checkpoint_data = pickle.loads(content)
            
            # Restore state
            machine_state = checkpoint_data["machine_state"]
            state_machine.current_state = machine_state["current_state"]
            state_machine.context = machine_state["context"]
            state_machine.running = machine_state["running"]
            
            logger.info(f"Loaded checkpoint {checkpoint_name} for {machine_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False
    
    async def list_checkpoints(self, machine_id: str) -> List[Dict[str, Any]]:
        """List available checkpoints"""
        checkpoint_dir = self.storage_dir / "checkpoints" / machine_id
        
        if not checkpoint_dir.exists():
            return []
        
        checkpoints = []
        
        for filepath in checkpoint_dir.glob("*.pkl"):
            try:
                async with aiofiles.open(filepath, 'rb') as f:
                    content = await f.read()
                    data = pickle.loads(content)
                    
                checkpoints.append({
                    "name": filepath.stem,
                    "timestamp": data["metadata"]["timestamp"],
                    "file": str(filepath)
                })
            except Exception as e:
                logger.error(f"Error reading checkpoint {filepath}: {e}")
        
        return sorted(checkpoints, key=lambda x: x["timestamp"], reverse=True)
    
    def _serialize_context(self, context: StateContext) -> Dict[str, Any]:
        """Serialize context for storage"""
        return {
            "data": context.data,
            "history": [
                {
                    "from_state": t.from_state,
                    "to_state": t.to_state,
                    "trigger": t.trigger,
                    "timestamp": t.timestamp.isoformat(),
                    "data": t.data,
                    "success": t.success,
                    "error": t.error,
                }
                for t in context.history
            ],
            "metadata": context.metadata,
        }
    
    def _deserialize_context(self, data: Dict[str, Any]) -> StateContext:
        """Deserialize context from storage"""
        context = StateContext()
        context.data = data["data"]
        context.metadata = data["metadata"]
        
        # Restore history
        for transition_data in data["history"]:
            transition = StateTransition(
                from_state=transition_data["from_state"],
                to_state=transition_data["to_state"],
                trigger=transition_data["trigger"],
                timestamp=datetime.fromisoformat(transition_data["timestamp"]),
                data=transition_data.get("data"),
                success=transition_data.get("success", True),
                error=transition_data.get("error"),
            )
            context.history.append(transition)
        
        return context

class AutoSaveStateMachine(StateMachine):
    """State machine with automatic persistence"""
    
    def __init__(
        self,
        name: str,
        machine_id: str,
        persistence: StatePersistence,
        save_interval: int = 10
    ):
        super().__init__(name)
        self.machine_id = machine_id
        self.persistence = persistence
        self.save_interval = save_interval
        self._save_task = None
        
        # Add transition callback for auto-save
        self.add_transition_callback(self._on_transition)
    
    async def start(self, context: Optional[StateContext] = None) -> Any:
        """Start with auto-save"""
        # Try to load previous state
        loaded = await self.persistence.load_state(self.machine_id, self)
        
        if loaded:
            logger.info(f"Resumed from saved state: {self.current_state}")
        
        # Start auto-save task
        self._save_task = asyncio.create_task(self._auto_save_loop())
        
        try:
            return await super().start(context)
        finally:
            # Stop auto-save
            if self._save_task:
                self._save_task.cancel()
                try:
                    await self._save_task
                except asyncio.CancelledError:
                    pass
    
    async def _auto_save_loop(self):
        """Periodically save state"""
        while self.running:
            await asyncio.sleep(self.save_interval)
            await self.persistence.save_state(self.machine_id, self)
    
    async def _on_transition(self, transition: StateTransition, context: StateContext):
        """Save state on transitions"""
        await self.persistence.save_state(self.machine_id, self)
    
    async def create_checkpoint(self, name: str) -> None:
        """Create a named checkpoint"""
        await self.persistence.save_checkpoint(self.machine_id, name, self)
    
    async def restore_checkpoint(self, name: str) -> bool:
        """Restore from checkpoint"""
        return await self.persistence.load_checkpoint(self.machine_id, name, self)
```

### Task 4: State Debugging and Visualization (10 min)
Create `src/state/debugging.py`:

```python
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import structlog
from dataclasses import dataclass

from src.state.core import StateMachine, StateTransition, StateContext

logger = structlog.get_logger()

@dataclass
class StateEvent:
    """Event in state machine execution"""
    timestamp: datetime
    event_type: str  # "enter", "exit", "transition", "error"
    state: str
    data: Dict[str, Any]

class StateDebugger:
    """Debug and monitor state machine execution"""
    
    def __init__(self, state_machine: StateMachine):
        self.state_machine = state_machine
        self.events: List[StateEvent] = []
        self.breakpoints: Set[str] = set()
        self.watch_expressions: Dict[str, str] = {}
        self.paused = False
        
        # Attach to state machine
        self._attach_callbacks()
    
    def _attach_callbacks(self):
        """Attach debugging callbacks"""
        # Add transition callback
        self.state_machine.add_transition_callback(self._on_transition)
        
        # Add entry/exit callbacks to all states
        for state in self.state_machine.states.values():
            state.add_entry_callback(self._on_state_enter)
            state.add_exit_callback(self._on_state_exit)
    
    async def _on_transition(self, transition: StateTransition, context: StateContext):
        """Handle state transition"""
        event = StateEvent(
            timestamp=datetime.now(),
            event_type="transition",
            state=transition.to_state,
            data={
                "from": transition.from_state,
                "to": transition.to_state,
                "trigger": transition.trigger,
            }
        )
        self.events.append(event)
        
        # Check breakpoint
        if transition.to_state in self.breakpoints:
            await self._handle_breakpoint(transition.to_state, context)
        
        # Evaluate watch expressions
        if self.watch_expressions:
            self._evaluate_watches(context)
    
    async def _on_state_enter(self, context: StateContext):
        """Handle state entry"""
        current_state = self.state_machine.current_state
        
        event = StateEvent(
            timestamp=datetime.now(),
            event_type="enter",
            state=current_state,
            data={"context_keys": list(context.data.keys())}
        )
        self.events.append(event)
    
    async def _on_state_exit(self, context: StateContext):
        """Handle state exit"""
        current_state = self.state_machine.current_state
        
        event = StateEvent(
            timestamp=datetime.now(),
            event_type="exit",
            state=current_state,
            data={"context_keys": list(context.data.keys())}
        )
        self.events.append(event)
    
    async def _handle_breakpoint(self, state: str, context: StateContext):
        """Handle breakpoint hit"""
        logger.info(f"Breakpoint hit at state: {state}")
        self.paused = True
        
        # In a real implementation, this would pause execution
        # and wait for debugger commands
        
        # For now, just log state
        self._print_debug_info(state, context)
    
    def _evaluate_watches(self, context: StateContext):
        """Evaluate watch expressions"""
        results = {}
        
        for name, expr in self.watch_expressions.items():
            try:
                # Simple evaluation - just get from context
                value = context.get(expr)
                results[name] = value
            except Exception as e:
                results[name] = f"Error: {e}"
        
        if results:
            logger.info("Watch expressions:", **results)
    
    def _print_debug_info(self, state: str, context: StateContext):
        """Print debug information"""
        print("\n" + "="*60)
        print(f"STATE: {state}")
        print("="*60)
        
        print("\nCONTEXT DATA:")
        for key, value in context.data.items():
            value_str = str(value)
            if len(value_str) > 100:
                value_str = value_str[:100] + "..."
            print(f"  {key}: {value_str}")
        
        print(f"\nHISTORY: {len(context.history)} transitions")
        
        print("\nPOSSIBLE TRANSITIONS:")
        for trigger in self.state_machine.get_possible_transitions():
            print(f"  - {trigger}")
        
        print("="*60 + "\n")
    
    def add_breakpoint(self, state: str):
        """Add breakpoint at state"""
        self.breakpoints.add(state)
        logger.info(f"Added breakpoint at state: {state}")
    
    def remove_breakpoint(self, state: str):
        """Remove breakpoint"""
        self.breakpoints.discard(state)
    
    def add_watch(self, name: str, expression: str):
        """Add watch expression"""
        self.watch_expressions[name] = expression
    
    def get_trace(self) -> List[Dict[str, Any]]:
        """Get execution trace"""
        return [
            {
                "timestamp": e.timestamp.isoformat(),
                "type": e.event_type,
                "state": e.state,
                "data": e.data,
            }
            for e in self.events
        ]
    
    def export_trace(self, filepath: str):
        """Export trace to file"""
        trace = self.get_trace()
        
        with open(filepath, 'w') as f:
            json.dump(trace, f, indent=2)
        
        logger.info(f"Exported trace to {filepath}")
    
    def generate_mermaid_diagram(self) -> str:
        """Generate Mermaid diagram of state machine"""
        lines = ["stateDiagram-v2"]
        
        # Add states
        for state_name, state in self.state_machine.states.items():
            if state_name == self.state_machine.initial_state:
                lines.append(f"    [*] --> {state_name}")
            
            if state_name in self.state_machine.final_states:
                lines.append(f"    {state_name} --> [*]")
        
        # Add transitions from execution history
        seen_transitions = set()
        
        for event in self.events:
            if event.event_type == "transition":
                from_state = event.data["from"]
                to_state = event.data["to"]
                
                transition_key = (from_state, to_state)
                if transition_key not in seen_transitions:
                    seen_transitions.add(transition_key)
                    lines.append(f"    {from_state} --> {to_state}")
        
        # Highlight current state
        if self.state_machine.current_state:
            lines.append(f"    {self.state_machine.current_state}:::current")
            lines.append(f"    classDef current fill:#f9f,stroke:#333,stroke-width:4px")
        
        return "\n".join(lines)
    
    def print_summary(self):
        """Print execution summary"""
        print("\n" + "="*60)
        print("STATE MACHINE EXECUTION SUMMARY")
        print("="*60)
        
        # Count events by type
        event_counts = {}
        state_visits = {}
        
        for event in self.events:
            event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1
            
            if event.event_type == "enter":
                state_visits[event.state] = state_visits.get(event.state, 0) + 1
        
        print("\nEvent Counts:")
        for event_type, count in event_counts.items():
            print(f"  {event_type}: {count}")
        
        print("\nState Visits:")
        for state, visits in sorted(state_visits.items(), key=lambda x: x[1], reverse=True):
            print(f"  {state}: {visits}")
        
        print(f"\nTotal Events: {len(self.events)}")
        
        if self.events:
            duration = self.events[-1].timestamp - self.events[0].timestamp
            print(f"Duration: {duration}")
        
        print("="*60 + "\n")

class StateMonitor:
    """Real-time state machine monitoring"""
    
    def __init__(self):
        self.active_machines: Dict[str, StateMachine] = {}
        self.machine_debuggers: Dict[str, StateDebugger] = {}
        
    def register_machine(self, machine_id: str, machine: StateMachine):
        """Register a state machine for monitoring"""
        self.active_machines[machine_id] = machine
        self.machine_debuggers[machine_id] = StateDebugger(machine)
        
        logger.info(f"Registered machine for monitoring: {machine_id}")
    
    def get_status(self, machine_id: str) -> Dict[str, Any]:
        """Get current status of a machine"""
        if machine_id not in self.active_machines:
            return {"error": "Machine not found"}
        
        machine = self.active_machines[machine_id]
        debugger = self.machine_debuggers[machine_id]
        
        return {
            "machine_id": machine_id,
            "name": machine.name,
            "current_state": machine.current_state,
            "running": machine.running,
            "event_count": len(debugger.events),
            "context_size": len(machine.context.data),
            "history_length": len(machine.context.history),
        }
    
    def get_all_status(self) -> List[Dict[str, Any]]:
        """Get status of all machines"""
        return [
            self.get_status(machine_id)
            for machine_id in self.active_machines
        ]
    
    def get_diagram(self, machine_id: str) -> str:
        """Get Mermaid diagram for a machine"""
        if machine_id in self.machine_debuggers:
            return self.machine_debuggers[machine_id].generate_mermaid_diagram()
        return ""
```

## 🌞 Afternoon Tasks (90-120 minutes)

### Task 5: Multi-Agent State Coordination (45 min)
Create `src/state/coordination.py`:

```python
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum, auto
import asyncio
from datetime import datetime
import structlog

from src.state.core import StateMachine, StateContext, State, StateType

logger = structlog.get_logger()

class CoordinationStrategy(Enum):
    """Multi-agent coordination strategies"""
    SEQUENTIAL = auto()      # Agents work one after another
    PARALLEL = auto()        # Agents work simultaneously
    HIERARCHICAL = auto()    # Master-worker pattern
    COLLABORATIVE = auto()   # Agents work together with communication
    COMPETITIVE = auto()     # Agents compete for resources

@dataclass
class AgentMessage:
    """Message between agents"""
    from_agent: str
    to_agent: Optional[str]  # None for broadcast
    message_type: str
    content: Any
    timestamp: datetime = field(default_factory=datetime.now)
    requires_response: bool = False

@dataclass
class SharedResource:
    """Resource shared between agents"""
    name: str
    value: Any
    owner: Optional[str] = None
    locked: bool = False
    version: int = 0

class MessageBus:
    """Message passing between agents"""
    
    def __init__(self):
        self.queues: Dict[str, asyncio.Queue] = {}
        self.subscribers: Dict[str, Set[str]] = {}
        self.message_history: List[AgentMessage] = []
        
    def register_agent(self, agent_id: str):
        """Register an agent with the message bus"""
        self.queues[agent_id] = asyncio.Queue()
        self.subscribers[agent_id] = set()
        logger.info(f"Registered agent: {agent_id}")
        
    async def send_message(self, message: AgentMessage):
        """Send a message"""
        self.message_history.append(message)
        
        if message.to_agent:
            # Direct message
            if message.to_agent in self.queues:
                await self.queues[message.to_agent].put(message)
                logger.debug(f"Message sent: {message.from_agent} -> {message.to_agent}")
            else:
                logger.warning(f"Unknown recipient: {message.to_agent}")
        else:
            # Broadcast
            for agent_id, queue in self.queues.items():
                if agent_id != message.from_agent:
                    await queue.put(message)
            logger.debug(f"Broadcast from {message.from_agent}")
    
    async def receive_message(self, agent_id: str, timeout: Optional[float] = None) -> Optional[AgentMessage]:
        """Receive a message for an agent"""
        if agent_id not in self.queues:
            return None
            
        try:
            if timeout:
                message = await asyncio.wait_for(
                    self.queues[agent_id].get(),
                    timeout=timeout
                )
            else:
                message = await self.queues[agent_id].get()
            
            return message
        except asyncio.TimeoutError:
            return None
    
    def subscribe(self, agent_id: str, topic: str):
        """Subscribe agent to a topic"""
        if agent_id in self.subscribers:
            self.subscribers[agent_id].add(topic)

class SharedMemory:
    """Shared memory for agent coordination"""
    
    def __init__(self):
        self.resources: Dict[str, SharedResource] = {}
        self.locks: Dict[str, asyncio.Lock] = {}
        
    async def get(self, resource_name: str, agent_id: str) -> Optional[Any]:
        """Get a shared resource"""
        if resource_name not in self.resources:
            return None
            
        resource = self.resources[resource_name]
        
        if resource.locked and resource.owner != agent_id:
            logger.warning(f"Resource {resource_name} is locked by {resource.owner}")
            return None
            
        return resource.value
    
    async def set(
        self,
        resource_name: str,
        value: Any,
        agent_id: str,
        require_lock: bool = True
    ) -> bool:
        """Set a shared resource"""
        if resource_name not in self.resources:
            self.resources[resource_name] = SharedResource(
                name=resource_name,
                value=value,
                owner=agent_id,
                version=1
            )
            self.locks[resource_name] = asyncio.Lock()
            return True
        
        resource = self.resources[resource_name]
        
        if require_lock and resource.locked and resource.owner != agent_id:
            return False
        
        resource.value = value
        resource.version += 1
        
        logger.debug(f"Resource {resource_name} updated by {agent_id}")
        return True
    
    async def acquire_lock(self, resource_name: str, agent_id: str) -> bool:
        """Acquire lock on a resource"""
        if resource_name not in self.resources:
            return False
            
        resource = self.resources[resource_name]
        lock = self.locks[resource_name]
        
        if await lock.acquire():
            resource.locked = True
            resource.owner = agent_id
            logger.debug(f"Lock acquired: {resource_name} by {agent_id}")
            return True
            
        return False
    
    async def release_lock(self, resource_name: str, agent_id: str) -> bool:
        """Release lock on a resource"""
        if resource_name not in self.resources:
            return False
            
        resource = self.resources[resource_name]
        
        if resource.owner != agent_id:
            return False
        
        resource.locked = False
        resource.owner = None
        self.locks[resource_name].release()
        
        logger.debug(f"Lock released: {resource_name}")
        return True

class CoordinatedAgent(StateMachine):
    """Agent that can coordinate with others"""
    
    def __init__(
        self,
        name: str,
        agent_id: str,
        message_bus: MessageBus,
        shared_memory: SharedMemory
    ):
        super().__init__(name)
        self.agent_id = agent_id
        self.message_bus = message_bus
        self.shared_memory = shared_memory
        
        # Register with message bus
        self.message_bus.register_agent(agent_id)
        
        # Add coordination states
        self._add_coordination_states()
    
    def _add_coordination_states(self):
        """Add states for coordination"""
        # These would be mixed into the regular states
        pass
    
    async def send_message(
        self,
        to_agent: Optional[str],
        message_type: str,
        content: Any,
        requires_response: bool = False
    ):
        """Send a message to another agent"""
        message = AgentMessage(
            from_agent=self.agent_id,
            to_agent=to_agent,
            message_type=message_type,
            content=content,
            requires_response=requires_response
        )
        
        await self.message_bus.send_message(message)
    
    async def wait_for_message(
        self,
        message_type: Optional[str] = None,
        from_agent: Optional[str] = None,
        timeout: float = 30
    ) -> Optional[AgentMessage]:
        """Wait for a specific message"""
        end_time = datetime.now().timestamp() + timeout
        
        while datetime.now().timestamp() < end_time:
            remaining_time = end_time - datetime.now().timestamp()
            message = await self.message_bus.receive_message(
                self.agent_id,
                timeout=min(remaining_time, 1)
            )
            
            if message:
                # Check if it matches criteria
                if message_type and message.message_type != message_type:
                    continue
                if from_agent and message.from_agent != from_agent:
                    continue
                    
                return message
        
        return None
    
    async def collaborate_on_task(
        self,
        task: str,
        collaborators: List[str]
    ) -> Any:
        """Collaborate with other agents on a task"""
        # Announce task
        await self.send_message(
            None,  # Broadcast
            "task_announcement",
            {"task": task, "looking_for": collaborators}
        )
        
        # Wait for collaborators
        confirmed_collaborators = []
        
        for collaborator in collaborators:
            response = await self.wait_for_message(
                "collaboration_accepted",
                from_agent=collaborator,
                timeout=10
            )
            
            if response:
                confirmed_collaborators.append(collaborator)
        
        if not confirmed_collaborators:
            logger.warning("No collaborators available")
            return None
        
        # Execute collaborative task
        return await self._execute_collaborative_task(
            task,
            confirmed_collaborators
        )
    
    async def _execute_collaborative_task(
        self,
        task: str,
        collaborators: List[str]
    ) -> Any:
        """Execute task with collaborators"""
        # This would contain the actual collaboration logic
        pass

class MultiAgentCoordinator:
    """Coordinate multiple agents"""
    
    def __init__(self, strategy: CoordinationStrategy = CoordinationStrategy.PARALLEL):
        self.strategy = strategy
        self.agents: Dict[str, CoordinatedAgent] = {}
        self.message_bus = MessageBus()
        self.shared_memory = SharedMemory()
        self.running = False
        
    def add_agent(self, agent: CoordinatedAgent):
        """Add an agent to the coordinator"""
        self.agents[agent.agent_id] = agent
        logger.info(f"Added agent to coordinator: {agent.agent_id}")
    
    async def execute_task(
        self,
        task: str,
        task_allocation: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute task across multiple agents"""
        self.running = True
        
        try:
            if self.strategy == CoordinationStrategy.SEQUENTIAL:
                return await self._execute_sequential(task, task_allocation)
            elif self.strategy == CoordinationStrategy.PARALLEL:
                return await self._execute_parallel(task, task_allocation)
            elif self.strategy == CoordinationStrategy.HIERARCHICAL:
                return await self._execute_hierarchical(task, task_allocation)
            else:
                raise ValueError(f"Unsupported strategy: {self.strategy}")
                
        finally:
            self.running = False
    
    async def _execute_sequential(
        self,
        task: str,
        task_allocation: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute task sequentially"""
        results = {}
        
        for agent_id, agent in self.agents.items():
            # Get subtask for this agent
            subtask = task_allocation.get(agent_id, task) if task_allocation else task
            
            # Set up context
            context = StateContext()
            context.set("task", subtask)
            context.set("previous_results", results.copy())
            
            # Execute
            result = await agent.start(context)
            results[agent_id] = result.get("final_answer")
            
            # Share result
            await self.shared_memory.set(
                f"result_{agent_id}",
                results[agent_id],
                agent_id,
                require_lock=False
            )
        
        return results
    
    async def _execute_parallel(
        self,
        task: str,
        task_allocation: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute task in parallel"""
        tasks = []
        
        for agent_id, agent in self.agents.items():
            # Get subtask for this agent
            subtask = task_allocation.get(agent_id, task) if task_allocation else task
            
            # Set up context
            context = StateContext()
            context.set("task", subtask)
            
            # Create async task
            task_coro = self._run_agent(agent, context)
            tasks.append(asyncio.create_task(task_coro))
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks)
        
        return {
            agent_id: result.get("final_answer")
            for agent_id, result in zip(self.agents.keys(), results)
        }
    
    async def _execute_hierarchical(
        self,
        task: str,
        task_allocation: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute task hierarchically (master-worker)"""
        # First agent is the master
        master_id = list(self.agents.keys())[0]
        master = self.agents[master_id]
        
        # Master creates work items
        context = StateContext()
        context.set("task", f"Plan and delegate: {task}")
        context.set("workers", list(self.agents.keys())[1:])
        
        await master.start(context)
        
        # Workers execute based on master's plan
        work_items = await self.shared_memory.get("work_items", master_id)
        
        if work_items:
            worker_tasks = []
            
            for worker_id, work_item in work_items.items():
                if worker_id in self.agents:
                    worker = self.agents[worker_id]
                    worker_context = StateContext()
                    worker_context.set("task", work_item)
                    
                    task_coro = self._run_agent(worker, worker_context)
                    worker_tasks.append(asyncio.create_task(task_coro))
            
            await asyncio.gather(*worker_tasks)
        
        # Master collects results
        final_context = StateContext()
        final_context.set("task", "Collect and synthesize results")
        
        return await master.start(final_context)
    
    async def _run_agent(
        self,
        agent: CoordinatedAgent,
        context: StateContext
    ) -> StateContext:
        """Run a single agent"""
        try:
            return await agent.start(context)
        except Exception as e:
            logger.error(f"Agent {agent.agent_id} failed: {e}")
            context.set("error", str(e))
            return context
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        return {
            "strategy": self.strategy.name,
            "running": self.running,
            "agents": {
                agent_id: {
                    "state": agent.current_state,
                    "running": agent.running
                }
                for agent_id, agent in self.agents.items()
            }
        }
```

### Task 6: State History and Rollback (30 min)
Create `src/state/history.py`:

```python
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import copy
import structlog

from src.state.core import StateContext, StateTransition, StateMachine

logger = structlog.get_logger()

@dataclass
class StateSnapshot:
    """Snapshot of state at a point in time"""
    timestamp: datetime
    state_name: str
    context: StateContext
    transition_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)

class StateHistory:
    """Manage state history with rollback capability"""
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.snapshots: List[StateSnapshot] = []
        self.rollback_points: Dict[str, int] = {}  # name -> snapshot index
        
    def take_snapshot(
        self,
        state_name: str,
        context: StateContext,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Take a snapshot of current state"""
        # Deep copy context to preserve state
        context_copy = self._deep_copy_context(context)
        
        snapshot = StateSnapshot(
            timestamp=datetime.now(),
            state_name=state_name,
            context=context_copy,
            transition_index=len(context.history),
            metadata=metadata or {}
        )
        
        self.snapshots.append(snapshot)
        
        # Limit history size
        if len(self.snapshots) > self.max_history:
            self.snapshots = self.snapshots[-self.max_history:]
            
        logger.debug(f"Snapshot taken at state {state_name}")
    
    def create_rollback_point(self, name: str) -> bool:
        """Create a named rollback point"""
        if not self.snapshots:
            return False
            
        self.rollback_points[name] = len(self.snapshots) - 1
        logger.info(f"Created rollback point: {name}")
        return True
    
    def rollback_to_point(self, name: str) -> Optional[StateSnapshot]:
        """Rollback to a named point"""
        if name not in self.rollback_points:
            logger.warning(f"Rollback point not found: {name}")
            return None
            
        index = self.rollback_points[name]
        if index < len(self.snapshots):
            return self.snapshots[index]
            
        return None
    
    def rollback_to_snapshot(self, index: int) -> Optional[StateSnapshot]:
        """Rollback to a specific snapshot"""
        if 0 <= index < len(self.snapshots):
            return self.snapshots[index]
        return None
    
    def get_recent_snapshots(self, count: int = 10) -> List[StateSnapshot]:
        """Get recent snapshots"""
        return self.snapshots[-count:]
    
    def find_snapshot_by_state(self, state_name: str) -> Optional[StateSnapshot]:
        """Find most recent snapshot for a state"""
        for snapshot in reversed(self.snapshots):
            if snapshot.state_name == state_name:
                return snapshot
        return None
    
    def get_state_path(self) -> List[str]:
        """Get path of states visited"""
        return [s.state_name for s in self.snapshots]
    
    def _deep_copy_context(self, context: StateContext) -> StateContext:
        """Deep copy a context"""
        new_context = StateContext()
        new_context.data = copy.deepcopy(context.data)
        new_context.metadata = copy.deepcopy(context.metadata)
        
        # Copy history
        new_context.history = []
        for transition in context.history:
            new_transition = StateTransition(
                from_state=transition.from_state,
                to_state=transition.to_state,
                trigger=transition.trigger,
                timestamp=transition.timestamp,
                data=copy.deepcopy(transition.data) if transition.data else None,
                success=transition.success,
                error=transition.error
            )
            new_context.history.append(new_transition)
        
        return new_context

class VersionedStateMachine(StateMachine):
    """State machine with versioning and rollback"""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.history = StateHistory()
        self.version = 0
        self.branches: Dict[str, Tuple[int, StateSnapshot]] = {}  # branch_name -> (version, snapshot)
        
        # Add transition callback for history
        self.add_transition_callback(self._record_history)
    
    async def _record_history(self, transition: StateTransition, context: StateContext):
        """Record state history on transitions"""
        self.history.take_snapshot(
            transition.to_state,
            context,
            metadata={"version": self.version}
        )
        self.version += 1
    
    async def rollback(self, steps: int = 1) -> bool:
        """Rollback state machine by n steps"""
        current_index = len(self.history.snapshots) - 1
        target_index = max(0, current_index - steps)
        
        snapshot = self.history.rollback_to_snapshot(target_index)
        if snapshot:
            return await self._restore_from_snapshot(snapshot)
            
        return False
    
    async def rollback_to_point(self, point_name: str) -> bool:
        """Rollback to named point"""
        snapshot = self.history.rollback_to_point(point_name)
        if snapshot:
            return await self._restore_from_snapshot(snapshot)
            
        return False
    
    async def rollback_to_state(self, state_name: str) -> bool:
        """Rollback to when machine was in specific state"""
        snapshot = self.history.find_snapshot_by_state(state_name)
        if snapshot:
            return await self._restore_from_snapshot(snapshot)
            
        return False
    
    async def _restore_from_snapshot(self, snapshot: StateSnapshot) -> bool:
        """Restore machine from snapshot"""
        try:
            # Exit current state
            if self.current_state:
                current = self.states[self.current_state]
                await current.on_exit(self.context)
            
            # Restore state
            self.current_state = snapshot.state_name
            self.context = self.history._deep_copy_context(snapshot.context)
            self.version = snapshot.metadata.get("version", 0) + 1
            
            # Enter restored state
            restored_state = self.states[self.current_state]
            await restored_state.on_enter(self.context)
            
            logger.info(f"Rolled back to state {self.current_state} (version {self.version})")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def create_checkpoint(self, name: str) -> bool:
        """Create a checkpoint for rollback"""
        return self.history.create_rollback_point(name)
    
    def create_branch(self, branch_name: str) -> bool:
        """Create a branch at current state"""
        if not self.history.snapshots:
            return False
            
        latest_snapshot = self.history.snapshots[-1]
        self.branches[branch_name] = (self.version, latest_snapshot)
        
        logger.info(f"Created branch: {branch_name} at version {self.version}")
        return True
    
    async def switch_branch(self, branch_name: str) -> bool:
        """Switch to a different branch"""
        if branch_name not in self.branches:
            logger.warning(f"Branch not found: {branch_name}")
            return False
            
        version, snapshot = self.branches[branch_name]
        
        # Save current state as branch
        current_branch = f"auto_save_{datetime.now().timestamp()}"
        self.create_branch(current_branch)
        
        # Restore branch state
        success = await self._restore_from_snapshot(snapshot)
        
        if success:
            self.version = version
            logger.info(f"Switched to branch: {branch_name}")
            
        return success
    
    def get_history_summary(self) -> Dict[str, Any]:
        """Get summary of state history"""
        path = self.history.get_state_path()
        
        # Count state visits
        state_counts = {}
        for state in path:
            state_counts[state] = state_counts.get(state, 0) + 1
        
        return {
            "total_transitions": len(path),
            "unique_states": len(set(path)),
            "current_version": self.version,
            "branches": list(self.branches.keys()),
            "checkpoints": list(self.history.rollback_points.keys()),
            "state_visits": state_counts,
            "path_summary": path[-10:] if len(path) > 10 else path  # Last 10 states
        }

class DiffTracker:
    """Track differences between state snapshots"""
    
    @staticmethod
    def diff_contexts(
        old_context: StateContext,
        new_context: StateContext
    ) -> Dict[str, Any]:
        """Compare two contexts and return differences"""
        diff = {
            "data_changes": {},
            "metadata_changes": {},
            "new_transitions": [],
        }
        
        # Compare data
        all_keys = set(old_context.data.keys()) | set(new_context.data.keys())
        
        for key in all_keys:
            old_value = old_context.data.get(key)
            new_value = new_context.data.get(key)
            
            if old_value != new_value:
                diff["data_changes"][key] = {
                    "old": old_value,
                    "new": new_value,
                    "action": "added" if old_value is None else "removed" if new_value is None else "modified"
                }
        
        # Compare metadata
        all_meta_keys = set(old_context.metadata.keys()) | set(new_context.metadata.keys())
        
        for key in all_meta_keys:
            old_value = old_context.metadata.get(key)
            new_value = new_context.metadata.get(key)
            
            if old_value != new_value:
                diff["metadata_changes"][key] = {
                    "old": old_value,
                    "new": new_value
                }
        
        # Compare transitions
        old_count = len(old_context.history)
        new_count = len(new_context.history)
        
        if new_count > old_count:
            diff["new_transitions"] = [
                {
                    "from": t.from_state,
                    "to": t.to_state,
                    "trigger": t.trigger
                }
                for t in new_context.history[old_count:]
            ]
        
        return diff
    
    @staticmethod
    def diff_snapshots(
        old_snapshot: StateSnapshot,
        new_snapshot: StateSnapshot
    ) -> Dict[str, Any]:
        """Compare two snapshots"""
        diff = {
            "state_change": old_snapshot.state_name != new_snapshot.state_name,
            "old_state": old_snapshot.state_name,
            "new_state": new_snapshot.state_name,
            "time_diff": (new_snapshot.timestamp - old_snapshot.timestamp).total_seconds(),
            "context_diff": DiffTracker.diff_contexts(
                old_snapshot.context,
                new_snapshot.context
            )
        }
        
        return diff
```

### Task 7: Integration Example (15 min)
Create `src/state/example_agent.py`:

```python
import asyncio
from typing import Optional

from src.state.agent_states import create_agent_state_machine
from src.state.persistence import AutoSaveStateMachine, StatePersistence
from src.state.history import VersionedStateMachine
from src.state.debugging import StateDebugger, StateMonitor
from src.state.coordination import (
    MultiAgentCoordinator,
    CoordinatedAgent,
    MessageBus,
    SharedMemory,
    CoordinationStrategy
)

async def example_single_agent():
    """Example of single agent with state machine"""
    print("=== Single Agent Example ===\n")
    
    # Create persistence
    persistence = StatePersistence()
    
    # Create versioned state machine with auto-save
    class VersionedAutoSaveStateMachine(VersionedStateMachine, AutoSaveStateMachine):
        pass
    
    sm = VersionedAutoSaveStateMachine(
        name="example_agent",
        machine_id="agent_001",
        persistence=persistence
    )
    
    # Add agent states
    agent_sm = create_agent_state_machine()
    
    # Copy states from agent state machine
    for state_name, state in agent_sm.states.items():
        sm.add_state(
            state,
            initial=(state_name == "initial"),
            final=(state_name == "finalizing")
        )
    
    # Create debugger
    debugger = StateDebugger(sm)
    debugger.add_breakpoint("thinking")
    debugger.add_watch("task", "task")
    
    # Create context with task
    context = StateContext()
    context.set("task", "Calculate the sum of 1 to 10")
    
    # Create checkpoint before execution
    sm.create_checkpoint("before_execution")
    
    # Run state machine
    print("Starting state machine...\n")
    result = await sm.start(context)
    
    # Print results
    print(f"\nFinal answer: {result.get('final_answer')}")
    print(f"\nState path: {' -> '.join(sm.history.get_state_path())}")
    
    # Print debug info
    debugger.print_summary()
    
    # Export trace
    debugger.export_trace("agent_trace.json")
    
    # Generate diagram
    print("\nState Machine Diagram:")
    print(debugger.generate_mermaid_diagram())
    
    return sm

async def example_multi_agent():
    """Example of multi-agent coordination"""
    print("\n\n=== Multi-Agent Example ===\n")
    
    # Create shared resources
    message_bus = MessageBus()
    shared_memory = SharedMemory()
    
    # Create coordinator
    coordinator = MultiAgentCoordinator(
        strategy=CoordinationStrategy.PARALLEL
    )
    
    # Create three agents
    for i in range(3):
        agent_id = f"agent_{i+1}"
        
        # Create coordinated agent
        agent = CoordinatedAgent(
            name=f"worker_{i+1}",
            agent_id=agent_id,
            message_bus=message_bus,
            shared_memory=shared_memory
        )
        
        # Add agent states
        agent_sm = create_agent_state_machine()
        for state_name, state in agent_sm.states.items():
            agent.add_state(
                state,
                initial=(state_name == "initial"),
                final=(state_name == "finalizing")
            )
        
        coordinator.add_agent(agent)
    
    # Define task allocation
    task_allocation = {
        "agent_1": "Calculate the sum of 1 to 10",
        "agent_2": "Calculate the product of 1 to 5",
        "agent_3": "Find the maximum of [15, 42, 8, 23]"
    }
    
    print("Starting multi-agent execution...\n")
    
    # Execute tasks
    results = await coordinator.execute_task(
        "Perform calculations",
        task_allocation
    )
    
    # Print results
    print("\nResults:")
    for agent_id, result in results.items():
        print(f"  {agent_id}: {result}")
    
    # Get status
    status = coordinator.get_status()
    print(f"\nCoordinator status: {status}")
    
    return coordinator

async def example_state_rollback():
    """Example of state rollback"""
    print("\n\n=== State Rollback Example ===\n")
    
    # Create versioned state machine
    sm = VersionedStateMachine("rollback_example")
    
    # Add simple states for demonstration
    class CounterState(State):
        def __init__(self, name: str, next_state: Optional[str] = None):
            super().__init__(name)
            self.next_state = next_state
            
        async def execute(self, context: StateContext) -> Optional[str]:
            count = context.get("count", 0)
            count += 1
            context.set("count", count)
            
            print(f"State {self.name}: count = {count}")
            
            if count >= 5:
                context.set("error", "Count too high!")
                return "error"
            
            return self.next_state
    
    # Add states
    sm.add_state(CounterState("state1", "state2"), initial=True)
    sm.add_state(CounterState("state2", "state3"))
    sm.add_state(CounterState("state3", "state1"))
    sm.add_state(State("error", state_type=StateType.ERROR), final=True)
    
    # Create context
    context = StateContext()
    
    # Run for a few iterations
    print("Running state machine...\n")
    
    # Manually step through states
    sm.current_state = sm.initial_state
    
    for i in range(6):
        # Take snapshot
        sm.history.take_snapshot(sm.current_state, sm.context)
        
        # Create checkpoint every 2 steps
        if i % 2 == 0:
            sm.create_checkpoint(f"checkpoint_{i}")
        
        # Execute current state
        current = sm.states[sm.current_state]
        next_state = await current.execute(sm.context)
        
        if next_state:
            sm.current_state = next_state
            
        if sm.current_state in sm.final_states:
            break
    
    print(f"\nReached error state with count: {sm.context.get('count')}")
    
    # Rollback to checkpoint
    print("\nRolling back to checkpoint_2...")
    await sm.rollback_to_point("checkpoint_2")
    print(f"After rollback, count: {sm.context.get('count')}")
    
    # Show history
    summary = sm.get_history_summary()
    print(f"\nHistory summary: {summary}")
    
    return sm

async def main():
    """Run all examples"""
    # Single agent example
    single_agent = await example_single_agent()
    
    # Multi-agent example
    multi_agent = await example_multi_agent()
    
    # Rollback example
    rollback = await example_state_rollback()
    
    print("\n\n=== Examples Complete ===")

if __name__ == "__main__":
    asyncio.run(main())
```

## 🌆 Evening Tasks (45-60 minutes)

### Task 8: Testing State Machines (20 min)
Create `tests/test_state_machines.py`:

```python
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from src.state.core import State, StateMachine, StateContext, StateType
from src.state.agent_states import (
    InitialState, ThinkingState, ActingState,
    create_agent_state_machine
)
from src.state.persistence import StatePersistence, AutoSaveStateMachine
from src.state.history import VersionedStateMachine

class TestStateMachine:
    @pytest.mark.asyncio
    async def test_basic_state_machine(self):
        """Test basic state machine functionality"""
        sm = StateMachine("test")
        
        # Create simple states
        class StartState(State):
            async def execute(self, context: StateContext) -> Optional[str]:
                context.set("started", True)
                return "end"
        
        class EndState(State):
            async def execute(self, context: StateContext) -> Optional[str]:
                context.set("ended", True)
                return None
        
        # Add states
        sm.add_state(StartState("start"), initial=True)
        sm.add_state(EndState("end"), final=True)
        
        # Run machine
        result = await sm.start()
        
        assert result.get("started") == True
        assert result.get("ended") == True
    
    @pytest.mark.asyncio
    async def test_state_transitions(self):
        """Test state transitions"""
        sm = StateMachine("test")
        transitions_recorded = []
        
        def record_transition(transition, context):
            transitions_recorded.append(transition)
        
        sm.add_transition_callback(record_transition)
        
        # Create states with explicit transitions
        class StateA(State):
            async def execute(self, context: StateContext) -> Optional[str]:
                return "b"
        
        class StateB(State):
            async def execute(self, context: StateContext) -> Optional[str]:
                return "c"
        
        class StateC(State):
            async def execute(self, context: StateContext) -> Optional[str]:
                return None
        
        sm.add_state(StateA("a"), initial=True)
        sm.add_state(StateB("b"))
        sm.add_state(StateC("c"), final=True)
        
        await sm.start()
        
        assert len(transitions_recorded) == 2
        assert transitions_recorded[0].from_state == "a"
        assert transitions_recorded[0].to_state == "b"
        assert transitions_recorded[1].from_state == "b"
        assert transitions_recorded[1].to_state == "c"

class TestAgentStates:
    @pytest.mark.asyncio
    async def test_initial_state(self):
        """Test initial state processing"""
        state = InitialState()
        context = StateContext()
        
        # No task provided
        next_state = await state.execute(context)
        assert next_state == "error"
        
        # Simple task
        context.set("task", "Calculate 2 + 2")
        next_state = await state.execute(context)
        assert next_state == "thinking"
        
        # Complex task requiring planning
        context.set("task", "Create a plan to solve this step by step")
        next_state = await state.execute(context)
        assert next_state == "planning"
    
    @pytest.mark.asyncio
    async def test_thinking_state(self):
        """Test thinking state"""
        # Mock agent
        mock_agent = Mock()
        mock_agent.llm = AsyncMock()
        
        state = ThinkingState(agent=mock_agent)
        context = StateContext()
        context.set("task", "Test task")
        
        # Mock LLM response with final answer
        mock_agent.llm.complete.return_value = "Final Answer: The result is 4"
        
        next_state = await state.execute(context)
        
        assert next_state == "finalizing"
        assert context.get("final_answer") == "The result is 4"

class TestStatePersistence:
    @pytest.mark.asyncio
    async def test_save_and_load_state(self, tmp_path):
        """Test state persistence"""
        persistence = StatePersistence(str(tmp_path))
        
        # Create state machine
        sm = StateMachine("test")
        sm.current_state = "test_state"
        sm.context.set("data", {"key": "value"})
        
        # Save state
        await persistence.save_state("test_machine", sm)
        
        # Create new machine and load state
        sm2 = StateMachine("test")
        loaded = await persistence.load_state("test_machine", sm2)
        
        assert loaded == True
        assert sm2.current_state == "test_state"
        assert sm2.context.get("data") == {"key": "value"}
    
    @pytest.mark.asyncio
    async def test_checkpoints(self, tmp_path):
        """Test checkpoint functionality"""
        persistence = StatePersistence(str(tmp_path))
        
        sm = StateMachine("test")
        sm.current_state = "state1"
        sm.context.set("value", 1)
        
        # Save checkpoint
        await persistence.save_checkpoint("test_machine", "checkpoint1", sm)
        
        # Modify state
        sm.current_state = "state2"
        sm.context.set("value", 2)
        
        # Load checkpoint
        loaded = await persistence.load_checkpoint("test_machine", "checkpoint1", sm)
        
        assert loaded == True
        assert sm.current_state == "state1"
        assert sm.context.get("value") == 1

class TestVersionedStateMachine:
    @pytest.mark.asyncio
    async def test_rollback(self):
        """Test state rollback"""
        sm = VersionedStateMachine("test")
        
        # Create states
        class IncrementState(State):
            async def execute(self, context: StateContext) -> Optional[str]:
                count = context.get("count", 0)
                context.set("count", count + 1)
                return None
        
        sm.add_state(IncrementState("increment"), initial=True, final=True)
        
        # Set initial context
        sm.context.set("count", 0)
        sm.current_state = "increment"
        
        # Take snapshots at different counts
        for i in range(5):
            sm.history.take_snapshot("increment", sm.context)
            await sm.states["increment"].execute(sm.context)
        
        assert sm.context.get("count") == 5
        
        # Rollback 3 steps
        await sm.rollback(3)
        assert sm.context.get("count") == 2
    
    @pytest.mark.asyncio
    async def test_branches(self):
        """Test branching functionality"""
        sm = VersionedStateMachine("test")
        
        # Set initial state
        sm.context.set("value", "main")
        sm.current_state = "test"
        
        # Create branch
        sm.create_branch("feature")
        
        # Modify main
        sm.context.set("value", "modified")
        
        # Switch to branch
        await sm.switch_branch("feature")
        
        assert sm.context.get("value") == "main"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### Task 9: Performance Monitoring (15 min)
Create `src/state/monitoring.py`:

```python
from typing import Dict, List, Any
from datetime import datetime, timedelta
import asyncio
import psutil
import structlog

from src.state.core import StateMachine, StateTransition

logger = structlog.get_logger()

class StateMetrics:
    """Collect metrics for state machine execution"""
    
    def __init__(self):
        self.state_durations: Dict[str, List[float]] = {}
        self.transition_counts: Dict[str, int] = {}
        self.error_counts: Dict[str, int] = {}
        self.memory_usage: List[Tuple[datetime, float]] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
    def record_state_duration(self, state: str, duration: float):
        """Record how long a state took"""
        if state not in self.state_durations:
            self.state_durations[state] = []
        self.state_durations[state].append(duration)
    
    def record_transition(self, from_state: str, to_state: str):
        """Record a transition"""
        key = f"{from_state}->{to_state}"
        self.transition_counts[key] = self.transition_counts.get(key, 0) + 1
    
    def record_error(self, state: str):
        """Record an error in a state"""
        self.error_counts[state] = self.error_counts.get(state, 0) + 1
    
    def record_memory_usage(self):
        """Record current memory usage"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.memory_usage.append((datetime.now(), memory_mb))
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        summary = {
            "execution_time": None,
            "states": {},
            "transitions": self.transition_counts,
            "errors": self.error_counts,
            "memory": {}
        }
        
        # Execution time
        if self.start_time and self.end_time:
            summary["execution_time"] = (self.end_time - self.start_time).total_seconds()
        
        # State statistics
        for state, durations in self.state_durations.items():
            if durations:
                summary["states"][state] = {
                    "count": len(durations),
                    "total_time": sum(durations),
                    "avg_time": sum(durations) / len(durations),
                    "min_time": min(durations),
                    "max_time": max(durations)
                }
        
        # Memory statistics
        if self.memory_usage:
            memory_values = [m[1] for m in self.memory_usage]
            summary["memory"] = {
                "min_mb": min(memory_values),
                "max_mb": max(memory_values),
                "avg_mb": sum(memory_values) / len(memory_values)
            }
        
        return summary

class MonitoredStateMachine(StateMachine):
    """State machine with performance monitoring"""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.metrics = StateMetrics()
        self._state_enter_times: Dict[str, datetime] = {}
        self._monitoring_task = None
        
        # Add callbacks
        self.add_transition_callback(self._on_transition_metrics)
    
    async def start(self, context=None):
        """Start with monitoring"""
        self.metrics.start_time = datetime.now()
        
        # Start memory monitoring
        self._monitoring_task = asyncio.create_task(self._monitor_memory())
        
        try:
            # Override state callbacks to track timing
            for state in self.states.values():
                state.add_entry_callback(self._on_state_enter_metrics)
                state.add_exit_callback(self._on_state_exit_metrics)
            
            result = await super().start(context)
            
            self.metrics.end_time = datetime.now()
            return result
            
        finally:
            # Stop monitoring
            if self._monitoring_task:
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass
    
    async def _monitor_memory(self):
        """Monitor memory usage periodically"""
        while self.running:
            self.metrics.record_memory_usage()
            await asyncio.sleep(1)  # Check every second
    
    async def _on_state_enter_metrics(self, context):
        """Record state entry time"""
        if self.current_state:
            self._state_enter_times[self.current_state] = datetime.now()
    
    async def _on_state_exit_metrics(self, context):
        """Record state exit time and duration"""
        if self.current_state and self.current_state in self._state_enter_times:
            enter_time = self._state_enter_times[self.current_state]
            duration = (datetime.now() - enter_time).total_seconds()
            self.metrics.record_state_duration(self.current_state, duration)
    
    async def _on_transition_metrics(self, transition: StateTransition, context):
        """Record transition metrics"""
        self.metrics.record_transition(transition.from_state, transition.to_state)
        
        if not transition.success:
            self.metrics.record_error(transition.from_state)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.metrics.get_summary()
    
    def print_metrics(self):
        """Print metrics summary"""
        summary = self.get_metrics_summary()
        
        print("\n" + "="*60)
        print("PERFORMANCE METRICS")
        print("="*60)
        
        if summary["execution_time"]:
            print(f"\nTotal execution time: {summary['execution_time']:.2f} seconds")
        
        print("\nState Performance:")
        for state, stats in summary["states"].items():
            print(f"  {state}:")
            print(f"    Visits: {stats['count']}")
            print(f"    Avg time: {stats['avg_time']:.3f}s")
            print(f"    Total time: {stats['total_time']:.3f}s")
        
        print("\nTransition Counts:")
        for transition, count in summary["transitions"].items():
            print(f"  {transition}: {count}")
        
        if summary["errors"]:
            print("\nErrors by State:")
            for state, count in summary["errors"].items():
                print(f"  {state}: {count}")
        
        if summary["memory"]:
            print(f"\nMemory Usage:")
            print(f"  Min: {summary['memory']['min_mb']:.1f} MB")
            print(f"  Max: {summary['memory']['max_mb']:.1f} MB")
            print(f"  Avg: {summary['memory']['avg_mb']:.1f} MB")
        
        print("="*60 + "\n")
```

### Task 10: Update Learning Journal (10 min)

Update your CLAUDE.md:

```markdown
## Day 5: State Machines

### What I Built
- ✅ Morning: Core state machine implementation
- ✅ Afternoon: Multi-agent coordination and history tracking  
- ✅ Evening: Testing and performance monitoring

### Key Learnings
1. **Technical**: 
   - State machines provide clear structure for agents
   - Async/await enables concurrent state execution
   - History tracking enables powerful debugging

2. **Architecture**:
   - State pattern for agent behavior
   - Message passing for multi-agent coordination
   - Versioning enables rollback and branching

3. **Performance**:
   - State transitions have measurable overhead
   - Memory snapshots need careful management
   - Concurrent agents improve throughput

### Challenges Faced
- **Issue**: Complex state dependencies
  **Solution**: Clear transition rules and validation
  **Lesson**: Explicit is better than implicit

- **Issue**: Debugging async state machines
  **Solution**: Comprehensive event tracking and visualization
  **Lesson**: Observability is crucial

### Code Metrics
- Lines written: ~2500
- States implemented: 9
- Test coverage: 85%
- Multi-agent speedup: 3x

### Tomorrow's Goal
- [ ] Build async patterns for agents
- [ ] Implement concurrent tool execution
- [ ] Add real-time streaming
```

## 📊 Deliverables Checklist
- [ ] Core state machine framework
- [ ] Agent-specific states
- [ ] State persistence and recovery
- [ ] Debugging and visualization tools
- [ ] Multi-agent coordination
- [ ] State history and rollback
- [ ] Performance monitoring
- [ ] Comprehensive testing

## 🎯 Success Metrics
You've succeeded if you can:
1. Build agents as proper state machines
2. Save and restore agent state
3. Coordinate multiple agents
4. Rollback to previous states
5. Debug state transitions visually

## 🚀 Extension Challenges
If you finish early:
1. Add hierarchical state machines (states within states)
2. Implement state machine DSL for easy definition
3. Build real-time state visualization web UI
4. Add distributed state synchronization
5. Create state machine optimizer

---

🎉 **Congratulations!** You've implemented production-grade state machines for agents. Tomorrow we'll focus on async patterns and concurrent execution.