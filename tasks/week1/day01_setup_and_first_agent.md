# Day 1: Setup and Your First Agent

## 🎯 Objectives
1. Set up development environment with production practices
2. Build a working agent that can reason and use tools
3. Understand the core agent loop: Think → Act → Observe
4. Implement proper error handling from the start
5. Create your first tool and integrate it

## 📋 Prerequisites
- Python 3.8+ installed
- Basic Python knowledge (functions, classes, async basics)
- Terminal/command line comfort
- 3 hours of focused time

## 🌄 Morning Tasks (45-60 minutes)

### Task 1: Environment Setup (20 min)
Create a production-ready project structure:

```bash
# Create project structure
mkdir -p agent47/{src,tests,tools,config,logs}
cd agent47

# Initialize git repository
git init
echo "*.pyc\n__pycache__/\n.env\nvenv/\nlogs/\n" > .gitignore

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Create requirements.txt
cat > requirements.txt << EOF
openai==1.12.0
anthropic==0.18.1
python-dotenv==1.0.0
pydantic==2.5.0
tenacity==8.2.3
structlog==24.1.0
pytest==8.0.0
pytest-asyncio==0.23.0
EOF

pip install -r requirements.txt
```

### Task 2: Configuration Management (15 min)
Create `src/config.py` with secure configuration:

```python
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv
import structlog

# Load environment variables
load_dotenv()

logger = structlog.get_logger()

@dataclass
class Config:
    """Production-ready configuration"""
    # API Configuration
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    llm_provider: str = "openai"  # or "anthropic"
    
    # Agent Configuration
    max_iterations: int = 10
    timeout_seconds: int = 300
    temperature: float = 0.7
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/agent.log"
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Load configuration from environment"""
        config = cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            llm_provider=os.getenv("LLM_PROVIDER", "openai"),
            max_iterations=int(os.getenv("MAX_ITERATIONS", "10")),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
        )
        
        # Validate configuration
        if config.llm_provider == "openai" and not config.openai_api_key:
            raise ValueError("OpenAI API key required when using OpenAI provider")
        if config.llm_provider == "anthropic" and not config.anthropic_api_key:
            raise ValueError("Anthropic API key required when using Anthropic provider")
            
        logger.info("Configuration loaded", provider=config.llm_provider)
        return config

# Global config instance
config = Config.from_env()
```

### Task 3: Logging Setup (10 min)
Create `src/logging_config.py`:

```python
import structlog
import logging
import sys
from pathlib import Path

def setup_logging(log_level: str = "INFO", log_file: str = "logs/agent.log"):
    """Configure structured logging for production"""
    
    # Create logs directory
    Path(log_file).parent.mkdir(exist_ok=True)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level),
    )
    
    # Add file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(file_handler)
```

### Task 4: Read About Agents (15 min)
Quick primer on agent architecture:

1. **What is an Agent?**
   - Autonomous system that can reason about tasks
   - Uses tools to interact with the world
   - Follows Think → Act → Observe loop

2. **ReAct Pattern**
   - **Re**asoning + **Act**ing
   - Agent thinks step-by-step
   - Each thought leads to an action
   - Each action produces an observation

3. **Key Components**
   - **LLM**: The "brain" for reasoning
   - **Tools**: Functions the agent can call
   - **Memory**: Context and history
   - **Loop**: The control flow

## 🌞 Afternoon Tasks (90-120 minutes)

### Task 5: Build Core Agent Class (45 min)
Create `src/agent.py`:

```python
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from src.llm_client import LLMClient
from src.tools.base import Tool, ToolResult

logger = structlog.get_logger()

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
    
    def __init__(
        self,
        llm_client: LLMClient,
        tools: List[Tool],
        max_iterations: int = 10
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
                if thought.startswith("Final Answer:"):
                    state.final_answer = thought.replace("Final Answer:", "").strip()
                    break
                
                # Act
                action = self._decide_action(state)
                state.actions.append(action)
                
                # Observe
                observation = self._execute_action(action)
                state.observations.append(observation)
                
            if not state.final_answer:
                state.final_answer = "I couldn't complete the task within the iteration limit."
                
            logger.info("Task completed", 
                       iterations=state.iteration_count,
                       answer_preview=state.final_answer[:100])
            return state.final_answer
            
        except Exception as e:
            logger.error("Agent error", error=str(e), exc_info=True)
            return f"An error occurred: {str(e)}"
    
    def _think(self, state: AgentState) -> str:
        """Generate next thought based on current state"""
        prompt = self._build_prompt(state)
        
        response = self.llm.complete(
            prompt,
            stop_sequences=["Action:", "Final Answer:"]
        )
        
        thought = response.strip()
        logger.debug("Generated thought", thought=thought)
        return thought
    
    def _decide_action(self, state: AgentState) -> Dict[str, Any]:
        """Decide which action to take"""
        # Get the last thought and extract action
        last_thought = state.thoughts[-1]
        
        # Ask LLM for action
        action_prompt = f"""
Based on this thought: "{last_thought}"

Available tools: {list(self.tools.keys())}

Respond with a JSON action:
{{"tool": "tool_name", "input": "tool_input"}}
"""
        
        response = self.llm.complete(action_prompt)
        
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
        tool_input = action.get("input")
        
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
        prompt = f"""You are an AI agent solving this task: {state.task}

Available tools: {list(self.tools.keys())}

Previous thoughts and actions:
"""
        
        for i, (thought, action, observation) in enumerate(
            zip(state.thoughts, state.actions, state.observations)
        ):
            prompt += f"\nThought {i+1}: {thought}"
            prompt += f"\nAction {i+1}: {action}"
            prompt += f"\nObservation {i+1}: {observation}\n"
        
        prompt += f"\nThought {len(state.thoughts) + 1}: "
        
        return prompt
```

### Task 6: Implement LLM Client (30 min)
Create `src/llm_client.py`:

```python
from abc import ABC, abstractmethod
from typing import List, Optional
import openai
from anthropic import Anthropic
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog

logger = structlog.get_logger()

class LLMClient(ABC):
    """Abstract base for LLM clients"""
    
    @abstractmethod
    def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """Generate completion for prompt"""
        pass

class OpenAIClient(LLMClient):
    """Production OpenAI client with retries"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(3),
        reraise=True
    )
    def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """Generate completion with retries"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI agent."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences
            )
            
            content = response.choices[0].message.content
            logger.debug("LLM completion", 
                        prompt_preview=prompt[:50],
                        response_preview=content[:50])
            return content
            
        except Exception as e:
            logger.error("OpenAI API error", error=str(e))
            raise

class AnthropicClient(LLMClient):
    """Production Anthropic client with retries"""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        self.client = Anthropic(api_key=api_key)
        self.model = model
        
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(3),
        reraise=True
    )
    def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """Generate completion with retries"""
        try:
            response = self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stop_sequences=stop_sequences or []
            )
            
            content = response.content[0].text
            logger.debug("LLM completion",
                        prompt_preview=prompt[:50],
                        response_preview=content[:50])
            return content
            
        except Exception as e:
            logger.error("Anthropic API error", error=str(e))
            raise

def create_llm_client(provider: str, api_key: str) -> LLMClient:
    """Factory for LLM clients"""
    if provider == "openai":
        return OpenAIClient(api_key)
    elif provider == "anthropic":
        return AnthropicClient(api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}")
```

### Task 7: Create Tool System (30 min)
Create `src/tools/base.py`:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional
import structlog

logger = structlog.get_logger()

@dataclass
class ToolResult:
    """Result from tool execution"""
    success: bool
    output: str
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class Tool(ABC):
    """Base class for all tools"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        logger.info("Tool initialized", name=name)
    
    @abstractmethod
    def execute(self, input_str: str) -> ToolResult:
        """Execute the tool with given input"""
        pass
    
    def validate_input(self, input_str: str) -> bool:
        """Validate input before execution"""
        return bool(input_str and input_str.strip())
    
    def __call__(self, input_str: str) -> ToolResult:
        """Make tool callable with validation"""
        if not self.validate_input(input_str):
            return ToolResult(
                success=False,
                output="",
                error="Invalid input"
            )
        
        try:
            return self.execute(input_str)
        except Exception as e:
            logger.error("Tool execution error", 
                        tool=self.name,
                        error=str(e),
                        exc_info=True)
            return ToolResult(
                success=False,
                output="",
                error=str(e)
            )
```

Create your first tool - `src/tools/calculator.py`:

```python
import ast
import operator
from typing import Any
import structlog

from src.tools.base import Tool, ToolResult

logger = structlog.get_logger()

class CalculatorTool(Tool):
    """Safe calculator tool using AST parsing"""
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform mathematical calculations"
        )
        
        # Safe operations
        self.operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
        }
    
    def execute(self, input_str: str) -> ToolResult:
        """Safely evaluate mathematical expression"""
        try:
            result = self._safe_eval(input_str)
            logger.info("Calculation performed", 
                       expression=input_str,
                       result=result)
            
            return ToolResult(
                success=True,
                output=str(result),
                metadata={"expression": input_str, "result": result}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Calculation error: {str(e)}"
            )
    
    def _safe_eval(self, expr: str) -> float:
        """Safely evaluate expression using AST"""
        node = ast.parse(expr, mode='eval')
        return self._eval_node(node.body)
    
    def _eval_node(self, node: Any) -> float:
        """Recursively evaluate AST node"""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return self.operators[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            return self.operators[type(node.op)](operand)
        else:
            raise ValueError(f"Unsupported operation: {type(node)}")
```

### Task 8: Wire Everything Together (15 min)
Create `src/main.py`:

```python
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import config
from src.logging_config import setup_logging
from src.agent import Agent
from src.llm_client import create_llm_client
from src.tools.calculator import CalculatorTool
import structlog

# Setup logging
setup_logging(config.log_level, config.log_file)
logger = structlog.get_logger()

def main():
    """Run the agent"""
    logger.info("Starting Agent47", config=config)
    
    # Create LLM client
    llm = create_llm_client(
        provider=config.llm_provider,
        api_key=config.openai_api_key or config.anthropic_api_key
    )
    
    # Create tools
    tools = [
        CalculatorTool()
    ]
    
    # Create agent
    agent = Agent(
        llm_client=llm,
        tools=tools,
        max_iterations=config.max_iterations
    )
    
    # Test with a simple task
    task = "What is 25 * 4 + 10?"
    logger.info("Running test task", task=task)
    
    result = agent.run(task)
    print(f"\nTask: {task}")
    print(f"Result: {result}")
    
    # Interactive mode
    print("\n" + "="*50)
    print("Agent47 Interactive Mode")
    print("Type 'exit' to quit")
    print("="*50 + "\n")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            break
            
        result = agent.run(user_input)
        print(f"Agent: {result}\n")

if __name__ == "__main__":
    main()
```

## 🌆 Evening Tasks (45-60 minutes)

### Task 9: Testing Your Agent (20 min)
Create `tests/test_agent.py`:

```python
import pytest
from unittest.mock import Mock, MagicMock

from src.agent import Agent, AgentState
from src.tools.base import Tool, ToolResult
from src.llm_client import LLMClient

class MockLLMClient(LLMClient):
    """Mock LLM for testing"""
    
    def __init__(self, responses: list):
        self.responses = responses
        self.call_count = 0
        
    def complete(self, prompt: str, **kwargs) -> str:
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response

class MockTool(Tool):
    """Mock tool for testing"""
    
    def __init__(self, name: str = "mock_tool", result: str = "mock result"):
        super().__init__(name, "Mock tool for testing")
        self.result = result
        self.calls = []
        
    def execute(self, input_str: str) -> ToolResult:
        self.calls.append(input_str)
        return ToolResult(success=True, output=self.result)

def test_agent_simple_task():
    """Test agent with simple task"""
    # Setup
    llm_responses = [
        "I need to calculate 2 + 2",
        '{"tool": "calculator", "input": "2 + 2"}',
        "Final Answer: The result is 4"
    ]
    
    llm = MockLLMClient(llm_responses)
    calculator = MockTool("calculator", "4")
    agent = Agent(llm, [calculator], max_iterations=5)
    
    # Execute
    result = agent.run("What is 2 + 2?")
    
    # Assert
    assert result == "The result is 4"
    assert len(calculator.calls) == 1
    assert calculator.calls[0] == "2 + 2"

def test_agent_max_iterations():
    """Test agent hits max iterations"""
    # Setup - LLM never gives final answer
    llm_responses = [
        "I need to think more",
        '{"tool": "calculator", "input": "1"}',
    ]
    
    llm = MockLLMClient(llm_responses)
    calculator = MockTool("calculator", "1")
    agent = Agent(llm, [calculator], max_iterations=3)
    
    # Execute
    result = agent.run("Solve the impossible")
    
    # Assert
    assert "iteration limit" in result
    assert agent.max_iterations == 3

def test_agent_error_handling():
    """Test agent handles errors gracefully"""
    # Setup - Tool that raises exception
    class ErrorTool(Tool):
        def execute(self, input_str: str) -> ToolResult:
            raise ValueError("Tool error")
    
    llm_responses = [
        "I'll use the error tool",
        '{"tool": "error_tool", "input": "test"}',
        "Final Answer: An error occurred"
    ]
    
    llm = MockLLMClient(llm_responses)
    error_tool = ErrorTool("error_tool", "Error tool")
    agent = Agent(llm, [error_tool])
    
    # Execute
    result = agent.run("Test error handling")
    
    # Should handle error gracefully
    assert "error occurred" in result.lower()

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### Task 10: Run and Debug (25 min)

1. **Create .env file**:
```bash
echo "OPENAI_API_KEY=your-key-here" > .env
# or
echo "ANTHROPIC_API_KEY=your-key-here" > .env
echo "LLM_PROVIDER=anthropic" >> .env
```

2. **Run your agent**:
```bash
python src/main.py
```

3. **Run tests**:
```bash
pytest tests/test_agent.py -v
```

4. **Check logs**:
```bash
tail -f logs/agent.log | jq '.'
```

### Task 11: Reflect and Document (15 min)

Update your learning journal in CLAUDE.md:

```markdown
## Day 1: Setup and First Agent

### What I Built
- ✅ Morning: Production-ready project structure with logging
- ✅ Afternoon: Working ReAct agent with calculator tool
- ✅ Evening: Test suite and debugging setup

### Key Learnings
1. **Technical**: 
   - Structured logging is crucial for debugging agents
   - Retry logic is essential for LLM calls
   - AST parsing makes calculator tool safe

2. **Architecture**:
   - Clean separation between agent, LLM, and tools
   - Factory pattern for LLM providers
   - State tracking helps with debugging

3. **Performance**:
   - Each LLM call costs time and money
   - Caching can help but needs careful design

### Challenges Faced
- **Issue**: LLM responses not always parseable as JSON
  **Solution**: Added fallback handling and better prompts
  **Lesson**: Always expect and handle malformed LLM output

### Tomorrow's Goal
- [ ] Implement robust output parsing
- [ ] Add more tools (file reader, web search)
- [ ] Improve agent prompting

### Code Metrics
- Lines written: ~500
- Tests added: 3
- Coverage: ~60%
- Agent success rate: 3/3 test tasks
```

## 📊 Deliverables Checklist
- [ ] Working agent that can solve math problems
- [ ] Production-ready project structure
- [ ] Comprehensive logging setup
- [ ] Error handling throughout
- [ ] Test suite with 3+ tests
- [ ] Clean, documented code
- [ ] Successfully ran agent interactively

## 🎯 Success Metrics
You've succeeded if you can:
1. Run `python src/main.py` and solve "What is 15 * 3 + 7?"
2. See structured JSON logs in `logs/agent.log`
3. Pass all tests with `pytest`
4. Add a new mathematical question and get correct answer
5. Understand how Think → Act → Observe loop works

## 🚀 Extension Challenges
If you finish early:
1. Add a `datetime` tool that tells current time
2. Implement token counting and cost tracking
3. Add prometheus metrics collection
4. Create a simple web UI with Flask
5. Add conversation memory that persists between runs

## 📚 Additional Resources
- [ReAct Paper](https://arxiv.org/abs/2210.03629) - The original paper
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling) - Modern approach
- [Structlog Documentation](https://www.structlog.org/) - For better logging
- [Tenacity Documentation](https://tenacity.readthedocs.io/) - Retry logic patterns

## 🤔 Common Issues & Solutions

### Issue: "No API key found"
```bash
# Make sure .env file exists and has:
OPENAI_API_KEY=sk-...
# or
ANTHROPIC_API_KEY=sk-ant-...
```

### Issue: "Import errors"
```bash
# Make sure you're in the venv:
which python  # Should show venv path
# If not:
source venv/bin/activate
```

### Issue: "Agent gets stuck"
- Check max_iterations is set
- Add better stop conditions in prompts
- Implement timeout handling

## 💡 Key Takeaways
1. **Start with production practices** - It's easier than retrofitting
2. **Logs are your best friend** - Structure them well
3. **Test early and often** - Mocking LLMs is essential
4. **Handle errors gracefully** - LLMs will surprise you
5. **Keep it simple** - Complexity can come later

---

🎉 **Congratulations!** You've built your first production-ready AI agent. Tomorrow we'll add more sophisticated parsing and additional tools.