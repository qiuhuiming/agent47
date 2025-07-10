# CLAUDE.md - Engineering-First AI Agent Development

> Your engineering companion for building production AI agents

## 🎯 Learning Philosophy

### Engineers Don't Need ML Theory
You're a software engineer. You understand:
- APIs and HTTP requests
- State machines and event loops  
- Design patterns and clean architecture
- Error handling and retries
- Testing and deployment

That's all you need. We treat LLMs as black-box APIs that:
- Accept text input (prompts)
- Return text output (completions)
- Sometimes fail (handle it)
- Cost money (optimize it)

### Production-First Mindset
From Day 1, write code like it's going to production because it will. This means:
- **Error handling** before happy path
- **Logging** from the start
- **Tests** as you build
- **Documentation** inline
- **Performance** considerations always

## 🏗️ Core Engineering Patterns

### 1. The Agent Loop Pattern
```python
class AgentStateMachine:
    """Treat your agent as a proper state machine"""
    
    states = ['IDLE', 'THINKING', 'ACTING', 'OBSERVING', 'COMPLETE', 'ERROR']
    
    def transition(self, from_state: str, to_state: str) -> None:
        if not self._valid_transition(from_state, to_state):
            raise InvalidStateTransition(f"{from_state} -> {to_state}")
        self.state = to_state
        self.emit_event(StateChanged(from_state, to_state))
```

### 2. Tool as Service Pattern
```python
class Tool(ABC):
    """Tools are microservices with a contract"""
    
    @abstractmethod
    def validate_input(self, input: Dict) -> bool:
        """Validate before execution"""
        
    @abstractmethod
    def execute(self, input: Dict) -> ToolResult:
        """Execute with proper error handling"""
        
    @abstractmethod
    def get_schema(self) -> Dict:
        """OpenAPI-style schema for LLM"""
```

### 3. Retry with Circuit Breaker
```python
class LLMClient:
    """Production-grade LLM client"""
    
    def __init__(self):
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=APIError
        )
        
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    @self.circuit_breaker
    async def complete(self, prompt: str) -> str:
        """Resilient API calls"""
```

## 🔧 Engineering Challenges & Solutions

### Challenge 1: LLM Output Parsing
**Problem**: LLMs return unstructured text that breaks parsers
**Solution**: Multiple fallback strategies
```python
def parse_llm_response(response: str) -> Action:
    strategies = [
        parse_json_block,      # Try ```json blocks first
        parse_xml_tags,        # Try <action> tags
        parse_regex_pattern,   # Try regex patterns
        parse_with_llm,        # Use another LLM call to parse
    ]
    
    for strategy in strategies:
        try:
            return strategy(response)
        except ParseError:
            continue
            
    raise UnparseableResponse(response)
```

### Challenge 2: Context Window Management
**Problem**: Agent memory exceeds token limits
**Solution**: Sliding window with importance scoring
```python
class ContextManager:
    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
        self.memories = PriorityQueue()
        
    def add_memory(self, content: str, importance: float):
        tokens = count_tokens(content)
        self.memories.put((-importance, tokens, content))
        self._trim_to_fit()
        
    def _trim_to_fit(self):
        total = 0
        kept = []
        while not self.memories.empty() and total < self.max_tokens:
            importance, tokens, content = self.memories.get()
            if total + tokens <= self.max_tokens:
                kept.append((importance, tokens, content))
                total += tokens
        
        # Restore kept memories
        for memory in kept:
            self.memories.put(memory)
```

### Challenge 3: Async Tool Execution
**Problem**: Tools block each other
**Solution**: Proper async orchestration
```python
class AsyncToolExecutor:
    async def execute_parallel(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """Execute independent tools in parallel"""
        tasks = []
        for call in tool_calls:
            if call.can_run_parallel:
                tasks.append(self._execute_tool(call))
            else:
                # Sequential execution for dependent tools
                result = await self._execute_tool(call)
                tasks.append(asyncio.create_task(asyncio.sleep(0)))
                
        return await asyncio.gather(*tasks, return_exceptions=True)
```

## 📊 Production Metrics

Track these from Day 1:
```python
class AgentMetrics:
    def __init__(self):
        self.counters = {
            'total_requests': Counter('agent_requests_total'),
            'errors': Counter('agent_errors_total'),
            'tool_calls': Counter('agent_tool_calls_total'),
        }
        self.histograms = {
            'response_time': Histogram('agent_response_seconds'),
            'tokens_used': Histogram('agent_tokens_used'),
            'cost': Histogram('agent_cost_dollars'),
        }
        self.gauges = {
            'active_agents': Gauge('agent_active_count'),
            'memory_usage': Gauge('agent_memory_bytes'),
        }
```

## 🔒 Security First

### Never Trust LLM Output
```python
class SafeToolExecutor:
    def execute(self, tool_name: str, args: Dict) -> Any:
        # Validate tool exists
        if tool_name not in self.allowed_tools:
            raise SecurityError(f"Tool {tool_name} not allowed")
            
        # Validate arguments
        schema = self.tools[tool_name].get_schema()
        if not validate_against_schema(args, schema):
            raise ValidationError("Invalid arguments")
            
        # Sandbox execution
        with ResourceLimits(cpu_time=30, memory=512*1024*1024):
            return self.tools[tool_name].execute(args)
```

### API Key Management
```python
class SecureConfig:
    """Never hardcode secrets"""
    
    def __init__(self):
        self.api_key = self._get_api_key()
        
    def _get_api_key(self) -> str:
        # Priority order
        sources = [
            lambda: os.environ.get('OPENAI_API_KEY'),
            lambda: keyring.get_password('agent47', 'openai'),
            lambda: self._prompt_for_key(),
        ]
        
        for source in sources:
            key = source()
            if key and self._validate_key_format(key):
                return key
                
        raise ConfigError("No valid API key found")
```

## 🚀 Performance Optimization

### 1. Cache Everything Safely
```python
class SmartCache:
    def __init__(self, ttl: int = 3600):
        self.cache = TTLCache(maxsize=1000, ttl=ttl)
        self.stats = CacheStats()
        
    def get_or_compute(self, key: str, compute_fn: Callable) -> Any:
        if key in self.cache:
            self.stats.hits += 1
            return self.cache[key]
            
        self.stats.misses += 1
        result = compute_fn()
        
        # Only cache successful results
        if result and not isinstance(result, Exception):
            self.cache[key] = result
            
        return result
```

### 2. Batch Operations
```python
class BatchProcessor:
    def __init__(self, batch_size: int = 10, flush_interval: float = 1.0):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.pending = []
        self._start_flush_timer()
        
    async def process(self, item: Any) -> Any:
        future = asyncio.Future()
        self.pending.append((item, future))
        
        if len(self.pending) >= self.batch_size:
            await self._flush()
            
        return await future
```

## 📈 Learning Progress Tracker

Use this template to track your daily progress:

```markdown
## Day X: [Topic]

### What I Built
- [ ] Morning: [Specific component]
- [ ] Afternoon: [Integration work]
- [ ] Evening: [Testing/polish]

### Key Learnings
1. **Technical**: [What worked/didn't work]
2. **Architecture**: [Design decisions made]
3. **Performance**: [Optimization opportunities]

### Challenges Faced
- **Issue**: [Description]
  **Solution**: [How you solved it]
  **Lesson**: [What you learned]

### Tomorrow's Goal
- [ ] [Specific objective]

### Code Metrics
- Lines written: X
- Tests added: Y
- Coverage: Z%
- Performance: N requests/sec
```

## 🎓 Advanced Patterns (Week 4+)

### Event Sourcing for Agents
```python
class EventSourcedAgent:
    """Every agent action is an event"""
    
    def __init__(self):
        self.events = []
        self.event_store = EventStore()
        
    def execute_action(self, action: Action) -> Result:
        event = ActionExecutedEvent(
            agent_id=self.id,
            action=action,
            timestamp=datetime.utcnow()
        )
        
        result = self._execute(action)
        
        event.result = result
        self.events.append(event)
        self.event_store.append(event)
        
        return result
        
    def replay_from_events(self, events: List[Event]) -> 'Agent':
        """Reconstruct agent state from events"""
```

### Multi-Agent Orchestration
```python
class AgentOrchestrator:
    """Manage multiple specialized agents"""
    
    def __init__(self):
        self.agents = {
            'researcher': ResearchAgent(),
            'coder': CodingAgent(),
            'reviewer': ReviewAgent(),
        }
        self.task_queue = asyncio.Queue()
        self.result_queue = asyncio.Queue()
        
    async def execute_pipeline(self, task: Task) -> Result:
        # Research phase
        research = await self.agents['researcher'].execute(task)
        
        # Coding phase (parallel if possible)
        coding_tasks = self._split_coding_work(research)
        code_results = await asyncio.gather(*[
            self.agents['coder'].execute(t) for t in coding_tasks
        ])
        
        # Review phase
        review = await self.agents['reviewer'].execute(code_results)
        
        return PipelineResult(research, code_results, review)
```

## 🔧 Debugging Tools

Build these as you go:
```python
class AgentDebugger:
    def __init__(self, agent: Agent):
        self.agent = agent
        self.traces = []
        
    def trace_execution(self, task: str) -> TraceResult:
        """Record everything for debugging"""
        with self.record_trace() as trace:
            result = self.agent.run(task)
            trace.add_result(result)
            
        return trace
        
    def replay_trace(self, trace_id: str) -> None:
        """Replay exact execution for debugging"""
        trace = self.load_trace(trace_id)
        
        for step in trace.steps:
            print(f"Step {step.number}: {step.action}")
            print(f"LLM Input: {step.llm_input}")
            print(f"LLM Output: {step.llm_output}")
            print(f"Tool Result: {step.tool_result}")
            print("-" * 80)
```

## 🏁 Remember

1. **You're an engineer** - Apply all your existing skills
2. **LLMs are just APIs** - Handle them like any external service
3. **Production from Day 1** - Build like it's going live
4. **Iterate fast** - Working code > perfect code
5. **Measure everything** - Data drives decisions

Happy building! 🚀