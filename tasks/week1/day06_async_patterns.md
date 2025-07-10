# Day 6: Async Patterns for Concurrent Agent Execution

## 🎯 Objectives
1. Master async/await patterns for concurrent tool execution
2. Build sophisticated task scheduling and prioritization
3. Implement backpressure and rate limiting mechanisms
4. Create async context managers and decorators
5. Design reactive agent architectures with event streams

## 📋 Prerequisites
- Completed Days 1-5
- Strong understanding of Python's asyncio
- Familiarity with concurrent programming concepts
- Basic knowledge of reactive programming (helpful)

## 🌄 Morning Tasks (60-75 minutes)

### Task 1: Advanced Async Patterns (25 min)
Create `src/async_patterns/core.py`:

```python
import asyncio
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, Coroutine
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
import inspect
import structlog
from contextlib import asynccontextmanager
from abc import ABC, abstractmethod

logger = structlog.get_logger()

T = TypeVar('T')
R = TypeVar('R')

# Async Pattern Decorators

def async_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """Retry async function with exponential backoff"""
    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            attempt = 0
            current_delay = delay
            
            while attempt < max_attempts:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        logger.error(f"Max retries exceeded for {func.__name__}")
                        raise
                    
                    logger.warning(
                        f"Retry {attempt}/{max_attempts} for {func.__name__} after {current_delay}s",
                        error=str(e)
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            
        return wrapper
    return decorator

def async_timeout(seconds: float):
    """Add timeout to async function"""
    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=seconds
                )
            except asyncio.TimeoutError:
                logger.error(f"Timeout after {seconds}s for {func.__name__}")
                raise
        
        return wrapper
    return decorator

def async_cached(ttl: Optional[float] = None):
    """Cache async function results"""
    cache: Dict[str, Tuple[Any, float]] = {}
    
    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            # Create cache key
            key = f"{args}:{kwargs}"
            
            # Check cache
            if key in cache:
                result, timestamp = cache[key]
                if ttl is None or (datetime.now().timestamp() - timestamp) < ttl:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return result
            
            # Call function
            result = await func(*args, **kwargs)
            
            # Store in cache
            cache[key] = (result, datetime.now().timestamp())
            
            return result
        
        wrapper.clear_cache = lambda: cache.clear()
        return wrapper
    
    return decorator

def async_throttle(rate: int, per: float = 1.0):
    """Throttle async function calls"""
    semaphore = asyncio.Semaphore(rate)
    
    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            async with semaphore:
                result = await func(*args, **kwargs)
                asyncio.create_task(release_after(per))
                return result
        
        async def release_after(delay: float):
            await asyncio.sleep(delay)
            semaphore.release()
        
        return wrapper
    return decorator

# Async Context Managers

@asynccontextmanager
async def async_timer(name: str):
    """Time async operations"""
    start = datetime.now()
    logger.info(f"Starting {name}")
    
    try:
        yield
    finally:
        duration = (datetime.now() - start).total_seconds()
        logger.info(f"Completed {name} in {duration:.2f}s")

@asynccontextmanager
async def async_resource_pool(
    create_resource: Callable[[], Coroutine[Any, Any, T]],
    max_size: int = 10
):
    """Manage a pool of async resources"""
    pool: List[T] = []
    available: asyncio.Queue[T] = asyncio.Queue()
    
    # Create initial resources
    for _ in range(max_size):
        resource = await create_resource()
        pool.append(resource)
        await available.put(resource)
    
    async def acquire() -> T:
        return await available.get()
    
    async def release(resource: T):
        await available.put(resource)
    
    try:
        yield (acquire, release)
    finally:
        # Cleanup resources
        for resource in pool:
            if hasattr(resource, 'close'):
                await resource.close()

# Async Primitives

class AsyncBatcher:
    """Batch async operations for efficiency"""
    
    def __init__(
        self,
        batch_size: int = 10,
        timeout: float = 1.0,
        process_batch: Callable[[List[T]], Coroutine[Any, Any, List[R]]] = None
    ):
        self.batch_size = batch_size
        self.timeout = timeout
        self.process_batch = process_batch
        self.pending: List[Tuple[T, asyncio.Future]] = []
        self._lock = asyncio.Lock()
        self._timer_task: Optional[asyncio.Task] = None
    
    async def add(self, item: T) -> R:
        """Add item to batch and wait for result"""
        future = asyncio.Future()
        
        async with self._lock:
            self.pending.append((item, future))
            
            if len(self.pending) >= self.batch_size:
                await self._process_pending()
            elif not self._timer_task:
                self._timer_task = asyncio.create_task(self._timeout_handler())
        
        return await future
    
    async def _timeout_handler(self):
        """Process batch after timeout"""
        await asyncio.sleep(self.timeout)
        async with self._lock:
            if self.pending:
                await self._process_pending()
            self._timer_task = None
    
    async def _process_pending(self):
        """Process pending batch"""
        if not self.pending:
            return
        
        batch = self.pending
        self.pending = []
        
        if self._timer_task:
            self._timer_task.cancel()
            self._timer_task = None
        
        try:
            # Extract items
            items = [item for item, _ in batch]
            
            # Process batch
            if self.process_batch:
                results = await self.process_batch(items)
            else:
                results = items  # Default: return items as-is
            
            # Resolve futures
            for (_, future), result in zip(batch, results):
                future.set_result(result)
                
        except Exception as e:
            # Reject all futures
            for _, future in batch:
                future.set_exception(e)

class AsyncRateLimiter:
    """Advanced rate limiter with multiple strategies"""
    
    def __init__(
        self,
        rate: int,
        per: float = 1.0,
        burst: Optional[int] = None,
        strategy: str = "sliding_window"
    ):
        self.rate = rate
        self.per = per
        self.burst = burst or rate
        self.strategy = strategy
        
        if strategy == "token_bucket":
            self._limiter = TokenBucketLimiter(rate, per, burst)
        elif strategy == "sliding_window":
            self._limiter = SlidingWindowLimiter(rate, per)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    async def acquire(self, tokens: int = 1):
        """Acquire permission to proceed"""
        await self._limiter.acquire(tokens)
    
    def __call__(self, func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        """Use as decorator"""
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            await self.acquire()
            return await func(*args, **kwargs)
        return wrapper

class TokenBucketLimiter:
    """Token bucket rate limiting"""
    
    def __init__(self, rate: int, per: float, burst: int):
        self.rate = rate
        self.per = per
        self.burst = burst
        self.tokens = burst
        self.last_update = asyncio.get_event_loop().time()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1):
        """Acquire tokens from bucket"""
        async with self._lock:
            # Refill tokens
            now = asyncio.get_event_loop().time()
            elapsed = now - self.last_update
            self.tokens = min(
                self.burst,
                self.tokens + elapsed * self.rate / self.per
            )
            self.last_update = now
            
            # Wait for tokens if needed
            while self.tokens < tokens:
                wait_time = (tokens - self.tokens) * self.per / self.rate
                await asyncio.sleep(wait_time)
                
                # Refill again
                now = asyncio.get_event_loop().time()
                elapsed = now - self.last_update
                self.tokens = min(
                    self.burst,
                    self.tokens + elapsed * self.rate / self.per
                )
                self.last_update = now
            
            # Consume tokens
            self.tokens -= tokens

class SlidingWindowLimiter:
    """Sliding window rate limiting"""
    
    def __init__(self, rate: int, per: float):
        self.rate = rate
        self.per = per
        self.requests: List[float] = []
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1):
        """Acquire permission"""
        async with self._lock:
            now = asyncio.get_event_loop().time()
            
            # Remove old requests
            cutoff = now - self.per
            self.requests = [t for t in self.requests if t > cutoff]
            
            # Check if we can proceed
            while len(self.requests) + tokens > self.rate:
                # Wait until oldest request expires
                if self.requests:
                    wait_time = self.requests[0] + self.per - now
                    await asyncio.sleep(wait_time)
                    
                    # Clean up again
                    now = asyncio.get_event_loop().time()
                    cutoff = now - self.per
                    self.requests = [t for t in self.requests if t > cutoff]
                else:
                    break
            
            # Record new requests
            for _ in range(tokens):
                self.requests.append(now)

# Async Patterns

class AsyncPipeline:
    """Chain async operations together"""
    
    def __init__(self):
        self.stages: List[Callable] = []
    
    def add_stage(self, func: Callable) -> 'AsyncPipeline':
        """Add a stage to the pipeline"""
        self.stages.append(func)
        return self
    
    async def execute(self, initial_value: Any) -> Any:
        """Execute pipeline with initial value"""
        result = initial_value
        
        for stage in self.stages:
            if inspect.iscoroutinefunction(stage):
                result = await stage(result)
            else:
                result = stage(result)
        
        return result
    
    def __or__(self, other: Callable) -> 'AsyncPipeline':
        """Use | operator to add stages"""
        return self.add_stage(other)

class AsyncCircuit:
    """Circuit pattern for async operations"""
    
    def __init__(self):
        self.operations: Dict[str, Callable] = {}
        self.connections: Dict[str, List[str]] = {}
    
    def add_operation(self, name: str, operation: Callable) -> 'AsyncCircuit':
        """Add an operation to the circuit"""
        self.operations[name] = operation
        return self
    
    def connect(self, from_op: str, to_op: str) -> 'AsyncCircuit':
        """Connect operations"""
        if from_op not in self.connections:
            self.connections[from_op] = []
        self.connections[from_op].append(to_op)
        return self
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute circuit with given inputs"""
        results = {}
        executed = set()
        
        async def execute_op(op_name: str):
            if op_name in executed:
                return
            
            executed.add(op_name)
            
            # Get operation
            operation = self.operations.get(op_name)
            if not operation:
                return
            
            # Get input (from previous results or initial inputs)
            op_input = inputs.get(op_name, results.get(op_name))
            
            # Execute
            if inspect.iscoroutinefunction(operation):
                result = await operation(op_input)
            else:
                result = operation(op_input)
            
            results[op_name] = result
            
            # Execute connected operations
            if op_name in self.connections:
                tasks = [
                    execute_op(next_op)
                    for next_op in self.connections[op_name]
                ]
                await asyncio.gather(*tasks)
        
        # Execute all operations
        tasks = [execute_op(op_name) for op_name in self.operations]
        await asyncio.gather(*tasks)
        
        return results
```

### Task 2: Concurrent Tool Execution (20 min)
Create `src/async_patterns/concurrent_tools.py`:

```python
from typing import Any, Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field
import asyncio
from datetime import datetime
import structlog

from src.tools.base import Tool, ToolResult
from src.async_patterns.core import AsyncRateLimiter, AsyncBatcher

logger = structlog.get_logger()

@dataclass
class ToolCall:
    """Represents a tool call request"""
    tool_name: str
    input_data: Any
    priority: int = 0
    timeout: Optional[float] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConcurrentToolResult:
    """Result from concurrent tool execution"""
    tool_name: str
    result: Optional[ToolResult]
    error: Optional[str] = None
    duration: float = 0.0
    retries: int = 0

class ConcurrentToolExecutor:
    """Execute tools concurrently with advanced control"""
    
    def __init__(
        self,
        tools: Dict[str, Tool],
        max_concurrent: int = 5,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        rate_limit: Optional[int] = None
    ):
        self.tools = tools
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Concurrency control
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Rate limiting
        self.rate_limiter = AsyncRateLimiter(
            rate=rate_limit or max_concurrent * 2,
            per=1.0
        ) if rate_limit else None
        
        # Metrics
        self.execution_count = 0
        self.error_count = 0
        
    async def execute_single(
        self,
        tool_call: ToolCall
    ) -> ConcurrentToolResult:
        """Execute a single tool call"""
        start_time = datetime.now()
        
        # Apply rate limiting if configured
        if self.rate_limiter:
            await self.rate_limiter.acquire()
        
        async with self.semaphore:
            for attempt in range(self.max_retries):
                try:
                    # Get tool
                    tool = self.tools.get(tool_call.tool_name)
                    if not tool:
                        return ConcurrentToolResult(
                            tool_name=tool_call.tool_name,
                            result=None,
                            error=f"Tool not found: {tool_call.tool_name}"
                        )
                    
                    # Execute with timeout
                    if tool_call.timeout:
                        result = await asyncio.wait_for(
                            tool.execute(tool_call.input_data),
                            timeout=tool_call.timeout
                        )
                    else:
                        result = await tool.execute(tool_call.input_data)
                    
                    # Success
                    duration = (datetime.now() - start_time).total_seconds()
                    self.execution_count += 1
                    
                    return ConcurrentToolResult(
                        tool_name=tool_call.tool_name,
                        result=result,
                        duration=duration,
                        retries=attempt
                    )
                    
                except asyncio.TimeoutError:
                    error = f"Tool {tool_call.tool_name} timed out"
                    logger.warning(error)
                    
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                        continue
                    
                    self.error_count += 1
                    return ConcurrentToolResult(
                        tool_name=tool_call.tool_name,
                        result=None,
                        error=error,
                        duration=(datetime.now() - start_time).total_seconds(),
                        retries=attempt
                    )
                    
                except Exception as e:
                    error = f"Tool {tool_call.tool_name} error: {str(e)}"
                    logger.error(error, exc_info=True)
                    
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                        continue
                    
                    self.error_count += 1
                    return ConcurrentToolResult(
                        tool_name=tool_call.tool_name,
                        result=None,
                        error=error,
                        duration=(datetime.now() - start_time).total_seconds(),
                        retries=attempt
                    )
    
    async def execute_batch(
        self,
        tool_calls: List[ToolCall]
    ) -> List[ConcurrentToolResult]:
        """Execute multiple tool calls concurrently"""
        # Sort by priority (higher first)
        sorted_calls = sorted(tool_calls, key=lambda x: x.priority, reverse=True)
        
        # Create tasks
        tasks = [
            asyncio.create_task(self.execute_single(call))
            for call in sorted_calls
        ]
        
        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for call, result in zip(sorted_calls, results):
            if isinstance(result, Exception):
                final_results.append(ConcurrentToolResult(
                    tool_name=call.tool_name,
                    result=None,
                    error=str(result)
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    async def execute_dag(
        self,
        tool_calls: Dict[str, ToolCall],
        dependencies: Dict[str, Set[str]]
    ) -> Dict[str, ConcurrentToolResult]:
        """Execute tools as a DAG with dependencies"""
        results = {}
        completed = set()
        in_progress = set()
        
        async def can_execute(tool_name: str) -> bool:
            """Check if tool can be executed"""
            if tool_name in completed or tool_name in in_progress:
                return False
            
            # Check dependencies
            deps = dependencies.get(tool_name, set())
            return all(dep in completed for dep in deps)
        
        async def execute_tool(tool_name: str):
            """Execute a tool and its dependents"""
            in_progress.add(tool_name)
            
            try:
                # Execute tool
                tool_call = tool_calls[tool_name]
                result = await self.execute_single(tool_call)
                results[tool_name] = result
                
            finally:
                in_progress.remove(tool_name)
                completed.add(tool_name)
            
            # Check for newly executable tools
            for next_tool in tool_calls:
                if await can_execute(next_tool):
                    asyncio.create_task(execute_tool(next_tool))
        
        # Start with tools that have no dependencies
        initial_tasks = []
        for tool_name in tool_calls:
            if await can_execute(tool_name):
                initial_tasks.append(execute_tool(tool_name))
        
        # Wait for all to complete
        if initial_tasks:
            await asyncio.gather(*initial_tasks)
        
        # Wait for any remaining tasks
        while len(completed) < len(tool_calls):
            await asyncio.sleep(0.1)
        
        return results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics"""
        return {
            "total_executions": self.execution_count,
            "total_errors": self.error_count,
            "error_rate": self.error_count / self.execution_count if self.execution_count > 0 else 0,
            "max_concurrent": self.max_concurrent,
        }

class AdaptiveConcurrencyExecutor(ConcurrentToolExecutor):
    """Executor that adapts concurrency based on performance"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Adaptive parameters
        self.min_concurrent = 1
        self.target_latency = 1.0  # Target latency in seconds
        self.adjustment_interval = 10  # Adjust every N executions
        
        # Metrics for adaptation
        self.recent_latencies = []
        self.executions_since_adjustment = 0
    
    async def execute_single(self, tool_call: ToolCall) -> ConcurrentToolResult:
        """Execute with adaptive concurrency tracking"""
        result = await super().execute_single(tool_call)
        
        # Track latency
        self.recent_latencies.append(result.duration)
        if len(self.recent_latencies) > 100:
            self.recent_latencies.pop(0)
        
        # Adjust concurrency periodically
        self.executions_since_adjustment += 1
        if self.executions_since_adjustment >= self.adjustment_interval:
            await self._adjust_concurrency()
            self.executions_since_adjustment = 0
        
        return result
    
    async def _adjust_concurrency(self):
        """Adjust concurrency based on performance"""
        if not self.recent_latencies:
            return
        
        avg_latency = sum(self.recent_latencies) / len(self.recent_latencies)
        
        if avg_latency > self.target_latency * 1.5:
            # Reduce concurrency if latency is too high
            new_value = max(self.min_concurrent, self.max_concurrent - 1)
            if new_value < self.max_concurrent:
                self.max_concurrent = new_value
                self.semaphore = asyncio.Semaphore(new_value)
                logger.info(f"Reduced concurrency to {new_value}")
                
        elif avg_latency < self.target_latency * 0.7:
            # Increase concurrency if latency is low
            new_value = self.max_concurrent + 1
            self.max_concurrent = new_value
            self.semaphore = asyncio.Semaphore(new_value)
            logger.info(f"Increased concurrency to {new_value}")

class ToolBatchProcessor:
    """Process tool calls in optimized batches"""
    
    def __init__(
        self,
        executor: ConcurrentToolExecutor,
        batch_size: int = 10,
        batch_timeout: float = 1.0
    ):
        self.executor = executor
        self.batcher = AsyncBatcher(
            batch_size=batch_size,
            timeout=batch_timeout,
            process_batch=self._process_batch
        )
    
    async def add_tool_call(self, tool_call: ToolCall) -> ConcurrentToolResult:
        """Add tool call to batch"""
        return await self.batcher.add(tool_call)
    
    async def _process_batch(
        self,
        tool_calls: List[ToolCall]
    ) -> List[ConcurrentToolResult]:
        """Process a batch of tool calls"""
        logger.info(f"Processing batch of {len(tool_calls)} tool calls")
        return await self.executor.execute_batch(tool_calls)
```

### Task 3: Task Scheduling and Priority (20 min)
Create `src/async_patterns/scheduling.py`:

```python
from typing import Any, Callable, Dict, List, Optional, Coroutine
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import heapq
from enum import Enum, auto
import structlog

logger = structlog.get_logger()

class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5

@dataclass
class ScheduledTask:
    """A task to be scheduled"""
    id: str
    coroutine: Coroutine[Any, Any, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    scheduled_time: Optional[datetime] = None
    deadline: Optional[datetime] = None
    dependencies: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """Compare tasks for priority queue"""
        # First by priority, then by scheduled time
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        if self.scheduled_time and other.scheduled_time:
            return self.scheduled_time < other.scheduled_time
        return False

@dataclass
class TaskResult:
    """Result of task execution"""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration: Optional[float] = None

class AsyncTaskScheduler:
    """Advanced async task scheduler"""
    
    def __init__(
        self,
        max_workers: int = 10,
        enable_priorities: bool = True,
        enable_deadlines: bool = True
    ):
        self.max_workers = max_workers
        self.enable_priorities = enable_priorities
        self.enable_deadlines = enable_deadlines
        
        # Task management
        self.pending_tasks: List[ScheduledTask] = []
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks: Dict[str, TaskResult] = {}
        
        # Worker pool
        self.worker_semaphore = asyncio.Semaphore(max_workers)
        self.running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        
        # Metrics
        self.total_scheduled = 0
        self.total_completed = 0
        self.total_failed = 0
    
    async def start(self):
        """Start the scheduler"""
        if self.running:
            return
        
        self.running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Task scheduler started")
    
    async def stop(self):
        """Stop the scheduler"""
        self.running = False
        
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        # Wait for running tasks
        if self.running_tasks:
            await asyncio.gather(
                *self.running_tasks.values(),
                return_exceptions=True
            )
        
        logger.info("Task scheduler stopped")
    
    async def schedule_task(
        self,
        task_id: str,
        coroutine: Coroutine[Any, Any, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        scheduled_time: Optional[datetime] = None,
        deadline: Optional[datetime] = None,
        dependencies: Optional[Set[str]] = None
    ) -> str:
        """Schedule a task for execution"""
        task = ScheduledTask(
            id=task_id,
            coroutine=coroutine,
            priority=priority,
            scheduled_time=scheduled_time or datetime.now(),
            deadline=deadline,
            dependencies=dependencies or set()
        )
        
        # Add to pending queue
        heapq.heappush(self.pending_tasks, task)
        self.total_scheduled += 1
        
        logger.info(
            f"Scheduled task {task_id}",
            priority=priority.name,
            scheduled_time=scheduled_time
        )
        
        return task_id
    
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> TaskResult:
        """Wait for a task to complete"""
        end_time = datetime.now() + timedelta(seconds=timeout) if timeout else None
        
        while task_id not in self.completed_tasks:
            if end_time and datetime.now() > end_time:
                raise asyncio.TimeoutError(f"Task {task_id} timed out")
            
            await asyncio.sleep(0.1)
        
        return self.completed_tasks[task_id]
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        # Remove from pending if not started
        self.pending_tasks = [
            t for t in self.pending_tasks
            if t.id != task_id
        ]
        
        # Cancel if running
        if task_id in self.running_tasks:
            self.running_tasks[task_id].cancel()
            return True
        
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status"""
        return {
            "running": self.running,
            "pending_tasks": len(self.pending_tasks),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks),
            "total_scheduled": self.total_scheduled,
            "total_completed": self.total_completed,
            "total_failed": self.total_failed,
            "worker_usage": len(self.running_tasks) / self.max_workers
        }
    
    async def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.running:
            try:
                # Check for ready tasks
                ready_tasks = self._get_ready_tasks()
                
                # Start tasks up to worker limit
                for task in ready_tasks:
                    if len(self.running_tasks) >= self.max_workers:
                        break
                    
                    # Start task execution
                    asyncio.create_task(self._execute_task(task))
                
                # Check deadlines
                if self.enable_deadlines:
                    self._check_deadlines()
                
                await asyncio.sleep(0.1)  # Small delay
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}", exc_info=True)
    
    def _get_ready_tasks(self) -> List[ScheduledTask]:
        """Get tasks ready for execution"""
        ready = []
        now = datetime.now()
        
        # Sort pending tasks
        heapq.heapify(self.pending_tasks)
        
        while self.pending_tasks:
            task = self.pending_tasks[0]
            
            # Check if scheduled time has arrived
            if task.scheduled_time and task.scheduled_time > now:
                break
            
            # Check dependencies
            if task.dependencies:
                completed_ids = set(self.completed_tasks.keys())
                if not task.dependencies.issubset(completed_ids):
                    # Dependencies not met, skip for now
                    heapq.heappop(self.pending_tasks)
                    heapq.heappush(self.pending_tasks, task)
                    break
            
            # Task is ready
            ready.append(heapq.heappop(self.pending_tasks))
        
        return ready
    
    def _check_deadlines(self):
        """Check for tasks approaching deadlines"""
        now = datetime.now()
        
        for task in self.pending_tasks:
            if task.deadline and task.deadline < now:
                logger.warning(
                    f"Task {task.id} missed deadline",
                    deadline=task.deadline
                )
                # Could implement deadline handling here
    
    async def _execute_task(self, task: ScheduledTask):
        """Execute a single task"""
        async with self.worker_semaphore:
            task_result = TaskResult(
                task_id=task.id,
                success=False,
                start_time=datetime.now()
            )
            
            # Create asyncio task
            asyncio_task = asyncio.create_task(task.coroutine)
            self.running_tasks[task.id] = asyncio_task
            
            try:
                # Execute with deadline if specified
                if task.deadline:
                    timeout = (task.deadline - datetime.now()).total_seconds()
                    if timeout > 0:
                        result = await asyncio.wait_for(asyncio_task, timeout=timeout)
                    else:
                        raise asyncio.TimeoutError("Deadline already passed")
                else:
                    result = await asyncio_task
                
                # Success
                task_result.success = True
                task_result.result = result
                self.total_completed += 1
                
            except asyncio.TimeoutError:
                task_result.error = "Task deadline exceeded"
                self.total_failed += 1
                logger.error(f"Task {task.id} exceeded deadline")
                
            except asyncio.CancelledError:
                task_result.error = "Task cancelled"
                self.total_failed += 1
                logger.info(f"Task {task.id} cancelled")
                raise
                
            except Exception as e:
                task_result.error = str(e)
                self.total_failed += 1
                logger.error(f"Task {task.id} failed: {e}", exc_info=True)
            
            finally:
                # Record completion
                task_result.end_time = datetime.now()
                task_result.duration = (
                    task_result.end_time - task_result.start_time
                ).total_seconds()
                
                self.completed_tasks[task.id] = task_result
                self.running_tasks.pop(task.id, None)
                
                logger.info(
                    f"Task {task.id} completed",
                    success=task_result.success,
                    duration=task_result.duration
                )

class FairShareScheduler(AsyncTaskScheduler):
    """Scheduler with fair sharing between users/groups"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user_queues: Dict[str, List[ScheduledTask]] = {}
        self.user_credits: Dict[str, float] = {}
        self.default_credits = 100.0
    
    async def schedule_task_for_user(
        self,
        user_id: str,
        task_id: str,
        coroutine: Coroutine[Any, Any, Any],
        cost: float = 1.0,
        **kwargs
    ) -> str:
        """Schedule task for specific user with fair sharing"""
        # Initialize user if needed
        if user_id not in self.user_queues:
            self.user_queues[user_id] = []
            self.user_credits[user_id] = self.default_credits
        
        # Check credits
        if self.user_credits[user_id] < cost:
            raise ValueError(f"Insufficient credits for user {user_id}")
        
        # Deduct credits
        self.user_credits[user_id] -= cost
        
        # Create task with user metadata
        task = ScheduledTask(
            id=task_id,
            coroutine=coroutine,
            metadata={"user_id": user_id, "cost": cost},
            **kwargs
        )
        
        self.user_queues[user_id].append(task)
        
        return task_id
    
    def _get_ready_tasks(self) -> List[ScheduledTask]:
        """Get tasks with fair sharing"""
        ready = []
        
        # Round-robin through users with available credits
        for user_id, queue in self.user_queues.items():
            if queue and self.user_credits[user_id] > 0:
                # Take one task from this user
                task = queue.pop(0)
                ready.append(task)
        
        return ready[:self.max_workers - len(self.running_tasks)]
    
    def add_credits(self, user_id: str, credits: float):
        """Add credits to user"""
        if user_id not in self.user_credits:
            self.user_credits[user_id] = 0
        self.user_credits[user_id] += credits
```

### Task 4: Backpressure and Flow Control (10 min)
Create `src/async_patterns/backpressure.py`:

```python
from typing import Any, Callable, Optional, TypeVar, Generic
from dataclasses import dataclass
import asyncio
from collections import deque
import structlog

logger = structlog.get_logger()

T = TypeVar('T')

class BackpressureQueue(Generic[T]):
    """Queue with backpressure support"""
    
    def __init__(
        self,
        max_size: int = 100,
        high_watermark: float = 0.8,
        low_watermark: float = 0.2
    ):
        self.max_size = max_size
        self.high_watermark = int(max_size * high_watermark)
        self.low_watermark = int(max_size * low_watermark)
        
        self.queue: deque[T] = deque()
        self._waiters: deque[asyncio.Future] = deque()
        self._pressure_high = False
        
        # Callbacks
        self.on_pressure_high: Optional[Callable] = None
        self.on_pressure_low: Optional[Callable] = None
    
    async def put(self, item: T) -> None:
        """Put item with backpressure"""
        while len(self.queue) >= self.max_size:
            # Queue full, wait
            waiter = asyncio.Future()
            self._waiters.append(waiter)
            
            try:
                await waiter
            except:
                self._waiters.remove(waiter)
                raise
        
        self.queue.append(item)
        
        # Check pressure
        if len(self.queue) >= self.high_watermark and not self._pressure_high:
            self._pressure_high = True
            if self.on_pressure_high:
                await self._call_callback(self.on_pressure_high)
    
    async def get(self) -> T:
        """Get item from queue"""
        while not self.queue:
            await asyncio.sleep(0.01)
        
        item = self.queue.popleft()
        
        # Wake up a waiter if any
        while self._waiters and len(self.queue) < self.max_size:
            waiter = self._waiters.popleft()
            if not waiter.done():
                waiter.set_result(None)
                break
        
        # Check pressure
        if len(self.queue) <= self.low_watermark and self._pressure_high:
            self._pressure_high = False
            if self.on_pressure_low:
                await self._call_callback(self.on_pressure_low)
        
        return item
    
    async def _call_callback(self, callback: Callable):
        """Call callback safely"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback()
            else:
                callback()
        except Exception as e:
            logger.error(f"Callback error: {e}")
    
    def size(self) -> int:
        """Get current queue size"""
        return len(self.queue)
    
    def is_full(self) -> bool:
        """Check if queue is full"""
        return len(self.queue) >= self.max_size
    
    def pressure_status(self) -> str:
        """Get pressure status"""
        size = len(self.queue)
        if size >= self.high_watermark:
            return "high"
        elif size <= self.low_watermark:
            return "low"
        else:
            return "normal"

class FlowController:
    """Control flow between producer and consumer"""
    
    def __init__(
        self,
        target_rate: float = 10.0,  # items per second
        burst_size: int = 20,
        adaptation_rate: float = 0.1
    ):
        self.target_rate = target_rate
        self.burst_size = burst_size
        self.adaptation_rate = adaptation_rate
        
        # Current state
        self.current_rate = target_rate
        self.tokens = burst_size
        self.last_update = asyncio.get_event_loop().time()
        
        # Metrics
        self.processed_count = 0
        self.dropped_count = 0
        self.rate_history = deque(maxlen=100)
    
    async def acquire(self) -> bool:
        """Try to acquire permission to process"""
        now = asyncio.get_event_loop().time()
        elapsed = now - self.last_update
        
        # Refill tokens
        self.tokens = min(
            self.burst_size,
            self.tokens + elapsed * self.current_rate
        )
        self.last_update = now
        
        if self.tokens >= 1:
            self.tokens -= 1
            self.processed_count += 1
            return True
        else:
            self.dropped_count += 1
            return False
    
    def adapt_rate(self, queue_size: int, max_queue_size: int):
        """Adapt rate based on queue pressure"""
        pressure = queue_size / max_queue_size
        
        if pressure > 0.8:
            # High pressure, reduce rate
            self.current_rate *= (1 - self.adaptation_rate)
        elif pressure < 0.2:
            # Low pressure, increase rate
            self.current_rate *= (1 + self.adaptation_rate)
        
        # Clamp rate
        self.current_rate = max(1.0, min(self.target_rate * 2, self.current_rate))
        
        # Record for history
        self.rate_history.append((asyncio.get_event_loop().time(), self.current_rate))
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get flow control metrics"""
        return {
            "current_rate": self.current_rate,
            "target_rate": self.target_rate,
            "processed": self.processed_count,
            "dropped": self.dropped_count,
            "drop_rate": self.dropped_count / (self.processed_count + self.dropped_count)
            if self.processed_count + self.dropped_count > 0 else 0
        }

class AsyncProducerConsumer:
    """Producer-consumer with backpressure"""
    
    def __init__(
        self,
        producer: Callable[[], AsyncIterator[T]],
        consumer: Callable[[T], Coroutine[Any, Any, None]],
        queue_size: int = 100,
        num_consumers: int = 5
    ):
        self.producer = producer
        self.consumer = consumer
        self.queue = BackpressureQueue[T](max_size=queue_size)
        self.num_consumers = num_consumers
        
        # Flow control
        self.flow_controller = FlowController()
        
        # State
        self.running = False
        self.producer_task: Optional[asyncio.Task] = None
        self.consumer_tasks: List[asyncio.Task] = []
        
        # Metrics
        self.produced_count = 0
        self.consumed_count = 0
        
        # Setup backpressure callbacks
        self.queue.on_pressure_high = self._on_high_pressure
        self.queue.on_pressure_low = self._on_low_pressure
    
    async def start(self):
        """Start producer and consumers"""
        if self.running:
            return
        
        self.running = True
        
        # Start producer
        self.producer_task = asyncio.create_task(self._produce())
        
        # Start consumers
        for i in range(self.num_consumers):
            task = asyncio.create_task(self._consume(i))
            self.consumer_tasks.append(task)
        
        logger.info(f"Started producer-consumer with {self.num_consumers} consumers")
    
    async def stop(self):
        """Stop producer and consumers"""
        self.running = False
        
        # Cancel producer
        if self.producer_task:
            self.producer_task.cancel()
            try:
                await self.producer_task
            except asyncio.CancelledError:
                pass
        
        # Cancel consumers
        for task in self.consumer_tasks:
            task.cancel()
        
        await asyncio.gather(*self.consumer_tasks, return_exceptions=True)
        
        logger.info("Stopped producer-consumer")
    
    async def _produce(self):
        """Producer coroutine"""
        try:
            async for item in self.producer():
                if not self.running:
                    break
                
                # Apply flow control
                if await self.flow_controller.acquire():
                    await self.queue.put(item)
                    self.produced_count += 1
                else:
                    logger.debug("Producer throttled")
                
                # Adapt rate based on queue pressure
                self.flow_controller.adapt_rate(
                    self.queue.size(),
                    self.queue.max_size
                )
                
        except Exception as e:
            logger.error(f"Producer error: {e}", exc_info=True)
    
    async def _consume(self, consumer_id: int):
        """Consumer coroutine"""
        try:
            while self.running:
                try:
                    item = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=1.0
                    )
                    
                    await self.consumer(item)
                    self.consumed_count += 1
                    
                except asyncio.TimeoutError:
                    continue
                    
        except Exception as e:
            logger.error(f"Consumer {consumer_id} error: {e}", exc_info=True)
    
    async def _on_high_pressure(self):
        """Handle high pressure"""
        logger.warning("Backpressure high - slowing producer")
        # Could implement additional actions here
    
    async def _on_low_pressure(self):
        """Handle low pressure"""
        logger.info("Backpressure low - resuming normal rate")
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "running": self.running,
            "queue_size": self.queue.size(),
            "queue_pressure": self.queue.pressure_status(),
            "produced": self.produced_count,
            "consumed": self.consumed_count,
            "flow_metrics": self.flow_controller.get_metrics()
        }
```

## 🌞 Afternoon Tasks (90-120 minutes)

### Task 5: Reactive Agent Architecture (45 min)
Create `src/async_patterns/reactive_agent.py`:

```python
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
from enum import Enum, auto
import structlog

logger = structlog.get_logger()

T = TypeVar('T')

class EventType(Enum):
    """Types of events in the system"""
    TASK_RECEIVED = auto()
    THOUGHT_GENERATED = auto()
    ACTION_REQUIRED = auto()
    TOOL_EXECUTED = auto()
    RESULT_AVAILABLE = auto()
    ERROR_OCCURRED = auto()
    STATE_CHANGED = auto()

@dataclass
class Event:
    """Base event class"""
    type: EventType
    source: str
    data: Any
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None

class EventStream:
    """Async event stream"""
    
    def __init__(self):
        self.subscribers: Dict[EventType, List[asyncio.Queue]] = {}
        self.all_subscribers: List[asyncio.Queue] = []
        self.event_history: List[Event] = []
        self.max_history = 1000
    
    async def publish(self, event: Event):
        """Publish event to subscribers"""
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history.pop(0)
        
        # Notify type-specific subscribers
        if event.type in self.subscribers:
            for queue in self.subscribers[event.type]:
                await queue.put(event)
        
        # Notify all-event subscribers
        for queue in self.all_subscribers:
            await queue.put(event)
        
        logger.debug(f"Published event: {event.type.name} from {event.source}")
    
    def subscribe(self, event_type: Optional[EventType] = None) -> asyncio.Queue:
        """Subscribe to events"""
        queue = asyncio.Queue()
        
        if event_type:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            self.subscribers[event_type].append(queue)
        else:
            self.all_subscribers.append(queue)
        
        return queue
    
    def unsubscribe(self, queue: asyncio.Queue, event_type: Optional[EventType] = None):
        """Unsubscribe from events"""
        if event_type and event_type in self.subscribers:
            self.subscribers[event_type].remove(queue)
        elif queue in self.all_subscribers:
            self.all_subscribers.remove(queue)

class ReactiveComponent:
    """Base class for reactive components"""
    
    def __init__(self, name: str, event_stream: EventStream):
        self.name = name
        self.event_stream = event_stream
        self.running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the component"""
        if self.running:
            return
        
        self.running = True
        self._task = asyncio.create_task(self._run())
        logger.info(f"Started reactive component: {self.name}")
    
    async def stop(self):
        """Stop the component"""
        self.running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"Stopped reactive component: {self.name}")
    
    async def _run(self):
        """Component main loop - override in subclasses"""
        pass
    
    async def emit(self, event_type: EventType, data: Any, correlation_id: Optional[str] = None):
        """Emit an event"""
        event = Event(
            type=event_type,
            source=self.name,
            data=data,
            correlation_id=correlation_id
        )
        await self.event_stream.publish(event)

class ReactiveThinkingComponent(ReactiveComponent):
    """Component that generates thoughts based on events"""
    
    def __init__(self, name: str, event_stream: EventStream, llm_client):
        super().__init__(name, event_stream)
        self.llm_client = llm_client
        self.task_queue = self.event_stream.subscribe(EventType.TASK_RECEIVED)
        self.result_queue = self.event_stream.subscribe(EventType.RESULT_AVAILABLE)
        self.context = {}
    
    async def _run(self):
        """Process events and generate thoughts"""
        while self.running:
            try:
                # Wait for events (with timeout to allow checking running state)
                try:
                    event = await asyncio.wait_for(
                        self._get_next_event(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                if event.type == EventType.TASK_RECEIVED:
                    await self._handle_task(event)
                elif event.type == EventType.RESULT_AVAILABLE:
                    await self._handle_result(event)
                    
            except Exception as e:
                logger.error(f"Thinking component error: {e}", exc_info=True)
                await self.emit(
                    EventType.ERROR_OCCURRED,
                    {"error": str(e), "component": self.name}
                )
    
    async def _get_next_event(self) -> Event:
        """Get next event from queues"""
        # Simple priority: tasks first, then results
        if not self.task_queue.empty():
            return await self.task_queue.get()
        elif not self.result_queue.empty():
            return await self.result_queue.get()
        else:
            # Wait for any event
            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(self.task_queue.get()),
                    asyncio.create_task(self.result_queue.get())
                ],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel pending
            for task in pending:
                task.cancel()
            
            # Return first completed
            return await list(done)[0]
    
    async def _handle_task(self, event: Event):
        """Handle new task"""
        task = event.data
        correlation_id = event.correlation_id or str(asyncio.current_task())
        
        # Store in context
        self.context[correlation_id] = {"task": task, "history": []}
        
        # Generate initial thought
        thought = await self._generate_thought(task, [])
        
        await self.emit(
            EventType.THOUGHT_GENERATED,
            {"thought": thought, "task": task},
            correlation_id
        )
    
    async def _handle_result(self, event: Event):
        """Handle tool result"""
        result = event.data
        correlation_id = event.correlation_id
        
        if correlation_id not in self.context:
            return
        
        # Update context
        ctx = self.context[correlation_id]
        ctx["history"].append(result)
        
        # Generate new thought based on result
        thought = await self._generate_thought(
            ctx["task"],
            ctx["history"]
        )
        
        await self.emit(
            EventType.THOUGHT_GENERATED,
            {"thought": thought, "history": ctx["history"]},
            correlation_id
        )
    
    async def _generate_thought(self, task: str, history: List[Any]) -> str:
        """Generate thought using LLM"""
        prompt = f"Task: {task}\n"
        
        if history:
            prompt += "\nPrevious results:\n"
            for item in history[-3:]:  # Last 3 items
                prompt += f"- {item}\n"
        
        prompt += "\nWhat should we do next?"
        
        return await self.llm_client.complete(prompt)

class ReactiveActionComponent(ReactiveComponent):
    """Component that executes actions based on thoughts"""
    
    def __init__(self, name: str, event_stream: EventStream, tools: Dict[str, Any]):
        super().__init__(name, event_stream)
        self.tools = tools
        self.thought_queue = self.event_stream.subscribe(EventType.THOUGHT_GENERATED)
    
    async def _run(self):
        """Process thoughts and execute actions"""
        while self.running:
            try:
                event = await asyncio.wait_for(
                    self.thought_queue.get(),
                    timeout=1.0
                )
                
                if event.type == EventType.THOUGHT_GENERATED:
                    await self._handle_thought(event)
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Action component error: {e}", exc_info=True)
    
    async def _handle_thought(self, event: Event):
        """Handle thought and execute action"""
        thought = event.data["thought"]
        correlation_id = event.correlation_id
        
        # Parse action from thought
        action = self._parse_action(thought)
        
        if action:
            await self.emit(
                EventType.ACTION_REQUIRED,
                action,
                correlation_id
            )
            
            # Execute tool
            if action["tool"] in self.tools:
                tool = self.tools[action["tool"]]
                result = await tool.execute(action["input"])
                
                await self.emit(
                    EventType.TOOL_EXECUTED,
                    {"tool": action["tool"], "result": result},
                    correlation_id
                )
                
                await self.emit(
                    EventType.RESULT_AVAILABLE,
                    result,
                    correlation_id
                )
    
    def _parse_action(self, thought: str) -> Optional[Dict[str, Any]]:
        """Parse action from thought"""
        # Simple parsing logic
        import re
        
        patterns = [
            r'use (\w+) tool with input "([^"]+)"',
            r'(\w+)\(([^)]+)\)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, thought, re.IGNORECASE)
            if match:
                return {
                    "tool": match.group(1).lower(),
                    "input": match.group(2)
                }
        
        return None

class ReactiveAgent:
    """Complete reactive agent system"""
    
    def __init__(self, name: str, llm_client, tools: Dict[str, Any]):
        self.name = name
        self.event_stream = EventStream()
        
        # Create components
        self.thinking = ReactiveThinkingComponent(
            f"{name}_thinking",
            self.event_stream,
            llm_client
        )
        
        self.action = ReactiveActionComponent(
            f"{name}_action",
            self.event_stream,
            tools
        )
        
        # Result collector
        self.results: Dict[str, List[Any]] = {}
        self.result_subscriber = self.event_stream.subscribe(EventType.RESULT_AVAILABLE)
        
        self.running = False
        self._result_collector_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the agent"""
        if self.running:
            return
        
        self.running = True
        
        # Start components
        await self.thinking.start()
        await self.action.start()
        
        # Start result collector
        self._result_collector_task = asyncio.create_task(self._collect_results())
        
        logger.info(f"Started reactive agent: {self.name}")
    
    async def stop(self):
        """Stop the agent"""
        self.running = False
        
        # Stop components
        await self.thinking.stop()
        await self.action.stop()
        
        # Stop result collector
        if self._result_collector_task:
            self._result_collector_task.cancel()
            try:
                await self._result_collector_task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"Stopped reactive agent: {self.name}")
    
    async def process_task(self, task: str) -> List[Any]:
        """Process a task and return results"""
        correlation_id = f"task_{datetime.now().timestamp()}"
        
        # Emit task event
        await self.event_stream.publish(Event(
            type=EventType.TASK_RECEIVED,
            source=self.name,
            data=task,
            correlation_id=correlation_id
        ))
        
        # Wait for results (with timeout)
        timeout = 30  # seconds
        start_time = datetime.now()
        
        while (datetime.now() - start_time).seconds < timeout:
            if correlation_id in self.results:
                return self.results[correlation_id]
            await asyncio.sleep(0.1)
        
        raise asyncio.TimeoutError(f"Task timeout after {timeout}s")
    
    async def _collect_results(self):
        """Collect results from event stream"""
        while self.running:
            try:
                event = await asyncio.wait_for(
                    self.result_subscriber.get(),
                    timeout=1.0
                )
                
                if event.correlation_id:
                    if event.correlation_id not in self.results:
                        self.results[event.correlation_id] = []
                    self.results[event.correlation_id].append(event.data)
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Result collector error: {e}")
    
    def get_event_history(self) -> List[Event]:
        """Get event history for debugging"""
        return self.event_stream.event_history

# Observable pattern for reactive updates
class Observable:
    """Observable that emits values to observers"""
    
    def __init__(self):
        self.observers: List[Callable] = []
        
    def subscribe(self, observer: Callable):
        """Subscribe an observer"""
        self.observers.append(observer)
        
    def unsubscribe(self, observer: Callable):
        """Unsubscribe an observer"""
        if observer in self.observers:
            self.observers.remove(observer)
    
    async def emit(self, value: Any):
        """Emit value to all observers"""
        tasks = []
        
        for observer in self.observers:
            if asyncio.iscoroutinefunction(observer):
                tasks.append(asyncio.create_task(observer(value)))
            else:
                observer(value)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

class ReactiveProperty:
    """Property that notifies on change"""
    
    def __init__(self, initial_value: Any = None):
        self._value = initial_value
        self.observable = Observable()
    
    @property
    def value(self) -> Any:
        return self._value
    
    async def set(self, new_value: Any):
        """Set value and notify observers"""
        if new_value != self._value:
            old_value = self._value
            self._value = new_value
            await self.observable.emit({
                "old": old_value,
                "new": new_value
            })
    
    def subscribe(self, observer: Callable):
        """Subscribe to value changes"""
        self.observable.subscribe(observer)
```

### Task 6: Async Integration Patterns (30 min)
Create `src/async_patterns/integration.py`:

```python
from typing import Any, Callable, Dict, List, Optional, TypeVar
import asyncio
from functools import wraps
import aiohttp
import structlog
from contextlib import asynccontextmanager

logger = structlog.get_logger()

T = TypeVar('T')

# Async HTTP Client Pool
class AsyncHTTPPool:
    """Managed HTTP connection pool"""
    
    def __init__(
        self,
        max_connections: int = 100,
        max_per_host: int = 10,
        timeout: int = 30
    ):
        self.max_connections = max_connections
        self.max_per_host = max_per_host
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(
            limit=self.max_connections,
            limit_per_host=self.max_per_host
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=self.timeout
        )
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def request(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> aiohttp.ClientResponse:
        """Make HTTP request"""
        if not self.session:
            raise RuntimeError("HTTPPool not initialized")
        
        return await self.session.request(method, url, **kwargs)
    
    async def get_json(self, url: str, **kwargs) -> Any:
        """GET request returning JSON"""
        async with await self.request('GET', url, **kwargs) as response:
            response.raise_for_status()
            return await response.json()
    
    async def post_json(self, url: str, data: Any, **kwargs) -> Any:
        """POST request with JSON data"""
        async with await self.request(
            'POST',
            url,
            json=data,
            **kwargs
        ) as response:
            response.raise_for_status()
            return await response.json()

# Async Database Connection Pool
class AsyncDatabasePool:
    """Generic async database pool"""
    
    def __init__(
        self,
        create_connection: Callable,
        min_size: int = 10,
        max_size: int = 20
    ):
        self.create_connection = create_connection
        self.min_size = min_size
        self.max_size = max_size
        self.pool: List[Any] = []
        self.available: asyncio.Queue = asyncio.Queue()
        self.size = 0
        self._lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize connection pool"""
        # Create minimum connections
        for _ in range(self.min_size):
            conn = await self.create_connection()
            self.pool.append(conn)
            await self.available.put(conn)
            self.size += 1
    
    async def acquire(self) -> Any:
        """Acquire connection from pool"""
        # Try to get available connection
        try:
            conn = self.available.get_nowait()
            return conn
        except asyncio.QueueEmpty:
            pass
        
        # Create new connection if under limit
        async with self._lock:
            if self.size < self.max_size:
                conn = await self.create_connection()
                self.pool.append(conn)
                self.size += 1
                return conn
        
        # Wait for available connection
        return await self.available.get()
    
    async def release(self, conn: Any):
        """Release connection back to pool"""
        await self.available.put(conn)
    
    @asynccontextmanager
    async def connection(self):
        """Context manager for connections"""
        conn = await self.acquire()
        try:
            yield conn
        finally:
            await self.release(conn)
    
    async def close(self):
        """Close all connections"""
        for conn in self.pool:
            if hasattr(conn, 'close'):
                await conn.close()

# Async Message Queue Integration
class AsyncMessageQueue:
    """Async message queue abstraction"""
    
    def __init__(self, broker_url: str):
        self.broker_url = broker_url
        self.subscribers: Dict[str, List[Callable]] = {}
        self.running = False
        self._consumer_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start message queue consumer"""
        if self.running:
            return
        
        self.running = True
        self._consumer_task = asyncio.create_task(self._consume_messages())
        logger.info("Started message queue consumer")
    
    async def stop(self):
        """Stop message queue consumer"""
        self.running = False
        
        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
    
    async def publish(self, topic: str, message: Any):
        """Publish message to topic"""
        # In real implementation, would publish to actual broker
        logger.info(f"Publishing to {topic}: {message}")
        
        # For demo, directly call subscribers
        if topic in self.subscribers:
            tasks = []
            for handler in self.subscribers[topic]:
                if asyncio.iscoroutinefunction(handler):
                    tasks.append(asyncio.create_task(handler(message)))
                else:
                    handler(message)
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
    
    def subscribe(self, topic: str, handler: Callable):
        """Subscribe to topic"""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(handler)
        logger.info(f"Subscribed to topic: {topic}")
    
    async def _consume_messages(self):
        """Consume messages from broker"""
        while self.running:
            # In real implementation, would consume from actual broker
            await asyncio.sleep(1)

# Async Service Discovery
class AsyncServiceDiscovery:
    """Discover and manage services"""
    
    def __init__(self, registry_url: str):
        self.registry_url = registry_url
        self.services: Dict[str, List[str]] = {}
        self.health_check_interval = 30
        self._health_check_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start service discovery"""
        await self._refresh_services()
        self._health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def stop(self):
        """Stop service discovery"""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
    
    async def get_service(self, service_name: str) -> Optional[str]:
        """Get healthy service instance"""
        if service_name not in self.services:
            await self._refresh_services()
        
        instances = self.services.get(service_name, [])
        if instances:
            # Simple round-robin
            instance = instances.pop(0)
            instances.append(instance)
            return instance
        
        return None
    
    async def _refresh_services(self):
        """Refresh service list from registry"""
        try:
            async with AsyncHTTPPool() as http:
                data = await http.get_json(f"{self.registry_url}/services")
                self.services = data
                logger.info(f"Refreshed services: {list(self.services.keys())}")
        except Exception as e:
            logger.error(f"Failed to refresh services: {e}")
    
    async def _health_check_loop(self):
        """Periodic health checks"""
        while True:
            await asyncio.sleep(self.health_check_interval)
            await self._check_health()
    
    async def _check_health(self):
        """Check health of all services"""
        for service_name, instances in self.services.items():
            healthy_instances = []
            
            for instance in instances:
                if await self._is_healthy(instance):
                    healthy_instances.append(instance)
                else:
                    logger.warning(f"Unhealthy instance: {instance}")
            
            self.services[service_name] = healthy_instances
    
    async def _is_healthy(self, instance_url: str) -> bool:
        """Check if instance is healthy"""
        try:
            async with AsyncHTTPPool() as http:
                response = await http.request('GET', f"{instance_url}/health")
                return response.status == 200
        except:
            return False

# Async Circuit Breaker for Service Calls
class AsyncServiceClient:
    """Client with circuit breaker for service calls"""
    
    def __init__(
        self,
        service_discovery: AsyncServiceDiscovery,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: int = 60
    ):
        self.service_discovery = service_discovery
        self.circuit_breakers: Dict[str, Any] = {}  # Per service
        self.threshold = circuit_breaker_threshold
        self.timeout = circuit_breaker_timeout
    
    async def call_service(
        self,
        service_name: str,
        endpoint: str,
        method: str = 'GET',
        **kwargs
    ) -> Any:
        """Call service with circuit breaker"""
        # Get service instance
        instance = await self.service_discovery.get_service(service_name)
        if not instance:
            raise ValueError(f"No instances available for {service_name}")
        
        # Get or create circuit breaker
        if service_name not in self.circuit_breakers:
            from src.async_patterns.core import CircuitBreaker
            self.circuit_breakers[service_name] = CircuitBreaker(
                failure_threshold=self.threshold,
                recovery_timeout=self.timeout
            )
        
        breaker = self.circuit_breakers[service_name]
        
        # Make call with circuit breaker
        @breaker.call
        async def make_request():
            async with AsyncHTTPPool() as http:
                url = f"{instance}{endpoint}"
                response = await http.request(method, url, **kwargs)
                response.raise_for_status()
                return await response.json()
        
        return await make_request()

# Async Event Bus for Integration
class AsyncEventBus:
    """Event bus for service integration"""
    
    def __init__(self):
        self.handlers: Dict[str, List[Callable]] = {}
        self.middleware: List[Callable] = []
    
    def use(self, middleware: Callable):
        """Add middleware"""
        self.middleware.append(middleware)
    
    def on(self, event_name: str):
        """Decorator to register event handler"""
        def decorator(func):
            if event_name not in self.handlers:
                self.handlers[event_name] = []
            self.handlers[event_name].append(func)
            return func
        return decorator
    
    async def emit(self, event_name: str, data: Any):
        """Emit event"""
        # Apply middleware
        for mw in self.middleware:
            data = await mw(event_name, data)
            if data is None:
                return  # Middleware cancelled event
        
        # Call handlers
        if event_name in self.handlers:
            tasks = []
            for handler in self.handlers[event_name]:
                if asyncio.iscoroutinefunction(handler):
                    tasks.append(asyncio.create_task(handler(data)))
                else:
                    handler(data)
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

# Integration orchestrator
class AsyncIntegrationOrchestrator:
    """Orchestrate multiple async integrations"""
    
    def __init__(self):
        self.services: Dict[str, Any] = {}
        self.event_bus = AsyncEventBus()
        self.running = False
    
    def register_service(self, name: str, service: Any):
        """Register a service"""
        self.services[name] = service
    
    async def start_all(self):
        """Start all services"""
        self.running = True
        
        for name, service in self.services.items():
            if hasattr(service, 'start'):
                await service.start()
                logger.info(f"Started service: {name}")
    
    async def stop_all(self):
        """Stop all services"""
        self.running = False
        
        for name, service in self.services.items():
            if hasattr(service, 'stop'):
                await service.stop()
                logger.info(f"Stopped service: {name}")
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all services"""
        health = {}
        
        for name, service in self.services.items():
            if hasattr(service, 'is_healthy'):
                health[name] = await service.is_healthy()
            else:
                health[name] = True  # Assume healthy if no check
        
        return health
```

### Task 7: Performance Optimization (15 min)
Create `src/async_patterns/optimization.py`:

```python
import asyncio
import time
from typing import Any, Callable, Dict, List, Optional
from functools import lru_cache, wraps
import structlog

logger = structlog.get_logger()

# Async memoization
def async_memoize(maxsize: int = 128):
    """Memoize async function results"""
    def decorator(func: Callable) -> Callable:
        cache = {}
        cache_info = {"hits": 0, "misses": 0, "size": 0}
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key
            key = (args, tuple(sorted(kwargs.items())))
            
            # Check cache
            if key in cache:
                cache_info["hits"] += 1
                return cache[key]
            
            # Cache miss
            cache_info["misses"] += 1
            result = await func(*args, **kwargs)
            
            # Store in cache with size limit
            if len(cache) >= maxsize:
                # Remove oldest entry (simple FIFO)
                cache.pop(next(iter(cache)))
            
            cache[key] = result
            cache_info["size"] = len(cache)
            
            return result
        
        wrapper.cache_info = lambda: cache_info
        wrapper.cache_clear = lambda: cache.clear()
        
        return wrapper
    return decorator

# Async connection pooling
class AsyncConnectionPool:
    """Reusable connection pool"""
    
    def __init__(
        self,
        create_func: Callable,
        max_size: int = 10,
        min_size: int = 2,
        max_idle_time: float = 300
    ):
        self.create_func = create_func
        self.max_size = max_size
        self.min_size = min_size
        self.max_idle_time = max_idle_time
        
        self.pool: List[Tuple[Any, float]] = []
        self.in_use: Set[Any] = set()
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the pool"""
        # Create minimum connections
        for _ in range(self.min_size):
            conn = await self.create_func()
            self.pool.append((conn, time.time()))
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def acquire(self) -> Any:
        """Acquire connection"""
        async with self._lock:
            # Find available connection
            while self.pool:
                conn, _ = self.pool.pop(0)
                if await self._is_valid(conn):
                    self.in_use.add(conn)
                    return conn
                else:
                    await self._close_connection(conn)
            
            # Create new connection if under limit
            if len(self.in_use) < self.max_size:
                conn = await self.create_func()
                self.in_use.add(conn)
                return conn
        
        # Wait and retry
        await asyncio.sleep(0.1)
        return await self.acquire()
    
    async def release(self, conn: Any):
        """Release connection"""
        async with self._lock:
            self.in_use.discard(conn)
            self.pool.append((conn, time.time()))
    
    async def _cleanup_loop(self):
        """Clean up idle connections"""
        while True:
            await asyncio.sleep(60)  # Check every minute
            await self._cleanup_idle()
    
    async def _cleanup_idle(self):
        """Remove idle connections"""
        async with self._lock:
            now = time.time()
            active_pool = []
            
            for conn, last_used in self.pool:
                if now - last_used > self.max_idle_time:
                    await self._close_connection(conn)
                else:
                    active_pool.append((conn, last_used))
            
            self.pool = active_pool
    
    async def _is_valid(self, conn: Any) -> bool:
        """Check if connection is valid"""
        if hasattr(conn, 'is_valid'):
            return await conn.is_valid()
        return True
    
    async def _close_connection(self, conn: Any):
        """Close a connection"""
        if hasattr(conn, 'close'):
            await conn.close()

# Async performance profiler
class AsyncProfiler:
    """Profile async code performance"""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
    
    def profile(self, name: str):
        """Decorator to profile async function"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start = time.perf_counter()
                
                try:
                    result = await func(*args, **kwargs)
                    duration = time.perf_counter() - start
                    
                    if name not in self.metrics:
                        self.metrics[name] = []
                    self.metrics[name].append(duration)
                    
                    return result
                    
                except Exception as e:
                    duration = time.perf_counter() - start
                    logger.error(f"Profiled function {name} failed after {duration:.3f}s")
                    raise
            
            return wrapper
        return decorator
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a profiled function"""
        if name not in self.metrics:
            return {}
        
        durations = self.metrics[name]
        if not durations:
            return {}
        
        return {
            "count": len(durations),
            "total": sum(durations),
            "mean": sum(durations) / len(durations),
            "min": min(durations),
            "max": max(durations),
            "p50": sorted(durations)[len(durations) // 2],
            "p95": sorted(durations)[int(len(durations) * 0.95)],
            "p99": sorted(durations)[int(len(durations) * 0.99)]
        }
    
    def print_report(self):
        """Print performance report"""
        print("\n" + "="*60)
        print("ASYNC PERFORMANCE REPORT")
        print("="*60)
        
        for name in sorted(self.metrics.keys()):
            stats = self.get_stats(name)
            print(f"\n{name}:")
            print(f"  Calls: {stats['count']}")
            print(f"  Total: {stats['total']:.3f}s")
            print(f"  Mean: {stats['mean']*1000:.2f}ms")
            print(f"  Min: {stats['min']*1000:.2f}ms")
            print(f"  Max: {stats['max']*1000:.2f}ms")
            print(f"  P95: {stats['p95']*1000:.2f}ms")

# Optimize async gather with limits
async def bounded_gather(
    *coros,
    limit: int = 10,
    return_exceptions: bool = False
) -> List[Any]:
    """Gather with concurrency limit"""
    semaphore = asyncio.Semaphore(limit)
    
    async def bounded_coro(coro):
        async with semaphore:
            return await coro
    
    return await asyncio.gather(
        *[bounded_coro(coro) for coro in coros],
        return_exceptions=return_exceptions
    )

# Async task deduplication
class AsyncTaskDeduplicator:
    """Deduplicate concurrent identical tasks"""
    
    def __init__(self):
        self.pending: Dict[str, asyncio.Future] = {}
    
    async def deduplicate(
        self,
        key: str,
        coro_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute or wait for existing execution"""
        if key in self.pending:
            # Wait for existing execution
            return await self.pending[key]
        
        # Create new execution
        future = asyncio.Future()
        self.pending[key] = future
        
        try:
            result = await coro_func(*args, **kwargs)
            future.set_result(result)
            return result
        except Exception as e:
            future.set_exception(e)
            raise
        finally:
            self.pending.pop(key, None)
```

## 🌆 Evening Tasks (45-60 minutes)

### Task 8: Complete Async Agent Example (20 min)
Create `src/async_patterns/async_agent.py`:

```python
import asyncio
from typing import Any, Dict, List, Optional
import structlog

from src.async_patterns.core import (
    AsyncRateLimiter,
    AsyncPipeline,
    async_retry,
    async_timeout
)
from src.async_patterns.concurrent_tools import (
    ConcurrentToolExecutor,
    ToolCall,
    AdaptiveConcurrencyExecutor
)
from src.async_patterns.scheduling import (
    AsyncTaskScheduler,
    TaskPriority
)
from src.async_patterns.reactive_agent import (
    ReactiveAgent,
    EventStream,
    EventType
)

logger = structlog.get_logger()

class AsyncProductionAgent:
    """Production-ready async agent"""
    
    def __init__(
        self,
        name: str,
        llm_client,
        tools: Dict[str, Any],
        max_concurrent_tools: int = 5,
        enable_reactive: bool = True
    ):
        self.name = name
        self.llm_client = llm_client
        self.tools = tools
        
        # Concurrent tool executor
        self.tool_executor = AdaptiveConcurrencyExecutor(
            tools=tools,
            max_concurrent=max_concurrent_tools
        )
        
        # Task scheduler
        self.scheduler = AsyncTaskScheduler(max_workers=3)
        
        # Rate limiter
        self.rate_limiter = AsyncRateLimiter(rate=10, per=1.0)
        
        # Reactive components
        self.reactive_agent = None
        if enable_reactive:
            self.reactive_agent = ReactiveAgent(
                name=f"{name}_reactive",
                llm_client=llm_client,
                tools=tools
            )
        
        # Pipeline for processing
        self.pipeline = AsyncPipeline()
        self._setup_pipeline()
        
        self.running = False
    
    def _setup_pipeline(self):
        """Setup processing pipeline"""
        self.pipeline.add_stage(self._validate_input)
        self.pipeline.add_stage(self._generate_plan)
        self.pipeline.add_stage(self._execute_plan)
        self.pipeline.add_stage(self._synthesize_results)
    
    async def start(self):
        """Start the agent"""
        if self.running:
            return
        
        self.running = True
        
        # Start scheduler
        await self.scheduler.start()
        
        # Start reactive agent if enabled
        if self.reactive_agent:
            await self.reactive_agent.start()
        
        logger.info(f"Started async agent: {self.name}")
    
    async def stop(self):
        """Stop the agent"""
        self.running = False
        
        # Stop scheduler
        await self.scheduler.stop()
        
        # Stop reactive agent
        if self.reactive_agent:
            await self.reactive_agent.stop()
        
        logger.info(f"Stopped async agent: {self.name}")
    
    @async_retry(max_attempts=3)
    @async_timeout(seconds=300)
    async def process_task(self, task: str) -> Any:
        """Process a task through the pipeline"""
        # Rate limiting
        await self.rate_limiter.acquire()
        
        # Schedule high-priority task
        task_id = f"task_{asyncio.current_task().get_name()}"
        
        await self.scheduler.schedule_task(
            task_id=task_id,
            coroutine=self.pipeline.execute(task),
            priority=TaskPriority.HIGH
        )
        
        # Wait for completion
        result = await self.scheduler.wait_for_task(task_id)
        
        if result.success:
            return result.result
        else:
            raise Exception(f"Task failed: {result.error}")
    
    async def process_batch(self, tasks: List[str]) -> List[Any]:
        """Process multiple tasks concurrently"""
        # Create task coroutines
        task_coros = [
            self.process_task(task)
            for task in tasks
        ]
        
        # Execute with bounded concurrency
        from src.async_patterns.optimization import bounded_gather
        
        results = await bounded_gather(
            *task_coros,
            limit=5,
            return_exceptions=True
        )
        
        # Process results
        successful_results = []
        for task, result in zip(tasks, results):
            if isinstance(result, Exception):
                logger.error(f"Task failed: {task}", error=str(result))
                successful_results.append(None)
            else:
                successful_results.append(result)
        
        return successful_results
    
    async def _validate_input(self, task: str) -> Dict[str, Any]:
        """Validate and parse input"""
        if not task or not task.strip():
            raise ValueError("Empty task")
        
        return {
            "task": task,
            "validated": True,
            "timestamp": asyncio.get_event_loop().time()
        }
    
    async def _generate_plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate execution plan"""
        task = context["task"]
        
        # Use LLM to generate plan
        plan_prompt = f"""
        Task: {task}
        
        Available tools: {list(self.tools.keys())}
        
        Create a step-by-step plan using the available tools.
        """
        
        plan = await self.llm_client.complete(plan_prompt)
        
        # Parse plan into tool calls
        tool_calls = self._parse_plan_to_tool_calls(plan)
        
        context["plan"] = plan
        context["tool_calls"] = tool_calls
        
        return context
    
    async def _execute_plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the plan using tools"""
        tool_calls = context.get("tool_calls", [])
        
        if not tool_calls:
            context["results"] = []
            return context
        
        # Execute tools concurrently
        results = await self.tool_executor.execute_batch(tool_calls)
        
        context["results"] = results
        return context
    
    async def _synthesize_results(self, context: Dict[str, Any]) -> Any:
        """Synthesize final result"""
        task = context["task"]
        results = context.get("results", [])
        
        # Build synthesis prompt
        synthesis_prompt = f"""
        Task: {task}
        
        Results from tools:
        {self._format_results(results)}
        
        Provide a final answer based on these results.
        """
        
        final_answer = await self.llm_client.complete(synthesis_prompt)
        
        return {
            "task": task,
            "answer": final_answer,
            "tool_results": results,
            "execution_time": asyncio.get_event_loop().time() - context["timestamp"]
        }
    
    def _parse_plan_to_tool_calls(self, plan: str) -> List[ToolCall]:
        """Parse plan into tool calls"""
        # Simple parsing - in practice would be more sophisticated
        tool_calls = []
        
        lines = plan.strip().split('\n')
        for i, line in enumerate(lines):
            for tool_name in self.tools:
                if tool_name in line.lower():
                    tool_calls.append(ToolCall(
                        tool_name=tool_name,
                        input_data={"line": line},
                        priority=len(lines) - i  # Earlier = higher priority
                    ))
                    break
        
        return tool_calls
    
    def _format_results(self, results: List[Any]) -> str:
        """Format results for synthesis"""
        formatted = []
        
        for i, result in enumerate(results):
            if hasattr(result, 'result') and hasattr(result.result, 'output'):
                formatted.append(f"{i+1}. {result.result.output}")
            else:
                formatted.append(f"{i+1}. {str(result)}")
        
        return "\n".join(formatted)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics"""
        metrics = {
            "name": self.name,
            "running": self.running,
            "scheduler_status": self.scheduler.get_status(),
            "tool_executor_metrics": self.tool_executor.get_metrics(),
        }
        
        return metrics

# Example usage
async def main():
    """Example of using async agent"""
    # Mock LLM client
    class MockLLMClient:
        async def complete(self, prompt: str) -> str:
            await asyncio.sleep(0.1)  # Simulate API call
            return "Sample response"
    
    # Mock tools
    class MockTool:
        def __init__(self, name: str):
            self.name = name
            
        async def execute(self, input_data: Any) -> Any:
            await asyncio.sleep(0.2)  # Simulate tool execution
            return {"output": f"Result from {self.name}"}
    
    # Create agent
    agent = AsyncProductionAgent(
        name="demo_agent",
        llm_client=MockLLMClient(),
        tools={
            "calculator": MockTool("calculator"),
            "search": MockTool("search"),
            "analyzer": MockTool("analyzer")
        }
    )
    
    # Start agent
    await agent.start()
    
    try:
        # Process single task
        print("Processing single task...")
        result = await agent.process_task("Calculate 2 + 2 and search for Python asyncio")
        print(f"Result: {result}")
        
        # Process batch
        print("\nProcessing batch...")
        tasks = [
            "Task 1: Simple calculation",
            "Task 2: Complex analysis",
            "Task 3: Data search"
        ]
        
        results = await agent.process_batch(tasks)
        for task, result in zip(tasks, results):
            print(f"{task}: {result}")
        
        # Get metrics
        print("\nAgent Metrics:")
        print(agent.get_metrics())
        
    finally:
        # Stop agent
        await agent.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### Task 9: Testing Async Patterns (15 min)
Create `tests/test_async_patterns.py`:

```python
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from src.async_patterns.core import (
    AsyncBatcher,
    AsyncRateLimiter,
    AsyncPipeline,
    async_retry,
    async_timeout
)
from src.async_patterns.concurrent_tools import (
    ConcurrentToolExecutor,
    ToolCall
)
from src.async_patterns.backpressure import BackpressureQueue

class TestAsyncPatterns:
    @pytest.mark.asyncio
    async def test_async_retry(self):
        """Test retry decorator"""
        call_count = 0
        
        @async_retry(max_attempts=3, delay=0.1)
        async def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"
        
        result = await flaky_function()
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_async_timeout(self):
        """Test timeout decorator"""
        @async_timeout(seconds=0.1)
        async def slow_function():
            await asyncio.sleep(1)
            return "done"
        
        with pytest.raises(asyncio.TimeoutError):
            await slow_function()
    
    @pytest.mark.asyncio
    async def test_async_batcher(self):
        """Test async batcher"""
        results = []
        
        async def process_batch(items):
            results.append(items)
            return [f"processed_{item}" for item in items]
        
        batcher = AsyncBatcher(
            batch_size=3,
            timeout=0.1,
            process_batch=process_batch
        )
        
        # Add items
        futures = []
        for i in range(5):
            futures.append(asyncio.create_task(batcher.add(i)))
        
        # Wait for results
        processed = await asyncio.gather(*futures)
        
        # Check batching occurred
        assert len(results) == 2  # Two batches
        assert len(results[0]) == 3  # First batch full
        assert len(results[1]) == 2  # Second batch partial
        
        # Check results
        assert processed == [
            "processed_0", "processed_1", "processed_2",
            "processed_3", "processed_4"
        ]
    
    @pytest.mark.asyncio
    async def test_rate_limiter(self):
        """Test rate limiter"""
        limiter = AsyncRateLimiter(rate=5, per=1.0)
        
        start_time = asyncio.get_event_loop().time()
        
        # Try to make 10 requests
        for i in range(10):
            await limiter.acquire()
        
        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time
        
        # Should take about 1 second for the second batch
        assert duration >= 0.9
        assert duration < 1.5
    
    @pytest.mark.asyncio
    async def test_pipeline(self):
        """Test async pipeline"""
        pipeline = AsyncPipeline()
        
        # Add stages
        pipeline.add_stage(lambda x: x * 2)
        pipeline.add_stage(lambda x: x + 1)
        pipeline.add_stage(lambda x: str(x))
        
        result = await pipeline.execute(5)
        assert result == "11"  # (5 * 2) + 1 = 11

class TestConcurrentTools:
    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """Test concurrent tool execution"""
        # Mock tools
        tools = {}
        for i in range(3):
            tool = Mock()
            tool.execute = AsyncMock(return_value=f"result_{i}")
            tools[f"tool_{i}"] = tool
        
        executor = ConcurrentToolExecutor(
            tools=tools,
            max_concurrent=2
        )
        
        # Create tool calls
        calls = [
            ToolCall(f"tool_{i}", f"input_{i}")
            for i in range(3)
        ]
        
        # Execute
        results = await executor.execute_batch(calls)
        
        # Verify results
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.tool_name == f"tool_{i}"
            assert result.result == f"result_{i}"

class TestBackpressure:
    @pytest.mark.asyncio
    async def test_backpressure_queue(self):
        """Test backpressure queue"""
        queue = BackpressureQueue(max_size=5, high_watermark=0.8)
        
        pressure_high_called = False
        pressure_low_called = False
        
        async def on_high():
            nonlocal pressure_high_called
            pressure_high_called = True
        
        async def on_low():
            nonlocal pressure_low_called
            pressure_low_called = True
        
        queue.on_pressure_high = on_high
        queue.on_pressure_low = on_low
        
        # Fill queue to trigger high pressure
        for i in range(4):
            await queue.put(i)
        
        assert pressure_high_called
        assert queue.pressure_status() == "high"
        
        # Drain queue to trigger low pressure
        for _ in range(4):
            await queue.get()
        
        assert pressure_low_called
        assert queue.pressure_status() == "low"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### Task 10: Update Learning Journal (10 min)

Update your CLAUDE.md:

```markdown
## Day 6: Async Patterns

### What I Built
- ✅ Morning: Core async patterns and primitives
- ✅ Afternoon: Reactive agent architecture
- ✅ Evening: Complete async agent with testing

### Key Learnings
1. **Technical**: 
   - Async/await enables massive concurrency
   - Backpressure prevents system overload
   - Rate limiting protects external APIs

2. **Architecture**:
   - Reactive patterns enable event-driven agents
   - Pipeline pattern structures async processing
   - Circuit breakers provide fault tolerance

3. **Performance**:
   - Concurrent tool execution: 5-10x speedup
   - Batching reduces API calls significantly
   - Connection pooling minimizes overhead

### Challenges Faced
- **Issue**: Debugging async code is complex
  **Solution**: Comprehensive logging and tracing
  **Lesson**: Observability is critical for async

- **Issue**: Resource exhaustion with unlimited concurrency
  **Solution**: Semaphores and rate limiting
  **Lesson**: Always bound concurrency

### Code Metrics
- Lines written: ~3000
- Async patterns implemented: 15+
- Test coverage: 80%
- Performance improvement: 5-10x

### Tomorrow's Goal
- [ ] Create week 1 project
- [ ] Build complete file analyzer agent
- [ ] Apply all week's learnings
```

## 📊 Deliverables Checklist
- [ ] Advanced async patterns and decorators
- [ ] Concurrent tool execution system
- [ ] Task scheduling with priorities
- [ ] Backpressure and flow control
- [ ] Reactive agent architecture
- [ ] Complete async agent example
- [ ] Comprehensive testing
- [ ] Performance optimizations

## 🎯 Success Metrics
You've succeeded if you can:
1. Execute 10+ tools concurrently without issues
2. Handle backpressure gracefully
3. Schedule tasks with priority and deadlines
4. Build reactive event-driven agents
5. Achieve 5x+ performance improvement

## 🚀 Extension Challenges
If you finish early:
1. Add distributed async execution with ray/dask
2. Implement async GraphQL subscriptions
3. Build real-time agent monitoring dashboard
4. Add async tracing with OpenTelemetry
5. Create async agent swarm coordination

---

🎉 **Congratulations!** You've mastered async patterns for high-performance agents. Tomorrow we'll build a complete project using everything from this week!