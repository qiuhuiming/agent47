# Day 2: API Client Engineering

## 🎯 Objectives
1. Build a production-grade LLM client with advanced features
2. Implement circuit breakers and rate limiting
3. Add response streaming for better UX
4. Create robust parsing with multiple fallback strategies
5. Build comprehensive error handling and recovery

## 📋 Prerequisites
- Completed Day 1 (working basic agent)
- Understanding of async Python
- Basic knowledge of HTTP and APIs

## 🌄 Morning Tasks (60-75 minutes)

### Task 1: Advanced LLM Client Base (25 min)
Create `src/llm/advanced_client.py`:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, AsyncIterator, Any
from datetime import datetime, timedelta
import asyncio
import aiohttp
from collections import deque
import structlog
import time

logger = structlog.get_logger()

@dataclass
class APICallMetrics:
    """Track API call metrics"""
    timestamp: datetime
    duration: float
    tokens_used: int
    cost: float
    success: bool
    error: Optional[str] = None

@dataclass
class RateLimiter:
    """Token bucket rate limiter"""
    capacity: int
    refill_rate: float  # tokens per second
    tokens: float = field(default_factory=lambda: 0)
    last_refill: float = field(default_factory=time.time)
    
    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens, return True if successful"""
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def _refill(self):
        """Refill tokens based on time passed"""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.refill_rate
        )
        self.last_refill = now

class CircuitBreaker:
    """Circuit breaker for fault tolerance"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_requests: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.half_open_attempts = 0
        
    def call(self, func):
        """Decorator for circuit breaker"""
        async def wrapper(*args, **kwargs):
            if not self._can_execute():
                raise Exception("Circuit breaker is OPEN")
                
            try:
                result = await func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise
                
        return wrapper
    
    def _can_execute(self) -> bool:
        """Check if we can execute request"""
        if self.state == "CLOSED":
            return True
            
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                self.half_open_attempts = 0
                return True
            return False
            
        if self.state == "HALF_OPEN":
            return self.half_open_attempts < self.half_open_requests
            
        return False
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should try to reset circuit"""
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self):
        """Handle successful call"""
        if self.state == "HALF_OPEN":
            self.half_open_attempts += 1
            if self.half_open_attempts >= self.half_open_requests:
                self.state = "CLOSED"
                self.failure_count = 0
                logger.info("Circuit breaker closed")
        else:
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == "HALF_OPEN":
            self.state = "OPEN"
            logger.warning("Circuit breaker opened from half-open")
        elif self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning("Circuit breaker opened", failures=self.failure_count)

@dataclass
class LLMConfig:
    """Configuration for LLM client"""
    api_key: str
    model: str
    base_url: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    rate_limit_rpm: int = 60  # requests per minute
    rate_limit_tpm: int = 90000  # tokens per minute
    circuit_breaker_enabled: bool = True
    
class AdvancedLLMClient(ABC):
    """Advanced LLM client with production features"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.metrics: deque[APICallMetrics] = deque(maxlen=1000)
        
        # Rate limiting
        self.request_limiter = RateLimiter(
            capacity=config.rate_limit_rpm,
            refill_rate=config.rate_limit_rpm / 60
        )
        self.token_limiter = RateLimiter(
            capacity=config.rate_limit_tpm,
            refill_rate=config.rate_limit_tpm / 60
        )
        
        # Circuit breaker
        self.circuit_breaker = CircuitBreaker() if config.circuit_breaker_enabled else None
        
        # Session for connection pooling
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    @abstractmethod
    async def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False,
        **kwargs
    ) -> Any:
        """Generate completion"""
        pass
    
    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        pass
    
    @abstractmethod
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for tokens"""
        pass
    
    async def _check_rate_limits(self, estimated_tokens: int):
        """Check and wait for rate limits"""
        # Wait for request limit
        while not self.request_limiter.consume(1):
            await asyncio.sleep(0.1)
            
        # Wait for token limit
        while not self.token_limiter.consume(estimated_tokens):
            await asyncio.sleep(0.1)
    
    def _record_metrics(
        self,
        start_time: float,
        tokens_used: int,
        cost: float,
        success: bool,
        error: Optional[str] = None
    ):
        """Record API call metrics"""
        metric = APICallMetrics(
            timestamp=datetime.now(),
            duration=time.time() - start_time,
            tokens_used=tokens_used,
            cost=cost,
            success=success,
            error=error
        )
        self.metrics.append(metric)
        
        logger.info(
            "API call completed",
            duration=metric.duration,
            tokens=tokens_used,
            cost=cost,
            success=success
        )
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of recent metrics"""
        if not self.metrics:
            return {}
            
        recent_metrics = list(self.metrics)
        successful_calls = [m for m in recent_metrics if m.success]
        
        return {
            "total_calls": len(recent_metrics),
            "successful_calls": len(successful_calls),
            "failure_rate": 1 - (len(successful_calls) / len(recent_metrics)),
            "avg_duration": sum(m.duration for m in recent_metrics) / len(recent_metrics),
            "total_tokens": sum(m.tokens_used for m in recent_metrics),
            "total_cost": sum(m.cost for m in recent_metrics),
        }
```

### Task 2: OpenAI Implementation (20 min)
Create `src/llm/openai_advanced.py`:

```python
import json
from typing import Any, AsyncIterator, Dict, Optional
import tiktoken
import openai
from openai import AsyncOpenAI
import structlog

from src.llm.advanced_client import AdvancedLLMClient, LLMConfig

logger = structlog.get_logger()

class AdvancedOpenAIClient(AdvancedLLMClient):
    """Advanced OpenAI client with streaming and metrics"""
    
    # Pricing per 1K tokens (update as needed)
    PRICING = {
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
    }
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = AsyncOpenAI(api_key=config.api_key)
        
        # Token counting
        try:
            self.encoding = tiktoken.encoding_for_model(config.model)
        except:
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    async def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Generate completion with optional streaming"""
        start_time = time.time()
        
        # Estimate tokens
        prompt_tokens = self.estimate_tokens(prompt)
        if system_prompt:
            prompt_tokens += self.estimate_tokens(system_prompt)
            
        # Check rate limits
        await self._check_rate_limits(prompt_tokens + max_tokens)
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            # Apply circuit breaker if enabled
            api_call = self._make_api_call
            if self.circuit_breaker:
                api_call = self.circuit_breaker.call(api_call)
            
            if stream:
                return await api_call(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                    **kwargs
                )
            else:
                response = await api_call(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                    **kwargs
                )
                
                # Record metrics
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
                cost = self.calculate_cost(prompt_tokens, completion_tokens)
                
                self._record_metrics(
                    start_time=start_time,
                    tokens_used=total_tokens,
                    cost=cost,
                    success=True
                )
                
                return response.choices[0].message.content
                
        except Exception as e:
            self._record_metrics(
                start_time=start_time,
                tokens_used=prompt_tokens,
                cost=0,
                success=False,
                error=str(e)
            )
            raise
    
    async def _make_api_call(self, **kwargs) -> Any:
        """Make API call with given parameters"""
        return await self.client.chat.completions.create(
            model=self.config.model,
            **kwargs
        )
    
    async def stream_complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream completion tokens as they arrive"""
        response = await self.complete(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            system_prompt=system_prompt,
            **kwargs
        )
        
        async for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using tiktoken"""
        return len(self.encoding.encode(text))
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on model pricing"""
        pricing = self.PRICING.get(self.config.model, self.PRICING["gpt-3.5-turbo"])
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        return input_cost + output_cost
```

### Task 3: Response Parser Engineering (20 min)
Create `src/parsing/robust_parser.py`:

```python
import json
import re
import ast
from typing import Any, Dict, List, Optional, Type, TypeVar
from dataclasses import dataclass
from abc import ABC, abstractmethod
import structlog
from pydantic import BaseModel, ValidationError

logger = structlog.get_logger()

T = TypeVar('T', bound=BaseModel)

@dataclass
class ParseResult:
    """Result of parsing attempt"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    parser_used: Optional[str] = None

class Parser(ABC):
    """Base parser interface"""
    
    @abstractmethod
    def parse(self, text: str) -> ParseResult:
        """Attempt to parse text"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Parser name for logging"""
        pass

class JSONBlockParser(Parser):
    """Parse JSON from code blocks"""
    
    name = "json_block"
    
    def parse(self, text: str) -> ParseResult:
        """Extract and parse JSON from code blocks"""
        # Try to find JSON in code blocks
        patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'{\s*".*?"\s*:.*?}',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    data = json.loads(match)
                    return ParseResult(
                        success=True,
                        data=data,
                        parser_used=self.name
                    )
                except json.JSONDecodeError:
                    continue
        
        return ParseResult(
            success=False,
            error="No valid JSON found in code blocks"
        )

class XMLParser(Parser):
    """Parse XML-style tags"""
    
    name = "xml"
    
    def parse(self, text: str) -> ParseResult:
        """Extract data from XML-style tags"""
        patterns = {
            'action': r'<action>(.*?)</action>',
            'tool': r'<tool>(.*?)</tool>',
            'input': r'<input>(.*?)</input>',
            'thought': r'<thought>(.*?)</thought>',
        }
        
        result = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.DOTALL)
            if match:
                result[key] = match.group(1).strip()
        
        if result:
            return ParseResult(
                success=True,
                data=result,
                parser_used=self.name
            )
        
        return ParseResult(
            success=False,
            error="No XML-style tags found"
        )

class PythonLiteralParser(Parser):
    """Parse Python literals safely"""
    
    name = "python_literal"
    
    def parse(self, text: str) -> ParseResult:
        """Parse using ast.literal_eval"""
        # Try to extract dict-like strings
        dict_pattern = r'{\s*["\'].*?["\']\s*:.*?}'
        matches = re.findall(dict_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                data = ast.literal_eval(match)
                return ParseResult(
                    success=True,
                    data=data,
                    parser_used=self.name
                )
            except (ValueError, SyntaxError):
                continue
        
        return ParseResult(
            success=False,
            error="No valid Python literal found"
        )

class KeyValueParser(Parser):
    """Parse key-value pairs"""
    
    name = "key_value"
    
    def parse(self, text: str) -> ParseResult:
        """Parse key: value format"""
        lines = text.strip().split('\n')
        result = {}
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                result[key.strip().lower()] = value.strip()
        
        if result:
            return ParseResult(
                success=True,
                data=result,
                parser_used=self.name
            )
        
        return ParseResult(
            success=False,
            error="No key-value pairs found"
        )

class LLMAssistedParser(Parser):
    """Use another LLM call to parse"""
    
    name = "llm_assisted"
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    async def parse_async(self, text: str, target_format: str) -> ParseResult:
        """Parse using LLM assistance"""
        prompt = f"""
Extract the following information from this text and return it as valid JSON:
{target_format}

Text to parse:
{text}

Return only the JSON object, no explanation.
"""
        
        try:
            response = await self.llm.complete(prompt, temperature=0.1)
            data = json.loads(response)
            return ParseResult(
                success=True,
                data=data,
                parser_used=self.name
            )
        except Exception as e:
            return ParseResult(
                success=False,
                error=f"LLM parsing failed: {str(e)}"
            )
    
    def parse(self, text: str) -> ParseResult:
        """Sync wrapper - use parse_async instead"""
        return ParseResult(
            success=False,
            error="Use parse_async for LLM-assisted parsing"
        )

class RobustParser:
    """Combine multiple parsers with fallback"""
    
    def __init__(self, parsers: List[Parser]):
        self.parsers = parsers
        
    def parse(self, text: str, expected_type: Optional[Type[T]] = None) -> ParseResult:
        """Try each parser in sequence"""
        errors = []
        
        for parser in self.parsers:
            try:
                result = parser.parse(text)
                if result.success:
                    # Validate with Pydantic if type provided
                    if expected_type and result.data:
                        try:
                            validated = expected_type(**result.data)
                            result.data = validated.dict()
                        except ValidationError as e:
                            errors.append(f"{parser.name}: {str(e)}")
                            continue
                    
                    logger.info("Parse successful", parser=parser.name)
                    return result
                else:
                    errors.append(f"{parser.name}: {result.error}")
                    
            except Exception as e:
                errors.append(f"{parser.name}: {str(e)}")
                logger.warning("Parser failed", parser=parser.name, error=str(e))
        
        return ParseResult(
            success=False,
            error=f"All parsers failed: {'; '.join(errors)}"
        )
    
    async def parse_with_llm_fallback(
        self,
        text: str,
        llm_parser: LLMAssistedParser,
        target_format: str
    ) -> ParseResult:
        """Try regular parsers first, fall back to LLM"""
        result = self.parse(text)
        
        if not result.success:
            logger.info("Falling back to LLM parser")
            result = await llm_parser.parse_async(text, target_format)
        
        return result

# Pydantic models for structured data
class ToolCall(BaseModel):
    """Structured tool call"""
    tool: str
    input: str
    reasoning: Optional[str] = None

class AgentAction(BaseModel):
    """Structured agent action"""
    action_type: str  # "tool", "response", "clarification"
    tool_call: Optional[ToolCall] = None
    message: Optional[str] = None
    confidence: float = 1.0

# Factory function
def create_robust_parser() -> RobustParser:
    """Create parser with all strategies"""
    return RobustParser([
        JSONBlockParser(),
        XMLParser(),
        PythonLiteralParser(),
        KeyValueParser(),
    ])
```

### Task 4: Integration Testing (10 min)
Create `src/llm/test_advanced_client.py`:

```python
import asyncio
import pytest
from unittest.mock import AsyncMock, patch

from src.llm.advanced_client import LLMConfig, RateLimiter, CircuitBreaker
from src.llm.openai_advanced import AdvancedOpenAIClient

@pytest.mark.asyncio
async def test_rate_limiter():
    """Test rate limiter functionality"""
    limiter = RateLimiter(capacity=10, refill_rate=5)
    limiter.tokens = 10
    
    # Should consume tokens
    assert limiter.consume(5) == True
    assert limiter.tokens == 5
    
    # Should fail when not enough tokens
    assert limiter.consume(6) == False
    
    # Wait and check refill
    await asyncio.sleep(1)
    limiter._refill()
    assert limiter.tokens > 5

def test_circuit_breaker():
    """Test circuit breaker states"""
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1)
    
    # Should start closed
    assert cb.state == "CLOSED"
    assert cb._can_execute() == True
    
    # Fail 3 times
    for _ in range(3):
        cb._on_failure()
    
    # Should be open
    assert cb.state == "OPEN"
    assert cb._can_execute() == False
    
    # Wait for recovery
    time.sleep(1.1)
    assert cb._can_execute() == True
    assert cb.state == "HALF_OPEN"

@pytest.mark.asyncio
async def test_openai_client_with_mocking():
    """Test OpenAI client with mocked API"""
    config = LLMConfig(
        api_key="test-key",
        model="gpt-3.5-turbo",
        rate_limit_rpm=60,
        rate_limit_tpm=90000
    )
    
    async with AdvancedOpenAIClient(config) as client:
        # Mock the API call
        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock(message=AsyncMock(content="Test response"))]
        mock_response.usage = AsyncMock(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15
        )
        
        with patch.object(client, '_make_api_call', return_value=mock_response):
            result = await client.complete("Test prompt")
            assert result == "Test response"
            
            # Check metrics recorded
            assert len(client.metrics) == 1
            assert client.metrics[0].success == True
            assert client.metrics[0].tokens_used == 15
```

## 🌞 Afternoon Tasks (90-120 minutes)

### Task 5: Streaming Implementation (30 min)
Create `src/llm/streaming.py`:

```python
from typing import AsyncIterator, Callable, Optional
import asyncio
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()

@dataclass
class StreamToken:
    """Single token from stream"""
    content: str
    index: int
    timestamp: float
    finish_reason: Optional[str] = None

class StreamProcessor:
    """Process streaming LLM responses"""
    
    def __init__(self):
        self.buffer = []
        self.callbacks = []
        
    def add_callback(self, callback: Callable[[str], None]):
        """Add callback for stream events"""
        self.callbacks.append(callback)
        
    async def process_stream(
        self,
        stream: AsyncIterator[str],
        print_output: bool = True
    ) -> str:
        """Process stream and return full response"""
        full_response = []
        
        async for token in stream:
            full_response.append(token)
            self.buffer.append(token)
            
            # Print to console if requested
            if print_output:
                print(token, end='', flush=True)
            
            # Call callbacks
            for callback in self.callbacks:
                try:
                    callback(token)
                except Exception as e:
                    logger.error("Stream callback error", error=str(e))
        
        if print_output:
            print()  # New line at end
            
        return ''.join(full_response)

class BufferedStreamWriter:
    """Buffer stream output for word-by-word display"""
    
    def __init__(self, delay: float = 0.05):
        self.delay = delay
        self.word_buffer = []
        
    async def write_stream(self, stream: AsyncIterator[str]):
        """Write stream with buffering"""
        current_word = []
        
        async for token in stream:
            if token.isspace():
                # Complete word
                if current_word:
                    word = ''.join(current_word)
                    print(word, end='', flush=True)
                    current_word = []
                    await asyncio.sleep(self.delay)
                print(token, end='', flush=True)
            else:
                current_word.append(token)
        
        # Final word
        if current_word:
            print(''.join(current_word), end='', flush=True)
        print()

class StreamingAgent:
    """Agent with streaming support"""
    
    def __init__(self, llm_client, tools):
        self.llm = llm_client
        self.tools = tools
        self.stream_processor = StreamProcessor()
        
    async def run_streaming(self, task: str) -> str:
        """Run agent with streaming output"""
        print(f"🤖 Task: {task}\n")
        
        # Initial thinking
        print("💭 Thinking", end='')
        for _ in range(3):
            print(".", end='', flush=True)
            await asyncio.sleep(0.3)
        print()
        
        # Generate response with streaming
        system_prompt = """You are a helpful AI agent. 
Think step by step and explain your reasoning.
Use available tools when needed."""
        
        print("\n📝 Response:\n")
        response = await self.stream_processor.process_stream(
            self.llm.stream_complete(
                task,
                system_prompt=system_prompt,
                temperature=0.7
            )
        )
        
        return response
```

### Task 6: Advanced Error Handling (30 min)
Create `src/llm/error_handling.py`:

```python
from enum import Enum
from typing import Any, Callable, Optional, Type, Union
from dataclasses import dataclass
import asyncio
import functools
import structlog
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log
)

logger = structlog.get_logger()

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ErrorContext:
    """Context for error handling"""
    error: Exception
    severity: ErrorSeverity
    retry_count: int = 0
    max_retries: int = 3
    metadata: dict = None
    
    def __post_init__(self):
        self.metadata = self.metadata or {}

class APIError(Exception):
    """Base API error"""
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code

class RateLimitError(APIError):
    """Rate limit exceeded"""
    pass

class AuthenticationError(APIError):
    """Authentication failed"""
    pass

class TokenLimitError(APIError):
    """Token limit exceeded"""
    pass

class NetworkError(APIError):
    """Network-related error"""
    pass

class ErrorHandler:
    """Centralized error handling"""
    
    def __init__(self):
        self.handlers = {
            RateLimitError: self._handle_rate_limit,
            AuthenticationError: self._handle_auth_error,
            TokenLimitError: self._handle_token_limit,
            NetworkError: self._handle_network_error,
        }
        
    async def handle(self, error: Exception, context: ErrorContext) -> Any:
        """Handle error with appropriate strategy"""
        error_type = type(error)
        
        if error_type in self.handlers:
            return await self.handlers[error_type](error, context)
        else:
            return await self._handle_generic_error(error, context)
    
    async def _handle_rate_limit(self, error: RateLimitError, context: ErrorContext):
        """Handle rate limit errors"""
        logger.warning("Rate limit hit", retry_count=context.retry_count)
        
        if context.retry_count < context.max_retries:
            # Exponential backoff
            wait_time = 2 ** context.retry_count
            logger.info(f"Waiting {wait_time}s before retry")
            await asyncio.sleep(wait_time)
            return "retry"
        else:
            return "fail"
    
    async def _handle_auth_error(self, error: AuthenticationError, context: ErrorContext):
        """Handle authentication errors"""
        logger.error("Authentication failed - check API key")
        return "fail"
    
    async def _handle_token_limit(self, error: TokenLimitError, context: ErrorContext):
        """Handle token limit errors"""
        logger.warning("Token limit exceeded")
        
        # Try to reduce context or split request
        if "reduce_context" in context.metadata:
            return "reduce_context"
        else:
            return "fail"
    
    async def _handle_network_error(self, error: NetworkError, context: ErrorContext):
        """Handle network errors"""
        logger.warning("Network error", error=str(error))
        
        if context.retry_count < context.max_retries:
            await asyncio.sleep(1)
            return "retry"
        else:
            return "fail"
    
    async def _handle_generic_error(self, error: Exception, context: ErrorContext):
        """Handle unknown errors"""
        logger.error("Unexpected error", error=str(error), exc_info=True)
        return "fail"

def with_error_handling(
    error_handler: ErrorHandler,
    max_retries: int = 3,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
):
    """Decorator for error handling"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            context = ErrorContext(
                error=None,
                severity=severity,
                max_retries=max_retries
            )
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    context.error = e
                    context.retry_count = attempt
                    
                    action = await error_handler.handle(e, context)
                    
                    if action == "retry" and attempt < max_retries:
                        continue
                    elif action == "reduce_context":
                        # Modify kwargs to reduce context
                        kwargs["max_tokens"] = kwargs.get("max_tokens", 1000) // 2
                        continue
                    else:
                        raise
                        
            raise Exception(f"Max retries ({max_retries}) exceeded")
        
        return wrapper
    return decorator

# Specific retry strategies
def create_retry_decorator(
    exception_types: tuple = (APIError,),
    max_attempts: int = 3
):
    """Create custom retry decorator"""
    return retry(
        retry=retry_if_exception_type(exception_types),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, structlog.WARNING),
        after=after_log(logger, structlog.INFO),
        reraise=True
    )

class SafeLLMClient:
    """LLM client with comprehensive error handling"""
    
    def __init__(self, base_client, error_handler: ErrorHandler):
        self.client = base_client
        self.error_handler = error_handler
        
    @with_error_handling(ErrorHandler(), max_retries=3)
    async def complete_safe(self, prompt: str, **kwargs) -> str:
        """Complete with error handling"""
        try:
            return await self.client.complete(prompt, **kwargs)
        except Exception as e:
            # Map to specific errors
            if "rate limit" in str(e).lower():
                raise RateLimitError(str(e))
            elif "authentication" in str(e).lower():
                raise AuthenticationError(str(e))
            elif "token" in str(e).lower() and "limit" in str(e).lower():
                raise TokenLimitError(str(e))
            elif "network" in str(e).lower() or "connection" in str(e).lower():
                raise NetworkError(str(e))
            else:
                raise
```

### Task 7: Cost Tracking System (30 min)
Create `src/llm/cost_tracker.py`:

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
from pathlib import Path
import structlog

logger = structlog.get_logger()

@dataclass
class UsageRecord:
    """Single usage record"""
    timestamp: datetime
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    task_id: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

class CostTracker:
    """Track and analyze LLM costs"""
    
    def __init__(self, storage_path: str = "logs/usage.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(exist_ok=True)
        self.records: List[UsageRecord] = []
        self._load_records()
        
    def record_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        task_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """Record usage"""
        record = UsageRecord(
            timestamp=datetime.now(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            task_id=task_id,
            metadata=metadata or {}
        )
        
        self.records.append(record)
        self._save_records()
        
        logger.info(
            "Usage recorded",
            model=model,
            tokens=input_tokens + output_tokens,
            cost=cost
        )
    
    def get_summary(self, period_days: int = 30) -> Dict:
        """Get usage summary for period"""
        cutoff = datetime.now() - timedelta(days=period_days)
        recent_records = [r for r in self.records if r.timestamp > cutoff]
        
        if not recent_records:
            return {
                "period_days": period_days,
                "total_cost": 0,
                "total_tokens": 0,
                "count": 0
            }
        
        total_cost = sum(r.cost for r in recent_records)
        total_input = sum(r.input_tokens for r in recent_records)
        total_output = sum(r.output_tokens for r in recent_records)
        
        # Group by model
        by_model = {}
        for record in recent_records:
            if record.model not in by_model:
                by_model[record.model] = {
                    "count": 0,
                    "cost": 0,
                    "tokens": 0
                }
            by_model[record.model]["count"] += 1
            by_model[record.model]["cost"] += record.cost
            by_model[record.model]["tokens"] += record.input_tokens + record.output_tokens
        
        return {
            "period_days": period_days,
            "total_cost": round(total_cost, 4),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "count": len(recent_records),
            "avg_cost_per_request": round(total_cost / len(recent_records), 4),
            "by_model": by_model,
            "daily_average": round(total_cost / period_days, 4)
        }
    
    def get_cost_by_task(self, task_id: str) -> float:
        """Get total cost for specific task"""
        task_records = [r for r in self.records if r.task_id == task_id]
        return sum(r.cost for r in task_records)
    
    def set_budget_alert(self, daily_limit: float, callback=None):
        """Set budget alert"""
        today_cost = self._get_today_cost()
        
        if today_cost > daily_limit:
            message = f"Daily budget exceeded: ${today_cost:.2f} > ${daily_limit:.2f}"
            logger.warning(message)
            
            if callback:
                callback(message, today_cost, daily_limit)
    
    def _get_today_cost(self) -> float:
        """Get today's total cost"""
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_records = [r for r in self.records if r.timestamp >= today_start]
        return sum(r.cost for r in today_records)
    
    def _load_records(self):
        """Load records from storage"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self.records = [
                        UsageRecord(
                            timestamp=datetime.fromisoformat(r["timestamp"]),
                            model=r["model"],
                            input_tokens=r["input_tokens"],
                            output_tokens=r["output_tokens"],
                            cost=r["cost"],
                            task_id=r.get("task_id"),
                            metadata=r.get("metadata", {})
                        )
                        for r in data
                    ]
                logger.info(f"Loaded {len(self.records)} usage records")
            except Exception as e:
                logger.error("Failed to load usage records", error=str(e))
                self.records = []
    
    def _save_records(self):
        """Save records to storage"""
        try:
            data = [
                {
                    "timestamp": r.timestamp.isoformat(),
                    "model": r.model,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "cost": r.cost,
                    "task_id": r.task_id,
                    "metadata": r.metadata
                }
                for r in self.records
            ]
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error("Failed to save usage records", error=str(e))

# Integration example
class CostAwareLLMClient:
    """LLM client with cost tracking"""
    
    def __init__(self, base_client, cost_tracker: CostTracker):
        self.client = base_client
        self.cost_tracker = cost_tracker
        
    async def complete(self, prompt: str, task_id: Optional[str] = None, **kwargs) -> str:
        """Complete with cost tracking"""
        response = await self.client.complete(prompt, **kwargs)
        
        # Extract usage info (depends on client implementation)
        if hasattr(response, 'usage'):
            self.cost_tracker.record_usage(
                model=self.client.config.model,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                cost=self.client.calculate_cost(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens
                ),
                task_id=task_id
            )
        
        return response
```

### Task 8: Put It All Together (30 min)
Create `src/agent_v2.py`:

```python
import asyncio
from typing import List, Optional
from uuid import uuid4
import structlog

from src.llm.advanced_client import LLMConfig
from src.llm.openai_advanced import AdvancedOpenAIClient
from src.llm.streaming import StreamingAgent, StreamProcessor
from src.llm.error_handling import ErrorHandler, SafeLLMClient
from src.llm.cost_tracker import CostTracker, CostAwareLLMClient
from src.parsing.robust_parser import create_robust_parser, ToolCall
from src.tools.base import Tool

logger = structlog.get_logger()

class AdvancedAgent:
    """Production agent with all advanced features"""
    
    def __init__(
        self,
        llm_config: LLMConfig,
        tools: List[Tool],
        enable_streaming: bool = True,
        enable_cost_tracking: bool = True,
        daily_cost_limit: Optional[float] = None
    ):
        # Initialize base client
        self.base_client = AdvancedOpenAIClient(llm_config)
        
        # Wrap with error handling
        self.error_handler = ErrorHandler()
        self.safe_client = SafeLLMClient(self.base_client, self.error_handler)
        
        # Add cost tracking
        if enable_cost_tracking:
            self.cost_tracker = CostTracker()
            self.llm = CostAwareLLMClient(self.safe_client, self.cost_tracker)
            
            if daily_cost_limit:
                self.cost_tracker.set_budget_alert(daily_cost_limit)
        else:
            self.llm = self.safe_client
        
        # Tools and parsing
        self.tools = {tool.name: tool for tool in tools}
        self.parser = create_robust_parser()
        
        # Streaming
        self.enable_streaming = enable_streaming
        self.stream_processor = StreamProcessor() if enable_streaming else None
        
        logger.info(
            "Advanced agent initialized",
            tools=list(self.tools.keys()),
            streaming=enable_streaming,
            cost_tracking=enable_cost_tracking
        )
    
    async def run(self, task: str, stream: bool = None) -> str:
        """Run task with all features"""
        task_id = str(uuid4())
        logger.info("Starting task", task=task, task_id=task_id)
        
        # Determine if we should stream
        should_stream = stream if stream is not None else self.enable_streaming
        
        if should_stream and self.stream_processor:
            return await self._run_streaming(task, task_id)
        else:
            return await self._run_standard(task, task_id)
    
    async def _run_standard(self, task: str, task_id: str) -> str:
        """Standard non-streaming execution"""
        context = [f"Task: {task}"]
        max_steps = 10
        
        for step in range(max_steps):
            # Generate thought
            thought_prompt = self._build_thought_prompt(context)
            thought = await self.llm.complete_safe(
                thought_prompt,
                task_id=task_id,
                temperature=0.7
            )
            
            context.append(f"Thought {step + 1}: {thought}")
            
            # Check for final answer
            if "final answer" in thought.lower():
                return self._extract_final_answer(thought)
            
            # Parse action
            action_result = self.parser.parse(thought, expected_type=ToolCall)
            
            if action_result.success and action_result.data:
                tool_call = ToolCall(**action_result.data)
                
                if tool_call.tool in self.tools:
                    # Execute tool
                    result = self.tools[tool_call.tool].execute(tool_call.input)
                    context.append(
                        f"Action {step + 1}: {tool_call.tool}({tool_call.input})\n"
                        f"Result: {result.output}"
                    )
                else:
                    context.append(f"Error: Unknown tool {tool_call.tool}")
            else:
                # Ask for clarification
                clarify_prompt = f"""
The last thought couldn't be parsed into an action.
Please provide a clear action in this format:
{{"tool": "tool_name", "input": "tool_input"}}

Available tools: {list(self.tools.keys())}
"""
                context.append(clarify_prompt)
        
        return "I couldn't complete the task within the allowed steps."
    
    async def _run_streaming(self, task: str, task_id: str) -> str:
        """Streaming execution"""
        print(f"\n🤖 Task: {task}\n")
        print("💭 Thinking...\n")
        
        prompt = f"""
You are an AI agent with access to these tools: {list(self.tools.keys())}

Task: {task}

Think step by step and use tools as needed.
When you have the final answer, start your response with "Final Answer:"
"""
        
        # Stream the response
        if self.base_client.session is None:
            await self.base_client.__aenter__()
        
        try:
            response = await self.stream_processor.process_stream(
                self.base_client.stream_complete(
                    prompt,
                    system_prompt="You are a helpful AI agent.",
                    task_id=task_id
                ),
                print_output=True
            )
            
            return response
            
        finally:
            await self.base_client.__aexit__(None, None, None)
    
    def _build_thought_prompt(self, context: List[str]) -> str:
        """Build prompt for next thought"""
        prompt = "\n".join(context)
        prompt += f"\n\nAvailable tools: {list(self.tools.keys())}"
        prompt += "\n\nWhat should I do next? If you have the final answer, start with 'Final Answer:'"
        return prompt
    
    def _extract_final_answer(self, thought: str) -> str:
        """Extract final answer from thought"""
        if "final answer:" in thought.lower():
            parts = thought.lower().split("final answer:")
            return parts[1].strip()
        return thought
    
    async def get_usage_summary(self) -> dict:
        """Get usage and cost summary"""
        if hasattr(self, 'cost_tracker'):
            return self.cost_tracker.get_summary()
        return {"message": "Cost tracking not enabled"}
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.base_client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.base_client.__aexit__(exc_type, exc_val, exc_tb)

# Example usage
async def main():
    config = LLMConfig(
        api_key="your-api-key",
        model="gpt-3.5-turbo",
        rate_limit_rpm=60,
        rate_limit_tpm=90000,
        circuit_breaker_enabled=True
    )
    
    from src.tools.calculator import CalculatorTool
    
    tools = [CalculatorTool()]
    
    async with AdvancedAgent(
        config,
        tools,
        enable_streaming=True,
        enable_cost_tracking=True,
        daily_cost_limit=1.0
    ) as agent:
        # Test standard execution
        result = await agent.run("What is 25 * 4 + 10?", stream=False)
        print(f"Result: {result}")
        
        # Test streaming execution
        result = await agent.run("Calculate the area of a circle with radius 5", stream=True)
        
        # Get usage summary
        summary = await agent.get_usage_summary()
        print(f"\nUsage Summary: {summary}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 🌆 Evening Tasks (45-60 minutes)

### Task 9: Advanced Testing (20 min)
Create `tests/test_advanced_features.py`:

```python
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from src.llm.advanced_client import RateLimiter, CircuitBreaker
from src.parsing.robust_parser import (
    JSONBlockParser,
    XMLParser,
    create_robust_parser,
    ToolCall
)
from src.llm.cost_tracker import CostTracker, UsageRecord
from datetime import datetime

class TestRateLimiter:
    def test_token_consumption(self):
        limiter = RateLimiter(capacity=10, refill_rate=5)
        limiter.tokens = 10
        
        assert limiter.consume(5) == True
        assert limiter.tokens == 5
        assert limiter.consume(6) == False
        
    def test_refill(self):
        limiter = RateLimiter(capacity=10, refill_rate=10)
        limiter.tokens = 0
        limiter.last_refill = time.time() - 1  # 1 second ago
        
        limiter._refill()
        assert limiter.tokens == 10  # Should be fully refilled

class TestParsing:
    def test_json_block_parser(self):
        parser = JSONBlockParser()
        
        text = '''
        Here's the action:
        ```json
        {"tool": "calculator", "input": "2 + 2"}
        ```
        '''
        
        result = parser.parse(text)
        assert result.success == True
        assert result.data["tool"] == "calculator"
        
    def test_xml_parser(self):
        parser = XMLParser()
        
        text = '''
        <action>calculate</action>
        <tool>calculator</tool>
        <input>5 * 5</input>
        '''
        
        result = parser.parse(text)
        assert result.success == True
        assert result.data["tool"] == "calculator"
        assert result.data["input"] == "5 * 5"
    
    def test_robust_parser_with_pydantic(self):
        parser = create_robust_parser()
        
        text = '{"tool": "calculator", "input": "10 / 2"}'
        
        result = parser.parse(text, expected_type=ToolCall)
        assert result.success == True
        assert isinstance(result.data, dict)
        assert result.data["tool"] == "calculator"

class TestCostTracking:
    def test_usage_recording(self, tmp_path):
        tracker = CostTracker(storage_path=str(tmp_path / "usage.json"))
        
        tracker.record_usage(
            model="gpt-3.5-turbo",
            input_tokens=100,
            output_tokens=50,
            cost=0.001,
            task_id="test-123"
        )
        
        assert len(tracker.records) == 1
        assert tracker.records[0].cost == 0.001
        
    def test_summary_calculation(self, tmp_path):
        tracker = CostTracker(storage_path=str(tmp_path / "usage.json"))
        
        # Add multiple records
        for i in range(5):
            tracker.record_usage(
                model="gpt-3.5-turbo",
                input_tokens=100,
                output_tokens=50,
                cost=0.001,
            )
        
        summary = tracker.get_summary(period_days=30)
        assert summary["total_cost"] == 0.005
        assert summary["count"] == 5
        assert summary["avg_cost_per_request"] == 0.001

@pytest.mark.asyncio
class TestAdvancedAgent:
    async def test_streaming_disabled(self):
        from src.agent_v2 import AdvancedAgent
        from src.llm.advanced_client import LLMConfig
        
        config = LLMConfig(
            api_key="test-key",
            model="gpt-3.5-turbo"
        )
        
        # Mock the LLM response
        with patch('src.agent_v2.AdvancedOpenAIClient') as mock_client:
            mock_instance = AsyncMock()
            mock_instance.complete_safe = AsyncMock(
                return_value="Final Answer: The result is 4"
            )
            mock_client.return_value = mock_instance
            
            agent = AdvancedAgent(
                config,
                tools=[],
                enable_streaming=False
            )
            
            result = await agent.run("What is 2 + 2?")
            assert "4" in result
```

### Task 10: Performance Benchmarking (15 min)
Create `benchmarks/benchmark_client.py`:

```python
import asyncio
import time
import statistics
from typing import List
import structlog

from src.llm.advanced_client import LLMConfig
from src.llm.openai_advanced import AdvancedOpenAIClient

logger = structlog.get_logger()

class LLMBenchmark:
    """Benchmark LLM client performance"""
    
    def __init__(self, client):
        self.client = client
        self.results = []
        
    async def run_benchmark(
        self,
        prompts: List[str],
        concurrent: int = 1
    ) -> dict:
        """Run benchmark with given prompts"""
        logger.info("Starting benchmark", 
                   prompts=len(prompts),
                   concurrent=concurrent)
        
        start_time = time.time()
        
        if concurrent == 1:
            # Sequential execution
            for prompt in prompts:
                await self._benchmark_single(prompt)
        else:
            # Concurrent execution
            semaphore = asyncio.Semaphore(concurrent)
            tasks = [
                self._benchmark_with_semaphore(prompt, semaphore)
                for prompt in prompts
            ]
            await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        return self._calculate_stats(total_time)
    
    async def _benchmark_single(self, prompt: str):
        """Benchmark single prompt"""
        start = time.time()
        
        try:
            response = await self.client.complete(prompt, max_tokens=100)
            duration = time.time() - start
            
            self.results.append({
                "success": True,
                "duration": duration,
                "prompt_length": len(prompt),
                "response_length": len(response)
            })
        except Exception as e:
            duration = time.time() - start
            self.results.append({
                "success": False,
                "duration": duration,
                "error": str(e)
            })
    
    async def _benchmark_with_semaphore(self, prompt: str, semaphore):
        """Benchmark with concurrency limit"""
        async with semaphore:
            await self._benchmark_single(prompt)
    
    def _calculate_stats(self, total_time: float) -> dict:
        """Calculate benchmark statistics"""
        successful = [r for r in self.results if r["success"]]
        failed = [r for r in self.results if not r["success"]]
        
        if not successful:
            return {"error": "No successful requests"}
        
        durations = [r["duration"] for r in successful]
        
        return {
            "total_requests": len(self.results),
            "successful": len(successful),
            "failed": len(failed),
            "total_time": round(total_time, 2),
            "avg_duration": round(statistics.mean(durations), 3),
            "min_duration": round(min(durations), 3),
            "max_duration": round(max(durations), 3),
            "median_duration": round(statistics.median(durations), 3),
            "requests_per_second": round(len(successful) / total_time, 2),
            "p95_duration": round(statistics.quantiles(durations, n=20)[18], 3),
            "p99_duration": round(statistics.quantiles(durations, n=100)[98], 3),
        }

async def main():
    """Run benchmarks"""
    config = LLMConfig(
        api_key="your-key",
        model="gpt-3.5-turbo",
        rate_limit_rpm=60,
        rate_limit_tpm=90000
    )
    
    # Test prompts
    prompts = [
        "What is 2 + 2?",
        "Explain quantum computing in one sentence.",
        "Write a haiku about programming.",
        "What's the capital of France?",
        "Convert 100 fahrenheit to celsius.",
    ] * 4  # 20 total prompts
    
    async with AdvancedOpenAIClient(config) as client:
        benchmark = LLMBenchmark(client)
        
        # Sequential benchmark
        print("\n=== Sequential Benchmark ===")
        seq_results = await benchmark.run_benchmark(prompts, concurrent=1)
        print(json.dumps(seq_results, indent=2))
        
        # Reset results
        benchmark.results = []
        
        # Concurrent benchmark
        print("\n=== Concurrent Benchmark (5 workers) ===")
        con_results = await benchmark.run_benchmark(prompts, concurrent=5)
        print(json.dumps(con_results, indent=2))
        
        # Get client metrics
        print("\n=== Client Metrics ===")
        print(json.dumps(client.get_metrics_summary(), indent=2))

if __name__ == "__main__":
    asyncio.run(main())
```

### Task 11: Documentation and Reflection (10 min)

Update your CLAUDE.md with today's learnings:

```markdown
## Day 2: API Client Engineering

### What I Built
- ✅ Morning: Advanced LLM client with rate limiting and circuit breakers
- ✅ Afternoon: Streaming support and robust parsing
- ✅ Evening: Cost tracking and comprehensive testing

### Key Learnings
1. **Technical**: 
   - Rate limiting prevents API quota issues
   - Circuit breakers provide fault tolerance
   - Streaming improves perceived performance

2. **Architecture**:
   - Layered client design (base → safe → cost-aware)
   - Parser chain of responsibility pattern
   - Async/await throughout for concurrency

3. **Performance**:
   - Connection pooling reduces latency
   - Concurrent requests improve throughput
   - Token estimation helps with cost prediction

### Challenges Faced
- **Issue**: Parsing inconsistent LLM outputs
  **Solution**: Multiple parser strategies with fallbacks
  **Lesson**: Never trust LLM output format

- **Issue**: Rate limit errors during testing
  **Solution**: Implemented token bucket algorithm
  **Lesson**: Proper rate limiting is essential

### Code Metrics
- Lines written: ~1200
- Tests added: 15
- Coverage: ~75%
- Benchmark: 5x speedup with concurrent requests

### Tomorrow's Goal
- [ ] Build sophisticated tool system
- [ ] Implement tool discovery and registration
- [ ] Add tool composition capabilities
```

## 📊 Deliverables Checklist
- [ ] Advanced LLM client with circuit breakers
- [ ] Rate limiting implementation
- [ ] Streaming response support
- [ ] Robust parsing with multiple strategies
- [ ] Cost tracking system
- [ ] Comprehensive error handling
- [ ] Performance benchmarking

## 🎯 Success Metrics
You've succeeded if you can:
1. Handle 20 concurrent requests without rate limit errors
2. Parse 5 different LLM output formats correctly
3. Track costs and set budget alerts
4. Stream responses in real-time
5. Recover from API failures automatically

## 🚀 Extension Challenges
If you finish early:
1. Add support for multiple LLM providers (Anthropic, Cohere)
2. Implement response caching with TTL
3. Add Prometheus metrics export
4. Build a simple dashboard for cost visualization
5. Create adaptive rate limiting based on response headers

---

🎉 **Congratulations!** You've built a production-grade LLM client that can handle real-world challenges. Tomorrow we'll focus on building a sophisticated tool system.