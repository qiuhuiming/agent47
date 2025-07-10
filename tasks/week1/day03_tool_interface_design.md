# Day 3: Tool Interface Design

## 🎯 Objectives
1. Design a clean, extensible tool system architecture
2. Implement tool discovery and automatic registration
3. Build tool validation and sandboxing
4. Create tool composition capabilities
5. Add comprehensive tool documentation system

## 📋 Prerequisites
- Completed Day 1-2 (working agent with API client)
- Understanding of design patterns (Factory, Strategy, Decorator)
- Basic knowledge of Python's inspection capabilities

## 🌄 Morning Tasks (60-75 minutes)

### Task 1: Advanced Tool Base Architecture (25 min)
Create `src/tools/advanced_base.py`:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union, Callable
from enum import Enum
import inspect
import json
from pydantic import BaseModel, Field, validator
import structlog

logger = structlog.get_logger()

class ToolCategory(Enum):
    """Tool categories for organization"""
    CALCULATION = "calculation"
    FILE_SYSTEM = "file_system"
    NETWORK = "network"
    DATA_PROCESSING = "data_processing"
    CODE_EXECUTION = "code_execution"
    EXTERNAL_API = "external_api"
    UTILITY = "utility"

class ToolPermission(Enum):
    """Security permissions for tools"""
    READ_FILES = "read_files"
    WRITE_FILES = "write_files"
    EXECUTE_CODE = "execute_code"
    NETWORK_ACCESS = "network_access"
    SYSTEM_COMMANDS = "system_commands"

@dataclass
class ToolMetadata:
    """Comprehensive tool metadata"""
    name: str
    description: str
    category: ToolCategory
    version: str = "1.0.0"
    author: str = "system"
    permissions: List[ToolPermission] = field(default_factory=list)
    examples: List[Dict[str, str]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "version": self.version,
            "author": self.author,
            "permissions": [p.value for p in self.permissions],
            "examples": self.examples,
            "tags": self.tags
        }

class ToolInput(BaseModel):
    """Base class for tool inputs with validation"""
    class Config:
        extra = "forbid"  # Reject extra fields
        validate_assignment = True

class ToolOutput(BaseModel):
    """Base class for tool outputs"""
    success: bool
    result: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    execution_time: Optional[float] = None

class AdvancedTool(ABC):
    """Advanced tool base with validation and introspection"""
    
    def __init__(self):
        self.metadata = self._build_metadata()
        self._validate_implementation()
        logger.info(f"Tool initialized: {self.metadata.name}")
    
    @abstractmethod
    def _build_metadata(self) -> ToolMetadata:
        """Build tool metadata"""
        pass
    
    @abstractmethod
    def get_input_schema(self) -> Type[ToolInput]:
        """Get Pydantic model for input validation"""
        pass
    
    @abstractmethod
    def get_output_schema(self) -> Type[ToolOutput]:
        """Get Pydantic model for output"""
        pass
    
    @abstractmethod
    async def _execute(self, validated_input: ToolInput) -> ToolOutput:
        """Execute tool with validated input"""
        pass
    
    async def execute(self, input_data: Union[str, Dict]) -> ToolOutput:
        """Execute tool with validation and error handling"""
        import time
        start_time = time.time()
        
        try:
            # Parse input if string
            if isinstance(input_data, str):
                try:
                    input_data = json.loads(input_data)
                except json.JSONDecodeError:
                    # Try to parse as single parameter
                    input_data = {"value": input_data}
            
            # Validate input
            input_schema = self.get_input_schema()
            validated_input = input_schema(**input_data)
            
            # Execute tool
            result = await self._execute(validated_input)
            
            # Add execution time
            result.execution_time = time.time() - start_time
            
            logger.info(
                "Tool executed successfully",
                tool=self.metadata.name,
                execution_time=result.execution_time
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                "Tool execution failed",
                tool=self.metadata.name,
                error=str(e),
                exc_info=True
            )
            
            output_schema = self.get_output_schema()
            return output_schema(
                success=False,
                result=None,
                error=str(e),
                execution_time=execution_time
            )
    
    def _validate_implementation(self):
        """Validate tool implementation"""
        # Check required methods
        required_methods = ['_execute', '_build_metadata', 'get_input_schema', 'get_output_schema']
        for method in required_methods:
            if not hasattr(self, method):
                raise NotImplementedError(f"Tool must implement {method}")
        
        # Validate metadata
        if not isinstance(self.metadata, ToolMetadata):
            raise ValueError("_build_metadata must return ToolMetadata instance")
    
    def get_openai_function_spec(self) -> Dict:
        """Get OpenAI function calling specification"""
        input_schema = self.get_input_schema()
        
        # Extract schema
        schema_dict = input_schema.schema()
        
        # Remove title and description from nested properties
        properties = schema_dict.get("properties", {})
        for prop in properties.values():
            prop.pop("title", None)
        
        return {
            "name": self.metadata.name,
            "description": self.metadata.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": schema_dict.get("required", [])
            }
        }
    
    def get_documentation(self) -> str:
        """Generate comprehensive documentation"""
        doc = f"# {self.metadata.name}\n\n"
        doc += f"**Category**: {self.metadata.category.value}\n"
        doc += f"**Version**: {self.metadata.version}\n"
        doc += f"**Author**: {self.metadata.author}\n\n"
        
        doc += f"## Description\n{self.metadata.description}\n\n"
        
        if self.metadata.permissions:
            doc += "## Required Permissions\n"
            for perm in self.metadata.permissions:
                doc += f"- {perm.value}\n"
            doc += "\n"
        
        doc += "## Input Schema\n```python\n"
        doc += str(self.get_input_schema().schema())
        doc += "\n```\n\n"
        
        if self.metadata.examples:
            doc += "## Examples\n"
            for i, example in enumerate(self.metadata.examples, 1):
                doc += f"### Example {i}\n"
                doc += f"**Input**: `{example.get('input')}`\n"
                doc += f"**Output**: `{example.get('output')}`\n"
                if example.get('description'):
                    doc += f"**Description**: {example.get('description')}\n"
                doc += "\n"
        
        return doc
```

### Task 2: Tool Registry and Discovery (20 min)
Create `src/tools/registry.py`:

```python
import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Dict, List, Optional, Type, Set
import structlog

from src.tools.advanced_base import AdvancedTool, ToolCategory, ToolPermission

logger = structlog.get_logger()

class ToolRegistry:
    """Central registry for tool discovery and management"""
    
    def __init__(self):
        self.tools: Dict[str, AdvancedTool] = {}
        self.categories: Dict[ToolCategory, List[str]] = {}
        self.permission_map: Dict[ToolPermission, List[str]] = {}
        
    def register(self, tool: AdvancedTool) -> None:
        """Register a tool"""
        name = tool.metadata.name
        
        if name in self.tools:
            logger.warning(f"Tool {name} already registered, overwriting")
        
        self.tools[name] = tool
        
        # Update category index
        category = tool.metadata.category
        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(name)
        
        # Update permission index
        for permission in tool.metadata.permissions:
            if permission not in self.permission_map:
                self.permission_map[permission] = []
            self.permission_map[permission].append(name)
        
        logger.info(
            "Tool registered",
            name=name,
            category=category.value,
            permissions=[p.value for p in tool.metadata.permissions]
        )
    
    def discover_tools(self, package_path: str) -> int:
        """Automatically discover and register tools from a package"""
        count = 0
        package = importlib.import_module(package_path)
        
        # Get package directory
        package_dir = Path(package.__file__).parent
        
        # Iterate through modules in package
        for _, module_name, _ in pkgutil.iter_modules([str(package_dir)]):
            if module_name.startswith('_'):
                continue
                
            full_module_name = f"{package_path}.{module_name}"
            
            try:
                module = importlib.import_module(full_module_name)
                
                # Find all tool classes in module
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, AdvancedTool) and 
                        obj != AdvancedTool):
                        
                        try:
                            tool_instance = obj()
                            self.register(tool_instance)
                            count += 1
                        except Exception as e:
                            logger.error(
                                f"Failed to instantiate tool {name}",
                                error=str(e)
                            )
                            
            except Exception as e:
                logger.error(
                    f"Failed to import module {full_module_name}",
                    error=str(e)
                )
        
        logger.info(f"Discovered {count} tools from {package_path}")
        return count
    
    def get_tool(self, name: str) -> Optional[AdvancedTool]:
        """Get tool by name"""
        return self.tools.get(name)
    
    def get_tools_by_category(self, category: ToolCategory) -> List[AdvancedTool]:
        """Get all tools in a category"""
        tool_names = self.categories.get(category, [])
        return [self.tools[name] for name in tool_names]
    
    def get_tools_by_permission(self, permission: ToolPermission) -> List[AdvancedTool]:
        """Get all tools requiring a specific permission"""
        tool_names = self.permission_map.get(permission, [])
        return [self.tools[name] for name in tool_names]
    
    def get_tools_with_permissions(self, allowed_permissions: Set[ToolPermission]) -> List[AdvancedTool]:
        """Get tools that only require allowed permissions"""
        result = []
        
        for tool in self.tools.values():
            required_permissions = set(tool.metadata.permissions)
            if required_permissions.issubset(allowed_permissions):
                result.append(tool)
        
        return result
    
    def search_tools(self, query: str) -> List[AdvancedTool]:
        """Search tools by name, description, or tags"""
        query_lower = query.lower()
        results = []
        
        for tool in self.tools.values():
            # Search in name
            if query_lower in tool.metadata.name.lower():
                results.append(tool)
                continue
            
            # Search in description
            if query_lower in tool.metadata.description.lower():
                results.append(tool)
                continue
            
            # Search in tags
            for tag in tool.metadata.tags:
                if query_lower in tag.lower():
                    results.append(tool)
                    break
        
        return results
    
    def get_all_function_specs(self) -> List[Dict]:
        """Get OpenAI function specs for all tools"""
        return [tool.get_openai_function_spec() for tool in self.tools.values()]
    
    def generate_catalog(self) -> str:
        """Generate a catalog of all tools"""
        catalog = "# Tool Catalog\n\n"
        
        for category in ToolCategory:
            tools = self.get_tools_by_category(category)
            if not tools:
                continue
                
            catalog += f"## {category.value.replace('_', ' ').title()}\n\n"
            
            for tool in tools:
                catalog += f"### {tool.metadata.name}\n"
                catalog += f"{tool.metadata.description}\n"
                catalog += f"- Version: {tool.metadata.version}\n"
                catalog += f"- Permissions: {', '.join(p.value for p in tool.metadata.permissions)}\n"
                catalog += "\n"
        
        return catalog

# Singleton registry
_registry = None

def get_registry() -> ToolRegistry:
    """Get the global tool registry"""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry
```

### Task 3: Tool Composition and Chaining (20 min)
Create `src/tools/composition.py`:

```python
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import asyncio
import structlog

from src.tools.advanced_base import AdvancedTool, ToolOutput

logger = structlog.get_logger()

@dataclass
class PipelineStep:
    """Single step in a tool pipeline"""
    tool_name: str
    input_mapping: Dict[str, str]  # Map previous outputs to inputs
    output_key: str  # Key to store output under
    condition: Optional[str] = None  # Python expression to evaluate

class ToolPipeline:
    """Chain multiple tools together"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.steps: List[PipelineStep] = []
        self.context: Dict[str, Any] = {}
        
    def add_step(
        self,
        tool_name: str,
        input_mapping: Dict[str, str],
        output_key: str,
        condition: Optional[str] = None
    ) -> 'ToolPipeline':
        """Add a step to the pipeline"""
        step = PipelineStep(
            tool_name=tool_name,
            input_mapping=input_mapping,
            output_key=output_key,
            condition=condition
        )
        self.steps.append(step)
        return self
    
    async def execute(
        self,
        initial_input: Dict[str, Any],
        tool_registry
    ) -> Dict[str, Any]:
        """Execute the pipeline"""
        self.context = {"input": initial_input, "outputs": {}}
        
        for i, step in enumerate(self.steps):
            logger.info(f"Executing pipeline step {i+1}/{len(self.steps)}", 
                       tool=step.tool_name)
            
            # Check condition
            if step.condition:
                if not self._evaluate_condition(step.condition):
                    logger.info(f"Skipping step {i+1} due to condition")
                    continue
            
            # Get tool
            tool = tool_registry.get_tool(step.tool_name)
            if not tool:
                raise ValueError(f"Tool {step.tool_name} not found")
            
            # Build input
            tool_input = self._build_input(step.input_mapping)
            
            # Execute tool
            result = await tool.execute(tool_input)
            
            # Store output
            self.context["outputs"][step.output_key] = result
            
            # Check for errors
            if not result.success:
                logger.error(f"Pipeline failed at step {i+1}", 
                           error=result.error)
                return {
                    "success": False,
                    "failed_step": i + 1,
                    "error": result.error,
                    "partial_results": self.context["outputs"]
                }
        
        return {
            "success": True,
            "results": self.context["outputs"]
        }
    
    def _evaluate_condition(self, condition: str) -> bool:
        """Safely evaluate a condition"""
        try:
            # Create safe evaluation context
            safe_context = {
                "input": self.context["input"],
                "outputs": self.context["outputs"]
            }
            return eval(condition, {"__builtins__": {}}, safe_context)
        except Exception as e:
            logger.error(f"Failed to evaluate condition: {condition}", 
                        error=str(e))
            return False
    
    def _build_input(self, mapping: Dict[str, str]) -> Dict[str, Any]:
        """Build input from mapping"""
        result = {}
        
        for param, source in mapping.items():
            value = self._resolve_value(source)
            if value is not None:
                result[param] = value
        
        return result
    
    def _resolve_value(self, source: str) -> Any:
        """Resolve a value from the context"""
        if source.startswith("input."):
            key = source[6:]
            return self.context["input"].get(key)
        elif source.startswith("outputs."):
            path = source[8:].split(".")
            value = self.context["outputs"]
            for key in path:
                if isinstance(value, dict):
                    value = value.get(key)
                elif hasattr(value, key):
                    value = getattr(value, key)
                else:
                    return None
            return value
        else:
            # Literal value
            return source

class ParallelToolExecutor:
    """Execute multiple tools in parallel"""
    
    def __init__(self, max_concurrency: int = 5):
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)
        
    async def execute_parallel(
        self,
        tool_calls: List[Dict[str, Any]],
        tool_registry
    ) -> List[ToolOutput]:
        """Execute multiple tool calls in parallel"""
        tasks = []
        
        for call in tool_calls:
            tool_name = call.get("tool")
            tool_input = call.get("input", {})
            
            tool = tool_registry.get_tool(tool_name)
            if not tool:
                logger.warning(f"Tool {tool_name} not found")
                continue
            
            task = self._execute_with_semaphore(tool, tool_input)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error outputs
        outputs = []
        for result in results:
            if isinstance(result, Exception):
                outputs.append(ToolOutput(
                    success=False,
                    result=None,
                    error=str(result)
                ))
            else:
                outputs.append(result)
        
        return outputs
    
    async def _execute_with_semaphore(
        self,
        tool: AdvancedTool,
        tool_input: Dict[str, Any]
    ) -> ToolOutput:
        """Execute tool with concurrency limit"""
        async with self.semaphore:
            return await tool.execute(tool_input)

class ToolRouter:
    """Route requests to appropriate tools based on intent"""
    
    def __init__(self, tool_registry):
        self.registry = tool_registry
        self.routes: Dict[str, List[str]] = {}  # intent -> tool names
        
    def add_route(self, intent: str, tool_names: List[str]) -> None:
        """Add routing rule"""
        self.routes[intent] = tool_names
        
    async def route_request(
        self,
        intent: str,
        input_data: Dict[str, Any]
    ) -> Optional[ToolOutput]:
        """Route request to appropriate tool"""
        tool_names = self.routes.get(intent, [])
        
        for tool_name in tool_names:
            tool = self.registry.get_tool(tool_name)
            if not tool:
                continue
                
            try:
                result = await tool.execute(input_data)
                if result.success:
                    return result
            except Exception as e:
                logger.error(f"Tool {tool_name} failed for intent {intent}",
                           error=str(e))
                continue
        
        return None
```

### Task 4: Security and Sandboxing (10 min)
Create `src/tools/security.py`:

```python
import resource
import signal
import subprocess
import tempfile
from contextlib import contextmanager
from typing import Any, Callable, Optional, Set
import structlog

from src.tools.advanced_base import ToolPermission

logger = structlog.get_logger()

class SecurityContext:
    """Security context for tool execution"""
    
    def __init__(
        self,
        allowed_permissions: Set[ToolPermission],
        max_execution_time: int = 30,
        max_memory_mb: int = 512
    ):
        self.allowed_permissions = allowed_permissions
        self.max_execution_time = max_execution_time
        self.max_memory_mb = max_memory_mb
        
    def has_permission(self, permission: ToolPermission) -> bool:
        """Check if permission is allowed"""
        return permission in self.allowed_permissions
    
    def validate_tool_permissions(self, required_permissions: List[ToolPermission]) -> bool:
        """Validate tool has required permissions"""
        for permission in required_permissions:
            if not self.has_permission(permission):
                logger.warning(
                    f"Permission denied: {permission.value}",
                    required=permission.value,
                    allowed=[p.value for p in self.allowed_permissions]
                )
                return False
        return True

@contextmanager
def resource_limits(max_time: int, max_memory_mb: int):
    """Apply resource limits to execution"""
    def timeout_handler(signum, frame):
        raise TimeoutError("Execution timeout exceeded")
    
    # Set CPU time limit
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(max_time)
    
    # Set memory limit (Linux only)
    if hasattr(resource, 'RLIMIT_AS'):
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(
            resource.RLIMIT_AS,
            (max_memory_mb * 1024 * 1024, hard)
        )
    
    try:
        yield
    finally:
        signal.alarm(0)  # Disable alarm
        if hasattr(resource, 'RLIMIT_AS'):
            resource.setrlimit(resource.RLIMIT_AS, (soft, hard))

class SandboxedExecutor:
    """Execute code in a sandboxed environment"""
    
    def __init__(self, security_context: SecurityContext):
        self.security_context = security_context
        
    async def execute_sandboxed(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute function in sandbox"""
        if not self.security_context.has_permission(ToolPermission.EXECUTE_CODE):
            raise PermissionError("Code execution not allowed")
        
        with resource_limits(
            self.security_context.max_execution_time,
            self.security_context.max_memory_mb
        ):
            return await func(*args, **kwargs)
    
    def execute_subprocess(
        self,
        command: List[str],
        input_data: Optional[str] = None
    ) -> subprocess.CompletedProcess:
        """Execute subprocess with restrictions"""
        if not self.security_context.has_permission(ToolPermission.SYSTEM_COMMANDS):
            raise PermissionError("System commands not allowed")
        
        # Create temporary directory for isolation
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                result = subprocess.run(
                    command,
                    input=input_data,
                    capture_output=True,
                    text=True,
                    timeout=self.security_context.max_execution_time,
                    cwd=tmpdir,  # Isolate to temp directory
                    env={"PATH": "/usr/bin:/bin"},  # Minimal environment
                )
                return result
            except subprocess.TimeoutExpired:
                raise TimeoutError("Command execution timeout")

class PermissionManager:
    """Manage tool permissions dynamically"""
    
    def __init__(self):
        self.user_permissions: Dict[str, Set[ToolPermission]] = {}
        self.default_permissions = {
            ToolPermission.READ_FILES,
            ToolPermission.NETWORK_ACCESS,
        }
        
    def grant_permission(
        self,
        user_id: str,
        permission: ToolPermission
    ) -> None:
        """Grant permission to user"""
        if user_id not in self.user_permissions:
            self.user_permissions[user_id] = self.default_permissions.copy()
        
        self.user_permissions[user_id].add(permission)
        logger.info(f"Granted permission {permission.value} to user {user_id}")
        
    def revoke_permission(
        self,
        user_id: str,
        permission: ToolPermission
    ) -> None:
        """Revoke permission from user"""
        if user_id in self.user_permissions:
            self.user_permissions[user_id].discard(permission)
            logger.info(f"Revoked permission {permission.value} from user {user_id}")
    
    def get_user_permissions(self, user_id: str) -> Set[ToolPermission]:
        """Get user's permissions"""
        return self.user_permissions.get(user_id, self.default_permissions.copy())
    
    def create_security_context(
        self,
        user_id: str,
        **kwargs
    ) -> SecurityContext:
        """Create security context for user"""
        permissions = self.get_user_permissions(user_id)
        return SecurityContext(permissions, **kwargs)
```

## 🌞 Afternoon Tasks (90-120 minutes)

### Task 5: Implement Example Tools (45 min)
Create `src/tools/implementations/calculator_advanced.py`:

```python
from typing import List, Optional
import ast
import math
from pydantic import Field, validator

from src.tools.advanced_base import (
    AdvancedTool, ToolMetadata, ToolCategory,
    ToolPermission, ToolInput, ToolOutput
)

class CalculatorInput(ToolInput):
    """Input for calculator tool"""
    expression: str = Field(..., description="Mathematical expression to evaluate")
    precision: Optional[int] = Field(2, description="Decimal precision for result")
    
    @validator('expression')
    def validate_expression(cls, v):
        """Validate expression is safe"""
        if len(v) > 1000:
            raise ValueError("Expression too long")
        
        # Check for forbidden characters
        forbidden = ['__', 'import', 'exec', 'eval', 'open', 'file']
        for word in forbidden:
            if word in v:
                raise ValueError(f"Forbidden keyword: {word}")
        
        return v

class CalculatorOutput(ToolOutput):
    """Output for calculator tool"""
    result: Optional[float]
    formatted_result: Optional[str]
    steps: List[str] = Field(default_factory=list)

class AdvancedCalculatorTool(AdvancedTool):
    """Advanced calculator with step tracking"""
    
    def _build_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="calculator",
            description="Perform mathematical calculations with step-by-step explanation",
            category=ToolCategory.CALCULATION,
            version="2.0.0",
            permissions=[],  # No special permissions needed
            examples=[
                {
                    "input": '{"expression": "2 + 2 * 3"}',
                    "output": '{"result": 8.0, "formatted_result": "8.00"}',
                    "description": "Basic arithmetic with order of operations"
                },
                {
                    "input": '{"expression": "sqrt(16) + pi", "precision": 4}',
                    "output": '{"result": 7.1416, "formatted_result": "7.1416"}',
                    "description": "Using mathematical functions and constants"
                }
            ],
            tags=["math", "arithmetic", "calculation"]
        )
    
    def get_input_schema(self) -> type[ToolInput]:
        return CalculatorInput
    
    def get_output_schema(self) -> type[ToolOutput]:
        return CalculatorOutput
    
    async def _execute(self, validated_input: CalculatorInput) -> CalculatorOutput:
        """Execute calculation"""
        try:
            # Parse and evaluate expression
            steps = []
            result = self._evaluate_expression(
                validated_input.expression,
                steps
            )
            
            # Format result
            formatted = f"{result:.{validated_input.precision}f}"
            
            return CalculatorOutput(
                success=True,
                result=result,
                formatted_result=formatted,
                steps=steps
            )
            
        except Exception as e:
            return CalculatorOutput(
                success=False,
                result=None,
                formatted_result=None,
                error=str(e)
            )
    
    def _evaluate_expression(self, expr: str, steps: List[str]) -> float:
        """Safely evaluate mathematical expression"""
        # Add safe math functions
        safe_dict = {
            'pi': math.pi,
            'e': math.e,
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'log': math.log,
            'log10': math.log10,
            'exp': math.exp,
            'pow': pow,
            'abs': abs,
            'round': round,
        }
        
        steps.append(f"Evaluating: {expr}")
        
        # Parse expression
        try:
            node = ast.parse(expr, mode='eval')
            
            # Validate AST (no function calls except allowed ones)
            for n in ast.walk(node):
                if isinstance(n, ast.Call):
                    if isinstance(n.func, ast.Name):
                        if n.func.id not in safe_dict:
                            raise ValueError(f"Function {n.func.id} not allowed")
            
            # Compile and evaluate
            code = compile(node, '<string>', 'eval')
            result = eval(code, {"__builtins__": {}}, safe_dict)
            
            steps.append(f"Result: {result}")
            return float(result)
            
        except Exception as e:
            raise ValueError(f"Invalid expression: {str(e)}")
```

Create `src/tools/implementations/file_tools.py`:

```python
from pathlib import Path
from typing import Optional, List
import os
import mimetypes
from pydantic import Field, validator

from src.tools.advanced_base import (
    AdvancedTool, ToolMetadata, ToolCategory,
    ToolPermission, ToolInput, ToolOutput
)

class FileReaderInput(ToolInput):
    """Input for file reader tool"""
    file_path: str = Field(..., description="Path to file to read")
    encoding: str = Field("utf-8", description="File encoding")
    max_size_mb: int = Field(10, description="Maximum file size in MB")
    
    @validator('file_path')
    def validate_path(cls, v):
        """Validate file path"""
        # Prevent directory traversal
        if '..' in v or v.startswith('/'):
            raise ValueError("Invalid file path")
        return v

class FileReaderOutput(ToolOutput):
    """Output for file reader tool"""
    content: Optional[str]
    file_size: Optional[int]
    mime_type: Optional[str]
    line_count: Optional[int]

class SecureFileReaderTool(AdvancedTool):
    """Secure file reader with size limits"""
    
    def __init__(self, base_directory: str = "."):
        super().__init__()
        self.base_directory = Path(base_directory).resolve()
    
    def _build_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="file_reader",
            description="Safely read files with size and path restrictions",
            category=ToolCategory.FILE_SYSTEM,
            version="1.0.0",
            permissions=[ToolPermission.READ_FILES],
            examples=[
                {
                    "input": '{"file_path": "data.txt"}',
                    "output": '{"content": "Hello World", "file_size": 11}',
                    "description": "Read a text file"
                }
            ],
            tags=["file", "read", "text"]
        )
    
    def get_input_schema(self) -> type[ToolInput]:
        return FileReaderInput
    
    def get_output_schema(self) -> type[ToolOutput]:
        return FileReaderOutput
    
    async def _execute(self, validated_input: FileReaderInput) -> FileReaderOutput:
        """Execute file reading"""
        try:
            # Resolve path safely
            file_path = (self.base_directory / validated_input.file_path).resolve()
            
            # Ensure path is within base directory
            if not str(file_path).startswith(str(self.base_directory)):
                raise ValueError("Access denied: Path outside allowed directory")
            
            # Check file exists
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {validated_input.file_path}")
            
            # Check file size
            file_size = file_path.stat().st_size
            if file_size > validated_input.max_size_mb * 1024 * 1024:
                raise ValueError(f"File too large: {file_size} bytes")
            
            # Read file
            with open(file_path, 'r', encoding=validated_input.encoding) as f:
                content = f.read()
            
            # Get mime type
            mime_type, _ = mimetypes.guess_type(str(file_path))
            
            # Count lines
            line_count = content.count('\n') + 1
            
            return FileReaderOutput(
                success=True,
                result=content,
                content=content,
                file_size=file_size,
                mime_type=mime_type,
                line_count=line_count
            )
            
        except Exception as e:
            return FileReaderOutput(
                success=False,
                result=None,
                content=None,
                error=str(e)
            )

class FileListInput(ToolInput):
    """Input for file listing tool"""
    directory: str = Field(".", description="Directory to list")
    pattern: str = Field("*", description="File pattern to match")
    recursive: bool = Field(False, description="List recursively")
    include_hidden: bool = Field(False, description="Include hidden files")

class FileListOutput(ToolOutput):
    """Output for file listing tool"""
    files: List[Dict[str, Any]]
    total_count: int
    total_size: int

class FileListTool(AdvancedTool):
    """List files in directory"""
    
    def __init__(self, base_directory: str = "."):
        super().__init__()
        self.base_directory = Path(base_directory).resolve()
    
    def _build_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="file_list",
            description="List files in a directory with filtering",
            category=ToolCategory.FILE_SYSTEM,
            version="1.0.0",
            permissions=[ToolPermission.READ_FILES],
            tags=["file", "list", "directory"]
        )
    
    def get_input_schema(self) -> type[ToolInput]:
        return FileListInput
    
    def get_output_schema(self) -> type[ToolOutput]:
        return FileListOutput
    
    async def _execute(self, validated_input: FileListInput) -> FileListOutput:
        """Execute file listing"""
        try:
            # Resolve directory path
            dir_path = (self.base_directory / validated_input.directory).resolve()
            
            # Ensure path is within base directory
            if not str(dir_path).startswith(str(self.base_directory)):
                raise ValueError("Access denied: Path outside allowed directory")
            
            # List files
            files = []
            total_size = 0
            
            if validated_input.recursive:
                paths = dir_path.rglob(validated_input.pattern)
            else:
                paths = dir_path.glob(validated_input.pattern)
            
            for path in paths:
                if path.is_file():
                    # Skip hidden files if requested
                    if not validated_input.include_hidden and path.name.startswith('.'):
                        continue
                    
                    stat = path.stat()
                    file_info = {
                        "name": path.name,
                        "path": str(path.relative_to(self.base_directory)),
                        "size": stat.st_size,
                        "modified": stat.st_mtime,
                        "is_file": True
                    }
                    files.append(file_info)
                    total_size += stat.st_size
            
            # Sort by name
            files.sort(key=lambda x: x["name"])
            
            return FileListOutput(
                success=True,
                result=files,
                files=files,
                total_count=len(files),
                total_size=total_size
            )
            
        except Exception as e:
            return FileListOutput(
                success=False,
                result=None,
                files=[],
                total_count=0,
                total_size=0,
                error=str(e)
            )
```

### Task 6: Tool Testing Framework (30 min)
Create `src/tools/testing.py`:

```python
import asyncio
from typing import Any, Dict, List, Optional, Type
from dataclasses import dataclass
import json
import structlog

from src.tools.advanced_base import AdvancedTool, ToolInput, ToolOutput

logger = structlog.get_logger()

@dataclass
class TestCase:
    """Test case for a tool"""
    name: str
    input_data: Dict[str, Any]
    expected_success: bool
    expected_output: Optional[Dict[str, Any]] = None
    expected_error_contains: Optional[str] = None
    description: Optional[str] = None

class ToolTester:
    """Test framework for tools"""
    
    def __init__(self):
        self.results = []
        
    async def test_tool(
        self,
        tool: AdvancedTool,
        test_cases: List[TestCase]
    ) -> Dict[str, Any]:
        """Run test cases against a tool"""
        logger.info(f"Testing tool: {tool.metadata.name}")
        
        passed = 0
        failed = 0
        
        for test_case in test_cases:
            result = await self._run_test_case(tool, test_case)
            self.results.append(result)
            
            if result["passed"]:
                passed += 1
            else:
                failed += 1
        
        return {
            "tool": tool.metadata.name,
            "total": len(test_cases),
            "passed": passed,
            "failed": failed,
            "success_rate": passed / len(test_cases) if test_cases else 0
        }
    
    async def _run_test_case(
        self,
        tool: AdvancedTool,
        test_case: TestCase
    ) -> Dict[str, Any]:
        """Run single test case"""
        logger.info(f"Running test: {test_case.name}")
        
        try:
            # Execute tool
            output = await tool.execute(test_case.input_data)
            
            # Check success
            if output.success != test_case.expected_success:
                return {
                    "test": test_case.name,
                    "passed": False,
                    "reason": f"Expected success={test_case.expected_success}, got {output.success}",
                    "output": output.dict()
                }
            
            # Check error message
            if test_case.expected_error_contains and output.error:
                if test_case.expected_error_contains not in output.error:
                    return {
                        "test": test_case.name,
                        "passed": False,
                        "reason": f"Error should contain '{test_case.expected_error_contains}'",
                        "actual_error": output.error
                    }
            
            # Check output fields
            if test_case.expected_output:
                for key, expected_value in test_case.expected_output.items():
                    actual_value = getattr(output, key, None)
                    if actual_value != expected_value:
                        return {
                            "test": test_case.name,
                            "passed": False,
                            "reason": f"Expected {key}={expected_value}, got {actual_value}",
                            "output": output.dict()
                        }
            
            return {
                "test": test_case.name,
                "passed": True,
                "output": output.dict()
            }
            
        except Exception as e:
            return {
                "test": test_case.name,
                "passed": False,
                "reason": f"Exception: {str(e)}",
                "exception": str(e)
            }
    
    def generate_report(self) -> str:
        """Generate test report"""
        report = "# Tool Test Report\n\n"
        
        # Group by tool
        by_tool = {}
        for result in self.results:
            tool_name = result.get("tool", "unknown")
            if tool_name not in by_tool:
                by_tool[tool_name] = []
            by_tool[tool_name].append(result)
        
        # Generate report for each tool
        for tool_name, tool_results in by_tool.items():
            report += f"## {tool_name}\n\n"
            
            passed = sum(1 for r in tool_results if r.get("passed", False))
            total = len(tool_results)
            
            report += f"**Results**: {passed}/{total} passed ({passed/total*100:.1f}%)\n\n"
            
            # List failures
            failures = [r for r in tool_results if not r.get("passed", False)]
            if failures:
                report += "### Failures\n\n"
                for failure in failures:
                    report += f"- **{failure['test']}**: {failure['reason']}\n"
                report += "\n"
        
        return report

class ToolBenchmark:
    """Benchmark tool performance"""
    
    async def benchmark_tool(
        self,
        tool: AdvancedTool,
        test_input: Dict[str, Any],
        iterations: int = 100
    ) -> Dict[str, Any]:
        """Benchmark tool performance"""
        import time
        
        logger.info(f"Benchmarking tool: {tool.metadata.name}")
        
        # Warmup
        await tool.execute(test_input)
        
        # Run benchmark
        times = []
        errors = 0
        
        for _ in range(iterations):
            start = time.time()
            try:
                output = await tool.execute(test_input)
                if not output.success:
                    errors += 1
            except Exception:
                errors += 1
            
            times.append(time.time() - start)
        
        # Calculate statistics
        times.sort()
        
        return {
            "tool": tool.metadata.name,
            "iterations": iterations,
            "errors": errors,
            "avg_time": sum(times) / len(times),
            "min_time": times[0],
            "max_time": times[-1],
            "p50_time": times[len(times) // 2],
            "p95_time": times[int(len(times) * 0.95)],
            "p99_time": times[int(len(times) * 0.99)],
        }

# Example test suite
def create_calculator_test_suite() -> List[TestCase]:
    """Create test suite for calculator"""
    return [
        TestCase(
            name="basic_addition",
            input_data={"expression": "2 + 2"},
            expected_success=True,
            expected_output={"result": 4.0}
        ),
        TestCase(
            name="order_of_operations",
            input_data={"expression": "2 + 2 * 3"},
            expected_success=True,
            expected_output={"result": 8.0}
        ),
        TestCase(
            name="invalid_expression",
            input_data={"expression": "2 + + 2"},
            expected_success=False,
            expected_error_contains="Invalid expression"
        ),
        TestCase(
            name="forbidden_function",
            input_data={"expression": "__import__('os')"},
            expected_success=False,
            expected_error_contains="Forbidden"
        ),
    ]
```

### Task 7: Integration with Agent (15 min)
Create `src/agent_with_tools.py`:

```python
import asyncio
from typing import List, Optional, Dict, Any
import structlog

from src.tools.registry import get_registry
from src.tools.security import SecurityContext, ToolPermission
from src.tools.composition import ToolPipeline, ParallelToolExecutor
from src.llm.advanced_client import LLMConfig
from src.llm.openai_advanced import AdvancedOpenAIClient

logger = structlog.get_logger()

class ToolAwareAgent:
    """Agent with advanced tool capabilities"""
    
    def __init__(
        self,
        llm_config: LLMConfig,
        security_context: SecurityContext,
        auto_discover: bool = True
    ):
        self.llm = AdvancedOpenAIClient(llm_config)
        self.security_context = security_context
        self.registry = get_registry()
        self.parallel_executor = ParallelToolExecutor()
        
        if auto_discover:
            # Auto-discover tools
            self.registry.discover_tools("src.tools.implementations")
        
        # Get allowed tools based on permissions
        self.allowed_tools = self.registry.get_tools_with_permissions(
            security_context.allowed_permissions
        )
        
        logger.info(
            "Tool-aware agent initialized",
            allowed_tools=[t.metadata.name for t in self.allowed_tools]
        )
    
    async def run(self, task: str) -> str:
        """Run task with tool support"""
        # Get function specs for allowed tools
        function_specs = [
            tool.get_openai_function_spec() 
            for tool in self.allowed_tools
        ]
        
        # Initial prompt
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant with access to tools."
            },
            {
                "role": "user",
                "content": task
            }
        ]
        
        async with self.llm:
            for _ in range(10):  # Max iterations
                # Get LLM response with function calling
                response = await self.llm.client.chat.completions.create(
                    model=self.llm.config.model,
                    messages=messages,
                    functions=function_specs,
                    function_call="auto"
                )
                
                message = response.choices[0].message
                
                # Check if LLM wants to call a function
                if message.function_call:
                    # Execute tool
                    tool_name = message.function_call.name
                    tool_args = json.loads(message.function_call.arguments)
                    
                    tool = self.registry.get_tool(tool_name)
                    if tool and tool in self.allowed_tools:
                        result = await tool.execute(tool_args)
                        
                        # Add result to conversation
                        messages.append({
                            "role": "function",
                            "name": tool_name,
                            "content": json.dumps(result.dict())
                        })
                    else:
                        messages.append({
                            "role": "function",
                            "name": tool_name,
                            "content": json.dumps({
                                "error": f"Tool {tool_name} not found or not allowed"
                            })
                        })
                else:
                    # Final response
                    return message.content
        
        return "I couldn't complete the task within the allowed iterations."
    
    async def execute_pipeline(
        self,
        pipeline: ToolPipeline,
        initial_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a tool pipeline"""
        return await pipeline.execute(initial_input, self.registry)
    
    async def execute_parallel_tools(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> List[Any]:
        """Execute multiple tools in parallel"""
        # Filter allowed tools
        allowed_calls = []
        for call in tool_calls:
            tool_name = call.get("tool")
            tool = self.registry.get_tool(tool_name)
            if tool and tool in self.allowed_tools:
                allowed_calls.append(call)
            else:
                logger.warning(f"Tool {tool_name} not allowed")
        
        return await self.parallel_executor.execute_parallel(
            allowed_calls,
            self.registry
        )

# Example usage
async def main():
    # Setup
    llm_config = LLMConfig(
        api_key="your-key",
        model="gpt-3.5-turbo"
    )
    
    # Create security context
    security_context = SecurityContext(
        allowed_permissions={
            ToolPermission.READ_FILES,
            ToolPermission.NETWORK_ACCESS
        }
    )
    
    # Create agent
    agent = ToolAwareAgent(llm_config, security_context)
    
    # Simple task
    result = await agent.run("Calculate the square root of 144")
    print(f"Result: {result}")
    
    # Pipeline example
    pipeline = ToolPipeline("data_processing", "Process data files")
    pipeline.add_step(
        tool_name="file_list",
        input_mapping={"directory": "data", "pattern": "*.txt"},
        output_key="file_list"
    )
    pipeline.add_step(
        tool_name="file_reader",
        input_mapping={"file_path": "outputs.file_list.files[0].path"},
        output_key="content",
        condition="len(outputs.file_list.files) > 0"
    )
    
    pipeline_result = await agent.execute_pipeline(
        pipeline,
        {"directory": "data"}
    )
    print(f"Pipeline result: {pipeline_result}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 🌆 Evening Tasks (45-60 minutes)

### Task 8: Tool Documentation Generator (20 min)
Create `src/tools/documentation.py`:

```python
from typing import List, Optional
import json
from pathlib import Path
import structlog

from src.tools.registry import ToolRegistry
from src.tools.advanced_base import ToolCategory

logger = structlog.get_logger()

class ToolDocumentationGenerator:
    """Generate documentation for tools"""
    
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        
    def generate_markdown_docs(self, output_dir: str) -> None:
        """Generate markdown documentation for all tools"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate index
        self._generate_index(output_path)
        
        # Generate category pages
        for category in ToolCategory:
            self._generate_category_page(output_path, category)
        
        # Generate individual tool pages
        for tool in self.registry.tools.values():
            self._generate_tool_page(output_path, tool)
            
        logger.info(f"Generated documentation in {output_dir}")
    
    def _generate_index(self, output_path: Path) -> None:
        """Generate index page"""
        content = "# Tool Documentation\n\n"
        content += "## Categories\n\n"
        
        for category in ToolCategory:
            tools = self.registry.get_tools_by_category(category)
            if tools:
                content += f"- [{category.value}](./{category.value}.md) ({len(tools)} tools)\n"
        
        content += "\n## All Tools\n\n"
        
        for tool in sorted(self.registry.tools.values(), key=lambda t: t.metadata.name):
            content += f"- [{tool.metadata.name}](./tools/{tool.metadata.name}.md) - {tool.metadata.description}\n"
        
        with open(output_path / "index.md", "w") as f:
            f.write(content)
    
    def _generate_category_page(self, output_path: Path, category: ToolCategory) -> None:
        """Generate category page"""
        tools = self.registry.get_tools_by_category(category)
        if not tools:
            return
        
        content = f"# {category.value.replace('_', ' ').title()} Tools\n\n"
        
        for tool in sorted(tools, key=lambda t: t.metadata.name):
            content += f"## [{tool.metadata.name}](./tools/{tool.metadata.name}.md)\n\n"
            content += f"{tool.metadata.description}\n\n"
            
            if tool.metadata.permissions:
                content += "**Permissions**: "
                content += ", ".join(p.value for p in tool.metadata.permissions)
                content += "\n\n"
        
        with open(output_path / f"{category.value}.md", "w") as f:
            f.write(content)
    
    def _generate_tool_page(self, output_path: Path, tool) -> None:
        """Generate individual tool page"""
        tools_dir = output_path / "tools"
        tools_dir.mkdir(exist_ok=True)
        
        content = tool.get_documentation()
        
        # Add JSON schema
        content += "\n## JSON Schema\n\n```json\n"
        content += json.dumps(tool.get_openai_function_spec(), indent=2)
        content += "\n```\n"
        
        with open(tools_dir / f"{tool.metadata.name}.md", "w") as f:
            f.write(content)
    
    def generate_openapi_spec(self) -> Dict[str, Any]:
        """Generate OpenAPI specification for tools"""
        spec = {
            "openapi": "3.0.0",
            "info": {
                "title": "Tool API",
                "version": "1.0.0",
                "description": "API specification for available tools"
            },
            "paths": {}
        }
        
        for tool in self.registry.tools.values():
            path = f"/tools/{tool.metadata.name}/execute"
            
            spec["paths"][path] = {
                "post": {
                    "summary": tool.metadata.description,
                    "operationId": f"execute_{tool.metadata.name}",
                    "tags": [tool.metadata.category.value],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": tool.get_input_schema().schema()
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Successful execution",
                            "content": {
                                "application/json": {
                                    "schema": tool.get_output_schema().schema()
                                }
                            }
                        }
                    }
                }
            }
        
        return spec
```

### Task 9: Tool Usage Analytics (15 min)
Create `src/tools/analytics.py`:

```python
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import json
from collections import defaultdict
import structlog

logger = structlog.get_logger()

@dataclass
class ToolUsageEvent:
    """Single tool usage event"""
    tool_name: str
    timestamp: datetime
    success: bool
    execution_time: float
    user_id: Optional[str] = None
    error: Optional[str] = None
    input_size: int = 0
    output_size: int = 0

class ToolAnalytics:
    """Track and analyze tool usage"""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.events: List[ToolUsageEvent] = []
        self.storage_path = storage_path
        
        if storage_path:
            self._load_events()
    
    def record_usage(self, event: ToolUsageEvent) -> None:
        """Record tool usage event"""
        self.events.append(event)
        
        if self.storage_path:
            self._save_events()
            
        logger.info(
            "Tool usage recorded",
            tool=event.tool_name,
            success=event.success,
            execution_time=event.execution_time
        )
    
    def get_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get usage summary for date range"""
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()
        
        # Filter events
        filtered_events = [
            e for e in self.events
            if start_date <= e.timestamp <= end_date
        ]
        
        if not filtered_events:
            return {"message": "No events in date range"}
        
        # Calculate statistics
        tool_stats = defaultdict(lambda: {
            "count": 0,
            "success_count": 0,
            "total_time": 0,
            "errors": []
        })
        
        for event in filtered_events:
            stats = tool_stats[event.tool_name]
            stats["count"] += 1
            if event.success:
                stats["success_count"] += 1
            else:
                stats["errors"].append(event.error)
            stats["total_time"] += event.execution_time
        
        # Calculate aggregates
        summary = {
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "total_events": len(filtered_events),
            "unique_tools": len(tool_stats),
            "by_tool": {}
        }
        
        for tool_name, stats in tool_stats.items():
            summary["by_tool"][tool_name] = {
                "usage_count": stats["count"],
                "success_rate": stats["success_count"] / stats["count"],
                "avg_execution_time": stats["total_time"] / stats["count"],
                "error_count": stats["count"] - stats["success_count"],
                "unique_errors": len(set(stats["errors"]))
            }
        
        return summary
    
    def get_popular_tools(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most popular tools"""
        tool_counts = defaultdict(int)
        
        for event in self.events:
            tool_counts[event.tool_name] += 1
        
        sorted_tools = sorted(
            tool_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        return [
            {"tool": tool, "usage_count": count}
            for tool, count in sorted_tools
        ]
    
    def get_performance_metrics(self, tool_name: str) -> Dict[str, Any]:
        """Get performance metrics for specific tool"""
        tool_events = [e for e in self.events if e.tool_name == tool_name]
        
        if not tool_events:
            return {"message": f"No events for tool {tool_name}"}
        
        execution_times = [e.execution_time for e in tool_events if e.success]
        
        if not execution_times:
            return {"message": "No successful executions"}
        
        execution_times.sort()
        
        return {
            "tool": tool_name,
            "total_executions": len(tool_events),
            "successful_executions": len(execution_times),
            "performance": {
                "min": execution_times[0],
                "max": execution_times[-1],
                "avg": sum(execution_times) / len(execution_times),
                "p50": execution_times[len(execution_times) // 2],
                "p95": execution_times[int(len(execution_times) * 0.95)],
                "p99": execution_times[int(len(execution_times) * 0.99)]
            }
        }
    
    def _load_events(self) -> None:
        """Load events from storage"""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.events = [
                    ToolUsageEvent(
                        tool_name=e["tool_name"],
                        timestamp=datetime.fromisoformat(e["timestamp"]),
                        success=e["success"],
                        execution_time=e["execution_time"],
                        user_id=e.get("user_id"),
                        error=e.get("error"),
                        input_size=e.get("input_size", 0),
                        output_size=e.get("output_size", 0)
                    )
                    for e in data
                ]
        except Exception as e:
            logger.error(f"Failed to load events: {e}")
            self.events = []
    
    def _save_events(self) -> None:
        """Save events to storage"""
        try:
            data = [
                {
                    "tool_name": e.tool_name,
                    "timestamp": e.timestamp.isoformat(),
                    "success": e.success,
                    "execution_time": e.execution_time,
                    "user_id": e.user_id,
                    "error": e.error,
                    "input_size": e.input_size,
                    "output_size": e.output_size
                }
                for e in self.events
            ]
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save events: {e}")
```

### Task 10: Update Learning Journal (10 min)

Update your CLAUDE.md:

```markdown
## Day 3: Tool Interface Design

### What I Built
- ✅ Morning: Advanced tool architecture with Pydantic validation
- ✅ Afternoon: Tool registry, composition, and security
- ✅ Evening: Documentation generator and analytics

### Key Learnings
1. **Technical**: 
   - Pydantic provides excellent input/output validation
   - Tool discovery via introspection is powerful
   - Security must be built-in from the start

2. **Architecture**:
   - Registry pattern for tool management
   - Composition enables complex workflows
   - Proper abstraction makes tools pluggable

3. **Performance**:
   - Parallel tool execution improves throughput
   - Caching tool schemas reduces overhead
   - Analytics help identify bottlenecks

### Challenges Faced
- **Issue**: Dynamic tool discovery complexity
  **Solution**: Used Python introspection and importlib
  **Lesson**: Metaclasses and inspection are powerful

- **Issue**: Security concerns with code execution
  **Solution**: Permission system and sandboxing
  **Lesson**: Never trust user input, always sandbox

### Code Metrics
- Lines written: ~1500
- Tests added: 20
- Tools implemented: 4
- Documentation pages: 10

### Tomorrow's Goal
- [ ] Build robust parsing system
- [ ] Implement conversation memory
- [ ] Add semantic tool selection
```

## 📊 Deliverables Checklist
- [ ] Advanced tool base with validation
- [ ] Tool registry with auto-discovery
- [ ] Security and permission system
- [ ] Tool composition framework
- [ ] Example tool implementations
- [ ] Testing framework for tools
- [ ] Documentation generator
- [ ] Usage analytics

## 🎯 Success Metrics
You've succeeded if you can:
1. Auto-discover and register tools from a package
2. Execute tools with proper validation
3. Compose tools into pipelines
4. Enforce security permissions
5. Generate comprehensive documentation

## 🚀 Extension Challenges
If you finish early:
1. Add tool versioning and compatibility checks
2. Implement tool result caching with TTL
3. Build a visual pipeline designer
4. Add WebAssembly support for sandboxed execution
5. Create a tool marketplace with ratings

---

🎉 **Congratulations!** You've built a sophisticated, production-ready tool system. Tomorrow we'll focus on parsing and state management.