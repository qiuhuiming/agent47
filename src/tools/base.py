from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional
import structlog

logger = structlog.get_logger(__name__)

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