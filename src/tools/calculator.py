import ast
import operator
from typing import Any
import structlog

from src.tools.base import Tool, ToolResult

logger = structlog.get_logger(__name__)

class CalculatorTool(Tool):
    """Safe calculator tool using AST parsing"""
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Performs mathematical calculations. Supports +, -, *, /, and ** (power) operations. Input should be a mathematical expression like '2 + 2' or '10 * 5 / 2'."
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