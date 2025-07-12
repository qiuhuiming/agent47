import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import os

# Set dummy API key before importing modules that need it
os.environ["OPENAI_API_KEY"] = "test-key-for-testing"

import structlog

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import Agent, AgentState
from src.tools.base import Tool, ToolResult
from src.llm_client import LLMClient
from src.config import config

from src.logging_config import setup_logging

setup_logging(config.log_level, config.log_file)
logger = structlog.get_logger()


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
        "Final Answer: The result is 4",
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
        "Final Answer: An error occurred",
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
