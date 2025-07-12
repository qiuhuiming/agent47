import sys
from pathlib import Path
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.logging_config import setup_logging
from src.agent import Agent
from src.llm_client import create_llm_client
from src.tools.base import Tool
from src.tools.calculator import CalculatorTool
import structlog

# Setup logging
setup_logging(config.log_level, config.log_file)
logger = structlog.get_logger(__name__)


def main():
    """Run the agent"""
    logger.info("Starting Agent47")

    # Create LLM client
    api_key = (
        config.openai_api_key
        if config.llm_provider == "openai"
        else config.anthropic_api_key
    )
    assert api_key is not None, f"API key not found for provider: {config.llm_provider}"
    llm = create_llm_client(provider=config.llm_provider, api_key=api_key)

    # Create tools
    tools: List[Tool] = [CalculatorTool()]

    # Create agent
    agent = Agent(llm_client=llm, tools=tools, max_iterations=config.max_iterations)

    # Test with a simple task
    task = "What is 25 * 4 + 10?"
    logger.info(f"Running test task: {task}")

    result = agent.run(task)
    print(f"\nTask: {task}")
    print(f"Result: {result}")

    # Interactive mode
    print("\n" + "=" * 50)
    print("Agent47 Interactive Mode")
    print("Type 'exit' to quit")
    print("=" * 50 + "\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break

        result = agent.run(user_input)
        print(f"Agent: {result}\n")


if __name__ == "__main__":
    main()
