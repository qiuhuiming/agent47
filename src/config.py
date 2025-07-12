import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv
import logging
import structlog

# Load environment variables
load_dotenv()

logger = structlog.get_logger(__name__)

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
            openai_api_key=os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_TOKEN"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            llm_provider=os.getenv("LLM_PROVIDER", "openai"),
            max_iterations=int(os.getenv("MAX_ITERATIONS", "10")),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=os.getenv("LOG_FILE", "logs/agent.log")
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