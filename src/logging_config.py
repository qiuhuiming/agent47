import logging
import sys
from pathlib import Path
import structlog

def setup_logging(log_level: str = "INFO", log_file: str = "logs/agent.log"):
    """Configure simple, clear logging"""
    
    # Create logs directory
    Path(log_file).parent.mkdir(exist_ok=True)
    
    # Simple format: time filename:line level message
    log_format = "%(asctime)s %(filename)s:%(lineno)d %(levelname)s %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Configure logging
    logging.basicConfig(
        format=log_format,
        datefmt=date_format,
        stream=sys.stdout,
        level=getattr(logging, log_level),
    )
    
    # Add file handler with same format
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, log_level))
    file_formatter = logging.Formatter(log_format, date_format)
    file_handler.setFormatter(file_formatter)
    logging.getLogger().addHandler(file_handler)
    
    # Configure structlog to use standard logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt=date_format),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )