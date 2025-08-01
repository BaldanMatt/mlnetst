import logging
from typing import Optional

# Define color constants
COLORS_FOR_LOGGER = {
    'DEBUG': '\033[36m',      # Cyan
    'INFO': '\033[32m',       # Green
    'WARNING': '\033[33m',    # Yellow
    'ERROR': '\033[31m',      # Red
    'CRITICAL': '\033[35m',   # Magenta
    'RESET': '\033[0m'        # Reset
}


def get_colored_logger(
    name: str, 
    level: Optional[int] = None, 
    mode: Optional[str] = None, 
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Create a logger with colored console output and optional file output.
    
    Args:
        name: Name of the logger (will be prefixed in log messages)
        level: Logging level as integer (e.g., logging.INFO, logging.DEBUG)
        mode: Logging level mode ('debug', 'info', 'warning', 'error', 'critical')
              Takes precedence over level parameter if both are provided
        log_file: Optional file path to write logs to (without colors)
        
    Returns:
        Configured logger instance with no propagation to parent loggers
        
    Examples:
        >>> logger = get_colored_logger("MAIN", level=logging.INFO)
        >>> logger = get_colored_logger("NETWORK", mode="debug")
        >>> logger = get_colored_logger("DATA", mode="info", log_file="app.log")
    """
    # Get or create logger
    logger = logging.getLogger(name)
    
    # Prevent propagation to parent loggers to avoid duplicate messages
    logger.propagate = False
    
    # Clear any existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Determine logging level with improved logic
    log_level = _determine_log_level(level, mode)
    logger.setLevel(log_level)
    
    # Create and add console handler
    console_handler = _create_console_handler(log_level)
    logger.addHandler(console_handler)
    
    # Create and add file handler if requested
    if log_file:
        file_handler = _create_file_handler(log_file, log_level)
        logger.addHandler(file_handler)
    
    return logger


def _determine_log_level(level: Optional[int], mode: Optional[str]) -> int:
    """
    Determine the appropriate logging level from level integer or mode string.
    
    Args:
        level: Integer logging level
        mode: String mode descriptor
        
    Returns:
        Integer logging level
    """
    # Mode takes precedence over level if both are provided
    if mode is not None:
        mode_lower = mode.lower()
        level_mapping = {
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'warn': logging.WARNING,  # Common abbreviation
            'error': logging.ERROR,
            'critical': logging.CRITICAL,
            'fatal': logging.CRITICAL  # Common alias
        }
        
        if mode_lower in level_mapping:
            return level_mapping[mode_lower]
        else:
            print(f"⚠️  Unknown logging mode: '{mode}'. Available modes: {list(level_mapping.keys())}. Defaulting to INFO.")
            return logging.INFO
    
    # Use provided level or default to INFO
    if level is not None:
        return level
    
    return logging.INFO


def _create_console_handler(log_level: int) -> logging.StreamHandler:
    """
    Create a console handler with colored formatting.
    
    Args:
        log_level: Logging level for the handler
        
    Returns:
        Configured console handler
    """
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(ColoredFormatter())
    return console_handler


def _create_file_handler(log_file: str, log_level: int) -> logging.FileHandler:
    """
    Create a file handler with plain text formatting (no colors).
    
    Args:
        log_file: Path to the log file
        log_level: Logging level for the handler
        
    Returns:
        Configured file handler
    """
    try:
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(log_level)
        
        # Use plain formatter for file output (no colors)
        file_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        file_formatter = logging.Formatter(file_format, datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        
        return file_handler
    except (OSError, IOError) as e:
        print(f"⚠️  Could not create file handler for '{log_file}': {e}")
        raise


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages based on level."""
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record with appropriate colors.
        
        Args:
            record: LogRecord instance to format
            
        Returns:
            Formatted string with color codes
        """
        # Get color for this log level, default to no color
        color = COLORS_FOR_LOGGER.get(record.levelname, '')
        reset = COLORS_FOR_LOGGER['RESET']
        
        # Create colored format string
        # Note: %(name)s will contain the logger name as requested
        colored_format = (
            f"{color}%(asctime)s - %(name)s - %(levelname)s - %(message)s{reset}"
        )
        
        # Create formatter with the colored format
        formatter = logging.Formatter(colored_format, datefmt="%Y-%m-%d %H:%M:%S")
        
        return formatter.format(record)


# Convenience functions for common use cases
def get_main_logger(verbose: bool = False, log_file: Optional[str] = None) -> logging.Logger:
    """
    Get a main application logger.
    
    Args:
        verbose: If True, set to DEBUG level, otherwise INFO
        log_file: Optional log file path
        
    Returns:
        Configured main logger
    """
    level = logging.DEBUG if verbose else logging.INFO
    return get_colored_logger("MAIN", level=level, log_file=log_file)


def get_module_logger(module_name: str, debug: bool = True, log_file: Optional[str] = None) -> logging.Logger:
    """
    Get a module-specific logger, typically set to DEBUG level.
    
    Args:
        module_name: Name of the module/component
        debug: If True, set to DEBUG level, otherwise INFO
        log_file: Optional log file path
        
    Returns:
        Configured module logger
    """
    level = logging.DEBUG if debug else logging.INFO
    return get_colored_logger(module_name.upper(), level=level, log_file=log_file)
