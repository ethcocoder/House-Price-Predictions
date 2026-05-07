# Logging Utility

`src/utils/logger.py`

A standardized logging utility used throughout the project to ensure consistent trace and error reporting.

## Component: `setup_logger` Function

### Definition
```python
def setup_logger(name: str, log_file: str = "logs/app.log", level=logging.INFO):
    """
    Creates a logger that outputs to both the console (STDOUT) and a file.
    Automatically creates the 'logs/' directory if it doesn't exist.
    """
```

### Features
- **Dual Output**: Simultaneous logging to file and console.
- **Formatting**: Includes timestamp, logger name, level, and message.
- **Singleton-like Behavior**: Ensures handlers are only added once per logger name to avoid duplicate messages.

## Usage
```python
from src.utils.logger import setup_logger

# Initialize a module-specific logger
logger = setup_logger("MyModule", log_file="logs/mymodule.log")

logger.info("Starting process...")
logger.error("Something went wrong!")
```
