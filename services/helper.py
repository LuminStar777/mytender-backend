import datetime
import json
import logging
import os
import sys
from logging import handlers
from time import time

import aiohttp
from pythonjsonlogger import jsonlogger  # pylint: disable=no-name-in-module
from langchain_core.rate_limiters import InMemoryRateLimiter

log = logging.getLogger(__name__)


def init_logger(screenlevel, filename=None, logdir=None, modulename=""):
    """Initialize Logger."""
    if not logdir:
        logdir = get_dir("log")

    # Clear existing handlers
    root = logging.getLogger()
    root.handlers = []

    # Set root logger level
    root.setLevel(screenlevel)

    # Formatter for console handler (keep original format)
    console_formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')

    # JSON formatter for file handler using python-json-logger
    json_formatter = jsonlogger.JsonFormatter(
        fmt='%(asctime)s %(levelname)s %(filename)s %(lineno)d %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(screenlevel)
    console_handler.setFormatter(console_formatter)
    root.addHandler(console_handler)

    # File handler if filename is provided
    if filename and filename != "None":
        filename = filename.replace("{date}", datetime.date.today().strftime("%Y%m%d"))

        # Main log file
        file_handler = handlers.RotatingFileHandler(
            os.path.join(logdir, f"{filename}.log"), maxBytes=300000, backupCount=20
        )
        file_handler.setLevel(screenlevel)
        file_handler.setFormatter(json_formatter)
        root.addHandler(file_handler)

    # Disable uvicorn access logger
    logging.getLogger("uvicorn.access").disabled = True

    # Set other loggers to use our configuration
    for name in ["uvicorn", "uvicorn.error", "fastapi"]:
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True

    return root


def get_dir(*paths):
    """
    Retrieve path (for subpath use multiple arguments).

    1. path from config file under Files section (relative to staging  folder), or
    2. 'codebase' for codebase base directory, or
    3. if neither of the above, custom directory relative to codebase

    """
    codebase = os.path.abspath(os.path.dirname(os.path.dirname(os.path.relpath(__file__))))
    if paths[0] == "codebase":  # pylint: disable=no-else-return
        return codebase
    else:
        return os.path.abspath(os.path.join(codebase, *paths))  # if path has multiple entries


def load_prompt(choice: str) -> str:
    """
    Load a prompt file based on the given choice.

    Args:
        choice (str): The name of the prompt file (without .txt extension)

    Returns:
        str: The content of the prompt file

    Raises:
        ValueError: If no valid prompt is found for the given choice
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_dir = os.path.join(current_dir, "prompts")
    filename = f"{choice}.txt"
    file_path = os.path.join(prompt_dir, filename)

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
            if content:
                return content
            else:
                log.warning(f"Prompt file is empty: {file_path}")
                return ""
    except FileNotFoundError:
        log.warning(f"Prompt file not found: {file_path}")
    except Exception as e:
        log.error(f"Error reading prompt file {file_path}: {str(e)}")

    raise ValueError(f"No valid prompt found for choice '{choice}'")


def load_query(query: str) -> str:
    """
    Load a prompt file based on the given choice.

    Args:
        choice (str): The name of the prompt file (without .txt extension)

    Returns:
        str: The content of the prompt file

    Raises:
        ValueError: If no valid prompt is found for the given choice
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_dir = os.path.join(current_dir, "chunking_search_queries_txts")
    filename = f"{query}.txt"
    file_path = os.path.join(prompt_dir, filename)

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
            if content:
                return content
            else:
                log.warning(f"Prompt file is empty: {file_path}")
                return ""
    except FileNotFoundError:
        log.warning(f"Prompt file not found: {file_path}")
    except Exception as e:
        log.error(f"Error reading prompt file {file_path}: {str(e)}")

    raise ValueError(f"No valid prompt found for choice '{query}'")


async def post_to_slack(message):
    # Create the payload with the passed message
    payload = json.dumps({"text": message})
    webhook_url = (
        "https://hooks.slack.com/services/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"
    )

    async with aiohttp.ClientSession() as session:
        async with session.post(
            webhook_url, data=payload, headers={"Content-Type": "application/json"}
        ) as response:
            if response.status != 200:
                log.error(f"Request to slack returned an error {response.status}")


class LoggingRateLimiter(InMemoryRateLimiter):
    """Rate limiter that logs when throttling occurs"""

    def __init__(self, name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.last_log = 0
        self.log_interval = 1  # Only log once per second for the same limiter

    def acquire(self, *, blocking: bool = True) -> bool:
        start_time = time()
        result = super().acquire(blocking=blocking)
        wait_time = time() - start_time

        # Only log if we actually had to wait and haven't logged recently
        if wait_time > 0.01 and (time() - self.last_log) > self.log_interval:
            log.warning(
                f"Rate limit applied for {self.name}: waited {wait_time:.2f}s "
                f"(current rate: {self.requests_per_second}/s, "
                f"bucket size: {self.max_bucket_size})"
            )
            self.last_log = time()

        return result
