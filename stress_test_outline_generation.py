import asyncio
import logging
from typing import List, Dict
from api_modules.proposal_outline import get_outline
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

async def test_concurrent_limit(num_concurrent: int, sample_docs: List[Dict[str, str]]):
    """
    Test how many concurrent get_outline calls we can make before hitting the warning.
    
    Args:
        num_concurrent: Number of concurrent calls to test
        sample_docs: Sample documents to use for testing
    Returns:
        bool: True if warning was received
    """
    warning_received = False

    def warning_handler(msg, *args, **kwargs):
        nonlocal warning_received
        if "Error using llm_outline after 3 retries" in str(msg):
            warning_received = True

    # Replace warning handler
    original_warning = logging.root.warning
    logging.root.warning = warning_handler

    try:
        tasks = [get_outline(sample_docs) for _ in range(num_concurrent)]
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        # Restore original warning handler
        logging.root.warning = original_warning

    return warning_received

async def find_concurrent_limit():
    """Find the number of concurrent calls that triggers the warning."""
    sample_docs = [{"text": "Sample document content"}]

    for num_concurrent in range(1, 5):
        print(f"Testing {num_concurrent} concurrent calls...")
        warning_hit = await test_concurrent_limit(num_concurrent, sample_docs)

        if warning_hit:
            print(f"\nWarning received at {num_concurrent} concurrent calls")
            return num_concurrent

    print("\nNo warning received up to 4 concurrent calls")
    return None

if __name__ == "__main__":
    asyncio.run(find_concurrent_limit())
