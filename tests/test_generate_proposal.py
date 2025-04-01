"""
Test module for the tender bid generation functions. i.e opportunity info, compliance requirements, 
exec summary, cover letter
"""

import pytest

from api_modules.generate_proposal import process_generate_proposal

pytestmark = pytest.mark.asyncio


@pytest.fixture
def sample_documents():
    """
    Fixture that provides sample documents for tests, containing realistic extracts from tender documents.

    Returns:
        list: A list of dictionaries containing sample document texts from various sections of tender documents.
    """
    return [
        {
            "text": """
"""
        }
    ]


@pytest.mark.asyncio
async def test_generate_proposal_integrated():
    """
    Test the get_compliance_requirements function with minimal mocking.
    This test uses the actual chain invocation to test the function's behavior.
    """
    try:
        # Call the function with sample documents
        bid_id = '66c08bd6c45a0408088d0211'  # Bracknell City Council bid in admin user
        selected_folders = ['default']
        current_user = 'adminuser'
        result = await process_generate_proposal(
            bid_id, selected_folders, current_user
        )
        # Assertions
        assert isinstance(result, dict)
        assert len(result) > 0
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        if hasattr(e, 'detail'):
            print(f"Error detail: {e.detail}")
        raise
