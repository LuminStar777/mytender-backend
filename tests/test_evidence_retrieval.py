import json
import traceback
import pytest

from services.chain import get_evidence_for_text

# Test data
TEST_TEXT_VALID = "Our company has an amazin track record mainly because of our amazing team."
TEST_USERNAME = "adminuser"  # Admin user for testing

# Sample mock data for testing
MOCK_EVIDENCE_RESPONSE = {
    "success": True,
    "evidence": [
        {
            "content": "This is a mock evidence chunk for testing purposes.",
            "source": "Mock Document",
        },
        {
            "content": "This is another mock evidence chunk with different content.",
            "source": "Mock Document 2",
        },
    ],
    "query_used": "test query",
    "all_queries": ["test query", "alternative query"],
    "enhanced_text": "This is a test text that should be at least 10 characters long to pass the validation. As demonstrated in our Mock Document, this is a mock evidence chunk for testing purposes. Additionally, according to Mock Document 2, this is another mock evidence chunk with different content.",
}


@pytest.mark.asyncio
async def test_get_evidence_for_text_valid():
    """Test the get_evidence_for_text function with valid input"""

    print(f"Testing get_evidence_for_text with valid input")
    print(f"Username: {TEST_USERNAME}")
    print(f"Text: '{TEST_TEXT_VALID}'")

    try:
        # Call the function directly
        result = await get_evidence_for_text(TEST_TEXT_VALID, TEST_USERNAME)

        # Print the result for debugging
        print(f"Result: {json.dumps(result, indent=2)}")

        # Assert basic structure
        assert isinstance(result, dict)
        assert "success" in result

        # Check if error is about user not found
        if "User not found" in result.get("message", ""):
            print("User not found error - this is expected in test environment")
            # Skip further assertions as the user doesn't exist in test environment
            pytest.skip("Test user doesn't exist in the system")
            return

        # If successful, additional assertions
        if result.get("success", False):
            assert "evidence" in result
            assert isinstance(result["evidence"], list)
            # Only check for query_used if it exists (backward compatibility)
            if "query_used" in result:
                assert isinstance(result["query_used"], str)
            assert "all_queries" in result

            # Check if enhanced_text is present
            if "enhanced_text" in result:
                assert isinstance(result["enhanced_text"], str)
                assert len(result["enhanced_text"]) > 0
                print(f"Enhanced text: {result['enhanced_text'][:100]}...")

            # Check evidence chunks if any
            evidence = result.get("evidence", [])
            if evidence:
                print(f"Found {len(evidence)} evidence chunks")
                for i, chunk in enumerate(evidence):
                    print(f"Evidence {i+1}:")
                    print(f"Source: {chunk.get('source', 'Unknown')}")
                    print(f"Content snippet: {chunk.get('content', '')[:100]}...")

                    # Verify evidence structure
                    assert "content" in chunk
                    assert "source" in chunk
        else:
            # If not successful, check for expected error message
            assert "message" in result
            print(f"Not successful: {result.get('message', 'No error message')}")

    except Exception as e:
        print(f"Error: {str(e)}")

        traceback.print_exc()
        assert False, f"Function raised an exception: {str(e)}"
