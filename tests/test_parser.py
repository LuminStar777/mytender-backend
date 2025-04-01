import os
import pytest
from services.parser import parse_file_content

# Test data directory
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")


@pytest.mark.asyncio
async def test_parse_word():
    # Create test Word file path
    test_word_path = os.path.join(TEST_DATA_DIR, "testdoc.docx")

    # Read the test file content
    with open(test_word_path, "rb") as f:
        file_content = f.read()

    result = await parse_file_content(file_content, ".docx", "test_doc")

    # Assertions
    assert result is not None
    assert "parsed_content" in result
    assert "metadata" in result
    assert result["metadata"]["file_type"] == "word"
    assert isinstance(result["parsed_content"], str)
    assert len(result["parsed_content"]) > 0


@pytest.mark.asyncio
async def test_parse_excel():
    # Create test Doc file path
    test_doc_path = os.path.join(TEST_DATA_DIR, "test.xlsx")

    # Read the test file content
    with open(test_doc_path, "rb") as f:
        file_content = f.read()

    result = await parse_file_content(file_content, ".xlsx", "test_excel")

    # Assertions
    assert result is not None
    assert "parsed_content" in result
    assert "metadata" in result
    assert result["metadata"]["file_type"] == "excel"
    assert isinstance(result["parsed_content"], str)
    assert len(result["parsed_content"]) > 0
