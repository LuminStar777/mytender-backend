import pytest
from config import load_user_config
from services.chain import invoke_graph, perplexity_question

# pylint: disable=too-many-positional-arguments

pytestmark = pytest.mark.asyncio

@pytest.fixture(scope="module")
async def user_config():
    config = await load_user_config("adminuser")
    yield config

@pytest.mark.parametrize("choice, input_text, extra_instructions, username, datasets, broadness, expected_word", [
    ("2",
     "Resource Management - details of directly employed staff and levels of expertise, demonstrate the ability to cope with volume of works?",
     "", "adminuser", ["test_suite2"], "1", "staff"),
    # ("2",
    #  "Resource Management - details of directly employed staff and levels of expertise, demonstrate the ability to cope with volume of works?",
    #  "Add 'NOTE:' before each paragraph", "adminuser", ["test_suite2"], "1", "note"),
    ("4translate to french",
     "Hello, how are you? I am doing well.",
     "", "adminuser", ["test_suite2"], "1", "bonjour"),
])
async def test_invoke_graph(choice, input_text, extra_instructions, username, datasets, broadness, expected_word):
    result = await invoke_graph(choice, input_text, extra_instructions, username, datasets, broadness)
    assert isinstance(result, str), "Result should be a string"
    assert len(result) > 0, "Result should not be empty"
    if expected_word:
        assert expected_word in result.lower(), f"Result should contain the word '{expected_word}'"
    if extra_instructions == "Add 'NOTE:' before each paragraph":
        assert "note:" in result.lower(), "Result should contain 'NOTE:' (case insensitive)"
    print(result)

async def test_perplexity():
    input_text = "how do you do? What's the news? where do you come from?"
    username = 'adminuser'
    datasets = ['default']
    result = await perplexity_question(input_text, username, datasets)
    assert isinstance(result, str), "Result should be a string"
    assert len(result) > 0, "Result should not be empty"
    print(result)  # Keep the print for debugging purposes

async def test_invoke_graph_error_handling():
    with pytest.raises(Exception):
        # Intentionally pass invalid parameters to test error handling
        await invoke_graph("invalid_choice", "", "", "", "", "")
