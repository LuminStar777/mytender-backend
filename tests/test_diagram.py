import pytest

from services.diagram import transform_text_to_mermaid, generate_mermaid_diagram


@pytest.mark.asyncio
async def test_transform_text_to_mermaid():
    input_text = """
    The software development process involves several stages:
    1. Requirements gathering
    2. Design
    3. Implementation
    4. Testing
    5. Deployment
    6. Maintenance
    
    Each stage feeds into the next, but there's often feedback and iteration between stages.
    """

    result = await transform_text_to_mermaid(input_text)

    assert result.strip(), "The result should not be empty"
    assert (
        "graph" in result.lower() or "flowchart" in result.lower()
    ), "The result should contain a graph or flowchart definition"

    key_stages = ["requirement", "design", "implement", "test", "deploy", "maintenance"]
    stages_found = sum(1 for stage in key_stages if stage.lower() in result.lower())
    assert stages_found >= 4, f"At least 4 key stages should be present, found {stages_found}"

    assert (
        "-->" in result or "->" in result
    ), "The diagram should contain connections between elements"

    print(f"Generated Mermaid code:\n{result}")


@pytest.mark.skip(reason="This test is too slow to run on every commit")
@pytest.mark.asyncio
async def test_generate_mermaid_diagram():
    input_text = """
    The water cycle consists of four main stages:
    1. Evaporation: Water turns into vapor due to heat from the sun.
    2. Condensation: Water vapor cools and forms clouds.
    3. Precipitation: Water falls as rain or snow.
    4. Collection: Water is collected in bodies of water or seeps into the ground.
    
    This cycle is continuous and repeats indefinitely.
    """

    # First, generate the Mermaid code
    mermaid_code = await transform_text_to_mermaid(input_text)

    assert mermaid_code.strip(), "The Mermaid code should not be empty"
    assert (
        "graph" in mermaid_code.lower() or "flowchart" in mermaid_code.lower()
    ), "The Mermaid code should contain a graph or flowchart definition"

    key_terms = ["evaporation", "condensation", "precipitation", "collection"]
    terms_found = sum(1 for term in key_terms if term.lower() in mermaid_code.lower())
    assert (
        terms_found >= 3
    ), f"At least 3 key water cycle terms should be present, found {terms_found}"

    print(f"Generated Mermaid code:\n{mermaid_code}")

    # Now, generate the diagram image
    diagram_image = await generate_mermaid_diagram(mermaid_code)

    assert diagram_image.startswith(
        "data:image/png;base64,"
    ), "The diagram image should be a base64 encoded PNG"

    print(f"Diagram image (truncated): {diagram_image[:100]}...")
