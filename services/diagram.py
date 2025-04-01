import base64

import aiohttp
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

from config import llm_diagram
from services.helper import load_prompt


async def transform_text_to_mermaid(input_text: str) -> str:
    # Use the load_prompt function to load the diagram prompt
    prompt_template = load_prompt("diagram")

    # Create a ChatPromptTemplate
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Create the chain using the new pattern
    chain = prompt | llm_diagram | StrOutputParser()

    # Run the chain
    diagram_description = await chain.ainvoke({
        "input_text": input_text
    })

    # Extract the Mermaid diagram code from the LLM response
    mermaid_code = diagram_description.strip().split("```mermaid")[1].split("```")[0].strip()
    return mermaid_code


async def generate_mermaid_diagram(diagram_text: str) -> str:
    try:
        # Convert diagram text to base64
        graphbytes = diagram_text.encode("utf8")
        base64_bytes = base64.urlsafe_b64encode(graphbytes)
        base64_string = base64_bytes.decode("ascii")

        # Generate URL and fetch the image
        url = f"https://mermaid.ink/img/{base64_string}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise RuntimeError(f"Failed to fetch diagram: HTTP {response.status}")
                image_data = await response.read()

        # Convert to base64 for returning
        base64_image = base64.b64encode(image_data).decode('utf-8')
        return f"data:image/png;base64,{base64_image}"

    except Exception as e:
        raise ValueError(f"Failed to generate diagram: {str(e)}")
