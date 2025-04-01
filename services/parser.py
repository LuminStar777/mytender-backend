import logging
import os
import re
from typing import Dict, Union
import pandas as pd
import pymupdf4llm  # For better markdown conversion

from config import markdown_client

log = logging.getLogger(__name__)


def extract_qa_pairs(text):
    # Compile regular expressions for finding questions and answers
    question_re = re.compile(r'"question":', re.IGNORECASE)
    answer_re = re.compile(r'"answer":', re.IGNORECASE)

    # Find all start positions of questions and answers
    question_positions = [match.start() for match in question_re.finditer(text)]
    answer_positions = [match.start() for match in answer_re.finditer(text)]

    # Initialize the list to hold question-answer pairs
    qa_set = []

    # Iterate over the questions and find corresponding answers
    for i, q_pos in enumerate(question_positions):
        # Extract question text
        if i < len(answer_positions):
            question_text = text[q_pos : answer_positions[i]].replace('"question":', "").strip()

            # Find end position for answer
            answer_end_pos = answer_positions[i + 1] if i + 1 < len(answer_positions) else len(text)

            # Extract answer text
            answer_text = (
                text[answer_positions[i] : answer_end_pos].replace('"answer":', "").strip()
            )

            # Add the QA pair to the list
            qa_set.append({"question": question_text, "answer": answer_text})

    return qa_set


async def parse_file_content(
    file_content: bytes, file_extension: str, random_filename: str
) -> Dict[str, Union[str, int]]:
    """
    Parse different types of files and return their content and metadata.
    """
    # Ensure uploads folder exists
    if not os.path.exists("uploads"):
        os.makedirs("uploads")

    file_path = f"uploads/{random_filename}{file_extension}"

    try:
        # Save file temporarily
        with open(file_path, "wb") as f:
            f.write(file_content)

        if file_extension == '.pdf':
            parsed_content = await parse_pdf(file_path)
            metadata = {'file_type': 'pdf'}

        elif file_extension in ['.doc', '.docx']:
            parsed_content = await parse_word(file_path)
            metadata = {'file_type': 'word'}

        elif file_extension in ['.xlsx', '.xls']:
            parsed_content, sheet_count = parse_excel(file_path)
            metadata = {'file_type': 'excel', 'sheet_count': sheet_count}
        else:
            raise ValueError("Unsupported file type")

        if not parsed_content:
            raise ValueError(f"No content could be extracted from the {file_extension} file")

        return {'parsed_content': parsed_content, 'metadata': metadata}
    except Exception as e:
        print(e)
    finally:
        # Clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)


async def parse_pdf(file_path: str) -> str:
    """
    Parse PDF file content with enhanced markdown formatting including tables.
    Uses PyMuPDF4LLM for better structured output.
    """
    # Convert the document to markdown format
    md_text = pymupdf4llm.to_markdown(file_path)
    return md_text


async def parse_word(file_path: str) -> str:
    """Parse Word document content using MarkItDown."""
    md = markdown_client
    result = md.convert(file_path)
    return result.text_content


def parse_excel(file_path: str) -> tuple[str, int]:
    """Parse Excel file content and return content and sheet count."""
    df = pd.read_excel(file_path, sheet_name=None)  # Read all sheets
    parsed_content = []

    # Process each sheet
    for sheet_name, sheet_data in df.items():
        # Convert sheet to string representation
        sheet_text = f"\nSheet: {sheet_name}\n"

        # Handle column headers
        headers = sheet_data.columns.tolist()
        sheet_text += f"Headers: {', '.join(str(h) for h in headers)}\n\n"

        # Convert DataFrame to string, handling NaN values
        for idx, row in sheet_data.iterrows():
            row_values = [str(val) if not pd.isna(val) else '' for val in row]
            sheet_text += f"Row {idx + 1}: {' | '.join(row_values)}\n"

        parsed_content.append(sheet_text)

    return "\n".join(parsed_content), len(df)
