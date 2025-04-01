# 3b Prompt Documentation

## Overview

The 3b prompt is a specialized prompt template used for generating winning tender responses. It's designed to create professional, targeted content that addresses specific client requirements, pain points, and evaluation criteria.

## Prompt Variants

The 3b prompt has several specialized variants:
- `3b.txt` - Generic version for general responses
- `3b_case_study.txt` - For questions about past performance, track record, and examples
- `3b_commercial.txt` - For general business, strategy, and commercial questions
- `3b_personnel.txt` - For team structure, staffing, and workforce management
- `3b_technical.txt` - For technical processes, methodology, and operational questions
- `3b_compliance.txt` - For compliance-related questions

## Variables Flowing into the 3b Prompt

The following variables are populated in the prompt template:

| Variable | Description | Source |
|----------|-------------|--------|
| `evaluation_criteria` | Win themes for the bid | From the `evaluation_criteria` parameter |
| `derived_insights` | Customer pain points | From the `derived_insights` parameter |
| `differentiation_factors` | Ways to stand out from competition | From the `differentation_factors` parameter |
| `company_name` | Name of the company | From user configuration |
| `business_usp` | Business's unique selling points | From user configuration (`company_objectives`) |
| `writingplan` | Plan for structuring the response | From the `writingplans` parameter |
| `sub_topic` | The specific sub-topic to address | From the `selected_choices` parameter |
| `except_sub_topics` | Topics to exclude | Generated from other sub-topics in `selected_choices` |
| `word_amounts` | Word count limit | From the `word_amounts` parameter |
| `context_content_library` | Company information to use as evidence | Retrieved from content library based on sub-topic |
| `context_bid_library` | Bid-specific information | Retrieved from bid library based on sub-topic |
| `question` | The main question being answered | From the original `input_text` |
| `compliance_requirements` | Compliance requirements for the section | From the `compliance_reqs` parameter |
| `case_studies` | Case studies to reference | From the `selected_case_studies` parameter |
| `tone_of_voice` | The writing tone | From bid metadata or defaults to "Professional" |

## Flow of Data

The data flows through the system in this sequence:

1. The user submits input with parameters via the API endpoints
2. The `invoke_graph` function in `chain.py` initializes the state with all variables
3. The workflow processes through several steps:
   - Document retrieval for relevant content
   - Relevance checking
   - Context processing
   - Getting the final instructions
   - Preparing the question
   - Processing the query using `process_multiple_headers`
   - Post-processing the response using `post_process_3b`

4. For each sub-topic, `process_multiple_headers`:
   - Retrieves relevant documents from content and bid libraries
   - Checks the relevance of retrieved documents
   - Formats the documents into contexts
   - Populates all variables into the prompt template
   - Invokes the language model to generate the response
   
5. The final response is post-processed before being returned to the user

## Technical Implementation

The main processing happens in the `process_multiple_headers` function in `services/chain.py`. This function:

1. Retrieves documents relevant to each sub-topic
2. Checks the relevance of these documents to ensure quality
3. Formats the documents into proper context strings
4. Combines all variables into a structured input for the language model
5. Processes each sub-topic in parallel using `asyncio.gather`
6. Combines all sub-topic responses into a final output

The prompt is loaded from the file system using the `load_prompt_from_file` function, which selects the appropriate prompt variant based on the chosen template. 