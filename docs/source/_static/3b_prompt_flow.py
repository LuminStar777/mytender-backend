import graphviz

def create_3b_prompt_flow_diagram():
    dot = graphviz.Digraph('3b_prompt_flow', comment='Data flow into 3b prompt')

    # Set graph attributes with higher resolution settings
    dot.attr(rankdir='LR', size='16,12', ratio='fill', fontname='Arial', nodesep='0.5')
    # Add DPI setting for higher resolution
    dot.attr('graph', dpi='300')
    dot.attr('node', shape='box', style='filled', fillcolor='lightblue',
             fontname='Arial', fontsize='14', margin='0.3,0.2')
    dot.attr('edge', fontname='Arial', fontsize='12')

    # Add input sources
    dot.node('user_input', 'User API Input', fillcolor='lightgreen')
    dot.node('user_config', 'User Configuration', fillcolor='lightgreen')
    dot.node('bid_metadata', 'Bid Metadata', fillcolor='lightgreen')
    dot.node('content_library', 'Content Library', fillcolor='lightgreen')
    dot.node('bid_library', 'Bid Library', fillcolor='lightgreen')

    # Add processing nodes
    dot.node('invoke_graph', 'invoke_graph()', fillcolor='lightyellow')
    dot.node('retrieve_docs', 'retrieve_documents()', fillcolor='lightyellow')
    dot.node('process_multi', 'process_multiple_headers()', fillcolor='lightyellow')
    dot.node('post_process', 'post_process_3b()', fillcolor='lightyellow')

    # Add variables
    dot.node('var_selected_choices', 'selected_choices', shape='ellipse', fillcolor='white')
    dot.node('var_word_amounts', 'word_amounts', shape='ellipse', fillcolor='white')
    dot.node('var_compliance', 'compliance_reqs', shape='ellipse', fillcolor='white')
    dot.node('var_eval_criteria', 'evaluation_criteria', shape='ellipse', fillcolor='white')
    dot.node('var_insights', 'derived_insights', shape='ellipse', fillcolor='white')
    dot.node('var_diff_factors', 'differentation_factors', shape='ellipse', fillcolor='white')
    dot.node('var_writingplans', 'writingplans', shape='ellipse', fillcolor='white')
    dot.node('var_case_studies', 'selected_case_studies', shape='ellipse', fillcolor='white')
    dot.node('var_company_name', 'company_name', shape='ellipse', fillcolor='white')
    dot.node('var_business_usp', 'business_usp', shape='ellipse', fillcolor='white')
    dot.node('var_tone', 'tone_of_voice', shape='ellipse', fillcolor='white')
    dot.node('var_question', 'question', shape='ellipse', fillcolor='white')
    dot.node('var_retrieved_content', 'context_content_library', shape='ellipse', fillcolor='white')
    dot.node('var_retrieved_bid', 'context_bid_library', shape='ellipse', fillcolor='white')

    # Add the prompt template
    dot.node('3b_prompt', '3b Prompt Template', shape='note', fillcolor='lightcoral')

    # Add final result
    dot.node('final_result', 'Final Response', fillcolor='lightgreen')

    # Connect inputs to processing
    dot.edge('user_input', 'invoke_graph')
    dot.edge('user_config', 'var_company_name')
    dot.edge('user_config', 'var_business_usp', label='company_objectives')
    dot.edge('bid_metadata', 'var_tone')

    # Connect processing steps
    dot.edge('invoke_graph', 'retrieve_docs')
    dot.edge('retrieve_docs', 'process_multi')
    dot.edge('process_multi', 'post_process')
    dot.edge('post_process', 'final_result')

    # Connect input variables from user input to invoke_graph
    dot.edge('user_input', 'var_selected_choices', label='selected_choices')
    dot.edge('user_input', 'var_word_amounts', label='word_amounts')
    dot.edge('user_input', 'var_compliance', label='compliance_reqs')
    dot.edge('user_input', 'var_eval_criteria', label='evaluation_criteria')
    dot.edge('user_input', 'var_insights', label='derived_insights')
    dot.edge('user_input', 'var_diff_factors', label='differentation_factors')
    dot.edge('user_input', 'var_writingplans', label='writingplans')
    dot.edge('user_input', 'var_case_studies', label='selected_case_studies')
    dot.edge('user_input', 'var_question', label='input_text')

    # Connect vector stores to variable retrieval
    dot.edge('content_library', 'var_retrieved_content')
    dot.edge('bid_library', 'var_retrieved_bid')

    # Connect all variables to the prompt template
    dot.edge('var_selected_choices', '3b_prompt', label='sub_topic')
    dot.edge('var_word_amounts', '3b_prompt', label='word_amounts')
    dot.edge('var_compliance', '3b_prompt', label='compliance_requirements')
    dot.edge('var_eval_criteria', '3b_prompt', label='evaluation_criteria')
    dot.edge('var_insights', '3b_prompt', label='derived_insights')
    dot.edge('var_diff_factors', '3b_prompt', label='differentiation_factors')
    dot.edge('var_writingplans', '3b_prompt', label='writingplan')
    dot.edge('var_case_studies', '3b_prompt', label='case_studies')
    dot.edge('var_company_name', '3b_prompt', label='company_name')
    dot.edge('var_business_usp', '3b_prompt', label='business_usp')
    dot.edge('var_tone', '3b_prompt', label='tone_of_voice')
    dot.edge('var_question', '3b_prompt', label='question')
    dot.edge('var_retrieved_content', '3b_prompt')
    dot.edge('var_retrieved_bid', '3b_prompt')

    # Connect process_multi to 3b_prompt
    dot.edge('process_multi', '3b_prompt', label='populates')

    # Connect 3b_prompt to post_process
    dot.edge('3b_prompt', 'post_process', label='generates response')

    # Save the diagram directly to the _static directory with higher quality
    dot.format = 'png'
    dot.render('3b_prompt_flow', cleanup=True)
    return dot

if __name__ == "__main__":
    diagram = create_3b_prompt_flow_diagram()
    print("High-resolution diagram created at 3b_prompt_flow.png")
