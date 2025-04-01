import graphviz

def create_state_management_diagram():
    dot = graphviz.Digraph('react_state_management', comment='React State Management Flow')
    
    # Set graph attributes
    dot.attr(rankdir='TB', size='14,12', ratio='fill', fontname='Arial')
    # Add DPI setting for higher resolution
    dot.attr('graph', dpi='300')
    dot.attr('node', shape='box', style='filled', fontname='Arial', fontsize='14', margin='0.3,0.2')
    dot.attr('edge', fontname='Arial', fontsize='12')
    
    # Create clusters for state organization
    with dot.subgraph(name='cluster_global') as c:
        c.attr(label='Global State', style='filled', fillcolor='#E3F2FD')
        c.node('auth_state', 'Authentication State\n(React Auth Kit)', fillcolor='#90CAF9')
        c.node('bid_context', 'Bid Context', fillcolor='#90CAF9')
        c.node('status_context', 'Status Labels Context', fillcolor='#90CAF9')
    
    with dot.subgraph(name='cluster_feature') as c:
        c.attr(label='Feature-Level State', style='filled', fillcolor='#F1F8E9')
        c.node('tab_context', 'Tab Context', fillcolor='#C5E1A5')
        c.node('url_params', 'URL Parameters', fillcolor='#C5E1A5')
        c.node('wizard_context', 'Wizard Context', fillcolor='#C5E1A5')
    
    with dot.subgraph(name='cluster_component') as c:
        c.attr(label='Component-Level State', style='filled', fillcolor='#FFF8E1')
        c.node('usestate', 'useState Hooks', fillcolor='#FFE082')
        c.node('usereducer', 'useReducer Hooks', fillcolor='#FFE082')
        c.node('usememo', 'useMemo/useCallback', fillcolor='#FFE082')
    
    with dot.subgraph(name='cluster_persistence') as c:
        c.attr(label='Persistence Layer', style='filled', fillcolor='#E8EAF6')
        c.node('localstorage', 'LocalStorage', fillcolor='#9FA8DA')
        c.node('api', 'Backend API', fillcolor='#9FA8DA')
    
    with dot.subgraph(name='cluster_components') as c:
        c.attr(label='UI Components', style='filled', fillcolor='#FFEBEE')
        c.node('question_crafter', 'Question Crafter', fillcolor='#FFCDD2')
        c.node('qa_generator', 'Q&A Generator', fillcolor='#FFCDD2')
        c.node('proposal', 'Proposal Editor', fillcolor='#FFCDD2')
        c.node('bid_pilot', 'Bid Pilot', fillcolor='#FFCDD2')
    
    # Connect components to state
    dot.edge('question_crafter', 'bid_context', label='reads/writes')
    dot.edge('qa_generator', 'usestate', label='manages UI')
    dot.edge('proposal', 'bid_context', label='reads/writes')
    dot.edge('proposal', 'tab_context', label='uses tabs')
    dot.edge('bid_pilot', 'usestate', label='manages chat')
    
    # Connect state to persistence
    dot.edge('usestate', 'localstorage', label='saves temp')
    dot.edge('bid_context', 'api', label='saves permanently')
    dot.edge('auth_state', 'localstorage', label='stores token')
    
    # Connect global state to component state
    dot.edge('bid_context', 'usestate', label='initializes')
    dot.edge('bid_context', 'usereducer', label='complex updates')
    
    # Connect feature level state
    dot.edge('tab_context', 'usestate', label='controls')
    dot.edge('url_params', 'bid_context', label='initializes')
    
    # Global state interactions
    dot.edge('auth_state', 'bid_context', label='protects access')
    dot.edge('status_context', 'bid_context', label='provides labels')
    
    # Save the diagram
    dot.format = 'png'
    dot.render('react_state_management', cleanup=True)
    return dot

if __name__ == "__main__":
    diagram = create_state_management_diagram()
    print("React state management diagram created at react_state_management.png") 