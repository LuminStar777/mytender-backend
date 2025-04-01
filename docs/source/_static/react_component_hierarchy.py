import graphviz

def create_react_component_hierarchy():
    dot = graphviz.Digraph('react_component_hierarchy', comment='React Component Hierarchy')
    
    # Set graph attributes
    dot.attr(rankdir='TB', size='14,12', ratio='fill', fontname='Arial')
    # Add DPI setting for higher resolution
    dot.attr('graph', dpi='300')
    dot.attr('node', shape='box', style='filled', fillcolor='lightblue', 
             fontname='Arial', fontsize='14', margin='0.3,0.2')
    dot.attr('edge', fontname='Arial', fontsize='12')
    
    # Create clusters for organization
    with dot.subgraph(name='cluster_providers') as c:
        c.attr(label='Global Providers', style='filled', fillcolor='#E8F4F8')
        c.node('app', 'App', fillcolor='#B3E5FC')
        c.node('auth_provider', 'AuthProvider', fillcolor='#81D4FA')
        c.node('app_content', 'AppContent', fillcolor='#81D4FA')
        c.node('status_provider', 'StatusLabelsProvider', fillcolor='#4FC3F7')
        c.node('update_checker', 'UpdateChecker', fillcolor='#4FC3F7')
        c.node('router', 'BrowserRouter', fillcolor='#4FC3F7')
        c.node('toast', 'ToastContainer', fillcolor='#81D4FA')
    
    # Create cluster for routes
    with dot.subgraph(name='cluster_routes') as c:
        c.attr(label='Routes', style='filled', fillcolor='#F1F8E9')
        c.node('routing', 'Routing', fillcolor='#C5E1A5')
        c.node('protected_route', 'ProtectedRoute', fillcolor='#AED581')
        c.node('public_route', 'PublicRoute', fillcolor='#AED581')
    
    # Create cluster for main views
    with dot.subgraph(name='cluster_views') as c:
        c.attr(label='Main Views', style='filled', fillcolor='#FFF8E1')
        c.node('dashboard', 'Dashboard', fillcolor='#FFE082')
        c.node('chat', 'Chat', fillcolor='#FFE082')
        c.node('login', 'Login', fillcolor='#FFE082')
        c.node('signup', 'Signup', fillcolor='#FFE082')
        c.node('bids', 'Bids', fillcolor='#FFE082')
        c.node('library', 'Library', fillcolor='#FFE082')
    
    # Create cluster for common components
    with dot.subgraph(name='cluster_components') as c:
        c.attr(label='Common Components', style='filled', fillcolor='#E1F5FE')
        c.node('header', 'Header', fillcolor='#81D4FA')
        c.node('sidebar', 'Sidebar', fillcolor='#81D4FA')
        c.node('footer', 'Footer', fillcolor='#81D4FA')
        c.node('modal', 'Modal', fillcolor='#81D4FA')
        c.node('button', 'Button', fillcolor='#81D4FA')
        c.node('input', 'Input', fillcolor='#81D4FA')
    
    # Connect the components
    dot.edge('app', 'auth_provider')
    dot.edge('auth_provider', 'app_content')
    dot.edge('auth_provider', 'toast')
    dot.edge('app_content', 'router')
    dot.edge('app_content', 'status_provider')
    dot.edge('app_content', 'update_checker')
    dot.edge('router', 'routing')
    dot.edge('routing', 'protected_route')
    dot.edge('routing', 'public_route')
    dot.edge('protected_route', 'dashboard')
    dot.edge('protected_route', 'chat')
    dot.edge('protected_route', 'bids')
    dot.edge('protected_route', 'library')
    dot.edge('public_route', 'login')
    dot.edge('public_route', 'signup')
    
    # Connect views to common components
    dot.edge('dashboard', 'header')
    dot.edge('dashboard', 'sidebar')
    dot.edge('dashboard', 'footer')
    dot.edge('bids', 'modal', style='dashed')
    dot.edge('library', 'button', style='dashed')
    dot.edge('login', 'input', style='dashed')
    
    # Save the diagram
    dot.format = 'png'
    dot.render('react_component_hierarchy', cleanup=True)
    return dot

if __name__ == "__main__":
    diagram = create_react_component_hierarchy()
    print("React component hierarchy diagram created at react_component_hierarchy.png") 