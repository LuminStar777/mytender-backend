import graphviz

def create_chunking_diagram():
    dot = graphviz.Digraph('chunking_process', comment='Document Chunking Process')
    
    # Set graph attributes
    dot.attr(rankdir='TB', size='12,10', ratio='fill', fontname='Arial')
    # Add DPI setting for higher resolution
    dot.attr('graph', dpi='300')
    dot.attr('node', shape='box', style='filled', fillcolor='lightblue', 
             fontname='Arial', fontsize='14', margin='0.3,0.2')
    dot.attr('edge', fontname='Arial', fontsize='12')
    
    # Add nodes for the chunking process
    dot.node('document', 'Original Document', fillcolor='#A8E6CE')
    dot.node('parse', 'Document Parsing', fillcolor='#DCEDC2')
    dot.node('extract', 'Text Extraction', fillcolor='#DCEDC2')
    dot.node('semantic', 'Semantic Analysis', fillcolor='#FFD3B5')
    dot.node('split', 'Text Splitting\n(MarkdownSplitter)', fillcolor='#FFD3B5')
    
    # Add chunk nodes
    dot.node('chunk1', 'Chunk 1\nSize: 200-2000 tokens', fillcolor='#FFAAA6')
    dot.node('chunk2', 'Chunk 2\nSize: 200-2000 tokens', fillcolor='#FFAAA6')
    dot.node('chunk3', 'Chunk 3\nSize: 200-2000 tokens', fillcolor='#FFAAA6')
    
    # Add vector database nodes
    dot.node('embed', 'Generate Embeddings', fillcolor='#A8C0E6')
    dot.node('store', 'Store in ChromaDB', fillcolor='#A8C0E6')
    dot.node('vector_db', 'Vector Database', shape='cylinder', fillcolor='#D3B5FF')
    
    # Connect the nodes
    dot.edge('document', 'parse')
    dot.edge('parse', 'extract')
    dot.edge('extract', 'semantic')
    dot.edge('semantic', 'split')
    
    # Connect split to chunks
    dot.edge('split', 'chunk1')
    dot.edge('split', 'chunk2')
    dot.edge('split', 'chunk3')
    
    # Connect chunks to embedding
    dot.edge('chunk1', 'embed')
    dot.edge('chunk2', 'embed')
    dot.edge('chunk3', 'embed')
    
    # Connect embedding to storage
    dot.edge('embed', 'store')
    dot.edge('store', 'vector_db')
    
    # Add a note about metadata
    dot.node('metadata', 'Metadata\n- Document ID\n- Chunk Index\n- Upload Date\n- Creator', 
             shape='note', fillcolor='#F7EEC1')
    dot.edge('metadata', 'store', style='dashed')
    
    # Save the diagram
    dot.format = 'png'
    dot.render('chunking_diagram', cleanup=True)
    return dot

if __name__ == "__main__":
    diagram = create_chunking_diagram()
    print("Chunking diagram created at chunking_diagram.png") 