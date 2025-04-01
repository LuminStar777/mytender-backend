# pylint: disable=consider-using-from-import,R1735

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Create a larger figure with white background
fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')

# Simpler, more contrasting colors
colors = {
    'ingestion': '#ffccd5',
    'chunking': '#e2e2e2',
    'storage': '#cce8d4',
    'retrieval': '#ccd5ff',
    'context': '#ffdac1',
    'generation': '#e0c1ff',
    'post': '#ffffcc',
}

# Fixed positions with grid layout approach
stages = [
    # Left column
    {'name': '1. Document Ingestion', 'color': colors['ingestion'], 'pos': (1.0, 6.5, 3.0, 0.8)},
    {'name': '2. Chunking & Embedding', 'color': colors['chunking'], 'pos': (1.0, 4.5, 3.0, 0.8)},
    {'name': '3. Vector Storage', 'color': colors['storage'], 'pos': (1.0, 2.5, 3.0, 0.8)},

    # Right column
    {'name': '4. Retrieval', 'color': colors['retrieval'], 'pos': (8.0, 6.5, 3.0, 0.8)},
    {'name': '5. Context Processing', 'color': colors['context'], 'pos': (8.0, 4.5, 3.0, 0.8)},
    {'name': '6. Response Generation', 'color': colors['generation'], 'pos': (8.0, 2.5, 3.0, 0.8)},
    {'name': '7. Post Processing', 'color': colors['post'], 'pos': (8.0, 1.0, 3.0, 0.8)},
]

# Draw stage boxes
for stage in stages:
    x, y, w, h = stage['pos']
    rect = patches.Rectangle(
        (x, y), w, h,
        linewidth=1,
        edgecolor='black',
        facecolor=stage['color'],
        alpha=0.8
    )
    ax.add_patch(rect)
    ax.text(
        x + w/2,
        y + h/2,
        stage['name'],
        ha='center',
        va='center',
        fontweight='bold',
        fontsize=10
    )

# Define step content
steps = {
    '1. Document Ingestion': ['• File Upload', '• Parse Documents', '• Extract Content'],
    '2. Chunking & Embedding': ['• Text Splitting', '• Generate Embeddings', '• Create Metadata'],
    '3. Vector Storage': ['• ChromaDB Storage', '• MongoDB Tracking'],
    '4. Retrieval': ['• Process Query', '• Retrieve Documents', '• Check Relevance', '• Filter Documents'],
    '5. Context Processing': ['• Assemble Context', '• Process Numbers'],
    '6. Response Generation': ['• Create Prompt', '• LLM Processing', '• Generate Response'],
    '7. Post Processing': ['• Post Process Result', '• Format Output']
}

# Select fewer detail boxes
details = {
    'Parse Documents': {
        'text': 'Parser:\n• pymupdf4llm (PDF)\n• MarkItDown (Word)\n• pandas (Excel)',
        'pos': (5.5, 6.0)
    },
    'Generate Embeddings': {
        'text': 'Embedding:\n• Model: embedder\n• Function: text_to_chromadb',
        'pos': (5.5, 3.8)
    },
    'Retrieve Documents': {
        'text': 'Functions:\n• retrieve_docs\n• query_vectorstore\n• Search Type: MMR',
        'pos': (5.5, 7.2)
    },
    'LLM Processing': {
        'text': 'Models:\n• llm_chain_default (main)\n• llm_tender_insights\n• llm_compliance',
        'pos': (5.5, 2.0)
    }
}

# Draw each detail box first (lowest layer)
for detail_name, detail_info in details.items():
    x, y = detail_info['pos']
    text = detail_info['text']
    ax.text(
        x, y, text,
        va='center',
        ha='center',
        fontsize=8,
        bbox=dict(
            facecolor='white',
            alpha=0.95,
            boxstyle='round,pad=0.5',
            edgecolor='lightgray'
        )
    )

# Draw steps for each stage with fixed positions
y_offsets = {
    '1. Document Ingestion': [6.2, 5.9, 5.6],
    '2. Chunking & Embedding': [4.2, 3.9, 3.6],
    '3. Vector Storage': [2.2, 1.9],
    '4. Retrieval': [6.2, 5.9, 5.6, 5.3],
    '5. Context Processing': [4.2, 3.9],
    '6. Response Generation': [2.2, 1.9, 1.6],
    '7. Post Processing': [0.7, 0.4]
}

# Fixed x positions for left and right columns
x_positions = {
    '1. Document Ingestion': 1.2,
    '2. Chunking & Embedding': 1.2,
    '3. Vector Storage': 1.2,
    '4. Retrieval': 8.2,
    '5. Context Processing': 8.2,
    '6. Response Generation': 8.2,
    '7. Post Processing': 8.2
}

# Draw all steps
for stage_name, stage_steps in steps.items():
    for i, step in enumerate(stage_steps):
        # Get the y position from our fixed position map
        y_pos = y_offsets[stage_name][i]
        x_pos = x_positions[stage_name]

        # Draw the step text
        ax.text(
            x_pos, y_pos,
            step,
            va='center',
            fontsize=9
        )

        # Connect to detail box if needed
        step_name = step[2:] if step.startswith('•') else step
        if step_name in details:
            detail_x, detail_y = details[step_name]['pos']

            # Draw dashed line connecting step to detail
            ax.plot(
                [x_pos + 2.0, detail_x - 1.0],
                [y_pos, detail_y],
                '--',
                color='gray',
                alpha=0.7
            )

# Define fixed arrow connections
arrows = [
    # Left column vertical flows
    ((2.5, 6.5), (2.5, 5.3)), # 1→2
    ((2.5, 4.5), (2.5, 3.3)), # 2→3

    # Connection from left to right
    ((4.0, 2.9), (8.0, 6.9)), # 3→4

    # Right column vertical flows
    ((9.5, 6.5), (9.5, 5.3)), # 4→5
    ((9.5, 4.5), (9.5, 3.3)), # 5→6
    ((9.5, 2.5), (9.5, 1.8))  # 6→7
]

# Draw arrows
for (start_x, start_y), (end_x, end_y) in arrows:
    ax.annotate(
        '',
        xy=(end_x, end_y),
        xytext=(start_x, start_y),
        arrowprops=dict(
            facecolor='black',
            width=1.5,
            headwidth=7,
            shrink=0.05
        )
    )

# Add title
plt.text(
    6.0, 7.5,
    'mytender.io RAG Pipeline Architecture',
    fontsize=14,
    ha='center',
    fontweight='bold'
)

# Ensure directory exists
os.makedirs('source/_static', exist_ok=True)

# Save the diagram
plt.savefig(
    'source/_static/rag_architecture.png',
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)

print("Generated RAG architecture diagram at source/_static/rag_architecture.png")
