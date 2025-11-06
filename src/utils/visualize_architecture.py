"""
Generate visual architecture diagram for LSTM model
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np


def draw_lstm_architecture():
    """Draw the LSTM network architecture diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors
    input_color = '#E1F5FF'
    lstm_color = '#FFF4E1'
    dense_color = '#E8F5E9'
    output_color = '#FFEBEE'
    
    # Input sequence box
    input_box = FancyBboxPatch((0.5, 7), 3, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor=input_color, 
                               edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(2, 8, 'Input Sequence\n[Batch, 24, Features]', 
            ha='center', va='center', fontsize=11, weight='bold')
    
    # Input time steps
    for i in range(4):
        x_pos = 0.8 + i * 0.7
        circle = Circle((x_pos, 6), 0.15, facecolor='lightblue', edgecolor='black')
        ax.add_patch(circle)
        ax.text(x_pos, 6, f't-{23-i}', ha='center', va='center', fontsize=8)
    ax.text(3.5, 6, '...', ha='center', va='center', fontsize=12)
    
    # LSTM Layer 1
    lstm1_box = FancyBboxPatch((5, 6.5), 2.5, 2, 
                               boxstyle="round,pad=0.1", 
                               facecolor=lstm_color, 
                               edgecolor='black', linewidth=2)
    ax.add_patch(lstm1_box)
    ax.text(6.25, 8, 'LSTM Layer 1', ha='center', va='center', 
            fontsize=11, weight='bold')
    ax.text(6.25, 7.5, 'hidden_size=128', ha='center', va='center', 
            fontsize=9)
    ax.text(6.25, 7.1, '[Batch, 24, 128]', ha='center', va='center', 
            fontsize=8, style='italic')
    
    # LSTM cells representation
    for i in range(3):
        x_pos = 5.5 + i * 0.7
        cell_box = FancyBboxPatch((x_pos, 6.6), 0.5, 0.3,
                                  boxstyle="round,pad=0.02",
                                  facecolor='white', edgecolor='gray')
        ax.add_patch(cell_box)
    ax.text(7.5, 6.75, '...', ha='center', va='center', fontsize=10)
    
    # Arrow from input to LSTM1
    arrow1 = FancyArrowPatch((4, 8), (5, 8), 
                            arrowstyle='->', lw=2, 
                            mutation_scale=20, color='black')
    ax.add_patch(arrow1)
    
    # LSTM Layer 2
    lstm2_box = FancyBboxPatch((8.5, 6.5), 2.5, 2, 
                               boxstyle="round,pad=0.1", 
                               facecolor=lstm_color, 
                               edgecolor='black', linewidth=2)
    ax.add_patch(lstm2_box)
    ax.text(9.75, 8, 'LSTM Layer 2', ha='center', va='center', 
            fontsize=11, weight='bold')
    ax.text(9.75, 7.5, 'hidden_size=128', ha='center', va='center', 
            fontsize=9)
    ax.text(9.75, 7.1, 'dropout=0.1', ha='center', va='center', 
            fontsize=9)
    ax.text(9.75, 6.8, '[Batch, 24, 128]', ha='center', va='center', 
            fontsize=8, style='italic')
    
    # LSTM cells representation
    for i in range(3):
        x_pos = 9 + i * 0.7
        cell_box = FancyBboxPatch((x_pos, 6.6), 0.5, 0.3,
                                  boxstyle="round,pad=0.02",
                                  facecolor='white', edgecolor='gray')
        ax.add_patch(cell_box)
    ax.text(11, 6.75, '...', ha='center', va='center', fontsize=10)
    
    # Arrow from LSTM1 to LSTM2
    arrow2 = FancyArrowPatch((7.5, 8), (8.5, 8), 
                            arrowstyle='->', lw=2, 
                            mutation_scale=20, color='black')
    ax.add_patch(arrow2)
    
    # Sequence pooling - last hidden state
    pool_text = ax.text(11.75, 8, 'Last Hidden\nState', ha='center', va='center', 
                       fontsize=10, weight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='black'))
    
    # Arrow from LSTM2 to pooling
    arrow3 = FancyArrowPatch((11, 8), (11.75, 8), 
                            arrowstyle='->', lw=2, 
                            mutation_scale=20, color='black')
    ax.add_patch(arrow3)
    ax.text(11.75, 7.4, '[Batch, 128]', ha='center', va='center', 
            fontsize=8, style='italic')
    
    # Dense Layer 1
    dense1_box = FancyBboxPatch((13.5, 7), 1.5, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor=dense_color, 
                                edgecolor='black', linewidth=2)
    ax.add_patch(dense1_box)
    ax.text(14.25, 7.7, 'Dense', ha='center', va='center', 
            fontsize=10, weight='bold')
    ax.text(14.25, 7.3, '128 → 128', ha='center', va='center', 
            fontsize=9)
    
    # ReLU
    relu_box = FancyBboxPatch((13.5, 5.8), 1.5, 0.8, 
                              boxstyle="round,pad=0.1", 
                              facecolor=dense_color, 
                              edgecolor='black', linewidth=2)
    ax.add_patch(relu_box)
    ax.text(14.25, 6.2, 'ReLU', ha='center', va='center', 
            fontsize=10, weight='bold')
    
    # Dense Layer 2
    dense2_box = FancyBboxPatch((13.5, 4.5), 1.5, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor=dense_color, 
                                edgecolor='black', linewidth=2)
    ax.add_patch(dense2_box)
    ax.text(14.25, 5.2, 'Dense', ha='center', va='center', 
            fontsize=10, weight='bold')
    ax.text(14.25, 4.8, '128 → 1', ha='center', va='center', 
            fontsize=9)
    
    # Arrows for output head
    arrow4 = FancyArrowPatch((12.5, 7.5), (13.5, 7.5), 
                            arrowstyle='->', lw=2, 
                            mutation_scale=20, color='black')
    ax.add_patch(arrow4)
    
    arrow5 = FancyArrowPatch((14.25, 7), (14.25, 6.6), 
                            arrowstyle='->', lw=2, 
                            mutation_scale=20, color='black')
    ax.add_patch(arrow5)
    
    arrow6 = FancyArrowPatch((14.25, 5.8), (14.25, 5.5), 
                            arrowstyle='->', lw=2, 
                            mutation_scale=20, color='black')
    ax.add_patch(arrow6)
    
    # Output
    output_box = FancyBboxPatch((13.5, 3), 1.5, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=output_color, 
                               edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(14.25, 3.7, 'PM2.5', ha='center', va='center', 
            fontsize=11, weight='bold')
    ax.text(14.25, 3.3, 'Prediction', ha='center', va='center', 
            fontsize=10, weight='bold')
    ax.text(14.25, 3, '[Batch, 1]', ha='center', va='center', 
            fontsize=8, style='italic')
    
    arrow7 = FancyArrowPatch((14.25, 4.5), (14.25, 4), 
                            arrowstyle='->', lw=2, 
                            mutation_scale=20, color='black')
    ax.add_patch(arrow7)
    
    # Title
    ax.text(8, 9.5, 'LSTM Network Architecture for Air Quality Prediction', 
            ha='center', va='center', fontsize=16, weight='bold')
    
    # Legend/Notes
    notes = [
        "Input: 24 hourly time steps with multiple features (pollutants, weather)",
        "LSTM layers: Capture temporal patterns and dependencies",
        "Output Head: Dense layers for regression to PM2.5 concentration"
    ]
    y_start = 2
    for i, note in enumerate(notes):
        ax.text(8, y_start - i*0.3, f"• {note}", 
                ha='center', va='center', fontsize=9, 
                style='italic', color='gray')
    
    plt.tight_layout()
    return fig, ax


def draw_lstm_cell_detail():
    """Draw detailed LSTM cell internal structure"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'LSTM Cell Internal Structure', 
            ha='center', va='center', fontsize=16, weight='bold')
    
    # Input
    ax.text(1, 8.5, 'x_t', ha='center', va='center', fontsize=12, weight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', edgecolor='black'))
    ax.text(1, 7.5, 'h_{t-1}', ha='center', va='center', fontsize=12, weight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='black'))
    ax.text(1, 6.5, 'c_{t-1}', ha='center', va='center', fontsize=12, weight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black'))
    
    # Concatenation
    concat_box = FancyBboxPatch((2.5, 7), 1.5, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor='white', 
                                edgecolor='black', linewidth=2)
    ax.add_patch(concat_box)
    ax.text(3.25, 7.5, '[h_{t-1}, x_t]', ha='center', va='center', fontsize=10)
    
    # Forget gate
    forget_box = FancyBboxPatch((5, 8), 2, 0.8, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#FFE5E5', 
                               edgecolor='red', linewidth=2)
    ax.add_patch(forget_box)
    ax.text(6, 8.4, 'Forget Gate', ha='center', va='center', 
            fontsize=10, weight='bold')
    ax.text(6, 8.1, 'f_t = σ(W_f · [h, x] + b_f)', ha='center', va='center', 
            fontsize=8)
    
    # Input gate
    input_box = FancyBboxPatch((5, 6.8), 2, 0.8, 
                              boxstyle="round,pad=0.1", 
                              facecolor='#E5F5FF', 
                              edgecolor='blue', linewidth=2)
    ax.add_patch(input_box)
    ax.text(6, 7.2, 'Input Gate', ha='center', va='center', 
            fontsize=10, weight='bold')
    ax.text(6, 6.9, 'i_t = σ(W_i · [h, x] + b_i)', ha='center', va='center', 
            fontsize=8)
    
    # Candidate
    candidate_box = FancyBboxPatch((5, 5.6), 2, 0.8, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor='#E5FFE5', 
                                  edgecolor='green', linewidth=2)
    ax.add_patch(candidate_box)
    ax.text(6, 6, 'Candidate', ha='center', va='center', 
            fontsize=10, weight='bold')
    ax.text(6, 5.7, 'g_t = tanh(W_g · [h, x] + b_g)', ha='center', va='center', 
            fontsize=8)
    
    # Cell state update (pointwise operations)
    cell_update = FancyBboxPatch((8.5, 6.5), 2.5, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#FFF5E5', 
                                edgecolor='orange', linewidth=3)
    ax.add_patch(cell_update)
    ax.text(9.75, 7.8, 'Cell State', ha='center', va='center', 
            fontsize=11, weight='bold')
    ax.text(9.75, 7.4, 'c_t = f_t ⊙ c_{t-1}', ha='center', va='center', 
            fontsize=9)
    ax.text(9.75, 7.1, '+ i_t ⊙ g_t', ha='center', va='center', 
            fontsize=9)
    
    # Output gate
    output_box = FancyBboxPatch((5, 4.4), 2, 0.8, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#F5E5FF', 
                               edgecolor='purple', linewidth=2)
    ax.add_patch(output_box)
    ax.text(6, 4.8, 'Output Gate', ha='center', va='center', 
            fontsize=10, weight='bold')
    ax.text(6, 4.5, 'o_t = σ(W_o · [h, x] + b_o)', ha='center', va='center', 
            fontsize=8)
    
    # Hidden state
    hidden_box = FancyBboxPatch((8.5, 4), 2.5, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#E5FFFF', 
                               edgecolor='teal', linewidth=3)
    ax.add_patch(hidden_box)
    ax.text(9.75, 4.7, 'Hidden State', ha='center', va='center', 
            fontsize=11, weight='bold')
    ax.text(9.75, 4.3, 'h_t = o_t ⊙ tanh(c_t)', ha='center', va='center', 
            fontsize=9)
    
    # Outputs
    ax.text(12, 7.5, 'c_t', ha='center', va='center', fontsize=12, weight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black'))
    ax.text(12, 4.5, 'h_t', ha='center', va='center', fontsize=12, weight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='black'))
    
    # Arrows
    arrows = [
        # Input to concat
        ((1.3, 8.5), (2.5, 7.7)),
        ((1.3, 7.5), (2.5, 7.5)),
        # Concat to gates
        ((4, 7.5), (5, 8.4)),
        ((4, 7.5), (5, 7.2)),
        ((4, 7.5), (5, 6)),
        ((4, 7.5), (5, 4.8)),
        # Previous cell state to update
        ((1.3, 6.5), (8.5, 7.2)),
        # Forget gate to cell update
        ((7, 8.4), (8.5, 7.8)),
        # Input gate and candidate to cell update
        ((7, 7.2), (8.5, 7.4)),
        ((7, 6), (8.5, 7.1)),
        # Cell state to hidden
        ((9.75, 7), (9.75, 5)),
        # Output gate to hidden
        ((7, 4.8), (8.5, 4.7)),
        # Cell state output
        ((11, 7.5), (12, 7.5)),
        # Hidden state output
        ((11, 4.5), (12, 4.5)),
    ]
    
    for (start, end) in arrows:
        arrow = FancyArrowPatch(start, end, 
                               arrowstyle='->', lw=1.5, 
                               mutation_scale=15, color='black',
                               connectionstyle='arc3,rad=0.1')
        ax.add_patch(arrow)
    
    # Legend
    legend_text = [
        "σ: Sigmoid activation",
        "tanh: Hyperbolic tangent",
        "⊙: Element-wise multiplication",
        "W, b: Weight matrices and biases"
    ]
    for i, text in enumerate(legend_text):
        ax.text(7, 2.5 - i*0.4, text, ha='center', va='center', 
                fontsize=9, style='italic')
    
    plt.tight_layout()
    return fig, ax


if __name__ == '__main__':
    import os
    
    # Create output directory
    output_dir = 'reports'
    os.makedirs(output_dir, exist_ok=True)
    
    # Draw architecture diagram
    fig1, ax1 = draw_lstm_architecture()
    fig1.savefig(os.path.join(output_dir, 'lstm_architecture.png'), 
                 dpi=300, bbox_inches='tight')
    print(f"Saved architecture diagram to {output_dir}/lstm_architecture.png")
    
    # Draw LSTM cell detail
    fig2, ax2 = draw_lstm_cell_detail()
    fig2.savefig(os.path.join(output_dir, 'lstm_cell_detail.png'), 
                 dpi=300, bbox_inches='tight')
    print(f"Saved LSTM cell detail to {output_dir}/lstm_cell_detail.png")
    
    plt.show()

