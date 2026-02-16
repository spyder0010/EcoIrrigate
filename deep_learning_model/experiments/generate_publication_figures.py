"""
EcoIrrigate Publication-Standard Figure Generator
=================================================
Generates high-precision, publication-quality architecture diagrams in two styles:
1. Modern 3D: Gradient-filled, shadowed, perspective-rich (for Presentations/Web)
2. Journal Standard: Flat, high-contrast, grayscale/monochrome (for IEEE/Springer Print)

Output: 
- ../manuscript/springer/figures/modern_3d/
- ../manuscript/springer/figures/journal_flat/
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, PathPatch
import matplotlib.path as mpath
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import os

# ── Setup ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, '..', 'manuscript', 'springer', 'figures')
MODERN_DIR = os.path.join(OUT_DIR, 'modern_3d')
JOURNAL_DIR = os.path.join(OUT_DIR, 'journal_flat')

os.makedirs(MODERN_DIR, exist_ok=True)
os.makedirs(JOURNAL_DIR, exist_ok=True)

# ── Style Configurations ───────────────────────────────────────────
STYLES = {
    'modern_3d': {
        'bg_color': '#FFFFFF',
        'text_color': '#333333',
        'edge_color': 'none',
        'shadow': True,
        'gradient': True,
        'font_family': 'sans-serif',
        'dpi': 300,
        'ext': '.png'
    },
    'journal_flat': {
        'bg_color': '#FFFFFF',
        'text_color': '#000000',
        'edge_color': '#000000',
        'shadow': False,
        'gradient': False,
        'font_family': 'serif',
        'dpi': 600,
        'ext': '.pdf' # Vector for print
    }
}

# ── Helper Functions ──────────────────────────────────────────────
def draw_box_3d(ax, x, y, w, h, depth=0.02, color='#444444', style='modern_3d', label=None, sublabel=None):
    """Draws a box. In 3D mode, adds simulated depth and shadow."""
    
    cfg = STYLES[style]
    z_order = 10
    
    if style == 'modern_3d':
        # Shadow
        shadow = FancyBboxPatch((x + depth, y - depth), w, h,
                                 boxstyle="round,pad=0.01",
                                 ec="none", fc='#00000030', zorder=z_order-2)
        ax.add_patch(shadow)
        
        # Main face
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.01",
                             ec="white", linewidth=0.5, fc=color, zorder=z_order)
        ax.add_patch(box)
        
        # shimmer/gradient effect (simplified as top highlight)
        highlight = FancyBboxPatch((x, y + h/2), w, h/2,
                                    boxstyle="round,pad=0.01",
                                    ec="none", fc='#FFFFFF20', zorder=z_order+1)
        # Clip highlight to top half would be complex with patches, 
        # so we just use a lighter color for the box or overlay
        
    else: # journal_flat
        # Simple high-contrast box
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.01",
                             ec='black', linewidth=1.5, fc='white', zorder=z_order)
        ax.add_patch(box)
    
    # Text
    if label:
        tx = x + w/2
        ty = y + h/2 + (0.01 if sublabel else 0)
        tc = 'white' if style == 'modern_3d' else 'black'
        font = 'Arial' if style == 'modern_3d' else 'Times New Roman'
        weight = 'bold'
        
        ax.text(tx, ty, label, ha='center', va='center', color=tc,
                fontsize=7, fontweight=weight, fontname=font, zorder=z_order+2)
        
    if sublabel:
        tx = x + w/2
        ty = y + h/2 - 0.015
        tc = '#E0E0E0' if style == 'modern_3d' else '#444444'
        font = 'Arial' if style == 'modern_3d' else 'Times New Roman'
        
        ax.text(tx, ty, sublabel, ha='center', va='center', color=tc,
                fontsize=5, fontname=font, zorder=z_order+2)
        
    return (x+w, y+h/2) # Return connection point (right, center)

def draw_arrow(ax, x1, y1, x2, y2, style='modern_3d', curve=0.0):
    """Draws a connecting arrow."""
    color = '#555555' if style == 'modern_3d' else 'black'
    lw = 1.5 if style == 'modern_3d' else 1.0
    
    arrow_style = f"Simple,tail_width=0.5,head_width=3,head_length=3"
    connection_style = f"arc3,rad={curve}"
    
    arrow = mpatches.FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle=arrow_style,
                                     connectionstyle=connection_style,
                                     color=color, linewidth=lw, zorder=5)
    ax.add_patch(arrow)

def setup_canvas(style):
    fig, ax = plt.subplots(figsize=(8, 4), dpi=STYLES[style]['dpi'])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.5)
    ax.axis('off')
    fill_color = STYLES[style]['bg_color']
    fig.patch.set_facecolor(fill_color)
    ax.set_facecolor(fill_color)
    return fig, ax

def save_fig(fig, name, style):
    folder = MODERN_DIR if style == 'modern_3d' else JOURNAL_DIR
    ext = STYLES[style]['ext']
    path = os.path.join(folder, name + ext)
    plt.savefig(path,bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Saved {style} figure: {path}")

# ── Figure Generators ─────────────────────────────────────────────

def gen_calibration_net(style):
    fig, ax = setup_canvas(style)
    
    # Colors
    c_in = '#1976D2'  # Blue
    c_emb = '#7B1FA2' # Purple
    c_dense = '#2E7D32' # Green
    c_out = '#00695C' # Teal
    
    # Components
    # Input
    draw_box_3d(ax, 0.05, 0.2, 0.1, 0.1, color=c_in, style=style, label='Input', sublabel='ADC, Volt')
    draw_box_3d(ax, 0.05, 0.05, 0.1, 0.08, color=c_emb, style=style, label='Farm ID', sublabel='Index')
    
    # Embedding
    draw_box_3d(ax, 0.2, 0.05, 0.12, 0.08, color=c_emb, style=style, label='Embedding', sublabel='Dim=16')
    draw_arrow(ax, 0.15, 0.09, 0.2, 0.09, style)
    
    # Concat
    draw_arrow(ax, 0.15, 0.25, 0.35, 0.25, style) # Input to Dense
    draw_arrow(ax, 0.32, 0.09, 0.35, 0.22, style, curve=-0.3) # Emb to Dense
    
    # Dense Layers
    x = 0.35
    dims = [256, 128, 64]
    for i, dim in enumerate(dims):
        draw_box_3d(ax, x, 0.2, 0.12, 0.1, color=c_dense, style=style, label=f'Dense {i+1}', sublabel=f'{dim} units\nBN+ReLU')
        if i < len(dims)-1:
            draw_arrow(ax, x+0.12, 0.25, x+0.16, 0.25, style)
        x += 0.16
        
    # Output
    draw_arrow(ax, x-0.04, 0.25, x, 0.25, style)
    draw_box_3d(ax, x, 0.22, 0.1, 0.06, color=c_out, style=style, label='Output', sublabel='Moisture %')
    
    # Title
    t_color = STYLES[style]['text_color']
    ax.text(0.5, 0.45, 'CalibrationNet Architecture', ha='center', fontsize=12, fontweight='bold', color=t_color)
    
    save_fig(fig, 'fig_calibrationnet', style)

def gen_multitask_net(style):
    fig, ax = setup_canvas(style)
    
    c_shared = '#F57F17' # Orange
    c_cal = '#2E7D32'    # Green
    c_for = '#1565C0'    # Blue
    
    # Shared Input
    draw_box_3d(ax, 0.05, 0.2, 0.1, 0.1, color=c_shared, style=style, label='Input', sublabel='A6 Features\n+ Farm ID')
    
    # Shared Embedding
    draw_box_3d(ax, 0.2, 0.2, 0.12, 0.1, color=c_shared, style=style, label='Shared\nEmbedding', sublabel='Farm-Specific')
    draw_arrow(ax, 0.15, 0.25, 0.2, 0.25, style)
    
    # Branch Split
    # Calibration Branch (Top)
    draw_arrow(ax, 0.32, 0.28, 0.4, 0.38, style, curve=0.2)
    draw_box_3d(ax, 0.4, 0.35, 0.15, 0.08, color=c_cal, style=style, label='Calibration\nTower', sublabel='MLP Layers')
    draw_arrow(ax, 0.55, 0.39, 0.65, 0.39, style)
    draw_box_3d(ax, 0.65, 0.36, 0.1, 0.06, color=c_cal, style=style, label='Calib\nOutput', sublabel='Current %')
    
    # Forecasting Branch (Bottom)
    draw_arrow(ax, 0.32, 0.22, 0.4, 0.12, style, curve=-0.2)
    draw_box_3d(ax, 0.4, 0.08, 0.15, 0.08, color=c_for, style=style, label='Forecasting\nTower', sublabel='BiLSTM+Attn')
    draw_arrow(ax, 0.55, 0.12, 0.65, 0.12, style)
    draw_box_3d(ax, 0.65, 0.05, 0.18, 0.14, color=c_for, style=style, label='Multi-Horizon\nOutput', sublabel='1h, 6h, 12h, 24h')
    
    # Title
    t_color = STYLES[style]['text_color']
    ax.text(0.5, 0.48, 'MultiTaskNet Architecture', ha='center', fontsize=12, fontweight='bold', color=t_color)
    
    save_fig(fig, 'fig_multitasknet', style)

def gen_system_overview(style):
    fig, ax = setup_canvas(style)
    
    c_hw = '#455A64' # Slate
    c_data = '#00838F' # Cyan
    c_ai = '#D84315' # Deep Orange
    
    # Hardware
    draw_box_3d(ax, 0.05, 0.2, 0.15, 0.1, color=c_hw, style=style, label='IoT Node', sublabel='Arduino + Sensors')
    draw_arrow(ax, 0.2, 0.25, 0.25, 0.25, style)
    
    # Data
    draw_box_3d(ax, 0.25, 0.2, 0.15, 0.1, color=c_data, style=style, label='Data\nPipeline', sublabel='Cleaning + Feat Eng')
    draw_arrow(ax, 0.4, 0.25, 0.45, 0.25, style)
    
    # AI Engine
    draw_box_3d(ax, 0.45, 0.15, 0.2, 0.2, color=c_ai, style=style, label='Deep Learning\nEngine', sublabel='MultiTaskNet\n(Calib + Forecast)')
    draw_arrow(ax, 0.65, 0.25, 0.75, 0.25, style)
    
    # Action
    c_green = '#2E7D32'
    draw_box_3d(ax, 0.75, 0.2, 0.15, 0.1, color=c_green if style=='modern_3d' else 'white', style=style, label='Irrigation\nDecision', sublabel='Schedule/Trigger')
    
    t_color = STYLES[style]['text_color']
    ax.text(0.5, 0.48, 'EcoIrrigate System Overview', ha='center', fontsize=12, fontweight='bold', color=t_color)
    
    save_fig(fig, 'fig_system_overview', style)

# ── Main ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Generating Modern 3D Figures...')
    gen_calibration_net('modern_3d')
    gen_multitask_net('modern_3d')
    gen_system_overview('modern_3d')
    
    print('Generating Journal Standard Figures (Flat)...')
    gen_calibration_net('journal_flat')
    gen_multitask_net('journal_flat')
    gen_system_overview('journal_flat')
    
    print('Done.')
