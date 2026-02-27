"""
MultiTaskNet 3D Architecture Diagram
======================================

Generates a publication-quality pseudo-3D architecture diagram of the
MultiTaskNet dual-branch model (calibration + forecasting) with shared
BiLSTM encoder, farm embedding, and multi-horizon attention heads.

Output
------
  results/figures/fig_multitasknet_3d.png  — 300 DPI PNG

Usage
-----
    python experiments/generate_multitasknet_3D.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Union, Tuple
import os

def rgb(r, g, b): return (r/255, g/255, b/255)

# ── colour palette ─────────────────────────────────────────────────────────────
C_BG        = rgb(245, 248, 252)
C_CALIB_IN  = rgb( 70, 170, 220)
C_DENSE     = rgb( 50, 100, 200)
C_TOP_B     = rgb(100, 180, 240);  C_SIDE_B  = rgb( 20,  60, 160)
C_CALIB_HD  = rgb( 25, 140, 120)
C_TOP_T     = rgb( 55, 200, 170);  C_SIDE_T  = rgb( 10,  95,  80)
C_BILSTM    = rgb( 90,  50, 180)
C_TOP_P     = rgb(140, 100, 220);  C_SIDE_P  = rgb( 55,  20, 130)
C_ATTN      = rgb(220, 100,  30)
C_TOP_O     = rgb(250, 160,  60);  C_SIDE_O  = rgb(160,  65,  10)
C_MFCAST    = rgb(150,  80, 210)
C_FARM      = rgb(180,  30,  80)
C_FARM_T    = rgb(210,  60, 100);  C_FARM_S  = rgb(130,  20,  60)
C_CALIB_TXT = rgb( 50, 100, 200);  C_FORE_TXT= rgb( 90,  50, 180)


# ══════════════════════════════════════════════════════════════════════════════
# PRIMITIVE: 3-D BOX
# ══════════════════════════════════════════════════════════════════════════════
def box3d(ax, x, y, w, h, dx: float = 0.25, dy: float = 0.18,
          face_c=C_DENSE, top_c=C_TOP_B, side_c=C_SIDE_B,
          label: str = '', sublabel: str = '', lfs: float = 10.5, sfs: float = 8.0,
          label_color: str = 'white', extra_content=None):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x+0.06, y-0.06), w, h, boxstyle='round,pad=0.01',
        facecolor=rgb(160,160,160), edgecolor='none', alpha=0.22, zorder=1))
    xs = [x+w, x+w+dx, x+w+dx, x+w]
    ys = [y,   y+dy,   y+h+dy, y+h]
    ax.fill(xs, ys, color=side_c, zorder=2)
    ax.plot(xs+[xs[0]], ys+[ys[0]], color='white', lw=0.35, zorder=3)
    xt = [x,   x+dx,   x+w+dx, x+w]
    yt = [y+h, y+h+dy, y+h+dy, y+h]
    ax.fill(xt, yt, color=top_c, zorder=2)
    ax.plot(xt+[xt[0]], yt+[yt[0]], color='white', lw=0.35, zorder=3)
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.01',
        facecolor=face_c, edgecolor='white', linewidth=0.9, zorder=2))
    cx_, cy_ = x + w/2, y + h/2
    if extra_content:
        extra_content(ax, cx_, cy_, w, h)
    # Ensure text renders above the box patches and is not clipped
    ax.text(cx_, cy_ + h*0.13, label,
            ha='center', va='center', fontsize=lfs,
            fontweight='bold', color=label_color, zorder=10, clip_on=False)
    ax.text(cx_, cy_ - h*0.18, sublabel,
            ha='center', va='center', fontsize=sfs,
            color=label_color, alpha=0.90, linespacing=1.35, zorder=10, clip_on=False)


def seq_stack(ax, x, y, w, h, n=4, step=0.12):
    """Stacked-pages visual for Seq-Input box."""
    for i in range(n, 0, -1):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x-i*step, y+i*step*0.5), w, h,
            boxstyle='round,pad=0.01', facecolor=C_CALIB_IN,
            edgecolor='white', linewidth=0.5, alpha=0.35+0.12*i, zorder=1))


def bilstm_content(ax, cx, cy, w, h):
    """Two opposing white arrows inside BiLSTM box."""
    aw = w * 0.38
    for sign in [0.35, -0.35]:
        d = 1 if sign > 0 else -1
        ay = cy + h * sign
        ax.annotate('', xy=(cx+d*aw, ay), xytext=(cx-d*aw, ay),
                    arrowprops=dict(arrowstyle='->', color='white',
                                   lw=1.5, mutation_scale=14, zorder=6))


# ══════════════════════════════════════════════════════════════════════════════
# L-SHAPED ELBOW ARROW
# Draws a multi-segment polyline through all waypoints, with an arrowhead
# at the final point. Use this for any non-straight routing.
#   pts   : list of (x, y) waypoints
#   color : line + arrowhead colour
#   lw    : line width
#   ms    : arrowhead mutation_scale
# ══════════════════════════════════════════════════════════════════════════════
def elbow(ax, pts, color: Union[str, Tuple[float, float, float]] = '#444', lw: float = 1.5, ms: int = 15):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    for i in range(len(pts)-2):
        ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]],
                color=color, lw=lw, zorder=6, solid_capstyle='round')
    ax.annotate('', xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                arrowprops=dict(arrowstyle='->', color=color,
                               lw=lw, mutation_scale=ms, zorder=6))


AP = dict(arrowstyle='->', color='#333', lw=1.4, mutation_scale=16, zorder=7)
def harrow(ax, x0, x1, y, **kw):
    ax.annotate('', xy=(x1,y), xytext=(x0,y),
                arrowprops=dict(**{**AP, **kw}))

def branch_header(ax, y, label, color, x0=2.10, right=12.0):
    mid = (x0 + right - 0.5) / 2
    ax.plot([x0, mid-0.85],        [y, y], color=color, lw=1.8, zorder=4)
    ax.plot([mid+0.85, right-0.5], [y, y], color=color, lw=1.8, zorder=4)
    ax.text(mid, y, f' {label} ', ha='center', va='center',
            fontsize=11, fontweight='bold', color=color,
            bbox=dict(facecolor=C_BG, edgecolor='none', pad=1.5), zorder=5)


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
DX, DY = 0.25, 0.18
GAP    = 0.28

Y_CAL  = 6.20
Y_FORE = 2.60

# Box sizes
BW_CI,  BH_CI   = 1.30, 1.20
BW_D256,BH_D256 = 1.65, 1.60
BW_D128,BH_D128 = 1.45, 1.40
BW_CH,  BH_CH   = 1.35, 1.20
BW_MOI, BH_MOI  = 1.30, 0.90
BW_SEQ, BH_SEQ  = 1.35, 1.30
BW_BIL, BH_BIL  = 1.80, 2.10
BW_ATT, BH_ATT  = 1.45, 1.80
BW_HRZ, BH_HRZ  = 0.90, 0.48
BW_MFC, BH_MFC  = 1.45, 1.10
BW_FM,  BH_FM   = 1.45, 1.10   # Farm Embed

# X positions (incremental)
X_CI   = 0.55
X_D256 = X_CI   + BW_CI   + DX + GAP + 0.45
X_D128 = X_D256 + BW_D256 + DX + GAP + 0.20
X_CH   = X_D128 + BW_D128 + DX + GAP + 0.20
X_MOI  = X_CH   + BW_CH   + DX + GAP + 0.25
X_SEQ  = 0.55
X_BIL  = X_D256 - 0.15
X_ATT  = X_BIL  + BW_BIL  + DX + GAP + 0.20
X_HRZ  = X_ATT  + BW_ATT  + DX + GAP + 0.35
X_MFC  = X_HRZ  + BW_HRZ  + DX + GAP + 0.35
X_FM   = X_CI   # same column as the two input boxes

# Farm Embed y: centred between Calib In bottom and Seq Input top
Y_FARM_TOP = Y_CAL             # bottom of calib row
Y_FARM_BOT = Y_FORE + BH_SEQ  # top of fore row
Y_FM       = (Y_FARM_TOP + Y_FARM_BOT - BH_FM) / 2

def ccy(y0, bh): return y0 + bh/2

CY_CI   = ccy(Y_CAL,  BH_CI)
CY_D256 = ccy(Y_CAL,  BH_D256)
CY_D128 = ccy(Y_CAL,  BH_D128)
CY_CH   = ccy(Y_CAL,  BH_CH)
CY_SEQ  = ccy(Y_FORE, BH_SEQ)
CY_BIL  = ccy(Y_FORE, BH_BIL)
CY_ATT  = ccy(Y_FORE, BH_ATT)
CY_FM   = ccy(Y_FM,   BH_FM)

HRZ_LABELS = ['24h','12h','6h','1h']
HRZ_YS     = [CY_ATT+1.05, CY_ATT+0.35, CY_ATT-0.35, CY_ATT-1.05]

RIGHT  = X_MFC + BW_MFC + DX + 0.35
TOP    = Y_CAL + BH_D256 + DY + 1.10
BOT    = Y_FORE - 1.20

SCALE  = 0.82
fig, ax = plt.subplots(figsize=((RIGHT-0.30)*SCALE, (TOP-BOT)*SCALE))
ax.set_facecolor(C_BG); fig.patch.set_facecolor(C_BG)
ax.set_xlim(0.30, RIGHT); ax.set_ylim(BOT, TOP)
ax.axis('off')

branch_header(ax, Y_CAL+BH_D256+DY+0.45 - 0.25,
              'Calibration Branch', C_CALIB_TXT, right=RIGHT)
branch_header(ax, Y_FORE+BH_BIL+DY+0.45 - 3.45,
              'Forecasting Branch', C_FORE_TXT,  right=RIGHT)

# ── Calibration branch boxes ──────────────────────────────────────────────────
box3d(ax, X_CI,   Y_CAL, BW_CI,   BH_CI,   dx=DX, dy=DY,
      face_c=C_CALIB_IN, top_c=C_TOP_B, side_c=C_SIDE_B,
      label='Calib In', sublabel='2 feat (A2)', lfs=12, sfs=8)
box3d(ax, X_D256, Y_CAL, BW_D256, BH_D256, dx=DX, dy=DY,
      label='Dense 256', sublabel='BN + ReLU + Drop', lfs=10, sfs=7.5)
box3d(ax, X_D128, Y_CAL, BW_D128, BH_D128, dx=DX, dy=DY,
      label='Dense 128', sublabel='ReLU + Drop', lfs=10, sfs=8.5)
box3d(ax, X_CH,   Y_CAL, BW_CH,   BH_CH,   dx=DX, dy=DY,
      face_c=C_CALIB_HD, top_c=C_TOP_T, side_c=C_SIDE_T,
      label='Calib Head', sublabel='Dense→1', lfs=9.5, sfs=8.5)

MY_moi = Y_CAL + (BH_CH - BH_MOI)/2
ax.add_patch(mpatches.FancyBboxPatch(
    (X_MOI, MY_moi), BW_MOI, BH_MOI, boxstyle='round,pad=0.05',
    facecolor=C_BG, edgecolor=C_CALIB_HD, linewidth=2.0, zorder=4))
ax.text(X_MOI+BW_MOI/2, MY_moi+BH_MOI*0.64, 'Moisture %',
        ha='center', va='center', fontsize=9,
        fontweight='bold', color=C_CALIB_HD, zorder=5)
ax.text(X_MOI+BW_MOI/2, MY_moi+BH_MOI*0.27, '(Calibrated)',
        ha='center', va='center', fontsize=8, color=C_CALIB_HD, zorder=5)

# ── Forecasting branch boxes ──────────────────────────────────────────────────
box3d(ax, X_SEQ, Y_FORE, BW_SEQ, BH_SEQ, dx=DX, dy=DY,
      face_c=C_CALIB_IN, top_c=C_TOP_B, side_c=C_SIDE_B,
      label='Seq Input', sublabel='96×7', lfs=10, sfs=10)
box3d(ax, X_BIL, Y_FORE, BW_BIL, BH_BIL, dx=DX, dy=DY,
      face_c=C_BILSTM, top_c=C_TOP_P, side_c=C_SIDE_P,
      label='BiLSTM', sublabel='h=128, 2 layers',
      lfs=12, sfs=9, extra_content=bilstm_content)
box3d(ax, X_ATT, Y_FORE, BW_ATT, BH_ATT, dx=DX, dy=DY,
      face_c=C_ATTN, top_c=C_TOP_O, side_c=C_SIDE_O,
      label='Attention', sublabel='Multi-Head', lfs=11, sfs=10)

for lbl, hy in zip(HRZ_LABELS, HRZ_YS):
    box3d(ax, X_HRZ, hy-BH_HRZ/2, BW_HRZ, BH_HRZ, dx=0.16, dy=0.12,
          face_c=C_CALIB_HD, top_c=C_TOP_T, side_c=C_SIDE_T,
          label=lbl, sublabel='', lfs=10, sfs=8)

MFC_Y = CY_ATT - BH_MFC/2
ax.add_patch(mpatches.FancyBboxPatch(
    (X_MFC, MFC_Y), BW_MFC, BH_MFC, boxstyle='round,pad=0.06',
    facecolor=C_BG, edgecolor=C_MFCAST, linewidth=2.0, zorder=4))
for txt, frac, fs, fw in [
        ('Multi-Horizon', 0.72, 9, 'bold'),
        ('Forecasts',     0.46, 9, 'bold'),
        ('(1h, 6h, 12h, 24h)', 0.20, 7.5, 'normal')]:
    ax.text(X_MFC+BW_MFC/2, MFC_Y+BH_MFC*frac, txt,
            ha='center', va='center',
            fontsize=fs, fontweight=fw, color=C_MFCAST, zorder=5)

# ── Farm Embed (centred between the two branches, same column as inputs) ──────
box3d(ax, X_FM - 0.25, Y_FM, BW_FM, BH_FM, dx=0.22, dy=0.16,
      face_c=C_FARM, top_c=C_FARM_T, side_c=C_FARM_S,
      label='Farm', sublabel='16-dim', lfs=12, sfs=8.5)

# ══════════════════════════════════════════════════════════════════════════════
# FARM → DENSE 256  (L-shaped: exit right → travel right → turn UP → enter)
# Uses a dedicated vertical lane just left of Dense 256
# ══════════════════════════════════════════════════════════════════════════════
FARM_RIGHT = X_FM + BW_FM + DX - 0.25      # rightmost point of Farm (incl. 3-D)
LANE_UP    = X_D256 - GAP * 0.55     # vertical lane for upward leg

elbow(ax,
      [(FARM_RIGHT + 0.05, CY_FM),    # pt1: exit Farm right face
       (LANE_UP-0.35,           CY_FM),    # pt2: travel right to lane
       (LANE_UP-0.35,           CY_CI),    # pt3: turn up to calib row height
       (X_D256 - GAP*0.35, CY_CI)],   # pt4: enter Dense 256 left face
      color=C_FARM, lw=1.5, ms=14)

# ══════════════════════════════════════════════════════════════════════════════
# FARM → BILSTM  (L-shaped: exit right → travel right → turn DOWN → enter)
# Uses a dedicated vertical lane just left of BiLSTM
# ══════════════════════════════════════════════════════════════════════════════
LANE_DN = X_BIL - GAP * 0.55

elbow(ax,
      [(FARM_RIGHT + 0.05, CY_FM),    # pt1: exit Farm right face
       (LANE_DN - 0.2,           CY_FM),    # pt2: travel right to lane
       (LANE_DN - 0.2,           CY_BIL),   # pt3: turn down to fore row height
       (X_BIL - GAP*0.3,  CY_BIL)], # pt4: enter BiLSTM left face
      color=C_FARM, lw=1.5, ms=14)

# ── Calibration branch arrows ─────────────────────────────────────────────────
harrow(ax, X_CI+BW_CI+GAP*0.4,     X_D256-GAP*0.4,  CY_CI + 0.25)
harrow(ax, X_D256+BW_D256+GAP*0.4, X_D128-GAP*0.4,  CY_D256)
harrow(ax, X_D128+BW_D128+GAP*0.4, X_CH-GAP*0.4,    CY_D128)
harrow(ax, X_CH+BW_CH+DX+GAP*0.4,  X_MOI-GAP*0.4,   CY_CH)

# ── Forecasting branch arrows ─────────────────────────────────────────────────
harrow(ax, X_SEQ+BW_SEQ+GAP*0.4,   X_BIL-GAP*0.4,   CY_SEQ)
harrow(ax, X_BIL+BW_BIL+GAP*0.4,   X_ATT-GAP*0.4,   CY_BIL)

attn_exit_x = X_ATT + BW_ATT + DX + GAP*0.3
for hy in HRZ_YS:
    ax.annotate('', xy=(X_HRZ-GAP*0.3, hy),
                xytext=(attn_exit_x, CY_ATT),
                arrowprops=dict(arrowstyle='->', color=rgb(200,100,20),
                               lw=1.3, mutation_scale=14, zorder=6))

hrz_exit_x = X_HRZ + BW_HRZ + DX + GAP*0.3
for hy in HRZ_YS:
    harrow(ax, hrz_exit_x, X_MFC-GAP*0.3, hy,
           color='#555', lw=1.2, mutation_scale=13)

# ── Title & footnote ──────────────────────────────────────────────────────────
MID     = (0.30 + RIGHT) / 2
TITLE_Y = Y_CAL + BH_D256 + DY + 0.80
ax.text(MID, TITLE_Y, 'MultiTaskNet Architecture',
        ha='center', fontsize=18, fontweight='bold',
        color=rgb(20,20,20), zorder=6)
ax.text(MID, TITLE_Y-0.35,
        'Multi-Task Learning: Joint Calibration + Multi-Horizon Forecasting  |  ~748K Parameters',
        ha='center', fontsize=9, color=rgb(100,100,100), zorder=6)

ax.text(MID, Y_FORE - 1.2,
        r'$\mathcal{L}_{total} = \lambda_{cal}\cdot\mathcal{L}_{Huber}'
        r'(y_{cal},\hat{y}_{cal}) + \lambda_{frc}\cdot\mathcal{L}_{MSE}'
        r'(y_{frc},\hat{y}_{frc}) \quad|\quad \lambda_{cal} = \lambda_{frc} = 1.0$',
        ha='center', va='center', fontsize=9, color=rgb(60,60,60), zorder=6,
        bbox=dict(boxstyle='round,pad=0.50',
                  facecolor=C_BG, edgecolor=rgb(170,170,170), lw=1.1))


fig_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
os.makedirs(fig_dir, exist_ok=True)

plt.savefig(os.path.join(fig_dir, 'fig_multitasknet_3d.png'), dpi=300, bbox_inches='tight', facecolor=C_BG)
plt.close()
print("    Done.")