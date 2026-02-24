"""
Graphical Abstract — EcoIrrigate
================================================
CHANGES FROM v7:
  - Canvas enlarged to 1600×960 px (y range 0–60) to fit image zone
  - New image zone y∈[5.5,21.5]: left=arch mini-diagram, right=performance charts
  - Title shifted down slightly; "MultiTaskNet" y adjusted
  - Hero box enlarged (h=7.8 was 5.8)
  - Conclusion banner reduced (h=6.0)
  - Encoder box widened + taller (EH=3.4, EX closer to edges)
  - Forecasting "1.7×" font reduced to fs=6.0; sub-text to fs=3.8
  - "Forecasting Degradation" sub-header box taller (h=2.5)
  - Embedded recreated charts (fig_multitasknet_3d + fig9_rule_vs_dl) as inset axes
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc, FancyBboxPatch, Circle, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np, os

OUT = r'graphical_abstract\results'
os.makedirs(OUT, exist_ok=True)

# ── CANVAS ───────────────────────────────────────────────────────────────────
DPI   = 300
W_PX  = 1600;  H_PX  = 960
W_IN  = W_PX / DPI;  H_IN  = H_PX / DPI   # 5.333 × 3.2 in

# ── PALETTE ──────────────────────────────────────────────────────────────────
WHITE = '#FFFFFF';  BG = '#EEF3F8'
C1H='#163A6B'; C1L='#D0E6F8'; C1A='#1D4ED8'; C1D='#0D2545'
C2H='#3A1069'; C2L='#E8DAFC'; C2A='#7C3AED'; C2D='#28074A'
C3H='#0E3D20'; C3L='#D2F5E2'; C3A='#16A34A'; C3D='#072B14'
NAVY='#080F1E'
RED='#B91C1C'; AMB='#B45309'; GRN='#15803D'; ORG='#D97706'
DRK='#1E293B'; MID='#475569'; LITE='#64748B'; SEP='#94A3B8'

# ── FIGURE ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI, facecolor=WHITE)
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100);  ax.set_ylim(0, 60)
ax.set_facecolor(BG);  ax.axis('off')

# ── PRIMITIVES ────────────────────────────────────────────────────────────────
def bx(x, y, w, h, fc, ec='none', lw=0.7, r=0.0, z=2, a=1.0):
    style = f'round,pad=0,rounding_size={r}' if r else 'square,pad=0'
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=style,
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z, alpha=a))

def t(x, y, s, c=DRK, fs=5.5, bold=False,
      ha='center', va='center', z=15, it=False):
    ax.text(x, y, s, color=c, fontsize=fs,
            fontweight='bold' if bold else 'normal',
            fontstyle='italic' if it else 'normal',
            ha=ha, va=va, zorder=z,
            fontfamily=['Arial', 'DejaVu Sans', 'sans-serif'])

def hl(x1, x2, y, c=SEP, lw=0.55):
    ax.plot([x1, x2], [y, y], color=c, lw=lw, zorder=6)

def arr_r(x1, x2, y, c, lw=2.8, ms=14):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
        arrowprops=dict(arrowstyle='->', color=c, lw=lw,
                        mutation_scale=ms), zorder=9)

def arr_d(x, y1, y2, c, lw=1.5, ms=9):
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
        arrowprops=dict(arrowstyle='->', color=c, lw=lw,
                        mutation_scale=ms), zorder=9)

# ── ICONS ─────────────────────────────────────────────────────────────────────
def icon_wifi(cx, cy, c, s=1.0):
    for r_ in [2.3, 1.5, 0.75]:
        ax.add_patch(Arc((cx, cy), r_*s*2, r_*s*2,
            angle=90, theta1=22.5, theta2=157.5,
            color=c, lw=1.5*s, zorder=10))
    ax.add_patch(Circle((cx, cy), 0.28*s, color=c, zorder=11))
    ax.plot([cx, cx], [cy-0.28*s, cy-1.5*s], color=c, lw=1.0*s, zorder=10)
    ax.plot([cx-0.95*s, cx+0.95*s], [cy-1.5*s]*2, color=c, lw=1.0*s, zorder=10)

def icon_nn(cx, cy, c_in, c_h, c_out, s=1.0):
    layers = [
        ([cy+1.2*s, cy+0.4*s, cy-0.4*s, cy-1.2*s], cx-2.5*s),
        ([cy+0.8*s, cy,       cy-0.8*s           ], cx),
        ([cy+0.55*s,          cy-0.55*s           ], cx+2.5*s),
    ]
    for li in range(len(layers)-1):
        ys0, x0 = layers[li]; ys1, x1 = layers[li+1]
        for y0 in ys0:
            for y1 in ys1:
                ax.plot([x0,x1],[y0,y1], color='#94A3B8',
                        lw=0.3, zorder=7, alpha=0.45)
    for (ys, xn), nc in zip(layers, [c_in, c_h, c_out]):
        for yn in ys:
            ax.add_patch(Circle((xn, yn), 0.32*s, color=nc, zorder=12))

# ════════════════════════════════════════════════════════════════════════════
# GLOBAL LAYOUT  (y ∈ [0, 60])
# ════════════════════════════════════════════════════════════════════════════
TY1  = 57.5;  TY2  = 60.0        # title
CY1  = 22.0;  CY2  = 57.5        # 3-column zone
HDR  =  3.8
BY1  = CY1;   BY2  = CY2 - HDR   # body: [22.0, 53.7]
IY1  =  4.0;  IY2  = 20.5        # image zone — shifted down by 1.5 units
COY1 =  0.0;  COY2 =  4.0        # conclusion banner — matches new IY1

GAP  = 1.4
C1x  = 0.3;   C1w = 32.1;  C1X = C1x + C1w
C2x  = C1X + GAP;  C2w = C1w;  C2X = C2x + C2w
C3x  = C2X + GAP;  C3X = 99.7;  C3w = C3X - C3x
C1c  = (C1x + C1X) / 2    # 16.35
C2c  = (C2x + C2X) / 2    # 49.85
C3c  = (C3x + C3X) / 2    # 83.5

# ════════════════════════════════════════════════════════════════════════════
# TITLE BANNER
# ════════════════════════════════════════════════════════════════════════════
bx(0, TY1, 100, TY2-TY1, fc=C1A, z=1)
bx(0, TY2-0.35, 100, 0.45, fc=C1A, z=2)
# Title slightly lower (was at midpoint, now −0.3)
t(50, (TY1+TY2)/2 - 0.2,
  'MultiTaskNet: Multi-Task Deep Learning for IoT Soil Moisture Calibration & Forecasting',
  c=WHITE, fs=7, bold=True)

# ════════════════════════════════════════════════════════════════════════════
# COLUMN BACKGROUNDS + HEADERS
# ════════════════════════════════════════════════════════════════════════════
for cx1, cw, hc, bc, lbl in [
        (C1x, C1w, C1H, C1L, '① IoT Platform & Dataset'),
        (C2x, C2w, C2H, C2L, '② MultiTaskNet Architecture'),
        (C3x, C3w, C3H, C3L, '③ Performance Results')]:
    bx(cx1, BY1, cw, BY2-BY1, fc=bc, z=1)
    bx(cx1, BY2, cw, HDR,     fc=hc, z=2)
    t(cx1+cw/2, BY2+HDR/2, lbl, c=WHITE, fs=6.5, bold=True)

# ════════════════════════════════════════════════════════════════════════════
# COLUMN 1 — IoT Platform & Dataset
# WiFi icon top ≈ 53.0 < BY2=53.7 ✓
# All positions shifted +15 from v7 base, verified overlap-free
# ════════════════════════════════════════════════════════════════════════════
WIFI_CY = 52.0;  WIFI_S = 0.42
icon_wifi(C1c, WIFI_CY, C1A, s=WIFI_S)
# stem bottom = 52.0-0.63 = 51.37

t(C1c, 49.2, '21,312', c=C1D, fs=10, bold=True)   # [47.64, 50.76]
t(C1c, 46.4, 'raw sensor readings', c=MID, fs=5.0) # gap: 47.64-47.18=0.46 ✓

hl(C1x+2, C1X-2, 45.0)
t(C1c, 44.1, '2 farms  ·  110 days  ·  15-min intervals', c=C1D, fs=4.8, bold=True)

hl(C1x+2, C1X-2, 42.7)
t(C1c, 41.7, 'Input Features', c=C1D, fs=5.0, bold=True)

# Feature tags — 2×2 grid
TGW = 12.8;  TGH = 1.8
TG_LEFT = C1c - TGW - 0.8
for i, tag in enumerate(['ADC Count', 'Supply Voltage', 'Temperature', 'Pressure']):
    col_i = i % 2;  row_i = i // 2
    tx_ = TG_LEFT + col_i * (TGW + 1.6)
    ty_ = 38.8 - row_i * (TGH + 1.2)
    bx(tx_, ty_, TGW, TGH, fc=C1A, ec='none', r=0.35, z=5)
    t(tx_+TGW/2, ty_+TGH/2, tag, c=WHITE, fs=4.8, bold=True)

bx(C1x+2, 33.5, C1w-4, 1.8, fc='#B3D4F0', ec=C1A, lw=0.6, r=0.3, z=5)  # y: 34.8→34.3
t(C1c, 34.5, '+ 22 engineered features', c=C1D, fs=4.8, bold=True)        # y: 35.7→35.2

bx(C1x+2, BY1+0.4, C1w-4, 2.5, fc=C1D, ec='none', r=0.4, z=5)
t(C1c, BY1+1.65, 'Capacitive Soil Moisture Sensors', c=WHITE, fs=5.0, bold=True)

# ════════════════════════════════════════════════════════════════════════════
# COLUMN 2 — MultiTaskNet Architecture
# Encoder box enlarged: EH=3.4, EX closer to edges (EX=C2x+1.8)
# Forecasting text font reduced to avoid overflow
# ════════════════════════════════════════════════════════════════════════════
icon_nn(C2c, 52.6, C2A, C2D, C3A, s=0.50)
# NN bottom = 52.6-0.6 = 52.0

# Encoder box — ENLARGED (EH=3.4, wider)
EX = C2x+1.8;  EY = 47.5;  EW = C2w-3.6;  EH = 3.4
bx(EX, EY, EW, EH, fc=WHITE, ec=C2A, lw=1.3, r=0.5, z=5)
# top = 50.9;  gap from NN bottom (52.0): 1.1 ✓
t(C2c, EY+EH/2+0.55, 'Shared BiLSTM-Attention Encoder', c=C2D, fs=5.0, bold=True)
t(C2c, EY+EH/2-0.65, 'shared latent representation',    c=LITE, fs=4.2, it=True)

# Task heads
HW = 13.0;  HH = 7.5;  HY = 30.0
CAL = C2x + 1.8
FOR = C2X - HW - 1.8
# Arrows from encoder bottom (EY=47.5) to head tops (HY+HH=37.5)
arr_d(CAL+HW/2, EY, HY+HH, '#1D4ED8', lw=1.5, ms=9)
arr_d(FOR+HW/2, EY, HY+HH, '#B45309', lw=1.5, ms=9)

# Calibration head
bx(CAL, HY, HW, HH, fc='#DBEAFE', ec='#2563EB', lw=1.0, r=0.5, z=5)
bx(CAL, HY+HH-2.0, HW, 2.0, fc='#1D4ED8', ec='none', r=0.4, z=6)
t(CAL+HW/2, HY+HH-1.0, 'Calibration', c=WHITE, fs=5.2, bold=True)
t(CAL+HW/2, HY+3.6,    'R²=0.912',    c='#1E40AF', fs=8.0, bold=True)
t(CAL+HW/2, HY+1.4,    'RMSE = 2.1%', c=MID, fs=4.5)

# Forecasting head — REDUCED font sizes (user request)
bx(FOR, HY, HW, HH, fc='#FEF3C7', ec='#D97706', lw=1.0, r=0.5, z=5)
bx(FOR, HY+HH-2.0, HW, 2.0, fc='#D97706', ec='none', r=0.4, z=6)
t(FOR+HW/2, HY+HH-1.0, 'Forecasting',       c=WHITE,    fs=5.2, bold=True)
t(FOR+HW/2, HY+3.6,    '1.7× degrad.',       c='#92400E', fs=6.0, bold=True)   # ↓ from 7.5
t(FOR+HW/2, HY+1.4,    'vs 18.6× rule-based', c=MID,     fs=3.8)               # ↓ from 4.2

# Ablation badge
bx(C2x+2, BY1+0.4, C2w-4, 2.5, fc=C2D, ec='none', r=0.4, z=5)
t(C2c, BY1+1.65, 'Ablation: A2 (BiLSTM+Attn) = optimal', c=WHITE, fs=4.8, bold=True)

# ════════════════════════════════════════════════════════════════════════════
# COLUMN 3 — Performance Results
# Sub-header ENLARGED (h=2.5); Hero box ENLARGED (h=7.8); badges shifted
# ════════════════════════════════════════════════════════════════════════════
# Sub-header — ENLARGED (h=2.5 was 2.25)
bx(C3x+1.5, 50.8, C3w-3, 2.7, fc=C3H, ec='none', r=0.3, z=4)
t(C3c, 52.5, 'Forecasting Degradation', c=WHITE,      fs=5.5, bold=True)
t(C3c, 51.2, '(lower = better)',        c='#A7F3D0', fs=4.5)

# Bar chart
BLEFT=76.0;  BMAXW=17.5;  MAXV=20.0;  BH=1.5
BAR_Y = [48.4, 46.4, 44.4]
for lbl, val, col, by_ in [
        ('Rule-Based',   18.6, RED, BAR_Y[0]),
        ('Persistence',   5.8, AMB, BAR_Y[1]),
        ('MultiTaskNet',  1.7, GRN, BAR_Y[2])]:
    bw_ = (val / MAXV) * BMAXW
    bx(BLEFT, by_, BMAXW, BH, fc='#CBD5E1', ec='none', r=0.15, z=5)
    bx(BLEFT, by_, bw_,   BH, fc=col,      ec='none', r=0.20, z=6, a=0.9)
    t(BLEFT-0.5, by_+BH/2, lbl, c=col, fs=4.2, bold=True, ha='right')
    t(BLEFT+bw_+0.5, by_+BH/2, f'{val}×', c=col, fs=5.0, bold=True, ha='left')

hl(C3x+2, C3X-2, 43.2)

# Hero box — ENLARGED (h=7.8 was 5.8); y recalculated
HERO_Y = 34.0;  HERO_H = 8.8     # top = 42.8 < 43.2-0.4=42.8 ✓ (right at limit)
bx(C3x+2.5, HERO_Y, C3w-5, HERO_H, fc=WHITE, ec=C3A, lw=1.1, r=0.7, z=4)
# "10.9×" fs=13, half_h=2.03; y=HERO_Y+6.3=40.3 → top=42.33 < 42.8 ✓
t(C3c, HERO_Y+6.3, '10.9×',                c=GRN, fs=13, bold=True)
# sub-text y=HERO_Y+3.0=37.0; gap above 10.9× bottom (38.27): 1.27 ✓
t(C3c, HERO_Y+3.0, 'better than',          c=C3D, fs=5.5, bold=True)
t(C3c, HERO_Y+1.5, 'rule-based baseline',  c=C3D, fs=5.5, bold=True)

# Three badges — y adjusted down from hero bottom (34.0) with gap
# badge1: top = 34.0-0.4 = 33.6 → y = 33.6-1.6 = 32.0
for i, (btxt, bfc, bec) in enumerate([
        ('✓  Cross-farm generalisation  (R² > 0.91)', '#B3F0CD', C3A),
        ('✓  No retraining needed',                    '#B3F0CD', C3A),
        ('Calibration: R²=0.912 · RMSE=2.1%',          '#DBEAFE', C1A)]):
    by_ = 32.0 - i * 2.1
    bx(C3x+2, by_, C3w-4, 1.6, fc=bfc, ec=bec, lw=0.8, r=0.35, z=5)
    t(C3c, by_+0.8, btxt, c=DRK, fs=4.5, bold=True)

bx(C3x+2, BY1+0.4, C3w-4, 2.5, fc=C3D, ec='none', r=0.4, z=5)
t(C3c, BY1+1.65, '21,312 readings · 2 farms · 110 days', c=WHITE, fs=5.0, bold=True)

# ════════════════════════════════════════════════════════════════════════════
# FLOW ARROWS
# ════════════════════════════════════════════════════════════════════════════
AY = 46.0
arr_r(C1X+0.2, C2x-0.2, AY, C2A, lw=2.8, ms=14)
arr_r(C2X+0.2, C3x-0.2, AY, C3A, lw=2.8, ms=14)

# ════════════════════════════════════════════════════════════════════════════
# SEPARATOR between column zone and image zone
# ════════════════════════════════════════════════════════════════════════════
bx(0, IY2+1, 100, 0.5, fc=NAVY, z=3)


# Image zone background
bx(0, IY1, 100, IY2-IY1, fc='#F0F4F8', z=1)

# ════════════════════════════════════════════════════════════════════════════
# LEFT INSET — MultiTaskNet Architecture Diagram (Fig. 1 recreation)
# Inset axes coords in figure fraction:
#   x: 0→100 maps to 0→1 in figure width
#   y: 0→60 maps to 0→1 in figure height
# Left inset: x∈[0.5,49], y∈[6.0,21.0]  →  fig coords:
#   left=0.5/100=0.005, bottom=6.0/60=0.1, width=48.5/100=0.485, height=15/60=0.25
# ════════════════════════════════════════════════════════════════════════════

# LEFT INSET — MultiTaskNet Architecture Diagram (Fig. 1 recreation)
ax_arch = fig.add_axes([0.01, 0.10, 0.477, 0.23])  # slightly shifted from [0.008, 0.068]
ax_arch.set_xlim(0, 10)
ax_arch.set_ylim(-0.8, 7.2)
ax_arch.set_facecolor('#F8FAFF')
ax_arch.axis('off')

OFFSET = 0.5

# Caption bar at top — now fully inside ylim and enclosing text
ax_arch.add_patch(FancyBboxPatch((0, 6.4), 10, 6,
    boxstyle='square,pad=0',
    facecolor=C2H, edgecolor='none', zorder=3, clip_on=True))

ax_arch.text(5, 6.8,  # was 5.7 → move up by 1
    'Fig. 1 — MultiTaskNet Architecture  (Calibration + Forecasting)',
    color=WHITE, fontsize=4.2, fontweight='bold',
    ha='center', va='center', zorder=4, clip_on=True,
    fontfamily=['Arial', 'DejaVu Sans', 'sans-serif'])

def ab(ax_, x,y,w,h,fc,ec='none',lw=0.6,r=0.2,z=2):
    s = f'round,pad=0,rounding_size={r}' if r else 'square,pad=0'
    ax_.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=s,
        facecolor=fc,edgecolor=ec,linewidth=lw,zorder=z))

def at(ax_,x,y,s,c='#1E293B',fs=3.8,bold=False,ha='center',va='center',z=10,it=False):
    ax_.text(x,y,s,color=c,fontsize=fs,
        fontweight='bold' if bold else 'normal',
        fontstyle='italic' if it else 'normal',
        ha=ha,va=va,zorder=z,
        fontfamily=['Arial','DejaVu Sans','sans-serif'])

def aarr(ax_,x1,y1,x2,y2,c,lw=0.8,ms=5):
    ax_.annotate('',xy=(x2,y2),xytext=(x1,y1),
        arrowprops=dict(arrowstyle='->',color=c,lw=lw,mutation_scale=ms),zorder=8)

# ── Calibration branch (top) — FIXED box heights & text positions ──────────
ab(ax_arch, 0.1, 3.65, 1.6, 1.1, fc='#93C5FD', ec=C1A, lw=0.7)  # height 0.95→1.05
at(ax_arch, 0.9, 4.15, 'Calib In\n6 feat', c=C1D, fs=3.3, bold=True)

ab(ax_arch, 2.1, 3.65, 2.0, 1.05, fc='#2563EB', ec='none')  # height 0.95→1.05
at(ax_arch, 3.1, 4.15, 'Dense 256\nBN+ReLU+Drop', c=WHITE, fs=3.1, bold=True)

ab(ax_arch, 4.5, 3.65, 1.8, 1.05, fc='#3B82F6', ec='none')  # height 0.95→1.05  
at(ax_arch, 5.4, 4.15, 'Dense 128\nReLU+Drop', c=WHITE, fs=3.1, bold=True)

ab(ax_arch, 6.7, 3.65, 1.5, 1.05, fc=C3H, ec='none')  # height 0.95→1.05
at(ax_arch, 7.45, 4.15, 'Calib\nHead', c=WHITE, fs=3.1, bold=True)

ab(ax_arch, 8.6, 3.65, 1.3, 1.05, fc='#D1FAE5', ec=C3A, lw=0.7)  # height 0.95→1.05
at(ax_arch, 9.25, 4.15, 'Moisture\n%', c=C3D, fs=3.2, bold=True)

for x1, x2 in [(1.7, 2.1), (4.1, 4.5), (6.3, 6.7), (8.2, 8.6)]:
    aarr(ax_arch, x1, 4.15, x2, 4.15, '#1D4ED8')  # arrows align with text centers

# "Calib Branch" label — sits just below caption bar bottom (5.4) with gap
at(ax_arch,4.8,5.0,'— Calibration Branch —',c=C1A,fs=3.3,bold=True)

# ── Farm Embed (centre) ───────────────────────────────────────────────
ab(ax_arch, 0.1,2,1.6,1.2, fc='#BE185D',ec='none',r=0.15); at(ax_arch,0.9,2.58,'Farm Embed\n16-dim',c=WHITE,fs=3.3,bold=True)
# curved arrows from Farm Embed to Dense256 and BiLSTM
ax_arch.annotate('',xy=(2.2,4.0),xytext=(0.9,3.05),
    arrowprops=dict(arrowstyle='->',color='#BE185D',lw=0.7,
                    connectionstyle='arc3,rad=-0.3',mutation_scale=5),zorder=8)
ax_arch.annotate('',xy=(2.4,1.65),xytext=(0.9,2.1),
    arrowprops=dict(arrowstyle='->',color='#BE185D',lw=0.7,
                    connectionstyle='arc3,rad=0.3',mutation_scale=5),zorder=8)
at(ax_arch,3.1,3.38,'⊕',c='#BE185D',fs=6.5)

# ── Forecasting branch (bottom) ───────────────────────────────────────
ab(ax_arch, 0.1,0.6,1.6,0.99, fc='#93C5FD',ec=C1A,lw=0.7); at(ax_arch,0.9,1.03,'Seq Input\n96×7',c=C1D,fs=3.3,bold=True)
ab(ax_arch, 2.1,0.55,2.2,1.2, fc='#7C3AED',ec='none');      at(ax_arch,3.2,1.03,'BiLSTM\nh=128,2L',c=WHITE,fs=3.3,bold=True)
ab(ax_arch, 4.7,0.55,1.8,1, fc='#F97316',ec='none');      at(ax_arch,5.6,1.03,'Attention\nMulti-Head',c=WHITE,fs=3.2,bold=True)
# forecast heads — all below calib branch bottom (3.7), so max top = 3.60
# Centers: 24h=3.05, 12h=2.35, 6h=1.65, 1h=0.95 — each box height=0.65
for i,(lbl,yp) in enumerate([('24h',3.05),('12h',2.35),('6h',1.65),('1h',0.95)]):
    ab(ax_arch, 7.0,yp-0.32,1.15,0.65, fc=C3H,ec='none',r=0.1)
    at(ax_arch,7.57,yp,lbl,c=WHITE,fs=3.3,bold=True)
    ax_arch.annotate('',xy=(7.0,yp),xytext=(6.5,1.03),
        arrowprops=dict(arrowstyle='->',color=ORG,lw=0.55,mutation_scale=4),zorder=8)
    aarr(ax_arch,8.15,yp,8.55,yp,C3A)
ab(ax_arch,8.55,0.73,1.35,2.64,fc='#C4B5FD',ec=C2A,lw=0.7,r=0.15)
at(ax_arch,9.22,2.05,'Multi\nHorizon\nForecast',c=C2D,fs=3.2,bold=True)
for x1,x2 in [(1.7,2.1),(4.3,4.7),(6.5,7.0)]:
    aarr(ax_arch,x1,1.03,x2,1.03,'#7C3AED')
at(ax_arch,3.5,2,'— Forecasting Branch —',c=C2A,fs=3.2,bold=True)

# Loss function — compact, fits within panel
ab(ax_arch,0.3,-0.25,9.4,0.65,fc='#FEF9C3',ec='#CA8A04',lw=0.5,r=0.1,z=5)
at(ax_arch,4.6,0.05,
   'ℒ = λ_cal·ℒ_Huber(y_cal,ŷ) + λ_frc·ℒ_MSE(y_frc,ŷ)   |   λ = 1.0',
   c='#78350F',fs=3.0,bold=False)

# ════════════════════════════════════════════════════════════════════════════
# RIGHT INSET — Performance Charts (Fig. 2: RMSE + Degradation)
# fig coords: left=0.515, bottom=0.103, width=0.477, height=0.245
# ════════════════════════════════════════════════════════════════════════════

ax_charts = fig.add_axes([0.515, 0.10, 0.477, 0.23])
ax_charts.axis('off')
ax_charts.set_facecolor('#F8FAFF')

# Two sub-panels with clear gap — leave top 14% for caption bar
ax_rmse = ax_charts.inset_axes([0.04, 0.15, 0.43, 0.71])  # slight nudge up
ax_deg  = ax_charts.inset_axes([0.55, 0.15, 0.42, 0.71])

# Caption bar — compact single line, clipped
ax_charts.add_patch(FancyBboxPatch((0, 0.88), 1.0, 0.12,
    transform=ax_charts.transAxes, boxstyle='square,pad=0',
    facecolor=C3H, edgecolor='none', zorder=3, clip_on=True))
ax_charts.text(0.5, 0.937,
    'Fig. 2 — RMSE by Forecast Horizon  &  Degradation Ratio (24h / 1h)',
    transform=ax_charts.transAxes, color=WHITE, fontsize=3.9, fontweight='bold',
    ha='center', va='center', zorder=4, clip_on=True,
    fontfamily=['DejaVu Sans', 'sans-serif'])

# ── Left: Forecast RMSE by Horizon ──────────────────────────────────────────
horizons = [1, 6, 12, 24]
labels   = ['1h','6h','12h','24h']
RMSE_DATA = {
    'Persistence':           [0.27, 0.77, 1.22, 1.70],
    'Moving Avg (6h)':       [0.58, 1.00, 1.28, 1.73],
    'Linear Trend':          [0.27, 1.07, 2.08, 3.77],
    'Rule-Based (EcoIrrigate)': [0.40, 1.85, 3.77, 7.20],
}
COLS_RMSE = ['#93C5FD','#F59E0B','#7C3AED','#B91C1C']
x = np.arange(len(horizons));  bw = 0.18
for i,(method,vals) in enumerate(RMSE_DATA.items()):
    offset = (i - 1.5) * bw
    ax_rmse.bar(x + offset, vals, bw*0.92, color=COLS_RMSE[i], alpha=0.88,
                label=method, edgecolor='none', zorder=3)

ax_rmse.set_xticks(x); ax_rmse.set_xticklabels(labels, fontsize=3.8)
ax_rmse.set_ylabel('RMSE (%)', fontsize=3.8, labelpad=1)
ax_rmse.set_xlabel('Forecast Horizon', fontsize=3.8, labelpad=1)
# In-axes title — stays within bounds (clip_on=True prevents overflow)
ax_rmse.text(0.5, 1.01, 'Forecast RMSE by Horizon',
    transform=ax_rmse.transAxes, fontsize=4.0, fontweight='bold',
    ha='center', va='bottom', clip_on=True, color=DRK,
    fontfamily=['DejaVu Sans', 'sans-serif'])
ax_rmse.tick_params(axis='both', labelsize=3.5, length=1.5, width=0.4)
ax_rmse.spines['top'].set_visible(False)
ax_rmse.spines['right'].set_visible(False)
ax_rmse.spines['left'].set_linewidth(0.4)
ax_rmse.spines['bottom'].set_linewidth(0.4)
ax_rmse.set_facecolor('#FAFBFC')
ax_rmse.yaxis.grid(True, lw=0.3, color='#CBD5E1', zorder=0)
ax_rmse.set_axisbelow(True)
leg = ax_rmse.legend(fontsize=3.2, loc='upper left', framealpha=0.9,
                     edgecolor='#CBD5E1', handlelength=0.8, handletextpad=0.3,
                     borderpad=0.3, labelspacing=0.2)
leg.get_frame().set_linewidth(0.3)

# ── Right: Performance Degradation ──────────────────────────────────────────
deg_methods = ['Persistence','Moving\nAverage','Linear\nTrend','Rule-Based']
deg_vals    = [5.8, 3.1, 12.2, 18.6]
deg_cols    = ['#93C5FD','#F59E0B','#7C3AED','#B91C1C']
bar_x = np.arange(len(deg_methods)) + 1
bars  = ax_deg.bar(bar_x, deg_vals, 0.6, color=deg_cols, alpha=0.88,
                   edgecolor='none', zorder=3)
ax_deg.axhline(1.0, color='#94A3B8', lw=0.7, ls='--', zorder=4)
for bar_, val_ in zip(bars, deg_vals):
    ax_deg.text(bar_.get_x()+bar_.get_width()/2, val_+0.3, f'{val_}×',
        ha='center', va='bottom', fontsize=3.8, fontweight='bold',
        color=bar_.get_facecolor(),
        fontfamily=['Arial','DejaVu Sans','sans-serif'])
ax_deg.set_xticks(bar_x)
ax_deg.set_xticklabels(deg_methods, fontsize=3.5)
ax_deg.set_ylabel('Degradation Ratio\n(24h/1h RMSE)', fontsize=3.5, labelpad=1)
# In-axes title — stays within bounds
ax_deg.text(0.5, 1.01, 'Degradation Over Horizon',
    transform=ax_deg.transAxes, fontsize=4.0, fontweight='bold',
    ha='center', va='bottom', clip_on=True, color=DRK,
    fontfamily=['DejaVu Sans', 'sans-serif'])
ax_deg.tick_params(axis='both', labelsize=3.5, length=1.5, width=0.4)
ax_deg.spines['top'].set_visible(False)
ax_deg.spines['right'].set_visible(False)
ax_deg.spines['left'].set_linewidth(0.4)
ax_deg.spines['bottom'].set_linewidth(0.4)
ax_deg.set_facecolor('#FAFBFC')
ax_deg.yaxis.grid(True, lw=0.3, color='#CBD5E1', zorder=0)
ax_deg.set_axisbelow(True)
ax_deg.set_ylim(0, 23)

# ════════════════════════════════════════════════════════════════════════════
# CONCLUSION BANNER — REDUCED height (COY2=5.5)
# ════════════════════════════════════════════════════════════════════════════
bx(0, COY1, 100, COY2-COY1, fc=NAVY, z=1)
bx(0, COY2-0.2, 100, 0.4, fc=GRN, z=3)

bx(1.0, COY1+0.4, 13.5, 3.2, fc=C3A, ec='none', r=0.5, z=4)
t(8.25, COY1+2.0, 'Conclusion', c=WHITE, fs=5.5, bold=True)

t(57.5, COY1+2.8,
  'Multi-task learning enables simultaneous calibration (R²=0.912) and IoT sensor forecasting,',
  c=WHITE, fs=4.8)
t(57.5, COY1+1.3,
  'outperforming rule-based baselines by 10.9× and generalising across farms — scalable precision irrigation.',
  c=WHITE, fs=4.8)

# ════════════════════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════════════════════
out_png = os.path.join(OUT, 'graphical_abstract.png')
out_pdf = os.path.join(OUT, 'graphical_abstract.pdf')
out_tiff = os.path.join(OUT, 'graphical_abstract.tiff')  # NEW

kw = dict(dpi=DPI, bbox_inches='tight', pad_inches=0.0,
          facecolor=WHITE, edgecolor='none')
fig.savefig(out_png, format='png', **kw)
fig.savefig(out_pdf, format='pdf', **kw)
fig.savefig(out_tiff, format='tiff', pil_kwargs={'compression': 'tiff_lzw'}, **kw)  # NEW
plt.close('all')
print(f'Saved PNG: {out_png}')
print(f'Saved PDF: {out_pdf}')
print(f'Saved TIFF: {out_tiff}')  # NEW