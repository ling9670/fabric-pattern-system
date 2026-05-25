#!/usr/bin/env python3
"""
Weave System  —  Image / Pattern → Weaving Draft
=================================================
Supports:
  • Rigid heddle loom + pickup stick  (beginner home weaving)
  • Multi-shaft loom                  (intermediate)
  • Jacquard / industrial loom        (advanced — WIF export)

Usage (CLI):
    python3 weave_system.py <image> [options]

    python3 weave_system.py cat.png --warp-color 255,245,220 --weft-color 80,40,10
    python3 weave_system.py logo.png --width-in 6 --epi 10 --height-in 8 --ppi 10
    python3 weave_system.py pattern.png --warp 60 --picks 80 --multi-weft 5

Usage (module):
    from weave_system import generate_weave
    result = generate_weave("image.png", warp_count=40, pick_count=50,
                            warp_color=(240,230,200), weft_color=(60,20,10))
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from sklearn.cluster import KMeans
import argparse, os, sys, textwrap, datetime
import warnings
warnings.filterwarnings('ignore')

# ── Label helpers ─────────────────────────────────────────────────────────────
def col_lbl(i):
    return chr(ord('A')+i) if i < 26 else 'A'+chr(ord('A')+i-26)

def row_lbl(i):
    return f"{i+1:02d}"

def _norm(c):
    return tuple(np.clip(np.array(c)/255.0, 0, 1))

# ── Scale calculator ──────────────────────────────────────────────────────────
def scale_info(warp_count, pick_count, epi, ppi):
    return {
        'warp_count': warp_count, 'pick_count': pick_count,
        'epi': epi,  'ppi': ppi,
        'width_in':  round(warp_count / epi, 2),
        'height_in': round(pick_count / ppi, 2),
        'width_cm':  round(warp_count / epi * 2.54, 1),
        'height_cm': round(pick_count / ppi * 2.54, 1),
    }

def threads_for_size(width_in, height_in, epi, ppi):
    return {
        'warp_count': round(width_in * epi),
        'pick_count': round(height_in * ppi),
        'width_in': width_in, 'height_in': height_in,
    }

# ── Image → liftplan ──────────────────────────────────────────────────────────
def image_to_liftplan(image_path, warp_count, pick_count, warp_color, weft_colors):
    """
    liftplan[pick][warp] = 1 → warp UP  (warp colour on surface)
    liftplan[pick][warp] = 0 → warp DOWN (weft colour on surface)
    """
    if isinstance(weft_colors, tuple) and isinstance(weft_colors[0], int):
        weft_colors = [weft_colors]

    img = Image.open(image_path).convert('RGB')
    img = img.resize((warp_count, pick_count), Image.LANCZOS)
    pixels = np.array(img, dtype=float)

    warp_arr = np.array(warp_color, dtype=float)
    weft_arr = np.array(weft_colors, dtype=float)

    pick_colors = []
    liftplan    = np.zeros((pick_count, warp_count), dtype=int)

    for p in range(pick_count):
        row_pixels  = pixels[p]
        weft_dists  = np.array([np.sum((row_pixels - w)**2, axis=1) for w in weft_arr])
        dom_weft    = int(np.argmin(weft_dists.sum(axis=1)))
        pick_colors.append(tuple(int(x) for x in weft_arr[dom_weft]))

        this_weft = weft_arr[dom_weft]
        d_warp = np.sum((row_pixels - warp_arr)**2, axis=1)
        d_weft = np.sum((row_pixels - this_weft)**2, axis=1)
        liftplan[p] = (d_warp < d_weft).astype(int)

    return liftplan, pick_colors

# ── Rigid heddle pickup instructions ─────────────────────────────────────────
def derive_pickup_instructions(liftplan, pick_colors, warp_color):
    """
    Threading: odd columns (0,2,4…) = HOLE threads (heddle)
               even columns (1,3,5…) = SLOT threads (pickup stick)
    """
    pick_count, warp_count = liftplan.shape
    slot_threads = list(range(1, warp_count, 2))

    instructions = []
    for p in range(pick_count):
        row = liftplan[p]
        expected_up   = np.array([1 if i%2==0 else 0 for i in range(warp_count)])
        is_plain_up   = np.array_equal(row, expected_up)
        is_plain_down = np.array_equal(row, 1 - expected_up)

        pickup = [slot_threads[j] for j, t in enumerate(slot_threads)
                  if t < warp_count and row[t] == 1]

        if is_plain_up:
            hedge, pickup_str, note = 'UP', '—', 'Plain weave'
        elif is_plain_down:
            hedge, pickup_str, note = 'DOWN', '—', 'Plain weave'
        elif not pickup:
            hedge, pickup_str, note = 'DOWN', '—', 'Pattern pick (all slots down)'
        else:
            hedge = 'NEUTRAL'
            pickup_str = '  '.join(col_lbl(t) for t in pickup)
            note = f'Pickup {len(pickup)} threads'

        instructions.append({
            'pick': p+1, 'weft_rgb': pick_colors[p],
            'heddle': hedge, 'pickup': pickup_str, 'note': note,
            'raw_up': [col_lbl(i) for i in range(warp_count) if row[i]==1],
        })
    return instructions

# ── Draw drawdown chart ───────────────────────────────────────────────────────
def draw_drawdown(liftplan, warp_color, pick_colors, scale, out_path, title=''):
    pick_count, warp_count = liftplan.shape
    wc = _norm(warp_color)

    fig = plt.figure(figsize=(max(14, warp_count*0.38+4), max(10, pick_count*0.30+4)),
                     facecolor='#0e0e22')
    gs = GridSpec(2, 2, figure=fig,
                  left=0.08, right=0.98, top=0.93, bottom=0.03,
                  hspace=0.10, wspace=0.04,
                  height_ratios=[1, pick_count], width_ratios=[warp_count, 1])

    # Warp colour header
    ax_warp = fig.add_subplot(gs[0, 0])
    ax_warp.set_facecolor('#181830'); ax_warp.axis('off')
    ax_warp.set_xlim(-0.5, warp_count-0.5); ax_warp.set_ylim(0, 1)
    for c in range(warp_count):
        ax_warp.add_patch(plt.Rectangle((c-0.5, 0.05), 1, 0.9,
                                         facecolor=wc, edgecolor='#333', lw=0.4))
        if warp_count <= 52:
            tc = 'black' if sum(wc) > 1.5 else 'white'
            ax_warp.text(c, 0.5, col_lbl(c), ha='center', va='center',
                         fontsize=max(4, 7-warp_count//15),
                         color=tc, fontfamily='monospace', fontweight='bold')
    ax_warp.set_title(
        f'WARP  ({warp_count} threads  |  EPI {scale["epi"]}  |  '
        f'{scale["width_in"]}" / {scale["width_cm"]} cm)',
        fontsize=6.5, color='#ccc', pad=3, fontfamily='monospace')

    # Main drawdown
    ax = fig.add_subplot(gs[1, 0])
    ax.set_facecolor('#f8f6f0')
    ax.set_xlim(-0.5, warp_count-0.5)
    ax.set_ylim(pick_count-0.5, -0.5)
    ax.set_aspect('equal')

    for p in range(pick_count):
        weft_c = _norm(pick_colors[p])
        for w in range(warp_count):
            ax.add_patch(plt.Rectangle((w-0.5, p-0.5), 1, 1,
                                        facecolor=wc if liftplan[p,w]==1 else weft_c,
                                        edgecolor='none'))

    for c in range(warp_count+1):
        lw = 0.6 if c%10==0 else 0.15
        ax.axvline(c-0.5, color='#666' if c%10==0 else '#bbb', lw=lw)
    for p in range(pick_count+1):
        lw = 0.6 if p%10==0 else 0.15
        ax.axhline(p-0.5, color='#666' if p%10==0 else '#bbb', lw=lw)

    step = max(1, warp_count//26)
    ax.set_xticks(range(0, warp_count, step))
    ax.set_xticklabels([col_lbl(i) for i in range(0, warp_count, step)],
                       fontsize=max(4, 6-warp_count//30), fontfamily='monospace', color='#444')
    ax.xaxis.tick_top()
    step_p = max(1, pick_count//40)
    ax.set_yticks(range(0, pick_count, step_p))
    ax.set_yticklabels([row_lbl(i) for i in range(0, pick_count, step_p)],
                       fontsize=max(4, 6-pick_count//50), fontfamily='monospace', color='#444')
    ax.tick_params(length=0, pad=1)

    # Weft colour sidebar
    ax_weft = fig.add_subplot(gs[1, 1])
    ax_weft.set_facecolor('#181830'); ax_weft.axis('off')
    ax_weft.set_xlim(0, 1); ax_weft.set_ylim(pick_count-0.5, -0.5)
    for p in range(pick_count):
        ax_weft.add_patch(plt.Rectangle((0.05, p-0.45), 0.9, 0.9,
                                         facecolor=_norm(pick_colors[p]),
                                         edgecolor='#333', lw=0.3))
    ax_weft.set_title(f'PPI {scale["ppi"]}\n{scale["height_in"]}"',
                       fontsize=5.5, color='#ccc', pad=2, fontfamily='monospace')

    info = (f'{title or "Weave Draft"}   |   {warp_count}W × {pick_count}P   |   '
            f'{scale["width_in"]}" × {scale["height_in"]}"  '
            f'({scale["width_cm"]} × {scale["height_cm"]} cm)   |   '
            f'EPI {scale["epi"]}  PPI {scale["ppi"]}')
    fig.text(0.5, 0.965, info, ha='center', fontsize=7.5,
             color='#ddd', fontfamily='monospace', fontweight='bold')
    fig.text(0.5, 0.952,
             '■ = Warp UP (warp colour on surface)   □ = Weft UP (weft colour on surface)',
             ha='center', fontsize=6, color='#888', fontfamily='monospace')

    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

# ── Draw pickup instruction sheet ─────────────────────────────────────────────
def draw_pickup_sheet(instructions, warp_color, warp_count, scale, out_path, title=''):
    n     = len(instructions)
    row_h = 0.30
    fig_h = max(8, n*row_h + 3.5)
    fig   = plt.figure(figsize=(18, fig_h), facecolor='#0e0e22')
    ax    = fig.add_subplot(111)
    ax.set_facecolor('#0e0e22'); ax.axis('off')
    ax.set_xlim(0, 18); ax.set_ylim(0, fig_h)

    ax.text(9, fig_h-0.5,
            f'PICKUP INSTRUCTION SHEET  —  {title or "Weave Draft"}',
            ha='center', va='top', fontsize=13, color='white',
            fontfamily='serif', fontweight='bold')
    ax.text(9, fig_h-1.0,
            f'Rigid Heddle Loom  |  {warp_count} warp threads  |  '
            f'{scale["width_in"]}" × {scale["height_in"]}"  '
            f'({scale["width_cm"]} × {scale["height_cm"]} cm)  |  '
            f'EPI {scale["epi"]}  PPI {scale["ppi"]}',
            ha='center', va='top', fontsize=7, color='#aaa', fontfamily='monospace')

    hdr_y = fig_h - 1.7
    for x, h in zip([0.5, 1.6, 3.2, 5.3, 7.0],
                    ['PICK', 'WEFT COLOR', 'HEDDLE', 'PICKUP THREADS (col letter)', 'NOTE']):
        ax.text(x, hdr_y, h, va='center', fontsize=7, color='#FFD700',
                fontfamily='monospace', fontweight='bold')
    ax.axhline(hdr_y-0.18, color='#FFD700', lw=0.7, xmin=0.02, xmax=0.98)

    hedge_col = {'UP': '#5adf7a', 'DOWN': '#f07070', 'NEUTRAL': '#a0c8ff'}

    for i, inst in enumerate(instructions):
        y  = hdr_y - 0.22 - i*row_h
        bg = '#1a1a35' if i%2==0 else '#151528'
        ax.add_patch(plt.Rectangle((0.1, y-row_h*0.45), 17.8, row_h*0.9,
                                    facecolor=bg, edgecolor='none'))
        wc = _norm(inst['weft_rgb'])
        ax.text(0.5,  y, f"{inst['pick']:03d}", va='center',
                fontsize=7, color='#ccc', fontfamily='monospace')
        ax.add_patch(plt.Rectangle((1.6, y-0.08), 0.3, 0.17,
                                    facecolor=wc, edgecolor='#555', lw=0.4))
        rgb = inst['weft_rgb']
        ax.text(2.05, y, f"rgb({rgb[0]},{rgb[1]},{rgb[2]})",
                va='center', fontsize=5.5, color='#bbb', fontfamily='monospace')
        ax.text(3.2,  y, inst['heddle'], va='center', fontsize=7,
                color=hedge_col.get(inst['heddle'], '#fff'),
                fontfamily='monospace', fontweight='bold')
        pickup = inst['pickup'][:55] + ('…' if len(inst['pickup'])>55 else '')
        ax.text(5.3,  y, pickup, va='center', fontsize=6, color='#eee',
                fontfamily='monospace')
        ax.text(7.0 + 10, y, inst['note'], va='center', fontsize=5.5,
                color='#888', fontfamily='monospace')

    ax.text(0.5, 0.55, 'HEDDLE:   UP = hole threads rise   |   DOWN = hole threads fall   |   NEUTRAL = use pickup stick only',
            va='center', fontsize=6, color='#888', fontfamily='monospace')

    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

# ── Draw warp setup guide ─────────────────────────────────────────────────────
def draw_warp_setup(warp_color, weft_colors_unique, warp_count, pick_count, scale, out_path, title=''):
    fig = plt.figure(figsize=(16, 7), facecolor='#0e0e22')
    gs  = GridSpec(2, 3, figure=fig,
                   left=0.04, right=0.98, top=0.90, bottom=0.06,
                   hspace=0.5, wspace=0.25)
    fig.text(0.5, 0.96, f'WARP SETUP GUIDE  —  {title or "Weave Draft"}',
             ha='center', fontsize=12, color='white', fontfamily='serif', fontweight='bold')
    wc = _norm(warp_color)

    # Threading strip
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.set_facecolor('#181830'); ax1.axis('off')
    ax1.set_title('WARP THREADING (top→bottom)', fontsize=7, color='#FFD700',
                  fontfamily='monospace', fontweight='bold', pad=4)
    ax1.set_xlim(0, 6); ax1.set_ylim(-0.5, warp_count-0.5)
    for i in range(warp_count):
        ax1.add_patch(plt.Rectangle((1, warp_count-1-i), 3, 0.85,
                                     facecolor=wc, edgecolor='#444', lw=0.3))
        if warp_count <= 60:
            tc = 'black' if sum(wc)>1.5 else 'white'
            ax1.text(2.5, warp_count-0.65-i, col_lbl(i), ha='center', va='center',
                     fontsize=5, color=tc, fontfamily='monospace')
            ax1.text(0.4, warp_count-0.65-i, str(i+1), ha='right', va='center',
                     fontsize=5, color='#888', fontfamily='monospace')
        label = 'H' if i%2==0 else 'S'
        ax1.text(4.8, warp_count-0.65-i, label, ha='center', va='center',
                 fontsize=5, color='#7af' if i%2==0 else '#fa7',
                 fontfamily='monospace', fontweight='bold')
    ax1.text(0.4, warp_count+0.1, '#', ha='right', va='center',
             fontsize=5, color='#666', fontfamily='monospace')
    ax1.text(2.5, warp_count+0.1, 'col', ha='center', va='center',
             fontsize=5, color='#666', fontfamily='monospace')
    ax1.text(4.8, warp_count+0.1, 'H/S', ha='center', va='center',
             fontsize=5, color='#aaa', fontfamily='monospace')

    # Scale calculator
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#181830'); ax2.axis('off')
    ax2.set_title('SCALE CALCULATOR', fontsize=7, color='#FFD700',
                  fontfamily='monospace', fontweight='bold', pad=4)
    for j, (k, v) in enumerate([
        ('Warp threads',   str(warp_count)),
        ('Weft picks',     str(pick_count)),
        ('EPI (ends/in)', str(scale['epi'])),
        ('PPI (picks/in)',str(scale['ppi'])),
        ('─'*20,           '─'*6),
        ('Width',          f'{scale["width_in"]}" / {scale["width_cm"]} cm'),
        ('Height',         f'{scale["height_in"]}" / {scale["height_cm"]} cm'),
    ]):
        y = 0.88 - j*0.12
        ax2.text(0.05, y, k, va='center', fontsize=7, color='#bbb',
                 fontfamily='monospace', transform=ax2.transAxes)
        ax2.text(0.95, y, v, va='center', ha='right', fontsize=7,
                 color='#7af' if j>=5 else '#eee',
                 fontweight='bold' if j>=5 else 'normal',
                 fontfamily='monospace', transform=ax2.transAxes)

    # Materials estimate
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor('#181830'); ax3.axis('off')
    ax3.set_title('MATERIALS ESTIMATE', fontsize=7, color='#FFD700',
                  fontfamily='monospace', fontweight='bold', pad=4)
    warp_yd = round((scale['height_in']*1.4 + 12) * warp_count / 36, 1)
    weft_yd = round(scale['width_in'] * 1.1 * pick_count / 36, 1)
    for j, (k, v) in enumerate([
        ('Warp yarn',  f'{warp_yd} yd  ({warp_count} threads)'),
        ('Weft yarn',  f'{weft_yd} yd  ({pick_count} picks)'),
        ('Loom waste', '~12" per warp thread'),
        ('Take-up',    '~10% width reduction'),
        ('Tip',        'Add 20% extra for samples'),
    ]):
        y = 0.88 - j*0.18
        ax3.text(0.05, y,      k+':', va='center', fontsize=6.5, color='#bbb',
                 fontfamily='monospace', transform=ax3.transAxes)
        ax3.text(0.05, y-0.08, v,    va='center', fontsize=6.5, color='#eee',
                 fontfamily='monospace', transform=ax3.transAxes)

    # Colour key
    ax4 = fig.add_subplot(gs[:, 2])
    ax4.set_facecolor('#181830'); ax4.axis('off')
    ax4.set_title('COLOUR KEY', fontsize=7, color='#FFD700',
                  fontfamily='monospace', fontweight='bold', pad=4)
    ax4.set_xlim(0, 10); ax4.set_ylim(0, len(weft_colors_unique)+2)
    ax4.text(5, len(weft_colors_unique)+1.5, 'WARP', ha='center',
             fontsize=6.5, color='#7af', fontfamily='monospace', fontweight='bold')
    ax4.add_patch(plt.Rectangle((0.5, len(weft_colors_unique)+0.7), 9, 0.6,
                                 facecolor=wc, edgecolor='#555', lw=0.6))
    ax4.text(5, len(weft_colors_unique)+1.0,
             f'rgb({warp_color[0]},{warp_color[1]},{warp_color[2]})',
             ha='center', va='center', fontsize=6,
             color='black' if sum(wc)>1.5 else 'white', fontfamily='monospace')
    ax4.text(5, len(weft_colors_unique)+0.4, 'WEFT', ha='center',
             fontsize=6.5, color='#fa7', fontfamily='monospace', fontweight='bold')
    for j, weft_c in enumerate(weft_colors_unique):
        y  = len(weft_colors_unique) - j - 0.3
        fc = _norm(weft_c)
        ax4.add_patch(plt.Rectangle((0.5, y-0.25), 9, 0.5,
                                     facecolor=fc, edgecolor='#555', lw=0.6))
        tc = 'black' if sum(fc)>1.5 else 'white'
        ax4.text(5, y, f'rgb({weft_c[0]},{weft_c[1]},{weft_c[2]})',
                 ha='center', va='center', fontsize=6, color=tc, fontfamily='monospace')

    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

# ── Export WIF ────────────────────────────────────────────────────────────────
def export_wif(liftplan, warp_color, pick_colors, scale, out_path, title=''):
    """Standard WIF 1.1 — compatible with PixeLoom, WeavePoint, WeaveMaker, AVL, TC2."""
    pick_count, warp_count = liftplan.shape
    unique_wefts = list({tuple(c) for c in pick_colors})
    weft_to_idx  = {c: i+2 for i, c in enumerate(unique_wefts)}
    def pct(v): return f"{round(v/255*100, 1)}"

    lines = [
        '[WIF]', 'Version=1.1',
        f'Date={datetime.datetime.now().strftime("%Y-%m-%d")}',
        'Developers=https://github.com/ling9670/fabric-pattern-system',
        'Source Program=Fabric Pattern System', 'Source Version=2.0', '',
        '[CONTENTS]',
        'COLOR PALETTE=true', 'COLOR TABLE=true',
        'WARP=true', 'WEFT=true', 'THREADING=true', 'LIFTPLAN=true', '',
        '[COLOR PALETTE]', f'Entries={1+len(unique_wefts)}', 'Form=RGB', 'Unit=Percent', '',
        '[COLOR TABLE]',
        f'1={pct(warp_color[0])},{pct(warp_color[1])},{pct(warp_color[2])}',
    ]
    for c, idx in weft_to_idx.items():
        lines.append(f'{idx}={pct(c[0])},{pct(c[1])},{pct(c[2])}')
    lines += [
        '', '[WARP]', f'Threads={warp_count}', 'Units=Inches',
        f'Units per dent={scale["epi"]}', 'Color=1',
        '', '[WEFT]', f'Threads={pick_count}', 'Units=Inches',
        f'Units per dent={scale["ppi"]}',
        '', '[THREADING]',
    ]
    for i in range(warp_count):
        lines.append(f'{i+1}={i+1}')
    lines += ['', '[LIFTPLAN]']
    for p in range(pick_count):
        raised = [str(w+1) for w in range(warp_count) if liftplan[p,w]==1]
        lines.append(f'{p+1}={",".join(raised) if raised else "0"}')
    lines += ['', '[WEFT COLORS]']
    for p, c in enumerate(pick_colors):
        lines.append(f'{p+1}={weft_to_idx[tuple(c)]}')

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))

# ── Main generate ─────────────────────────────────────────────────────────────
def generate_weave(image_path,
                   warp_count=40, pick_count=50,
                   warp_color=(240,230,200), weft_color=(60,30,10),
                   n_weft_colors=1, epi=10, ppi=10,
                   output_prefix=None, title=''):
    """
    Full pipeline: image → drawdown + pickup sheet + warp setup + WIF.

    Size-first workflow:
        t = threads_for_size(width_in=6, height_in=8, epi=10, ppi=10)
        generate_weave("img.png", warp_count=t['warp_count'], pick_count=t['pick_count'])
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    if output_prefix is None:
        base    = os.path.splitext(os.path.basename(image_path))[0]
        out_dir = os.path.dirname(os.path.abspath(image_path))
        output_prefix = os.path.join(out_dir, f"{base}_weave")

    if isinstance(weft_color, list):
        weft_colors = weft_color
    elif n_weft_colors > 1:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((warp_count, pick_count), Image.LANCZOS)
        px  = np.array(img).reshape(-1, 3).astype(float)
        km  = KMeans(n_clusters=n_weft_colors, random_state=42, n_init=10)
        km.fit(px)
        weft_colors = [tuple(int(v) for v in c) for c in km.cluster_centers_]
    else:
        weft_colors = [weft_color]

    scale = scale_info(warp_count, pick_count, epi, ppi)
    print(f"Processing: {image_path}")
    print(f"Grid: {warp_count}W × {pick_count}P  |  "
          f'{scale["width_in"]}" × {scale["height_in"]}"  '
          f'({scale["width_cm"]} × {scale["height_cm"]} cm)  |  '
          f'EPI {epi}  PPI {ppi}  |  Weft colours: {len(weft_colors)}')

    liftplan, pick_colors = image_to_liftplan(
        image_path, warp_count, pick_count, warp_color, weft_colors)
    instructions  = derive_pickup_instructions(liftplan, pick_colors, warp_color)
    unique_wefts  = list({tuple(c) for c in pick_colors})

    paths = {
        'drawdown': output_prefix + '_drawdown.png',
        'pickup':   output_prefix + '_pickup_sheet.png',
        'setup':    output_prefix + '_warp_setup.png',
        'wif':      output_prefix + '.wif',
    }
    draw_drawdown(liftplan, warp_color, pick_colors, scale, paths['drawdown'], title)
    draw_pickup_sheet(instructions, warp_color, warp_count, scale, paths['pickup'], title)
    draw_warp_setup(warp_color, unique_wefts, warp_count, pick_count, scale, paths['setup'], title)
    export_wif(liftplan, warp_color, pick_colors, scale, paths['wif'], title)

    for k, v in paths.items():
        print(f"  {k:<12} {v}")
    return paths

# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Weave System — Image to Weaving Draft',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python3 weave_system.py cat.png --warp 40 --picks 50 --warp-color 240,230,200 --weft-color 60,30,10
          python3 weave_system.py logo.png --width-in 6 --height-in 8 --epi 10 --ppi 10
          python3 weave_system.py photo.jpg --warp 80 --picks 100 --multi-weft 6 --epi 12
        """))
    parser.add_argument('image')
    parser.add_argument('--warp',       type=int,   default=None)
    parser.add_argument('--picks',      type=int,   default=None)
    parser.add_argument('--width-in',   type=float, default=None)
    parser.add_argument('--height-in',  type=float, default=None)
    parser.add_argument('--epi',        type=int,   default=10)
    parser.add_argument('--ppi',        type=int,   default=10)
    parser.add_argument('--warp-color', type=str,   default='240,230,200')
    parser.add_argument('--weft-color', type=str,   default='60,30,10')
    parser.add_argument('--multi-weft', type=int,   default=1)
    parser.add_argument('--name',       type=str,   default=None)
    parser.add_argument('--title',      type=str,   default='')
    args = parser.parse_args()

    warp_rgb = tuple(int(x) for x in args.warp_color.split(','))
    weft_rgb = tuple(int(x) for x in args.weft_color.split(','))

    if args.width_in and args.height_in:
        t = threads_for_size(args.width_in, args.height_in, args.epi, args.ppi)
        warp_count, pick_count = t['warp_count'], t['pick_count']
        print(f"Size {args.width_in}\" × {args.height_in}\" → {warp_count}W × {pick_count}P")
    else:
        warp_count = args.warp or 40
        pick_count = args.picks or 50

    generate_weave(image_path=args.image, warp_count=warp_count, pick_count=pick_count,
                   warp_color=warp_rgb, weft_color=weft_rgb,
                   n_weft_colors=args.multi_weft, epi=args.epi, ppi=args.ppi,
                   output_prefix=args.name, title=args.title)
