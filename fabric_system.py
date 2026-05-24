#!/usr/bin/env python3
"""
Fabric Pattern System  —  Image → Cross-Stitch Grid
====================================================
Usage:
    python3 fabric_system.py <image_path> [--width 40] [--height 40] [--colors 8] [--bg white]

Output (saved next to input image):
    {name}_chart.png        Cross-stitch grid chart with X marks + legend
    {name}_coordinates.txt  Coordinate list by thread color
    {name}_preview.png      Tiled fabric preview (3×2)

Options:
    --width   INT   Grid width  in cells  (default 40)
    --height  INT   Grid height in cells  (default 40)
    --colors  INT   Max thread colors     (default 8, max 20)
    --bg      STR   Background: 'white' | 'none' | 'r,g,b'  (default white)
    --name    STR   Output file prefix    (default: derived from image name)
    --gap     INT   Tile gap in cells for preview  (default 1)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from sklearn.cluster import KMeans
import argparse, os, sys, textwrap
import warnings
warnings.filterwarnings('ignore')

# ── DMC Thread Palette (code → (R, G, B, name)) ──────────────────────────────
DMC = {
    'blanc': (255,255,255,'White'),         'ecru':  (240,234,218,'Ecru'),
    '310':   (35, 35, 35, 'Black'),         '318':   (171,171,171,'Light Steel Gray'),
    '321':   (196,29, 52, 'Christmas Red'), '347':   (179,20, 38, 'Very Dark Salmon'),
    '349':   (204,49, 54, 'Dark Coral'),    '350':   (220,80, 75, 'Medium Coral'),
    '351':   (226,109,93, 'Coral'),         '352':   (244,148,128,'Light Coral'),
    '353':   (253,193,176,'Peach'),         '402':   (254,189,142,'Very Light Mahogany'),
    '413':   (89, 89, 98, 'Dark Pewter'),   '433':   (106,62, 22, 'Medium Brown'),
    '434':   (133,82, 32, 'Light Brown'),   '435':   (160,100,40, 'Very Light Brown'),
    '436':   (175,116,55, 'Tan'),           '437':   (205,158,96, 'Light Tan'),
    '444':   (255,208,0,  'Dark Lemon'),    '445':   (255,240,128,'Light Lemon'),
    '469':   (82, 104,43, 'Avocado Green'), '471':   (149,166,87, 'Lt Avocado'),
    '498':   (172,10, 45, 'Dk Christmas Red'),'500': (29, 73, 40, 'Dk Blue Green'),
    '501':   (53, 104,62, 'Dk Blue Green 2'),'503': (131,170,139,'Med Blue Green'),
    '550':   (89, 21, 107,'Very Dk Violet'),'552':  (136,55, 156,'Med Violet'),
    '554':   (198,150,210,'Light Violet'),  '600':   (204,0,  100,'Very Dk Cranberry'),
    '602':   (221,40, 115,'Med Cranberry'), '604':   (244,137,171,'Light Cranberry'),
    '606':   (255,50, 0,  'Bright Orange'), '666':   (228,0,  50, 'Bright Red'),
    '700':   (3,  127,39, 'Bright Green'),  '702':   (53, 169,71, 'Kelly Green'),
    '704':   (122,204,118,'Bright Chartreuse'),'712':(255,252,231,'Cream'),
    '718':   (163,28, 120,'Plum'),          '720':   (229,91, 18, 'Dark Orange Spice'),
    '721':   (241,119,56, 'Med Orange Spice'),'722':(247,152,98, 'Lt Orange Spice'),
    '725':   (255,197,68, 'Topaz'),         '726':   (255,216,101,'Light Topaz'),
    '740':   (255,130,0,  'Tangerine'),     '741':   (255,156,0,  'Med Tangerine'),
    '742':   (255,178,54, 'Light Tangerine'),'743':  (254,207,88, 'Med Yellow'),
    '744':   (254,224,129,'Pale Yellow'),   '745':   (254,237,175,'Lt Pale Yellow'),
    '762':   (224,224,224,'Very Lt Pearl Gray'),'775':(207,231,245,'Very Lt Baby Blue'),
    '796':   (18, 68, 142,'Dk Royal Blue'), '797':   (24, 86, 174,'Royal Blue'),
    '798':   (65, 116,183,'Dk Delft Blue'), '809':   (101,152,202,'Delft Blue'),
    '817':   (193,19, 41, 'Very Dk Coral'), '819':   (255,222,222,'Lt Baby Pink'),
    '820':   (19, 53, 122,'Very Dk Royal Blue'),'827':(191,223,242,'Very Lt Blue'),
    '838':   (77, 52, 24, 'Very Dk Beige Brown'),'839':(104,74,36,'Dk Beige Brown'),
    '840':   (140,103,58, 'Med Beige Brown'),'841':  (177,143,100,'Lt Beige Brown'),
    '894':   (248,141,165,'Very Lt Carnation'),'895':(28,72,38,  'Very Dk Hunter Green'),
    '898':   (60, 22, 5,  'Very Dk Coffee Brown'),'900':(210,66,0,'Dk Burnt Orange'),
    '907':   (163,213,93, 'Lt Parrot Green'),'909': (27, 115,69, 'Very Dk Emerald'),
    '911':   (43, 143,86, 'Med Emerald'),   '913':   (119,183,126,'Med Nile Green'),
    '946':   (240,99, 0,  'Med Burnt Orange'),'947' :(255,130,42, 'Burnt Orange'),
    '956':   (255,122,125,'Geranium'),      '957':   (255,181,183,'Pale Geranium'),
    '963':   (255,215,213,'Ultra Lt Dusty Rose'),'996':(0,188,227,'Med Electric Blue'),
    '3325':  (176,210,235,'Light Baby Blue'),'3340': (255,113,82, 'Med Apricot'),
    '3341':  (255,156,132,'Apricot'),       '3345':  (50, 95, 22, 'Dk Hunter Green'),
    '3347':  (112,153,76, 'Med Yellow Green'),'3348':(193,214,151,'Lt Yellow Green'),
    '3607':  (196,45, 129,'Light Plum'),    '3608':  (229,133,181,'Very Lt Plum'),
    '3712':  (246,108,100,'Med Salmon'),    '3713':  (255,220,214,'Very Lt Salmon'),
    '3716':  (255,182,185,'Very Lt Dusty Rose'),'3746':(108,93,167,'Dk Blue Violet'),
    '3747':  (200,198,230,'Very Lt Blue Violet'),'3750':(28,56,100,'Very Dk Antique Blue'),
    '3755':  (118,167,211,'Baby Blue'),     '3760':  (42, 107,152,'Med Wedgwood'),
    '3801':  (227,40, 73, 'Very Dk Melon'),'3820':   (222,172,56, 'Dark Straw'),
    '3821':  (243,198,88, 'Straw'),         '3822':  (249,220,144,'Light Straw'),
    '3823':  (255,253,225,'Ultra Pale Yellow'),'3825':(255,183,116,'Pale Pumpkin'),
    '3826':  (179,113,44, 'Golden Brown'),  '3827':  (248,193,122,'Pale Golden Brown'),
    '3854':  (247,150,78, 'Med Autumn Gold'),'3855': (252,199,134,'Lt Autumn Gold'),
    '3862':  (111,82, 50, 'Dk Mocha Beige'),'3863': (148,113,76, 'Med Mocha Beige'),
    '3864':  (196,170,137,'Lt Mocha Beige'),'3865': (250,250,246,'Winter White'),
}

# Symbols for each color slot (for B&W readability)
SYMBOLS = ['■','●','▲','◆','★','▼','◉','⊕','⊗','✦','◈','⊞','⊠','◎','❖','✿','⊡','◑','◐','◒']

# ── Helper: column/row label ──────────────────────────────────────────────────
def col_lbl(i):
    return chr(ord('A')+i) if i < 26 else 'A'+chr(ord('A')+i-26)

def row_lbl(i):
    return f"{i+1:02d}"

# ── Helper: nearest DMC color ─────────────────────────────────────────────────
_dmc_array = None
_dmc_keys  = None

def _build_dmc_index():
    global _dmc_array, _dmc_keys
    _dmc_keys  = list(DMC.keys())
    _dmc_array = np.array([[DMC[k][0], DMC[k][1], DMC[k][2]] for k in _dmc_keys], dtype=float)

def nearest_dmc(rgb):
    if _dmc_array is None:
        _build_dmc_index()
    diff = _dmc_array - np.array(rgb, dtype=float)
    return _dmc_keys[int(np.argmin((diff**2).sum(axis=1)))]

# ── Draw X cross in a cell ────────────────────────────────────────────────────
def draw_x(ax, col, row, color, lw=1.3, margin=0.36):
    x0, x1 = col - margin, col + margin
    y0, y1 = row - margin, row + margin
    ax.plot([x0, x1], [y0, y1], color=color, lw=lw, solid_capstyle='round')
    ax.plot([x1, x0], [y0, y1], color=color, lw=lw, solid_capstyle='round')

# ── Core: image → label grid ──────────────────────────────────────────────────
def image_to_label_grid(image_path, grid_w, grid_h, n_colors, bg_color):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((grid_w, grid_h), Image.LANCZOS)
    pixels = np.array(img)

    # Background mask
    if bg_color == 'white':
        bg_mask = np.all(pixels > 215, axis=2)
    elif bg_color == 'none':
        bg_mask = np.zeros((grid_h, grid_w), dtype=bool)
    else:
        r, g, b = [int(x) for x in bg_color.split(',')]
        diff = np.abs(pixels.astype(int) - np.array([r,g,b]))
        bg_mask = np.all(diff < 35, axis=2)

    fg_pixels = pixels[~bg_mask]
    if len(fg_pixels) == 0:
        raise ValueError("No foreground pixels — try --bg none")

    n_colors = min(n_colors, len(np.unique(fg_pixels.reshape(-1,3), axis=0)), 20)
    km = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    km.fit(fg_pixels)

    label_grid = np.full((grid_h, grid_w), -1, dtype=int)
    fg_flat    = label_grid.ravel()
    fg_idx     = np.where(~bg_mask.ravel())[0]
    fg_flat[fg_idx] = km.labels_
    label_grid = fg_flat.reshape(grid_h, grid_w)

    centers = km.cluster_centers_.astype(int)
    return label_grid, centers, n_colors

# ── Draw chart ────────────────────────────────────────────────────────────────
def draw_chart(label_grid, centers, n_colors, grid_w, grid_h, out_path, title=""):
    dmc_codes  = [nearest_dmc(tuple(c)) for c in centers]
    dmc_rgb_f  = [(DMC[cd][0]/255, DMC[cd][1]/255, DMC[cd][2]/255) for cd in dmc_codes]

    # Sort clusters by count for legend
    coord_by = {i: [] for i in range(n_colors)}
    for r in range(grid_h):
        for c in range(grid_w):
            lbl = label_grid[r,c]
            if lbl >= 0:
                coord_by[lbl].append((r,c))
    sorted_c = sorted(range(n_colors), key=lambda i: -len(coord_by[i]))

    # Figure
    fig = plt.figure(figsize=(20, 14), facecolor='#0e0e22')
    gs  = GridSpec(1, 2, figure=fig,
                   left=0.05, right=0.98, top=0.93, bottom=0.03,
                   wspace=0.06, width_ratios=[3.2, 1])

    # ── Grid panel ──
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor('#f8f6f0')   # graph-paper cream
    ax.set_xlim(-0.5, grid_w-0.5)
    ax.set_ylim(grid_h-0.5, -0.5)
    ax.set_aspect('equal')

    # Graph-paper grid lines
    for c in range(grid_w):
        lw = 0.7 if c % 10 == 0 else 0.2
        col_c = '#888' if c % 10 == 0 else '#c8c4be'
        ax.axvline(c-0.5, color=col_c, lw=lw)
    ax.axvline(grid_w-0.5, color='#888', lw=0.7)
    for r in range(grid_h):
        lw = 0.7 if r % 10 == 0 else 0.2
        col_c = '#888' if r % 10 == 0 else '#c8c4be'
        ax.axhline(r-0.5, color=col_c, lw=lw)
    ax.axhline(grid_h-0.5, color='#888', lw=0.7)

    # Draw X marks
    for r in range(grid_h):
        for c in range(grid_w):
            lbl = label_grid[r,c]
            if lbl >= 0:
                draw_x(ax, c, r, dmc_rgb_f[lbl], lw=1.1)

    # Axis labels
    ax.set_xticks(range(grid_w))
    ax.set_xticklabels([col_lbl(i) for i in range(grid_w)],
                       fontsize=5.5, fontfamily='monospace', color='#333')
    ax.xaxis.tick_top()
    ax.set_yticks(range(grid_h))
    ax.set_yticklabels([row_lbl(i) for i in range(grid_h)],
                       fontsize=5.5, fontfamily='monospace', color='#333')
    ax.tick_params(length=0, pad=1.5)
    for sp in ax.spines.values():
        sp.set_edgecolor('#666'); sp.set_linewidth(0.8)

    # Title
    ttl = title or out_path.split('/')[-1].replace('_chart.png','')
    colored = int(np.sum(label_grid >= 0))
    ax.set_title(f"{ttl}   |   {colored}/{grid_w*grid_h} cells   |   {n_colors} colors   |   {grid_w}×{grid_h} grid",
                 fontsize=7, color='#ccc', pad=8, fontfamily='monospace', fontweight='bold')

    # ── Legend panel ──
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor('#181830'); ax2.axis('off')
    ax2.set_xlim(0,10); ax2.set_ylim(0, n_colors+3)

    ax2.text(5, n_colors+2.5, "THREAD KEY", ha='center', va='center',
             fontsize=9, color='#FFD700', fontfamily='monospace', fontweight='bold')
    ax2.text(5, n_colors+1.9, "DMC  /  Color name  /  Stitch count",
             ha='center', va='center', fontsize=6, color='#888', fontfamily='monospace')

    for rank, idx in enumerate(sorted_c):
        y  = n_colors - rank - 0.5
        fc = dmc_rgb_f[idx]
        code = dmc_codes[idx]
        name = DMC[code][3]
        cnt  = len(coord_by[idx])
        sym  = SYMBOLS[rank % len(SYMBOLS)]

        # Color swatch
        ax2.add_patch(plt.Rectangle((0.3, y-0.32), 1.2, 0.64,
                                     facecolor=fc, edgecolor='#555', lw=0.7))
        # Symbol
        tc = 'black' if sum(fc) > 1.8 else 'white'
        ax2.text(0.9, y, sym, ha='center', va='center',
                 fontsize=7, color=tc)
        # DMC code
        ax2.text(1.8, y+0.15, f"DMC {code}", va='center',
                 fontsize=6.5, color='#eee', fontfamily='monospace', fontweight='bold')
        # Color name
        ax2.text(1.8, y-0.15, name[:22], va='center',
                 fontsize=5.5, color='#bbb', fontfamily='monospace')
        # Count
        ax2.text(9.5, y, f"{cnt}x", ha='right', va='center',
                 fontsize=6.5, color='#aaa', fontfamily='monospace')

    fig.text(0.5, 0.005,
             "Fabric Pattern System  |  Each X = one cross-stitch  |  Coordinate: col-letter + row-number",
             ha='center', fontsize=6, color='#555', fontfamily='monospace')

    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    return dmc_codes, dmc_rgb_f, coord_by, sorted_c

# ── Write coordinates text ────────────────────────────────────────────────────
def write_coordinates(coord_by, sorted_c, dmc_codes, dmc_rgb_f, centers, out_path):
    lines = [
        "FABRIC PATTERN — COORDINATE REFERENCE",
        "=" * 60,
        "Format:  COL-ROW  (e.g. A01 = column A, row 1)",
        "Each coordinate = one cross-stitch (X) in that thread color.",
        "=" * 60, ""
    ]
    for rank, idx in enumerate(sorted_c):
        code = dmc_codes[idx]
        name = DMC[code][3]
        coords = coord_by[idx]
        r_vals, g_vals, b_vals = centers[idx]
        sym = SYMBOLS[rank % len(SYMBOLS)]
        lines += [
            f"COLOR {rank+1}  {sym}",
            f"  DMC #{code}  —  {name}",
            f"  RGB: ({r_vals}, {g_vals}, {b_vals})",
            f"  Stitches: {len(coords)}",
            "  Coordinates:"
        ]
        # Group into rows of 10 per line
        cell_strs = [f"{col_lbl(c)}{row_lbl(r)}" for (r,c) in sorted(coords, key=lambda x:(x[0],x[1]))]
        for i in range(0, len(cell_strs), 10):
            lines.append("    " + "  ".join(cell_strs[i:i+10]))
        lines.append("")

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

# ── Draw tiled preview ────────────────────────────────────────────────────────
def draw_preview(label_grid, dmc_rgb_f, grid_w, grid_h, gap, out_path):
    # Render pixel image
    img_arr = np.ones((grid_h, grid_w, 3))
    for r in range(grid_h):
        for c in range(grid_w):
            lbl = label_grid[r,c]
            if lbl >= 0:
                img_arr[r,c] = dmc_rgb_f[lbl]

    # Tile with gap
    pad_h = np.ones((gap, grid_w, 3))
    pad_v = np.ones((grid_h+gap, gap, 3))
    tile  = np.vstack([img_arr, pad_h])          # add bottom gap row
    tile  = np.hstack([tile, np.ones((grid_h+gap, gap, 3))])  # add right gap col
    row3  = np.hstack([tile, tile, tile])
    tiled = np.vstack([row3, row3])

    fig, ax = plt.subplots(figsize=(14, 10), facecolor='#0e0e22')
    ax.imshow(np.clip(tiled, 0, 1), interpolation='nearest', aspect='auto')
    ax.set_title(f"Tiled Fabric Preview  (3×2 repeats, {gap}-cell gap between tiles)",
                 fontsize=9, color='#bbb', pad=6, fontfamily='monospace')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

# ── Main entry point ──────────────────────────────────────────────────────────
def generate(image_path, grid_w=40, grid_h=40, n_colors=8,
             bg_color='white', output_prefix=None, gap=1, title=""):
    """
    Full pipeline: image → chart PNG + coordinates TXT + preview PNG.
    Returns dict with output file paths.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if output_prefix is None:
        base    = os.path.splitext(os.path.basename(image_path))[0]
        out_dir = os.path.dirname(os.path.abspath(image_path))
        output_prefix = os.path.join(out_dir, f"{base}_fabric")

    print(f"Processing: {image_path}")
    print(f"Grid: {grid_w}×{grid_h}  Colors: {n_colors}  BG: {bg_color}")

    label_grid, centers, n_colors = image_to_label_grid(
        image_path, grid_w, grid_h, n_colors, bg_color)

    chart_path = output_prefix + "_chart.png"
    coord_path = output_prefix + "_coordinates.txt"
    prev_path  = output_prefix + "_preview.png"

    dmc_codes, dmc_rgb_f, coord_by, sorted_c = draw_chart(
        label_grid, centers, n_colors, grid_w, grid_h, chart_path,
        title or os.path.basename(output_prefix))

    write_coordinates(coord_by, sorted_c, dmc_codes, dmc_rgb_f, centers, coord_path)

    draw_preview(label_grid, dmc_rgb_f, grid_w, grid_h, gap, prev_path)

    print(f"  Chart:       {chart_path}")
    print(f"  Coordinates: {coord_path}")
    print(f"  Preview:     {prev_path}")
    return {'chart': chart_path, 'coordinates': coord_path, 'preview': prev_path}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fabric Pattern System — Image to Cross-Stitch Grid',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python3 fabric_system.py cat.png
          python3 fabric_system.py building.jpg --width 50 --height 60 --colors 12 --bg none
          python3 fabric_system.py logo.png --bg 255,255,255 --colors 6 --name my_pattern
        """))
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('--width',  type=int,   default=40,      help='Grid width in cells')
    parser.add_argument('--height', type=int,   default=40,      help='Grid height in cells')
    parser.add_argument('--colors', type=int,   default=8,       help='Number of thread colors')
    parser.add_argument('--bg',     type=str,   default='white', help='Background: white | none | r,g,b')
    parser.add_argument('--name',   type=str,   default=None,    help='Output file prefix')
    parser.add_argument('--gap',    type=int,   default=1,       help='Tile gap in cells')
    parser.add_argument('--title',  type=str,   default='',      help='Chart title')
    args = parser.parse_args()

    generate(
        image_path    = args.image,
        grid_w        = args.width,
        grid_h        = args.height,
        n_colors      = args.colors,
        bg_color      = args.bg,
        output_prefix = args.name,
        gap           = args.gap,
        title         = args.title,
    )
