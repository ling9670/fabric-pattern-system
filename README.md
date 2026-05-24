# Fabric Pattern System

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=ling9670/fabric-pattern-system&branch=main&mainModule=app.py)

**Image → Cross-Stitch Fabric Pattern Generator**

Upload any image (photo, drawing, pixel art) and get a print-ready cross-stitch chart with thread coordinates, DMC color mapping, and a tiled fabric preview.

---

## How It Works

```
Your Image  →  Resize to grid  →  Color quantize (K-means)  →  DMC mapping  →  3 outputs
```

Each cell in the output grid = one cross-stitch (X) on fabric.
Coordinates use the system: **column letter + row number** (e.g. `A01`, `B12`, `AF24`).

---

## Outputs

| File | Description |
|------|-------------|
| `_chart.png` | Graph-paper grid with X marks, column/row labels (A–AZ × 01–NN), DMC thread legend |
| `_coordinates.txt` | Every stitch listed by coordinate grouped by thread color + DMC number |
| `_preview.png` | 3×2 tiled fabric preview with configurable gap between repeats |

---

## Quick Start

```bash
# Install dependencies
pip install pillow scikit-learn matplotlib numpy

# Basic usage
python3 fabric_system.py your_image.png

# With options
python3 fabric_system.py photo.jpg --width 50 --height 60 --colors 10 --bg white
python3 fabric_system.py logo.png --bg none --colors 6 --gap 2
python3 fabric_system.py drawing.png --bg 255,255,255 --colors 8 --title "My Pattern"
```

---

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--width` | 40 | Grid width in cells |
| `--height` | 40 | Grid height in cells |
| `--colors` | 8 | Number of thread colors (max 20) |
| `--bg` | white | Background: `white` / `none` / `r,g,b` |
| `--gap` | 1 | Gap cells between tile repeats in preview |
| `--name` | auto | Output file prefix |
| `--title` | auto | Chart title |

---

## Use as a Module

```python
from fabric_system import generate

result = generate(
    image_path = "cat.png",
    grid_w     = 40,
    grid_h     = 40,
    n_colors   = 8,
    bg_color   = "white",
    gap        = 1,
    title      = "Siamese Cat"
)

# result = {
#   'chart':       'cat_fabric_chart.png',
#   'coordinates': 'cat_fabric_coordinates.txt',
#   'preview':     'cat_fabric_preview.png'
# }
```

---

## Coordinate System

```
     A    B    C    D  ...  Z    AA   AB  ...
01  [ ]  [X]  [ ]  [X]     [ ]  [X]  [ ]
02  [X]  [ ]  [X]  [ ]     [X]  [ ]  [X]
03  [ ]  [X]  [ ]  [X]     [ ]  [X]  [ ]
...
```

- Columns: A → Z, then AA → AZ (up to 52 columns)
- Rows: 01 → 99
- Each `X` = one cross-stitch in the assigned DMC thread color

---

## DMC Color Mapping

The system includes ~120 DMC thread colors. For each K-means cluster (color region in your image), it finds the nearest DMC color by Euclidean distance in RGB space.

---

## Requirements

```
pillow
scikit-learn
matplotlib
numpy
```

---

## Example

Input: pixel art cat  
Grid: 30×36, 7 colors, white background removed

Output chart shows 7 DMC thread colors, each cell labeled with X in the correct color, full coordinate reference for stitching.

---

## Business Context

This system is part of a fabric-making business that converts any visual — mathematical patterns, drawings, photos — into thread-by-thread fabric instructions. Like sheet music for textiles.

Workflow: `Claude (pattern code) → this tool → Canva AI (recolor) → Spoonflower (print/sell) → Etsy`
