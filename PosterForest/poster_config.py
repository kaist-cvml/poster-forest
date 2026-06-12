# PosterForest/poster_config.py
# Single source of truth for all layout and typography constants.
# Import from here in main.py, pipeline.py, step_01, step_04, and debug_pipeline.py.

UNITS_PER_INCH = 25

# PDF rendering resolution scale (docling images_scale).
# Higher = sharper figures on poster; 5.0 gives ~360 DPI equivalent.
IMAGE_RESOLUTION_SCALE = 5.0

TITLE_HEIGHT_RATIO = 0.14

# Title bar heights (in layout units)
# actual textbox height = bar_height - SHRINK_MARGIN(6)
SECTION_TITLE_HEIGHT = 36     # → textbox 30u (46pt × 1.35 lh / 72 × 25 ≈ 22u text + 8u padding)
SUBSECTION_TITLE_HEIGHT = 28  # → textbox 22u (40pt × 1.35 lh / 72 × 25 ≈ 19u text + 3u padding)

# Font sizes (pt)
TITLE_FONT_SIZE = 60
TITLE_FONT_SIZE_2 = 40
SECTION_TITLE_FONT_SIZE = 48      # colored background bar → prominent
SUBSECTION_TITLE_FONT_SIZE = 42   # bold navy text → 6pt below section
CONT_FONT_SIZE = 36               # body bullets → readable on poster

SHRINK_MARGIN = 6
FONT_NAME = "Arial"
MAX_ATTEMPT = 2
# MAX_ATTEMPT = 1

# Bottom margin: empty space reserved below all content panels (in layout units).
# 10 units ≈ 0.4 in ≈ 0.75 lines at 36pt — prevents content from reaching the
# very edge of the slide and looks cleaner on printed/displayed posters.
BOTTOM_MARGIN = 6

# Gap inserted between sibling subsections (horizontal split, group_depth >= 1).
# Added at the top of the lower subsection so the upper subsection keeps its
# full allocated height. 6 units ≈ 0.24 in — visually one narrow line gap.
SUBSEC_GAP = 3

# Extra bottom padding inside each content panel (after the last bullet).
# Keeps body text from sitting flush against the subsection title below.
TEXT_BOX_BOTTOM_MARGIN = 4

# Figure natural-size scaling factor.
# natural_fig_w = poster_width × NATURAL_FIG_K × sqrt(paper_size_ratio)
# A figure with psr=1.0 (median area) renders at NATURAL_FIG_K × poster_width.
# Panels narrower than natural_fig_w render the figure at 100% (fill); wider
# panels cap the figure at natural_fig_w so small 1-column figures stay visually
# smaller than 2-column overview figures regardless of how large their panel grows.
NATURAL_FIG_K = 0.55

# Table figure weight: how much a table's pixel area contributes to panel sp.
# Tables are primary scientific evidence and need proportionally more space than
# figures to remain readable (wide landscape layout, dense content).
# Set higher than figure_weight (0.15) so table panels compete for height.
TABLE_FIGURE_WEIGHT = 0.25

# Minimum body textbox height: 2 lines at 36pt body font.
# Simulation uses DPI=72.5 with margin_bottom≈10px, so available height
# for text = height_px - 10.  2 lines need 2 × (36pt × 1.35) / 72 × 72.5 ≈ 97px,
# plus 10px margin → total 107px → 107/72.5 × 25 ≈ 37u.  Use 40u for buffer.
MIN_BODY_TEXT_H = 40

# Theme colors
THEME_TITLE_TEXT_COLOR = (255, 255, 255)
THEME_TITLE_FILL_COLOR = (63, 86, 147)
THEME_SUBSEC_TITLE_TEXT_COLOR = (32, 56, 100)
