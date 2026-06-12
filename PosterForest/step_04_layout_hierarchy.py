"""
STEP 4: Binary-split Layout Hierarchy

Converts the poster panel tree from Step 3 into pixel-precise bounding boxes.
Uses a recursive binary-split optimizer to assign (x, y, w, h) coordinates to
every panel, textbox, and figure placeholder; outputs inch-unit arrangements
consumed by Step 5 (bullet generation) and Step 7 (PPTX rendering).

Input:  Poster panel tree + poster dimensions (inches)
Output: panel_arrangement, text_arrangement, figure_arrangement (all in inches)
"""

from lxml import etree
import os
import copy
import glob
import re
import uuid
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse

from PosterForest.poster_config import (
    UNITS_PER_INCH,
    TITLE_HEIGHT_RATIO, SECTION_TITLE_HEIGHT, SUBSECTION_TITLE_HEIGHT,
    TITLE_FONT_SIZE, TITLE_FONT_SIZE_2,
    SECTION_TITLE_FONT_SIZE, SUBSECTION_TITLE_FONT_SIZE, CONT_FONT_SIZE,
    SHRINK_MARGIN, FONT_NAME, MAX_ATTEMPT, TEXT_BOX_BOTTOM_MARGIN, MIN_BODY_TEXT_H,
    THEME_TITLE_TEXT_COLOR, THEME_TITLE_FILL_COLOR, THEME_SUBSEC_TITLE_TEXT_COLOR,
    SUBSEC_GAP, NATURAL_FIG_K, TABLE_FIGURE_WEIGHT,
)


_SECTION_NUM_RE = re.compile(r'^\s*\d+(?:\.\d+)*\.?\s+')

def strip_section_numbering(title: str) -> str:
    """Remove leading section numbers (e.g. '3.1 Method' → 'Method')."""
    return _SECTION_NUM_RE.sub('', title).strip()


def _make_text_box(panel_id, panel_name, x, y, width, height, textbox_id, textbox_name_suffix, content=None):
    """Build a textbox dict with normalised float coordinates."""
    return {
        "panel_id": panel_id, "x": float(x), "y": float(y),
        "width": float(width), "height": float(height),
        "textbox_id": textbox_id,
        "textbox_name": f'p<{panel_name}>_{textbox_name_suffix}',
        "content": content,
    }


def _compute_subsec_title_height(section_name: str, panel_usable_width: float, base_title_height: float) -> float:
    """Compute actual subsection title height accounting for text wrapping.

    When a subsection title wraps onto multiple lines, the title box must be
    taller than base_title_height to avoid overlapping the content below.
    This mirrors the expansion logic in place_content_in_panel_for_parent so
    reconstruct_layout_with_groups and place_content_in_panel_for_parent agree
    on the title height used when positioning child panels.
    """
    _line_h = SUBSECTION_TITLE_FONT_SIZE * 1.35 / 72 * UNITS_PER_INCH
    _usable_w_pt = panel_usable_width / UNITS_PER_INCH * 72 - 20
    _chars_per_line = max(1, _usable_w_pt / (SUBSECTION_TITLE_FONT_SIZE * 0.60))
    _title_clean = strip_section_numbering(section_name)
    _n_lines = max(1, int(np.ceil(len(_title_clean) / _chars_per_line)))
    if _n_lines > 1:
        return base_title_height + (_n_lines - 1) * _line_h
    return base_title_height


# ---- Step 4-1: Unit Conversion Utilities ------------------------------------
def to_inches(value_in_units, units_per_inch=72):
    """
    Convert a single coordinate or dimension from 'units' to inches.
    For example, if your units are 'points' (72 points = 1 inch),
    then units_per_inch=72.
    If your units are 'pixels' at 96 DPI, then units_per_inch=96.
    """
    return value_in_units / units_per_inch


def from_inches(value_in_inches, units_per_inch=72):
    """
    Convert from inches back to the original 'units'.
    """
    return value_in_inches * units_per_inch


def softmax(logits):
    """Compute softmax probabilities from a list of logits."""
    s = sum(np.exp(logits))
    return [np.exp(l)/s for l in logits]


def visualize_complete_layout(
    panels, text_boxes, figure_boxes, poster_width, poster_height
):
    """Render panel/textbox/figure bounding boxes with matplotlib for debugging."""
    fig, ax = plt.subplots(figsize=(12,8))
    ax.set_xlim(0, poster_width)
    ax.set_ylim(0, poster_height)
    ax.set_aspect('equal')

    # Draw panels
    for panel in panels:
        rect = patches.Rectangle(
            (panel["x"], panel["y"]), panel["width"], panel["height"],
            linewidth=1, edgecolor='black', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            panel["x"] + 5, panel["y"] + panel["height"] - 5,
            f'Panel {panel["panel_id"]}', fontsize=8, va='top', color='black'
        )

    # Draw text boxes
    for txt in text_boxes:
        rect = patches.Rectangle(
            (txt["x"], txt["y"]), txt["width"], txt["height"],
            linewidth=1, edgecolor='green', linestyle='-.', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            txt["x"] + 2, txt["y"] + txt["height"] - 2,
            f'Text {txt["panel_id"]}', fontsize=7, color='green', va='top'
        )

    # Draw figures
    for fig_box in figure_boxes:
        rect = patches.Rectangle(
            (fig_box["x"], fig_box["y"]), fig_box["width"], fig_box["height"],
            linewidth=1, edgecolor='blue', linestyle='--', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            fig_box["x"] + 2, fig_box["y"] + 2,
            f'Fig {fig_box["panel_id"]}', fontsize=7, color='blue', va='bottom'
        )

    plt.gca().invert_yaxis()  # optional: invert y-axis if needed
    plt.show()


# ---- Step 4-5: Convert Arrangements to Inches --------------------------------
def _convert_arr_to_inches(arr, units_per_inch):
    """Deep-copy an arrangement list and convert x/y/width/height to inches."""
    result = copy.deepcopy(arr)
    for item in result:
        item["x"]      = to_inches(item["x"],      units_per_inch)
        item["y"]      = to_inches(item["y"],      units_per_inch)
        item["width"]  = to_inches(item["width"],  units_per_inch)
        item["height"] = to_inches(item["height"], units_per_inch)
    return result


def get_arrangments_in_inches(
    width, height,
    panel_arrangement, figure_arrangement, text_arrangement,
    units_per_inch=72,
):
    """Convert panel/figure/textbox coordinates from layout units to inches."""
    return (
        to_inches(width,  units_per_inch),
        to_inches(height, units_per_inch),
        _convert_arr_to_inches(panel_arrangement,  units_per_inch),
        _convert_arr_to_inches(figure_arrangement, units_per_inch),
        _convert_arr_to_inches(text_arrangement,   units_per_inch),
    )


# ---- Panel / Figure / Textbox Generation Functions --------------------------

def split_textbox(textbox, ratio):
    """Split a textbox horizontally: top gets ratio/(ratio+1) of height, bottom gets 1/(ratio+1)."""
    total_ratio = ratio + 1
    top_height = textbox['height'] * ratio / total_ratio
    bottom_height = textbox['height'] * 1 / total_ratio

    base_name = textbox['textbox_name'].rsplit('_t', 1)[0]

    top_box = dict(textbox)
    top_box['height'] = top_height
    top_box['textbox_name'] = f"{base_name}_t0"
    top_box['is_title'] = 40

    bottom_box = dict(textbox)
    bottom_box['y'] = textbox['y'] + top_height
    bottom_box['height'] = bottom_height
    bottom_box['textbox_name'] = f"{base_name}_t1"

    return top_box, bottom_box


# ---- Step 4-4: Content Placement in Leaf Panel --------------------------------
def place_content_in_panel(panel_dict, shrink_margin=3, top_margin=None):
    """Compute figure and textbox positions for a leaf panel. Returns (text_boxes, figure_boxes)."""
    if top_margin is None:
        top_margin = shrink_margin

    # Basic panel information
    x_p, y_p = panel_dict["x"], panel_dict["y"]
    w_p, h_p = panel_dict["width"], panel_dict["height"]

    panel_id = panel_dict["panel_id"]
    panel_name = panel_dict.get("section_name", panel_dict.get("panel_name", "")) # check both keys for compatibility

    figure_boxes, text_boxes = [], []
    has_title_in_name = "title" in panel_name.lower()

    def make_text_box(x, y, width, height, textbox_id, textbox_name_suffix, content=None):
        return _make_text_box(panel_id, panel_name, x, y, width, height, textbox_id, textbox_name_suffix, content)

    # -----------------------------------------------------------------------
    # Case 1 — Panel has no Figure (keep existing logic)
    # -----------------------------------------------------------------------
    # Extra bottom padding inside the text box so the last bullet line is never
    # flush against the panel edge.  With SHRINK_MARGIN=6, total bottom gap =
    # shrink_margin + TEXT_BOX_BOTTOM_MARGIN = 6 + 7 = 13 units ≈ 1 line at 36pt.
    CONTENT_BOTTOM_EXTRA = TEXT_BOX_BOTTOM_MARGIN
    if 'figure_size' not in panel_dict.keys() or panel_dict["figure_size"] <= 0:
        if has_title_in_name: # Title panel
            text_boxes.append(make_text_box(x_p, y_p, w_p, h_p - shrink_margin, 0, 't0'))
        else:
            text_boxes.append(make_text_box(
                x_p + shrink_margin, y_p + top_margin,
                w_p - 2 * shrink_margin, max(MIN_BODY_TEXT_H, h_p - top_margin - shrink_margin - CONTENT_BOTTOM_EXTRA),
                1, 'body', panel_dict['content']))
        return text_boxes, figure_boxes

    # -----------------------------------------------------------------------
    # Case 2 — Panel has a Figure (logic modified as requested)
    # -----------------------------------------------------------------------

    # Minimum figure dimensions: a figure smaller than this is not worth rendering
    # and indicates the panel itself is too small to be useful.  Increased from
    # 100/75 so we fall back to text-only before producing a postage-stamp figure.
    MIN_FIG_W = 150  # units (~6 in at 25 u/in)
    MIN_FIG_H = 100  # units (~4 in)

    # 1. Compute figure size (maintaining aspect ratio)
    # This follows the size calculation logic from the provided sample code.

    aspect = float(panel_dict["figure_aspect"]) if panel_dict.get("figure_aspect") else 1.0
    if aspect <= 0:
        aspect = 1.0

    # Target figure width: capped at the figure's "natural" poster size derived from its
    # relative area in the paper (natural_fig_w set in layout_initialization):
    #   - is_main_figure → natural_fig_w uses sqrt(3.0) = max cap → fills any typical panel
    #   - is_table       → natural_fig_w uses table_size_ratio (normalized to other tables)
    #   - regular figure → natural_fig_w uses paper_size_ratio (normalized to other figures)
    # If the panel is narrower than natural_fig_w the figure fills it (100%);
    # if wider, it is capped so smaller figures don't appear as large as the main figure.
    available_w = w_p - 2 * shrink_margin
    _natural_fig_w = panel_dict.get('natural_fig_w', None)
    # Figure-only panels (no text content): bypass natural_fig_w cap and fill the full
    # available width.  The cap exists to prevent small paper figures from looking as
    # large as the main figure when they share a wide panel with text — for figure-only
    # panels there is no text, so the figure should always fill the entire panel.
    _is_figure_only = (panel_dict.get('tp', 1) == 0 and
                       not panel_dict.get('content', '').strip())
    if _is_figure_only:
        _target_fig_w = available_w
    elif _natural_fig_w is not None:
        _target_fig_w = min(float(_natural_fig_w), available_w)
    else:
        _target_fig_w = available_w   # fallback for panels without natural_fig_w

    fig_w = _target_fig_w
    fig_h = fig_w / aspect

    # Determine layout mode.
    # Layout rule by aspect ratio:
    #   Portrait  (aspect < 0.8) : prefer SBS — tall figure sits naturally on the left,
    #                               text fills the right; stacked would overflow and shrink it.
    #   Near-square (0.8–1.3)    : SBS when stacked would height-constrain and downscale the figure.
    #                               Condition: fig_h (at full scaled width) > stacked max_fig_h.
    #                               This keeps SBS as long as stacked gives a smaller figure,
    #                               even as the panel grows taller.
    #   Landscape figure (aspect >= 1.3) : prefer stacked — wide figure is short, fits cleanly on top.
    #   Landscape table  (aspect >= 1.3) : SBS when stacked would waste vertical space (fill < 65%).
    #                               Wide tables have small fig_h; stacked leaves a large empty gap below
    #                               text, so SBS (table left, text right) gives better space efficiency.
    #
    # In all cases, SBS is blocked when the panel is too narrow to give text ≥ MIN_SIDE_TEXT_W.
    MIN_SIDE_TEXT_W = max(50, w_p * 0.20)
    _sbs_fig_w = fig_w
    _trial_fw = _sbs_fig_w
    while w_p - _trial_fw - shrink_margin * 3 < MIN_SIDE_TEXT_W and _trial_fw > MIN_SIDE_TEXT_W * 0.3:
        _trial_fw *= 0.9
    _effective_text_w = w_p - _trial_fw - shrink_margin * 3

    if aspect < 0.8:
        _sbs_height_cond = True              # portrait: always prefer SBS
    elif aspect < 1.3:
        # Use the same min_text_h formula as the stacked block below so the
        # threshold is consistent: SBS is preferred exactly when stacked would
        # height-constrain the figure (making it narrower than in SBS).
        if _is_figure_only:
            _stacked_min_text_h = 0
        elif aspect >= 1.2:
            _stacked_min_text_h = max(15, h_p * 0.06)
        else:
            _stacked_min_text_h = max(15, h_p * 0.08)
        _stacked_max_fig_h = h_p - _stacked_min_text_h - shrink_margin * 3
        _sbs_height_cond = fig_h > max(0.0, _stacked_max_fig_h)
    elif panel_dict.get('is_table'):
        # Landscape table: use SBS when stacked would waste vertical space.
        # Estimate stacked fill = (table_h + text_h) / panel_h; if < 65% → SBS.
        # Wide tables have small fig_h (= fig_w / aspect), so stacked leaves empty space below.
        _est_text_h = max(15, h_p * 0.10)
        _projected_stacked_fill = (fig_h + _est_text_h) / h_p
        _sbs_height_cond = _projected_stacked_fill < 0.65
    else:
        _sbs_height_cond = fig_h > h_p * 2.0    # landscape figure: prefer stacked
    use_side_by_side = _sbs_height_cond and (_effective_text_w >= MIN_SIDE_TEXT_W)

    if use_side_by_side:
        # --- Side-by-side: figure LEFT, text RIGHT ---
        max_fig_h = h_p - shrink_margin * 2
        if fig_h > max_fig_h:
            scale = max_fig_h / fig_h
            fig_w *= scale
            fig_h = max_fig_h

        # Shrink the figure until there is at least MIN_SIDE_TEXT_W of text space on the right.
        # Figures should fill as much of the panel as possible; text gets whatever is left.
        # For figure-only panels there is no text on the right — skip the shrink loop.
        min_text_w = MIN_SIDE_TEXT_W
        if not _is_figure_only:
            while w_p - fig_w - shrink_margin * 3 < min_text_w and fig_w > min_text_w * 0.3:
                fig_w *= 0.9
                fig_h = fig_w / aspect

        # Portrait figures (aspect < 0.8) are height-constrained in SBS: their width
        # shrinks with panel height.  Stacked fills the full panel width but is also
        # height-constrained (by max_fig_h_stacked).  Cancel SBS only when stacked
        # would produce a strictly wider figure — otherwise SBS is at least as good.
        if aspect < 0.8:
            _stk_min_text_h = max(15, h_p * 0.09)
            _stk_max_fig_h  = max(0.0, h_p - _stk_min_text_h - shrink_margin * 3)
            _stk_fig_w = min(_target_fig_w / aspect, _stk_max_fig_h) * aspect
            cancel_sbs = _stk_fig_w > fig_w   # stacked gives a wider figure → switch
        else:
            cancel_sbs = fig_w < MIN_FIG_W or fig_h < MIN_FIG_H
        if cancel_sbs:
            use_side_by_side = False  # fall through to stacked block below
        else:
            current_x, current_y = x_p + shrink_margin, y_p + top_margin
            figure_boxes.append({
                "panel_id": panel_id, "x": float(current_x), "y": float(current_y),
                "panel_name": panel_name, "width": float(fig_w), "height": float(fig_h),
                "figure_id": 0, "figure_name": f'p<{panel_name}>_f0',
                "figure_path": panel_dict['asset_path']
            })
            current_x += (fig_w + shrink_margin)
            if panel_dict.get('content', '').strip():
                text_w = max(10.0, (x_p + w_p) - current_x - shrink_margin)
                text_h = max(MIN_BODY_TEXT_H, h_p - top_margin - shrink_margin - CONTENT_BOTTOM_EXTRA)
                text_boxes.append(make_text_box(current_x, current_y, text_w, max(10.0, text_h), 0, 'body', panel_dict['content']))

    if not use_side_by_side:
        # --- Stacked: figure TOP, text BOTTOM ---
        # min_text_h: minimum text height below the figure.  Floor is MIN_BODY_TEXT_H
        # (2 lines at body font) so the text box is always at least minimally readable.
        # Tables are primary content — reserve less space for text so the table fills more
        # of the panel. For figure-only panels (tp=0) or table panels: skip the fraction.
        _is_table_panel = panel_dict.get('is_table', False)
        if _is_figure_only:
            min_text_h = 0  # no text at all — figure fills the full panel height
        elif _is_table_panel:
            min_text_h = MIN_BODY_TEXT_H  # table panels keep a text reserve
        elif aspect >= 1.2:
            min_text_h = max(MIN_BODY_TEXT_H, h_p * 0.06)   # landscape figure
        else:
            min_text_h = max(MIN_BODY_TEXT_H, h_p * 0.08)   # portrait figure

        # Correct overhead formula: top_margin (figure top gap) + shrink_margin
        # (gap between figure and text) + shrink_margin (text bottom gap) +
        # CONTENT_BOTTOM_EXTRA.  Previous code used shrink_margin*3 (18u) but
        # actual overhead is top_margin+2*shrink_margin+CONTENT_BOTTOM_EXTRA (27u),
        # leaving text 9u shorter than min_text_h and triggering MIN_BODY_TEXT_H
        # clamp → overflow cascade → 10u text box.
        _stacked_overhead = top_margin + 2 * shrink_margin + CONTENT_BOTTOM_EXTRA
        max_fig_h = h_p - min_text_h - _stacked_overhead

        # When max_fig_h is below the absolute minimum figure dimension the panel
        # is too short to show figure+text.  Show figure/table filling the full
        # available height (figure-only, no text).  Never drop a figure/table image
        # in favor of text — visuals are primary content on a poster.
        _ABS_MIN_FIG_EARLY = 30
        if max_fig_h < _ABS_MIN_FIG_EARLY:
            _avail_h = h_p - top_margin - shrink_margin
            _avail_w = w_p - 2 * shrink_margin
            _fh = min(_avail_h, _avail_w / aspect)
            _fw = _fh * aspect
            if _fw > _avail_w:
                _fw = _avail_w
                _fh = _fw / aspect
            if _fw >= 30 and _fh >= 30:
                _fig_x = x_p + (_avail_w - _fw) / 2 + shrink_margin
                figure_boxes.append({
                    "panel_id": panel_id, "x": float(_fig_x), "y": float(y_p + top_margin),
                    "panel_name": panel_name, "width": float(_fw), "height": float(_fh),
                    "figure_id": 0, "figure_name": f'p<{panel_name}>_f0',
                    "figure_path": panel_dict['asset_path']
                })
            elif panel_dict.get('content', '').strip():
                text_boxes.append(make_text_box(
                    x_p + shrink_margin, y_p + top_margin,
                    _avail_w,
                    max(MIN_BODY_TEXT_H, _avail_h - CONTENT_BOTTOM_EXTRA),
                    0, 'body', panel_dict['content']))
            return text_boxes, figure_boxes

        # Recompute fig dimensions for stacked mode using same target width
        fig_w = _target_fig_w
        fig_h = fig_w / aspect

        if fig_h > max_fig_h:
            fig_h = max_fig_h
            fig_w = fig_h * aspect

        # Figure must not exceed panel width
        if fig_w > w_p - 2 * shrink_margin:
            fig_w = w_p - 2 * shrink_margin
            fig_h = fig_w / aspect

        # Clamp to absolute minimum — always render the figure, even if small.
        # A small figure is far better than a missing one (posters need visuals).
        # Exception: when max_fig_h is below ABS_MIN_FIG the panel is genuinely too
        # short for any figure — fall back to text-only so text fills the panel.
        ABS_MIN_FIG = 30  # units (~1.2 in) — smallest acceptable figure dimension
        if fig_w < ABS_MIN_FIG or fig_h < ABS_MIN_FIG:
            # Panel is extremely small; render figure-only (no text) at whatever fits
            fig_w = max(ABS_MIN_FIG, min(w_p - 2 * shrink_margin, ABS_MIN_FIG * aspect))
            fig_h = fig_w / aspect
            current_x, current_y = x_p + shrink_margin, y_p + top_margin
            fig_x = x_p + (w_p - fig_w) / 2
            figure_boxes.append({
                "panel_id": panel_id, "x": float(fig_x), "y": float(current_y),
                "panel_name": panel_name, "width": float(fig_w), "height": float(fig_h),
                "figure_id": 0, "figure_name": f'p<{panel_name}>_f0',
                "figure_path": panel_dict['asset_path']
            })
            return text_boxes, figure_boxes

        current_x, current_y = x_p + shrink_margin, y_p + top_margin
        fig_x = x_p + (w_p - fig_w) / 2  # horizontally centered
        figure_boxes.append({
            "panel_id": panel_id, "x": float(fig_x), "y": float(current_y),
            "panel_name": panel_name, "width": float(fig_w), "height": float(fig_h),
            "figure_id": 0, "figure_name": f'p<{panel_name}>_f0',
            "figure_path": panel_dict['asset_path']
        })
        current_y += (fig_h + shrink_margin)
        # Only create a text area when the panel actually has content.
        # Panels with tp=0 (figure-only) have empty content — skipping the text box
        # prevents hallucination in step 5 and lets the figure fill the panel fully.
        if panel_dict.get('content', '').strip():
            text_w = w_p - shrink_margin * 2
            # text_h is now guaranteed ≥ min_text_h by the max_fig_h formula above.
            text_h = (y_p + h_p) - current_y - shrink_margin - CONTENT_BOTTOM_EXTRA
            text_h = max(min_text_h, text_h)
            text_boxes.append(make_text_box(current_x, current_y, text_w, max(10.0, text_h), 0, 'body', panel_dict['content']))


    return text_boxes, figure_boxes


def place_content_in_panel_for_parent(panel_dict, title_height=32, shrink_margin=3, is_subsec=False):
    """Place section/subsection title textbox in a parent panel. Returns (text_boxes, [])."""
    # section title / subsection title
    shrink_margin_ = shrink_margin
    if is_subsec:
        shrink_margin_ = 0

    x_p, y_p = panel_dict["x"] + shrink_margin, panel_dict["y"] + shrink_margin_
    w_p, h_p = panel_dict["width"] - 2 * shrink_margin, panel_dict["height"] - shrink_margin

    panel_id = panel_dict["panel_id"]
    panel_name = panel_dict.get("section_name", panel_dict.get("panel_name", "")) # check both keys for compatibility

    figure_boxes, text_boxes = [], []

    def make_text_box(x, y, width, height, textbox_id, textbox_name_suffix, content=None):
        return _make_text_box(panel_id, panel_name, x, y, width, height, textbox_id, textbox_name_suffix, content)

    # Small bottom pad inside the title box to prevent text from feeling flush.
    # Subsections use 0 pad so the title box fills the allocated height cleanly
    # and the gap between subsection title and content body stays minimal.
    TITLE_BOTTOM_PAD = 0
    title_h = min(title_height - shrink_margin - TITLE_BOTTOM_PAD, h_p)
    title_h = max(10.0, title_h)

    # Expand title_h when subsection title wraps to multiple lines
    if is_subsec:
        _line_h = SUBSECTION_TITLE_FONT_SIZE * 1.35 / 72 * UNITS_PER_INCH
        _usable_w_pt = w_p / UNITS_PER_INCH * 72 - 20  # subtract left(15)+right(5) pt margins from add_textbox
        _chars_per_line = max(1, _usable_w_pt / (SUBSECTION_TITLE_FONT_SIZE * 0.60))
        _n_lines = max(1, int(np.ceil(len(strip_section_numbering(panel_dict['section_name'])) / _chars_per_line)))
        if _n_lines > 1:
            title_h = min(title_h + (_n_lines - 1) * _line_h, h_p)

    text_boxes.append(make_text_box(x_p, y_p, w_p, title_h, 0, 'title', strip_section_numbering(panel_dict['section_name'])))
    return text_boxes, figure_boxes


# ---- Step 4-3: Binary-split Hierarchical Layout Tree ------------------------
def generate_layout_tree_hierarchical(panels, x, y, w, h, force_split_type=None, depth=0, first_split=False):
    """Recursively binary-split panels into a layout tree, minimizing aspect-ratio loss."""

    # Base case: only one panel
    # Absolute minimum dimensions: sections must be tall/wide enough to show title + content.
    # MIN_PANEL_H applies to any node (leaf or group) — too-short panels are unreadable.
    # MIN_PANEL_W applies to vertical-split columns — too-narrow panels can't show text.
    # MIN_TABLE_PANEL_W is raised for panels containing tables: tables need width to be legible,
    # and this penalty steers the optimizer to stack table subsections vertically (each gets
    # full column width) rather than side-by-side (each gets half width → unreadably narrow).
    MIN_PANEL_H = 70          # units ≈ 2.8 in
    MIN_PANEL_W = 120         # units ≈ 4.8 in: minimum readable column width
    MIN_TABLE_PANEL_W = 220   # units ≈ 8.8 in: minimum width when panel contains a table
    # Subsection panels need more width than generic panels so their titles don't
    # word-wrap at single characters.  When a section column is too narrow to give
    # each subsection this width side-by-side, the penalty forces horizontal stacking
    # (each subsection as a row with full section width) instead.
    MIN_SUBSEC_PANEL_W = 160  # units ≈ 6.4 in: minimum subsection column width

    # Whether the panels being laid out are subsections (node_type='subsection').
    # Subsections get a higher minimum width than generic leaf panels.
    _splitting_subsecs = len(panels) > 0 and panels[0].get('node_type') == 'subsection'

    def _panel_has_table(p):
        if p.get('is_table'):
            return True
        for ch in p.get('children', []):
            if _panel_has_table(ch):
                return True
        return False

    if len(panels) == 1:
        panel = panels[0]
        if panel.get('children'):
            # Run two candidate layouts and pick the better one with a horizontal
            # stacking bonus.  Forcing horizontal (subsections stacked as rows) prevents
            # wide section groups from creating narrow vertical sub-columns whose aspect
            # ratios look acceptable to the optimizer but produce crowded content.
            # The bonus (5500) means vertical sub-columns are only chosen when they are
            # substantially better — in practice only when a section has so many
            # subsections that stacking would produce extremely flat panels.
            loss_h, tree_h = generate_layout_tree_hierarchical(
                panel['children'], x, y, w, h, force_split_type='horizontal', depth=0
            )
            loss_v, tree_v = generate_layout_tree_hierarchical(
                panel['children'], x, y, w, h, force_split_type=None, depth=0
            )
            _GROUP_HORIZ_BONUS = 3000
            if loss_h <= loss_v + _GROUP_HORIZ_BONUS:
                loss, sub_tree = loss_h, tree_h
            else:
                loss, sub_tree = loss_v, tree_v
            # Apply the same aspect-ratio penalty to GROUP nodes as to leaf nodes,
            # so the optimizer avoids creating very narrow tall section columns.
            # Use a large but FINITE cap so l1+l2 never becomes float('inf'),
            # which would leave best_node=None when all splits are degenerate.
            rev_ratio  = h / max(w, 1e-6)   # avoid division by zero → cap ratio
            flat_ratio = w / max(h, 1e-6)
            group_penalty = 0

            # Hard stop for panels physically too small to display content
            if _panel_has_table(panel):
                _eff_min_w = MIN_TABLE_PANEL_W
            elif _splitting_subsecs:
                _eff_min_w = MIN_SUBSEC_PANEL_W
            else:
                _eff_min_w = MIN_PANEL_W
            if h < MIN_PANEL_H:
                group_penalty += 1e6
            if w < _eff_min_w:
                group_penalty += 1e6

            # Penalty for extreme aspect ratios (aligned with leaf thresholds)
            if rev_ratio > 5.0:
                group_penalty += min(80000 * rev_ratio, 1e6)
            elif rev_ratio > 3.0:
                group_penalty += rev_ratio * 1000
            elif rev_ratio > 2.0:
                group_penalty += rev_ratio * 300
            if flat_ratio > 6.0:
                group_penalty += min(80000 * flat_ratio, 1e6)
            elif flat_ratio > 4.0:
                group_penalty += flat_ratio * 1000
            elif flat_ratio > 3.0:
                group_penalty += flat_ratio * 300
            group_node = {"type": "group", "panel_data": panel, "child_tree": sub_tree}
            return loss + group_penalty, group_node
        else:
            # Penalty for extreme aspect ratios (too tall-narrow OR too flat-wide)
            rev_ratio  = h / w if w > 1e-9 else float('inf')   # tall: h/w
            flat_ratio = w / h if h > 1e-9 else float('inf')   # flat: w/h
            penalty = 0

            # Hard stop for panels physically too small to display content
            if _panel_has_table(panel):
                _eff_min_w = MIN_TABLE_PANEL_W
            elif _splitting_subsecs:
                _eff_min_w = MIN_SUBSEC_PANEL_W
            else:
                _eff_min_w = MIN_PANEL_W
            if h < MIN_PANEL_H:
                penalty += 1e6
            if w < _eff_min_w:
                penalty += 1e6

            # Too tall and narrow — relaxed threshold (5.0) to match old version spirit
            if rev_ratio > 5.0:
                penalty += 80000 * rev_ratio
            elif rev_ratio > 3.0:
                penalty += rev_ratio * 500
            elif rev_ratio > 2.0:
                penalty += rev_ratio * 150

            # Too flat and wide
            if flat_ratio > 6.0:
                penalty += min(80000 * flat_ratio, 1e6)
            elif flat_ratio > 4.0:
                penalty += flat_ratio * 800
            elif flat_ratio > 3.0:
                penalty += flat_ratio * 200

            if panel.get('asset_path') is None: # loss when there is no figure
                cur_rp = min(5.0, (w / h) if h > 1e-9 else 5.0)
                loss = abs(1 - cur_rp) + penalty
            else: # loss when there is a figure
                cur_rp = min(5.0, (w / h) if h > 1e-9 else 5.0)
                rp = panel.get('figure_aspect', panel.get("rp", 1))
                # Clamp to a sane range without distorting the target.
                # Portrait figures target their actual aspect (tall panels are fine for tall figures).
                # Very extreme landscape figures are capped at 3.0 to avoid overly flat panels.
                effective_rp = max(0.4, min(3.0, rp))
                loss = abs(effective_rp - cur_rp) + penalty

            leaf_node = {"type": "leaf", "panel_data": panel}
            return loss, leaf_node


    best_loss, best_node = float('inf'), None
    total_sp = sum(p.get("sp", 0) for p in panels)
    n = len(panels)

    for i in range(1, n):
        subset1, subset2 = panels[:i], panels[i:]
        sp1 = sum(p.get("sp", 0) for p in subset1)
        ratio = sp1 / total_sp if total_sp > 1e-9 else len(subset1) / n

        # 1. Try horizontal split
        if force_split_type != 'vertical':
            h_top = ratio * h
            if 0 < h_top < h:
                l1, node1 = generate_layout_tree_hierarchical(subset1, x, y, w, h_top, force_split_type=None, depth=depth + 1, first_split=first_split)
                l2, node2 = generate_layout_tree_hierarchical(subset2, x, y + h_top, w, h - h_top, force_split_type=None, depth=depth + 1, first_split=first_split)
                if (l1 + l2) < best_loss:
                    best_loss, best_node = l1 + l2, {"type": f"internal_{uuid.uuid4().hex}", "split_type": "horizontal", "split_ratio": ratio, "children": [node1, node2]}

        # 2. Try vertical split
        if force_split_type != 'horizontal':
            w_left = ratio * w
            if 0 < w_left < w:
                # By default, children of a vertical split are forced to use horizontal splits (prevent consecutive vertical splits)
                next_force_for_left_child = 'horizontal'
                next_force_for_right_child = 'horizontal'

                # Exception rule: for the first split (depth == 0), determine the right child's split direction based on canvas ratio
                if first_split and depth == 0:
                    if w > h:  # if wider than tall, also split the right child vertically
                        next_force_for_right_child = 'vertical'
                        if i == n-1:
                            continue  # skip the last panel since there is no right child
                    else:  # if taller than or equal to wide, split the right child horizontally
                        next_force_for_right_child = 'horizontal'

                # Pass the computed constraints to recursive calls
                l1, node1 = generate_layout_tree_hierarchical(subset1, x, y, w_left, h, force_split_type=next_force_for_left_child, depth=depth + 1, first_split=first_split)
                l2, node2 = generate_layout_tree_hierarchical(subset2, x + w_left, y, w - w_left, h, force_split_type=next_force_for_right_child, depth=depth + 1, first_split=first_split)

                if (l1 + l2) < best_loss:
                    best_loss, best_node = l1 + l2, {"type": f"internal_{uuid.uuid4().hex}", "split_type": "vertical", "split_ratio": ratio, "children": [node1, node2]}

    return best_loss, best_node

def reconstruct_layout_with_groups(
    node,
    x, y, w, h,
    initial_title_height=32,
    decay_factor=0.85,
    group_depth=0
):
    """Walk the layout tree and assign (x, y, w, h) to each panel. Returns flat list of panel dicts."""
    if node is None:
        return []
    layouts = []
    node_type = node.get("type")

    if node_type == "leaf":
        layout_info = node["panel_data"].copy()
        layout_info.update({"x": x, "y": y, "width": w, "height": h})
        layouts.append(layout_info)

    # elif node_type == "internal":
    elif "internal" in node_type:
        c1, c2, r = node["children"][0], node["children"][1], node["split_ratio"]
        # 'internal' nodes pass through without incrementing depth.
        if node["split_type"] == "horizontal":
            # Insert a gap between sibling subsections (group_depth >= 1 means we are
            # inside a section's child_tree, so c1/c2 are subsections not sections).
            _gap = SUBSEC_GAP if group_depth >= 1 else 0
            h_top = h * r

            # If the top child (c1) is a figure-only leaf, cap its allocated height to
            # the actual figure height so empty space below is redistributed to c2.
            # Applies to both landscape (stacked) and portrait (SBS height is also capped).
            if c1.get('type') == 'leaf':
                _pd = c1['panel_data']
                _fig_only = (
                    _pd.get('tp', 1) == 0 and
                    not (_pd.get('content') or '').strip() and
                    _pd.get('asset_path') and
                    (_pd.get('figure_size') or 0) > 0
                )
                if _fig_only:
                    _aspect = max(float(_pd.get('figure_aspect', 1.0) or 1.0), 0.1)
                    _nat_w = _pd.get('natural_fig_w')
                    _avail_w = w - 2 * SHRINK_MARGIN
                    _fig_w = min(float(_nat_w), _avail_w) if _nat_w else _avail_w
                    if _aspect >= 1.3:
                        # Landscape → stacked mode: figure height = fig_w / aspect
                        _fig_h = _fig_w / _aspect
                    else:
                        # Portrait/square → SBS mode: figure height constrained by panel height
                        # Use panel height-based estimate; SBS max_fig_h = h_top - 2*SHRINK_MARGIN
                        # so estimate fig_h = min(fig_w/aspect, h_top - 2*SHRINK_MARGIN)
                        _fig_h_sbs = _fig_w / _aspect
                        _fig_h = min(_fig_h_sbs, h_top - 2 * SHRINK_MARGIN)
                    # Minimum panel height = stacked overhead + fig_h
                    # overhead = top_margin(SHRINK_MARGIN) + shrink(between fig/text) + shrink(bottom) + CONTENT_BOTTOM_EXTRA
                    _overhead = 3 * SHRINK_MARGIN + TEXT_BOX_BOTTOM_MARGIN
                    _h_needed = _overhead + _fig_h
                    if _h_needed < h_top:
                        h_top = _h_needed

            layouts.extend(reconstruct_layout_with_groups(c1, x, y, w, h_top, initial_title_height, decay_factor, group_depth))
            layouts.extend(reconstruct_layout_with_groups(c2, x, y + h_top + _gap, w, h - h_top - _gap, initial_title_height, decay_factor, group_depth))
        elif node["split_type"] == "vertical":
            layouts.extend(reconstruct_layout_with_groups(c1, x, y, w * r, h, initial_title_height, decay_factor, group_depth))
            layouts.extend(reconstruct_layout_with_groups(c2, x + w * r, y, w - w * r, h, initial_title_height, decay_factor, group_depth))

    elif node_type == "group":
        # Compute the title height for the current depth.
        current_title_height = initial_title_height * (decay_factor ** group_depth)

        # For subsection groups (depth >= 1), expand title height dynamically when
        # the title wraps onto multiple lines, so child panels are pushed down far
        # enough to avoid overlapping the title text.
        if group_depth >= 1:
            sec_name = node["panel_data"].get("section_name", "")
            usable_w = w - 2 * SHRINK_MARGIN
            current_title_height = _compute_subsec_title_height(sec_name, usable_w, current_title_height)

        # Add the group's own layout.
        group_layout = node["panel_data"].copy()
        group_layout.update({"x": x, "y": y, "width": w, "height": h})
        layouts.append(group_layout)

        # Recursively reconstruct the child tree.
        # Clamp child height to ≥ 0 so extremely squeezed panels never produce
        # negative-height textboxes (which cause phantom overflow at 200%+).
        child_h = max(0.0, h - current_title_height)
        layouts.extend(reconstruct_layout_with_groups(
            node['child_tree'],
            x, y + current_title_height, w, child_h,
            initial_title_height,
            decay_factor,
            group_depth + 1
        ))

    return layouts

# ---- Step 4-2: Layout Initialization ----------------------------------------
def layout_initialization(
    paper_panels,
    poster_width=1200,
    poster_height=800,
    title_h=None,
    smoothness=0.1,
    figure_weight=0.20,
):
    """Initialize poster layout: infer sp from tp/figure size and build the title + content panel tree."""
    import math as _math

    # Pre-pass: collect all leaf-node figure/table sizes so we can scale fp proportionally
    # to the original PDF figure area instead of using fixed step-function thresholds.
    # figure_size = pixel_width × pixel_height (set in step_01b_asset_extraction.py).
    # Tables are collected separately so their scale reference is median TABLE size,
    # not figure size — tables tend to be smaller in pixel area than figures.
    _fig_sizes = []
    _table_sizes = []
    def _collect_sizes(nodes):
        for n in nodes:
            ch = n.get('children') or []
            if not ch:
                s = n.get('figure_size', 0) or 0
                if s > 0 and n.get('asset_path'):
                    if n.get('is_table'):
                        _table_sizes.append(float(s))
                    else:
                        _fig_sizes.append(float(s))
            else:
                _collect_sizes(ch)
    _collect_sizes(paper_panels)
    if len(_fig_sizes) >= 2:
        _sorted = sorted(_fig_sizes)
        _ref_size = _sorted[len(_sorted) // 2]   # median
    elif _fig_sizes:
        _ref_size = _fig_sizes[0]
    else:
        _ref_size = 1.0
    if len(_table_sizes) >= 2:
        _tsorted = sorted(_table_sizes)
        _table_ref_size = _tsorted[len(_tsorted) // 2]
    elif _table_sizes:
        _table_ref_size = _table_sizes[0]
    else:
        _table_ref_size = _ref_size

    # Section-type lookup for method-section figure boost
    _M_SEC_KW = {'method', 'approach', 'model', 'architecture', 'framework', 'algorithm', 'proposed'}

    def _set_natural_fig_w(node):
        """Compute natural_fig_w for a leaf node that has a figure/table asset.
        Mirrors the depth-3 logic so all depths behave consistently.
        Tables use table_size_ratio (normalized to other tables).
        is_main_figure always uses sr=3.0 (max cap) so the main figure fills
        any typical panel regardless of its raw pixel area in the PDF."""
        if not node.get('asset_path'):
            return
        s = max(float(node.get('figure_size', 1) or 1), 1.0)
        if node.get('is_table'):
            tsr = max(0.3, min(3.0, s / _table_ref_size))
            node['natural_fig_w'] = poster_width * NATURAL_FIG_K * _math.sqrt(tsr)
        else:
            sr = max(0.1, min(3.0, s / _ref_size))
            if node.get('is_main_figure'):
                node['natural_fig_w'] = poster_width * NATURAL_FIG_K * _math.sqrt(3.0)
            else:
                # sr^0.8 (vs sqrt=sr^0.5) gives more aggressive width reduction for
                # small figures: a figure at 0.335× median maps to 0.42× instead of 0.58×.
                node['natural_fig_w'] = poster_width * NATURAL_FIG_K * (sr ** 0.8)

    def _has_main_figure(node):
        """Return True if the node or any descendant has is_main_figure=True."""
        if node.get('is_main_figure'):
            return True
        return any(_has_main_figure(c) for c in node.get('children', []))

    def _main_sec_sp_boost(node):
        """Compute section sp boost proportional to the largest main figure's size_ratio.

        Bigger figures (higher figure_size in PDF) deserve wider poster columns.
        sr=0.3 → ×1.3  |  sr=1.0 → ×1.4  |  sr=1.8 → ×1.7  |  sr=3.0 → ×2.0
        """
        max_sr = 0.0
        def _walk(n):
            nonlocal max_sr
            if n.get('is_main_figure') and n.get('figure_size', 0) > 0:
                s = max(float(n.get('figure_size', 1) or 1), 1.0)
                sr = max(0.3, min(3.0, s / _ref_size))
                if sr > max_sr:
                    max_sr = sr
            for c in n.get('children', []):
                _walk(c)
        _walk(node)
        if max_sr <= 0:
            return 1.0
        return max(1.3, min(2.0, 1.0 + max_sr * 0.4))

    def _has_main_table(node):
        """Return True if the node or any descendant has is_table=True with an asset."""
        if node.get('is_table') and node.get('asset_path'):
            return True
        return any(_has_main_table(c) for c in node.get('children', []))

    def _table_sec_sp_boost(node):
        """Compute section sp boost proportional to the largest table's table_size_ratio.

        Tables are primary scientific evidence — sections containing them get extra space.
        tsr=0.3 → ×1.3  |  tsr=1.0 → ×1.4  |  tsr=2.0 → ×1.8  |  tsr=3.0 → ×2.0
        """
        max_tsr = 0.0
        def _walk(n):
            nonlocal max_tsr
            if n.get('is_table') and n.get('figure_size', 0) > 0:
                s = max(float(n.get('figure_size', 1) or 1), 1.0)
                tsr = max(0.3, min(3.0, s / _table_ref_size))
                if tsr > max_tsr:
                    max_tsr = tsr
            for c in n.get('children', []):
                _walk(c)
        _walk(node)
        if max_tsr <= 0:
            return 1.0
        return max(1.3, min(2.0, 1.0 + max_tsr * 0.4))

    for sec in paper_panels:
        if 'children' not in sec.keys() or sec['children'] == []:
            _set_natural_fig_w(sec)
            _fp_1 = smoothness
            if sec.get('asset_path'):
                _s_1 = max(float(sec.get('figure_size', 1) or 1), 1.0)
                _fa_1 = sec.get('figure_aspect', 1.0)
                _sec_method_1 = any(k in sec.get('section_name', '').lower() for k in _M_SEC_KW)
                if sec.get('is_table'):
                    _tsr_1 = max(0.3, min(3.0, _s_1 / _table_ref_size))
                    sec['table_size_ratio'] = _tsr_1
                    _fp_1 = TABLE_FIGURE_WEIGHT * _tsr_1
                else:
                    _sr_1 = max(0.3, min(3.0, _s_1 / _ref_size))
                    sec['paper_size_ratio'] = _sr_1
                    _fp_1 = figure_weight * _sr_1
                    if sec.get('is_main_figure'):
                        _fp_1 *= 1.3
                    elif _sec_method_1:
                        _fp_1 *= 1.2
                    elif _fa_1 >= 1.5:
                        _fp_1 *= 1.1
                    elif _fa_1 >= 1.2:
                        _fp_1 *= 1.05
            sec["sp"] = sec["tp"] + _fp_1
            if _has_main_figure(sec):
                _boost = _main_sec_sp_boost(sec)
                sec["sp"] *= _boost
                print(f"   🚀 Section main-fig sp boost ×{_boost:.2f} on '{sec.get('section_name','?')}'")
            elif _has_main_table(sec):
                _boost = _table_sec_sp_boost(sec)
                sec["sp"] *= _boost
                print(f"   📊 Section table sp boost ×{_boost:.2f} on '{sec.get('section_name','?')}'")

        else:
            sec_sp = 0
            _sec_is_method = any(k in sec.get('section_name', '').lower() for k in _M_SEC_KW)

            for subsec in sec['children']:

                if 'children' not in subsec.keys() or subsec['children'] == []:
                    _set_natural_fig_w(subsec)
                    _fp_2 = smoothness
                    if subsec.get('asset_path'):
                        _s_2 = max(float(subsec.get('figure_size', 1) or 1), 1.0)
                        _fa_2 = subsec.get('figure_aspect', 1.0)
                        if subsec.get('is_table'):
                            _tsr_2 = max(0.3, min(3.0, _s_2 / _table_ref_size))
                            subsec['table_size_ratio'] = _tsr_2
                            _fp_2 = TABLE_FIGURE_WEIGHT * _tsr_2
                        else:
                            _sr_2 = max(0.3, min(3.0, _s_2 / _ref_size))
                            subsec['paper_size_ratio'] = _sr_2
                            _fp_2 = figure_weight * _sr_2
                            if subsec.get('is_main_figure'):
                                _fp_2 *= 1.3
                            elif _sec_is_method:
                                _fp_2 *= 1.2
                            elif _fa_2 >= 1.5:
                                _fp_2 *= 1.1
                            elif _fa_2 >= 1.2:
                                _fp_2 *= 1.05
                    subsec["sp"] = subsec["tp"] + _fp_2
                    sec_sp += subsec["sp"]

                else:
                    subsec_sp = 0
                    for leaf in subsec['children']:

                        fp = 0
                        if leaf['asset_path'] is not None and leaf['asset_path'] != '':
                            size = max(float(leaf.get("figure_size", 1) or 1), 1.0)
                            fig_aspect = leaf.get("figure_aspect", 1.0)

                            if leaf.get('is_table'):
                                # Tables: normalize fp and natural_fig_w against other tables,
                                # not figure sizes. Use TABLE_FIGURE_WEIGHT (higher than figure_weight)
                                # so table panels compete for height — tables are primary evidence.
                                table_size_ratio = size / _table_ref_size
                                table_size_ratio = max(0.3, min(3.0, table_size_ratio))
                                leaf['table_size_ratio'] = table_size_ratio
                                leaf['natural_fig_w'] = poster_width * NATURAL_FIG_K * _math.sqrt(table_size_ratio)
                                fp = TABLE_FIGURE_WEIGHT * table_size_ratio
                                print(f"   📊 Table fp={fp:.3f} (tsr={table_size_ratio:.2f}) on panel {leaf.get('panel_id', '?')}")
                            else:
                                # Use LINEAR size ratio (not sqrt) so figures that are proportionally
                                # larger in the paper get proportionally more space in the poster.
                                size_ratio = size / _ref_size
                                size_ratio = max(0.1, min(3.0, size_ratio))
                                leaf['paper_size_ratio'] = size_ratio
                                if leaf.get('is_main_figure'):
                                    leaf['natural_fig_w'] = poster_width * NATURAL_FIG_K * _math.sqrt(3.0)
                                else:
                                    # sr^0.8 for more aggressive small-figure shrinkage vs sqrt(sr)
                                    leaf['natural_fig_w'] = poster_width * NATURAL_FIG_K * (size_ratio ** 0.8)
                                fp = figure_weight * size_ratio  # fp still uses actual size_ratio

                            # Hierarchical boost for figures: main > method-section > landscape
                            # (Tables handled above with TABLE_FIGURE_WEIGHT, no further boost)
                            if leaf.get('is_main_figure'):
                                fp *= 1.3
                                print(f"   🔍 Main figure boost ×1.3 on panel {leaf.get('panel_id', '?')} "
                                      f"(aspect={fig_aspect:.2f}, size_ratio={size_ratio:.2f})")
                            elif _sec_is_method:
                                fp *= 1.2
                                print(f"   🔍 Method-section figure boost ×1.2 on panel {leaf.get('panel_id', '?')} "
                                      f"(aspect={fig_aspect:.2f}, size_ratio={size_ratio:.2f})")
                            elif fig_aspect >= 1.5:
                                fp *= 1.1
                            elif fig_aspect >= 1.2:
                                fp *= 1.05
                        else:
                            # Text-only depth-3 leaves were missing smoothness, making them
                            # disproportionately small vs. figure leaves. Add smoothness so
                            # text panels get baseline space.
                            fp = smoothness

                        leaf["sp"] = leaf["tp"] + fp
                        subsec_sp += leaf["sp"]

                    subsec["sp"] = subsec_sp
                    sec_sp += subsec_sp

            sec["sp"] = sec_sp + smoothness
            if _has_main_figure(sec):
                _boost = _main_sec_sp_boost(sec)
                sec["sp"] *= _boost
                print(f"   🚀 Section main-fig sp boost ×{_boost:.2f} on '{sec.get('section_name','?')}'")
            elif _has_main_table(sec):
                _boost = _table_sec_sp_boost(sec)
                sec["sp"] *= _boost
                print(f"   📊 Section table sp boost ×{_boost:.2f} on '{sec.get('section_name','?')}'")

    # Enforce minimum sp per subsection to prevent extremely narrow sub-columns.
    # Formula: each subsection gets ≥ max(floor, 1/(n+2)) of siblings' total.
    # 2 children→25%, 3→20%, 4→17%, 5+→clamped at floor.
    _MIN_SUBSEC_FRAC_DYNAMIC = 0.10
    for sec in paper_panels:
        children = sec.get('children', [])
        if len(children) >= 2:
            total_child_sp = sum(c.get('sp', 0) for c in children)
            if total_child_sp > 0:
                n_ch = len(children)
                min_frac = max(_MIN_SUBSEC_FRAC_DYNAMIC, 1.0 / (n_ch + 2))
                min_sp_ch = min_frac * total_child_sp
                if any(c.get('sp', 0) < min_sp_ch for c in children):
                    for c in children:
                        if c.get('sp', 0) < min_sp_ch:
                            print(f"   📐 Subsec min-width: '{c.get('section_name','?')}' "
                                  f"sp {c.get('sp',0):.3f}→{min_sp_ch:.3f} (min {min_frac:.0%})")
                            c['sp'] = min_sp_ch
                    new_total = sum(c.get('sp', 0) for c in children)
                    scale = total_child_sp / new_total if new_total > 0 else 1.0
                    for c in children:
                        c['sp'] *= scale
                    sec['sp'] = sum(c.get('sp', 0) for c in children) + smoothness

    # Enforce a loose minimum per top-level section so extremely short sections
    # (e.g. a one-line Conclusion) still get a readable column width.
    # Kept deliberately small (0.06) so the natural sp distribution is preserved.
    MIN_SECTION_FRACTION = 0.06
    total_sp = sum(sec.get("sp", 0) for sec in paper_panels)
    if total_sp > 0:
        min_sp = MIN_SECTION_FRACTION * total_sp
        adjusted = any(sec.get("sp", 0) < min_sp for sec in paper_panels)
        if adjusted:
            for sec in paper_panels:
                if sec.get("sp", 0) < min_sp:
                    sec["sp"] = min_sp
            new_total = sum(sec.get("sp", 0) for sec in paper_panels)
            scale = total_sp / new_total if new_total > 0 else 1.0
            for sec in paper_panels:
                sec["sp"] = sec["sp"] * scale

    # Enforce minimum COLUMN WIDTH to prevent unnaturally tall-narrow sections.
    # 0.13 floor: each section gets ≥ 13% of poster width (~6.2" for 48" poster).
    # Relaxed from 0.17 so natural content-proportional allocation is preserved.
    n_sections = len(paper_panels)
    min_col_frac = max(0.13, 1.0 / (n_sections + 2))
    total_sp2 = sum(sec.get("sp", 0) for sec in paper_panels)
    if total_sp2 > 0:
        min_sp_col = min_col_frac * total_sp2
        if any(sec.get("sp", 0) < min_sp_col for sec in paper_panels):
            for sec in paper_panels:
                if sec.get("sp", 0) < min_sp_col:
                    sec["sp"] = min_sp_col
            new_total2 = sum(sec.get("sp", 0) for sec in paper_panels)
            scale2 = total_sp2 / new_total2 if new_total2 > 0 else 1.0
            for sec in paper_panels:
                sec["sp"] = sec["sp"] * scale2

    # Enforce minimum space per subsection so no child panel becomes too short.
    # Each subsection gets at least _MIN_SUBSEC_FRAC_STATIC of its parent's sp.
    # Also cap the max/min ratio of siblings to avoid extreme panel imbalance.
    _MIN_SUBSEC_FRAC_STATIC = 0.15
    MAX_SIBLING_RATIO = 6.0  # largest sibling can be at most 6× the smallest
    for sec in paper_panels:
        children = sec.get("children", [])
        if not children:
            continue
        total_child_sp = sum(c.get("sp", 0) for c in children)
        if total_child_sp <= 0:
            continue
        min_child_sp = _MIN_SUBSEC_FRAC_STATIC * total_child_sp
        adjusted = any(c.get("sp", 0) < min_child_sp for c in children)
        if adjusted:
            for c in children:
                if c.get("sp", 0) < min_child_sp:
                    c["sp"] = min_child_sp
            new_total = sum(c.get("sp", 0) for c in children)
            scale = total_child_sp / new_total if new_total > 0 else 1.0
            for c in children:
                c["sp"] = c["sp"] * scale

        # Cap max/min ratio to prevent extreme panel imbalance
        sp_vals = [c.get("sp", 0) for c in children]
        min_sp = max(min(sp_vals), 1e-9)
        if max(sp_vals) / min_sp > MAX_SIBLING_RATIO:
            cap = min_sp * MAX_SIBLING_RATIO
            for c in children:
                if c.get("sp", 0) > cap:
                    c["sp"] = cap
            # Re-normalise
            new_total = sum(c.get("sp", 0) for c in children)
            scale = total_child_sp / new_total if new_total > 0 else 1.0
            for c in children:
                c["sp"] = c["sp"] * scale

    # Enforce absolute minimum column width for top-level sections only.
    # Subsections within a section are stacked horizontally (they share the
    # parent column's full width), so applying a minimum-width constraint to
    # them is incorrect — it would inflate the sp of small subsections and
    # shrink the dominant figure subsection's height allocation.
    MIN_ABS_W = 125  # units — ~5 inches, minimum readable text column width

    def _enforce_abs_width(items, avail_w):
        if not items or avail_w <= 0:
            return
        total = sum(i.get("sp", 0) for i in items)
        if total <= 0:
            return
        for item in items:
            child_w = (item.get("sp", 0) / total) * avail_w
            if child_w < MIN_ABS_W:
                item["sp"] = (MIN_ABS_W / avail_w) * total
        new_total = sum(i.get("sp", 0) for i in items)
        if new_total > 0:
            scale = total / new_total
            for i in items:
                i["sp"] = i["sp"] * scale
        # Do NOT recurse into subsections: they share the parent column width
        # and must not receive artificial width inflation.

    _enforce_abs_width(paper_panels, poster_width)

    # Generate recursive layout on remaining space for other panels
    layout_loss, content_panel_tree = generate_layout_tree_hierarchical(
        paper_panels,
        x=0, y=title_h,
        w=poster_width, h=poster_height - title_h,
        force_split_type='vertical',
        depth=0,
        first_split=True
    )

    title_panel = {'panel_id': 0, 'section_name': 'Title', 'x' : 0, 'y': 0, 'width': poster_width, 'height': title_h, 'figure_size': 0, 'tp': 0, 'gp': 0, 'figure_aspect': 1}

    return title_panel, content_panel_tree

# ---- Step 4 [Entry]: Panel Tree → Coordinate Arrangements -------------------
# Calls: 4-2 layout_initialization (pre-called from main.py)
#        4-3 generate_layout_tree_hierarchical
#        4-4 place_content_in_panel
#        4-1 to_inches / get_arrangments_in_inches (post-step)
def generate_complete_layout(
    title_panel,
    content_panel_tree,
    poster_width=1200,
    poster_height=800,
    title_h=None,
    section_title_h=32,
    subsection_title_h=24,
    shrink_margin=3,
):
    """Run binary-split layout on the panel tree and return final panel/textbox/figure arrangements."""

    original_arrangement = reconstruct_layout_with_groups(
        node=content_panel_tree, x=0, y=title_h, w=poster_width, h=poster_height - title_h,
        initial_title_height=section_title_h, decay_factor=subsection_title_h/section_title_h,group_depth=0
    )

    # Combine title panel with others
    panel_arrangement = [title_panel] + original_arrangement


    # generate box for text and figure
    text_arrangement = []
    figure_arrangement = []

    for p in panel_arrangement:
        if 'children' in p.keys() and p['children'] != []: # not leaf node
            if len(str(p['panel_id'])) == 1:
                text_boxes, _ = place_content_in_panel_for_parent(p, title_height=section_title_h, shrink_margin=shrink_margin)
            else:
                text_boxes, _ = place_content_in_panel_for_parent(p, title_height=subsection_title_h, shrink_margin=shrink_margin, is_subsec=True)

            text_arrangement.extend(text_boxes)          # text arrangement

        else: # leaf node
            # Reduce the top margin for leaf panels inside subsections (panel_id ≥ 3 digits,
            # e.g. "111", "121") so the gap between the subsection title and the first
            # line of content is visually tight rather than having a large empty band.
            _pid_str = str(p.get('panel_id', ''))
            _is_subsection_leaf = _pid_str.isdigit() and len(_pid_str) >= 3
            _top_margin = max(2, shrink_margin // 2) if _is_subsection_leaf else shrink_margin
            text_boxes, fig_boxes = place_content_in_panel(p, shrink_margin=shrink_margin, top_margin=_top_margin)

            text_arrangement.extend(text_boxes)          # text arrangement
            figure_arrangement.extend(fig_boxes)       # figure arrangement

    # special processing for title text box
    text_arrangement_title = text_arrangement[0]
    text_arrangement = text_arrangement[1:]
    text_arrangement_title_top, text_arrangement_title_bottom = split_textbox(
        text_arrangement_title, 
        0.85
    )

    # Add the split textboxes back to the list
    text_arrangement = [text_arrangement_title_top, text_arrangement_title_bottom] + text_arrangement

    return panel_arrangement, figure_arrangement, text_arrangement

# ------------------------------------------------------------


if __name__ == '__main__':

    # example command to run this script:
    # PYTHONPATH=$PYTHONPATH:. python PosterForest/step_04_layout_hierarchy.py --poster_tree_path outputs/<run_id>/03_generate_outline/<paper_name>_poster_panels.json --poster_bullet_path outputs/<run_id>/05_generate_content/<paper_name>_bullet_contents.json

    units_per_inch = 25

    parser = argparse.ArgumentParser()
    parser.add_argument('--poster_width_inches', type=int, default=48)
    parser.add_argument('--poster_height_inches', type=int, default=36)
    parser.add_argument('--poster_name', type=str, default='NeurIPS2024_VAR')

    parser.add_argument('--index', type=int, default=1)
    parser.add_argument('--tmp_dir', type=str, default='tmp_layout')
    args = parser.parse_args()

    args.poster_tree_path = f'outputs/4o_4o_{args.poster_name}/03_generate_outline/{args.poster_name}_poster_panels.json'
    args.poster_bullet_path = f'outputs/4o_4o_{args.poster_name}/05_generate_content/{args.poster_name}_bullet_contents.json'

    poster_width = args.poster_width_inches * units_per_inch
    poster_height = args.poster_height_inches * units_per_inch

    # Typography and layout constants imported from PosterForest.poster_config

    import shutil
    import os
    import json
    from utils.wei_utils import get_agent_config, utils_functions, run_code, style_bullet_content, scale_to_target_area, char_capacity  # General utilities
    from PosterForest.step_07_pptx_generation import generate_poster_code  # Step 7 - PowerPoint generation
    from utils.src.utils import ppt_to_images_with_title

    # Remove static file dependency: read from step_dirs
    with open(args.poster_tree_path, 'r') as f:
        poster_tree = json.load(f)
    panels = poster_tree['tree_data']['children']
    panels = panels[1:]

    for p in panels:
        if 'abstract' in p['section_name'].lower():
            panels.remove(p)
            break


    title_height_ratio = 0.15
    title_panel, content_panel_tree = layout_initialization(
        panels,
        poster_width,
        poster_height,
        title_h = poster_height * title_height_ratio,
    )

    panel_arrangement, figure_arrangement, text_arrangement = generate_complete_layout(
        title_panel,
        content_panel_tree,
        poster_width=poster_width,
        poster_height=poster_height,
        title_h=poster_height * title_height_ratio,
        section_title_h=SECTION_TITLE_HEIGHT,  # Use 32 for section titles
        subsection_title_h=SUBSECTION_TITLE_HEIGHT,  # Use 25 for subsection titles
        shrink_margin=SHRINK_MARGIN
    )

    # Save layout results
    tree_split_results = {
        'poster_width': poster_width,
        'poster_height': poster_height,
        'panels': panels,
        'panel_arrangement': panel_arrangement,
        'figure_arrangement': figure_arrangement,
        'text_arrangement': text_arrangement,

        'title_panel': title_panel,
        'content_panel_tree': content_panel_tree,
        'title_h': poster_height * TITLE_HEIGHT_RATIO,

        'section_title_h': SECTION_TITLE_HEIGHT,
        'subsection_title_h': SUBSECTION_TITLE_HEIGHT,
        'title_font_size': TITLE_FONT_SIZE,
        'title_font_size_2': TITLE_FONT_SIZE_2,
        'section_title_font_size': SECTION_TITLE_FONT_SIZE,
        'subsection_title_font_size': SUBSECTION_TITLE_FONT_SIZE,
        'cont_font_size': CONT_FONT_SIZE,
        'shrink_margin': SHRINK_MARGIN,
        'units_per_inch': UNITS_PER_INCH,
        'font_name': FONT_NAME,
    }

    theme_title_text_color = THEME_TITLE_TEXT_COLOR
    theme_title_fill_color = THEME_TITLE_FILL_COLOR

    theme = {
        'panel_visible': True,
        'textbox_visible': False,
        'figure_visible': False,
        'panel_theme': {
            'color': (0, 255, 0),
            'thickness': 5,
            'line_style': 'solid',
        },
        'split_theme': {
            'color': (255, 0, 0),
            'thickness': 30,
            'line_style': 'solid',
        },
        'textbox_theme': None,
        'figure_theme': None,
    }

    try:
        with open(args.poster_bullet_path, 'r') as f:
            bullet_content = json.load(f)

        for i in range(len(text_arrangement)):
            text_arrangement[i]['content_for_ppt'] = bullet_content[i]['content_for_ppt']

        text_arrangement[0]['is_title'] = True
        style_bullet_content(text_arrangement[0]['content_for_ppt'], theme_title_text_color, theme_title_fill_color)
        style_bullet_content(text_arrangement[1]['content_for_ppt'], theme_title_text_color, theme_title_fill_color)

        for i in range(2, len(text_arrangement)):
            curr_content = text_arrangement[i]
            if 'title' in curr_content['textbox_name'].lower():
                if int(curr_content['panel_id']) < 10:
                    style_bullet_content(curr_content['content_for_ppt'], theme_title_text_color, theme_title_fill_color)
            else:
                text_arrangement[i]['content_for_ppt'] = [{"alignment": "center", "bullet": False, "level": 0, "font_size": 60, "runs": [{"text": " ", "bold": False}]}]

        visibility = True
        theme['figure_visible'] = True
        theme['textbox_visible'] = True
        theme['textbox_theme'] = {'color': (255, 0, 0), 'thickness': 5, 'line_style': 'solid'}
        theme['figure_theme'] = {'color': (0, 0, 255), 'thickness': 5, 'line_style': 'solid'}

    except:
        for i in range(len(text_arrangement)):
            text_arrangement[i]['content_for_ppt'] = None


    width_inch, height_inch, panel_arrangement_inches, figure_arrangement_inches, text_arrangement_inches = get_arrangments_in_inches(
        poster_width, poster_height, panel_arrangement, figure_arrangement, text_arrangement, 25
    )

    poster_code = generate_poster_code(
        panel_arrangement_inches,
        text_arrangement_inches,
        figure_arrangement_inches,
        presentation_object_name='poster_presentation',
        slide_object_name='poster_slide',
        utils_functions=utils_functions,
        slide_width=width_inch,
        slide_height=height_inch,
        img_path=None,
        save_path=f'{args.tmp_dir}/poster_final.pptx',
        visible=visibility,
        content=text_arrangement,
        theme=theme,
        tmp_dir=args.tmp_dir,
    )

    with open(f'{args.tmp_dir}/poster_code.py', 'w', encoding='utf-8') as f:
        f.write(poster_code)

    output, err = run_code(poster_code)
    ppt_to_images_with_title(f'{args.tmp_dir}/poster_final.pptx', args.tmp_dir, output_type='jpg', title=f"final")

