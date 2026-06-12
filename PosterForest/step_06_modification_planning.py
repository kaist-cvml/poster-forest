"""
STEP 6: Modification Planning — VLM/LLM Feedback Loop

Iteratively renders the poster, detects text overflow and layout imbalances,
and applies targeted fixes (font scaling, panel resizing, content trimming)
until the poster meets quality thresholds or max iterations are reached.

Input:  text_arrangement + figure_arrangement + panel_arrangement (Step 4/5)
Output: Corrected arrangements + final PPTX file
"""

from dotenv import load_dotenv
from utils.src.utils import get_json_from_response
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time

from PosterForest.step_07_pptx_generation import generate_poster_code
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from utils.src.utils import ppt_to_images_with_title
from PIL import Image, ImageFont

from utils.wei_utils import *

from utils.pptx_utils import *
from utils.critic_utils import *
import yaml
from jinja2 import Environment, StrictUndefined
import argparse
from collections import deque

from PosterForest.step_04_layout_hierarchy import get_arrangments_in_inches, generate_complete_layout, strip_section_numbering

from PIL import Image, ImageDraw, ImageOps
import matplotlib.font_manager as fm

load_dotenv()


# ---- Module-level helpers (shared by multiple call sites in step_06) ---------

def _render_to_jpg(
        poster_width, poster_height, panel_arr, fig_arr, text_arr,
        stem, tmp_dir, visible, theme, units_per_inch=25
    ):
    """Convert arrangements to PPTX, run it, export to JPG. Returns (output, err)."""
    wi, hi, pa, fa, ta = get_arrangments_in_inches(
        poster_width, poster_height, panel_arr, fig_arr, text_arr, units_per_inch
    )
    code = generate_poster_code(
        pa, ta, fa,
        presentation_object_name='poster_presentation',
        slide_object_name='poster_slide',
        utils_functions=utils_functions,
        slide_width=wi, slide_height=hi,
        img_path=None,
        save_path=f'{tmp_dir}/{stem}.pptx',
        visible=visible,
        content=text_arr,
        theme=theme,
        tmp_dir=tmp_dir,
    )
    output, err = run_code(code)
    ppt_to_images_with_title(f'{tmp_dir}/{stem}.pptx', tmp_dir, output_type='jpg', stem=stem)
    return output, err


def _apply_title_styles(text_arr, text_color, fill_color, subsec_text_color=None):
    """Apply section/subsection title styles in-place.

    subsec_text_color=None skips subsection styling (used in intermediate
    visualizations where only the section-level title color matters).
    """
    style_bullet_content(text_arr[0]['content_for_ppt'], text_color, fill_color)
    style_bullet_content(text_arr[1]['content_for_ppt'], text_color, fill_color)
    for entry in text_arr[2:]:
        if 'title' not in entry.get('textbox_name', '').lower():
            continue
        if _pid_is_section(entry['panel_id']):
            style_bullet_content(entry['content_for_ppt'], text_color, fill_color)
        elif subsec_text_color is not None:
            for _item in entry.get('content_for_ppt', []):
                _item['runs'][0]['color'] = subsec_text_color
                _item['runs'][0]['fill_color'] = None


def _set_content(bc_entry, panel_id, tname, result_json, lookup):
    """Update bc_entry and lookup together (always modified as a pair)."""
    bc_entry['content_for_ppt'] = result_json
    lookup[(str(panel_id), tname)] = result_json


def _bank_push(bank, panel_id, tname, items):
    """Prepend removed items to the content bank for this textbox."""
    if items:
        key = (str(panel_id), tname)
        bank[key] = items + bank.get(key, [])


# ---- Overflow Detection (merged from step_06_overflow_detection) -------------

def inches_to_px(inches, dpi=50):
    """Convert inches to pixels."""
    return int(inches * dpi)


def get_font_path(font_name, bold=False, italic=False):
    """Return system font path by name (cross-platform via matplotlib)."""
    prop = fm.FontProperties(
        family=font_name,
        style='italic' if italic else 'normal',
        weight='bold' if bold else 'normal'
    )
    try:
        font_path = fm.findfont(prop, fallback_to_default=False)
        if 'not found' in font_path:
            raise Exception
        return font_path
    except Exception:
        return fm.findfont(fm.FontProperties(family=['sans-serif']))


def tokenize_for_wrapping(text):
    """Split text into word-wrap units, preserving hyphen boundaries."""
    final_units = []
    for word in text.split():
        parts = word.split('-')
        if len(parts) > 1:
            for i, part in enumerate(parts):
                if part:
                    final_units.append(part + '-' if i < len(parts) - 1 else part)
        else:
            final_units.append(word)
    return final_units


def render_ppt_content_with_pillow(draw, panel_data, default_font="Arial", font_color=(0, 0, 0), dpi=50):
    """Simulate PPTX text rendering with Pillow; returns (status, overflow_p_idx, available_h, occupied_h).
    Status is one of: 'FIT', 'OVERFLOW', 'UNDERFLOW'."""
    status = 'FIT'

    left_px   = inches_to_px(panel_data['x'], dpi)
    top_px    = inches_to_px(panel_data['y'], dpi)
    width_px  = inches_to_px(panel_data['width'], dpi)
    height_px = inches_to_px(panel_data['height'], dpi)

    # Match actual PPTX add_textbox margins (utils_functions):
    #   margin_top = Pt(-5) → clips to ~0; margin_bottom = Pt(10); left = Pt(15); right = Pt(5)
    margin_top_px    = 0
    margin_left_px   = max(1, int(15 / 72 * dpi))   # Pt(15) in PPTX
    margin_right_px  = max(1, int(5 / 72 * dpi))    # Pt(5) in PPTX
    margin_bottom_px = max(1, int(10 / 72 * dpi))   # Pt(10) in PPTX (includes pptx_extra)
    indent_px        = 36
    # Calibrated line height: Arial with inter-paragraph spacing in PPTX.
    # 1.25 underestimates actual rendering (PPTX adds ~8% extra per paragraph for
    # space-before/after and internal rendering differences). 1.35 is empirically closer.
    line_height_multiplier = 1.35
    pptx_extra_bottom_px   = 0  # already included in margin_bottom_px above

    start_y_cursor  = top_px + margin_top_px
    bottom_boundary = top_px + height_px - margin_bottom_px - pptx_extra_bottom_px
    y_cursor        = start_y_cursor
    overflow_p_idx  = -1

    for p_idx, paragraph in enumerate(panel_data['content_for_ppt']):
        if status == 'OVERFLOW':
            break

        font_size_px = paragraph.get('font_size', 40)
        font_name    = paragraph.get('font_name', 'Courier New')
        font_bold    = paragraph['runs'][0].get('bold', False)
        level        = paragraph.get('level', 0)
        alignment    = paragraph.get('alignment', 'left')
        has_bullet   = paragraph.get('bullet', False)

        font        = ImageFont.truetype(get_font_path(font_name, font_bold), font_size_px)
        line_height = font_size_px * line_height_multiplier

        indent_px_      = level * indent_px
        text_start_x    = left_px + margin_left_px + indent_px_
        drawable_width  = width_px - margin_left_px - margin_right_px - indent_px_

        full_text       = "".join(run['text'] for run in paragraph['runs'])
        wrappable_units = tokenize_for_wrapping(full_text)

        text_lines = []
        if wrappable_units:
            if has_bullet:
                bullet_text = {0: '• ', 1: '◦ '}.get(level, '▪ ')
            else:
                bullet_text = ""
            current_line = bullet_text
            for unit in wrappable_units:
                sep = '' if not current_line or current_line.endswith('-') else ' '
                if not current_line:
                    current_line = unit
                    continue
                if font.getlength(current_line + sep + unit) <= drawable_width:
                    current_line += sep + unit
                else:
                    text_lines.append(current_line)
                    current_line = unit
            text_lines.append(current_line)

        overflow = False
        for line in text_lines:
            if not overflow and (y_cursor + line_height) > bottom_boundary and y_cursor > top_px:
                status = 'OVERFLOW'
                overflow = True
                overflow_p_idx = p_idx

            line_width = font.getlength(line)
            final_x = text_start_x
            if alignment == 'center':
                final_x = text_start_x + (drawable_width - line_width) / 2
            elif alignment == 'right':
                final_x = text_start_x + (drawable_width - line_width)

            draw.text((final_x, y_cursor), line, font=font, fill=font_color)
            y_cursor += line_height

    # bottom_boundary already excludes the margin, so text_occupied is simply
    # the y distance drawn — adding margin again would double-count and inflate
    # fill ratio, causing false OVERFLOW on content that actually fits in PPTX.
    text_occupied_height  = y_cursor - start_y_cursor
    total_available_height = bottom_boundary - start_y_cursor
    empty_space_height    = total_available_height - text_occupied_height

    if status == 'FIT' and y_cursor > start_y_cursor:
        # Trigger UNDERFLOW when empty space exceeds 25% of available height (i.e., fill < 75%).
        # Scientific posters should use space efficiently; 55% fill leaves too much dead area.
        _underflow_ratio = empty_space_height / max(total_available_height, 1.0)
        if empty_space_height > 2.0 * line_height and _underflow_ratio > 0.25:
            status = 'UNDERFLOW'

    border_color = {'OVERFLOW': 'blue', 'UNDERFLOW': 'orange'}.get(status, 'green')
    draw.rectangle([(left_px, top_px), (left_px + width_px, top_px + height_px)],
                   outline=border_color, width=5)

    return (status, overflow_p_idx, total_available_height, text_occupied_height)



def _pid(panel_id):
    """Zero-pad panel_id for lexicographic ordering.
    0 -> '00', 1 -> '01', '1_1' -> '01_01', '12_3' -> '12_03'.
    Non-numeric parts (e.g. 'title') are left as-is."""
    parts = []
    for p in str(panel_id).split("_"):
        try:
            parts.append(f"{int(p):02d}")
        except ValueError:
            parts.append(p)
    return "_".join(parts)


def _pid_is_section(panel_id):
    """Return True when panel_id is a single-digit integer (top-level section)."""
    try:
        return int(str(panel_id)) < 10
    except (ValueError, TypeError):
        return False


# ---- Step 6-2: Textbox Overflow Simulation ----------------------------------
def render_textbox(
        text_arrangement, textbox_content, tmp_dir,
        shrink_margin=3, units_per_inch=25, num_attempt=0, simulation_only=False
    ):
    """Render a textbox and return overflow status.

    When simulation_only=True, skip PPTX generation (slow LibreOffice conversion)
    and only run the fast Pillow simulation. Use for intermediate overflow checks.
    Set simulation_only=False only for the final render saved as debug output.
    """
    panel_id = text_arrangement['panel_id']

    arrangement = copy.deepcopy(text_arrangement)
    arrangement['x'] = 1.5
    arrangement['y'] = 0.6
    arrangement['width'] = (arrangement['width'] / units_per_inch)
    arrangement['height'] = (arrangement['height'] / units_per_inch)

    # PPTX slide includes padding around the textbox for context.
    # These are always ≥ textbox dims + padding, so they need no separate guard.
    width_inch  = arrangement['width']  + 3
    height_inch = arrangement['height'] + 1.4

    # Only skip truly degenerate (non-positive) dimensions.
    # Small-but-positive panels are valid — the overflow detector handles them.
    if arrangement['width'] <= 0 or arrangement['height'] <= 0:
        print(f"  ✗ Panel {panel_id}: degenerate textbox "
              f"({arrangement['width']:.3f}\" × {arrangement['height']:.3f}\") — skipping.")
        return None
    if arrangement['height'] < 0.5:
        print(f"  ⚠ Panel {panel_id}: textbox height {arrangement['height']:.3f}\" is small "
              f"(< 0.5\") — content will likely overflow severely.")

    if not simulation_only:
        poster_code = generate_poster_code(
            [],
            [arrangement],
            [],
            presentation_object_name='poster_presentation',
            slide_object_name='poster_slide',
            utils_functions=utils_functions,
            slide_width=width_inch,
            slide_height=height_inch,
            img_path='placeholder.jpg',
            save_path=f'{tmp_dir}/step6_panel_{_pid(panel_id)}_textbox_{num_attempt:02d}.pptx',
            visible=True,
            content=textbox_content,
            only_textbox=True,
            tmp_dir=tmp_dir,
        )
        stem = f"step6_panel_{_pid(panel_id)}_textbox_{num_attempt:02d}"
        output, err = run_code(poster_code)
        ppt_to_images_with_title(f'{tmp_dir}/{stem}.pptx', tmp_dir, output_type='jpg', stem=stem)

    # Pillow simulation for overflow detection (fast, no PPTX needed)

    if textbox_content:
        arrangement['content_for_ppt'] = textbox_content

    _sim_dpi = 72.5  # Pillow simulation DPI (not the layout units_per_inch parameter)

    img_width = inches_to_px(width_inch, dpi=_sim_dpi)
    img_height = inches_to_px(height_inch, dpi=_sim_dpi)

    image = Image.new("RGB", (img_width, img_height), "white")
    drawer = ImageDraw.Draw(image)

    status, p_idx, available_height, occupied_height = render_ppt_content_with_pillow(
        drawer,
        arrangement,
        dpi=_sim_dpi,
    )

    image.save(f"{tmp_dir}/step6_panel_{_pid(panel_id)}_textbox_{num_attempt:02d}_simulation.png")

    return status, p_idx, available_height, occupied_height


# ---- Step 6-1: Panel Rendering ----------------------------------------------
def render_panel(
        panel_arrangement, text_arrangement_list, figure_arrangement_list,
        tmp_dir, shrink_margin=3, units_per_inch=25
    ):
    """Render a single panel with all its textboxes and figures to a PNG image for VLM review."""
    # scaling
    panel_id = panel_arrangement['panel_id']

    width_inch = (panel_arrangement['width'] / units_per_inch) + 1
    height_inch = (panel_arrangement['height'] / units_per_inch) + 1

    # width_inch = panel_width_inches + 1, so <= 1 means non-positive panel width.
    if width_inch <= 1 or height_inch <= 1:
        print(f"  ✗ Panel {panel_id}: non-positive panel dimensions — skipping render.")
        return None

    p_arr = copy.deepcopy(panel_arrangement)

    scale_t = []
    for t in text_arrangement_list:
        t_arr = copy.deepcopy(t)
        t_arr['x'] = ((t_arr['x'] - p_arr['x'])/ units_per_inch ) + 0.5
        t_arr['y'] = ((t_arr['y'] - p_arr['y'])/ units_per_inch ) + 0.5
        t_arr['width'] = (t_arr['width'] / units_per_inch)
        t_arr['height'] = (t_arr['height'] / units_per_inch)
        # Skip degenerate textboxes; continue rendering the rest of the panel.
        if t_arr['width'] <= 0 or t_arr['height'] <= 0:
            print(f"  ⚠ Panel {panel_id}: skipping degenerate textbox "
                  f"({t_arr['width']:.3f}\" × {t_arr['height']:.3f}\")")
            continue
        scale_t.append(t_arr)

    scale_f = []
    for f in figure_arrangement_list:
        f_arr = copy.deepcopy(f)
        f_arr['x'] = ((f_arr['x'] - p_arr['x'])/ units_per_inch ) + 0.5
        f_arr['y'] = ((f_arr['y'] - p_arr['y'])/ units_per_inch ) + 0.5
        f_arr['width'] = (f_arr['width'] / units_per_inch)
        f_arr['height'] = (f_arr['height'] / units_per_inch)
        # Skip degenerate figures; continue rendering the rest of the panel.
        if f_arr['width'] <= 0 or f_arr['height'] <= 0:
            print(f"  ⚠ Panel {panel_id}: skipping degenerate figure "
                  f"({f_arr['width']:.3f}\" × {f_arr['height']:.3f}\")")
            continue
        scale_f.append(f_arr)

    p_arr['x'] = 0.5
    p_arr['y'] = 0.5
    p_arr['width'] = (p_arr['width'] / units_per_inch)
    p_arr['height'] = (p_arr['height'] / units_per_inch)


    if p_arr['width'] <= 0 or p_arr['height'] <= 0:
        print(f"  ✗ Panel {panel_id}: non-positive panel dimensions after scaling — skipping render.")
        return None

    poster_code = generate_poster_code(
        [p_arr],
        scale_t,
        scale_f,
        presentation_object_name='poster_presentation',
        slide_object_name='poster_slide',
        utils_functions=utils_functions,
        slide_width=width_inch,
        slide_height=height_inch,
        img_path=None,
        save_path=f'{tmp_dir}/step6_panel_{_pid(panel_id)}.pptx',
        visible=True,
        content=scale_t,
        only_textbox=False,
        tmp_dir=tmp_dir,
    )

    output, err = run_code(poster_code)
    stem = f"step6_panel_{_pid(panel_id)}"
    ppt_to_images_with_title(f'{tmp_dir}/{stem}.pptx', tmp_dir, output_type='jpg', stem=stem)
    img = Image.open(f'{tmp_dir}/{stem}.jpg')

    return img


# ---- Internal Node Helper Functions -----------------------------------------

def _get_all_panels_in_subtree(start_node):
    """Collect panel_data from all leaf/group nodes in subtree (DFS)."""
    panels = []

    def _traverse(node):
        if not node:
            return
        node_type = node.get("type", "")
        if node_type == "leaf" or node_type == "group":
            panels.append(node.get("panel_data"))
        if "internal" in node_type:
            for child in node.get("children", []):
                _traverse(child)
        elif node_type == "group":
            _traverse(node.get("child_tree"))

    _traverse(start_node)
    return panels


def get_internal_nodes_with_panels_bfs(root_node):
    """BFS the layout tree; return list of {internal_node_id, left_panels, right_panels}."""
    if not root_node:
        return []

    results = []
    queue = deque([root_node])

    while queue:
        current_node = queue.popleft()
        node_type = current_node.get("type", "")

        if "internal" in node_type:
            internal_node_id = node_type
            children = current_node.get("children", [])
            left_panels, right_panels = [], []
            if len(children) == 2:
                left_panels  = _get_all_panels_in_subtree(children[0])
                right_panels = _get_all_panels_in_subtree(children[1])
            results.append({
                "internal_node_id": internal_node_id,
                "left_panels": left_panels,
                "right_panels": right_panels
            })

        if "internal" in node_type:
            queue.extend(current_node.get("children", []))
        elif node_type == "group":
            child_tree = current_node.get("child_tree")
            if child_tree:
                queue.append(child_tree)

    return results


def get_internal_split_coords(root_node, target_node_id, initial_x=0, initial_y=0, initial_w=1920, initial_h=1080, section_title_h=32, decay_factor=0.85):
    """Find an internal node and return (c1_coords, c2_coords, split_type, ratio), or None."""
    def _find_and_calculate(current_node, x, y, w, h, group_depth=0):
        if current_node.get("type") == target_node_id:
            if "internal" not in current_node.get("type"):
                return None
            r = current_node["split_ratio"]
            if current_node["split_type"] == "horizontal":
                c1_info = {"panel_id": -1, "textbox_name": "split_test", "x": x, "y": y, "width": w, "height": h * r}
                c2_info = {"panel_id": -1, "textbox_name": "split_test", "x": x, "y": y + h * r, "width": w, "height": h - h * r}
            elif current_node["split_type"] == "vertical":
                c1_info = {"panel_id": -1, "textbox_name": "split_test", "x": x, "y": y, "width": w * r, "height": h}
                c2_info = {"panel_id": -1, "textbox_name": "split_test", "x": x + w * r, "y": y, "width": w - w * r, "height": h}
            return (c1_info, c2_info, current_node["split_type"], r)

        node_type = current_node.get("type")
        if node_type == "leaf":
            return None
        elif "internal" in node_type:
            c1, c2, r = current_node["children"][0], current_node["children"][1], current_node["split_ratio"]
            if current_node["split_type"] == "horizontal":
                found = _find_and_calculate(c1, x, y, w, h * r, group_depth)
                if found: return found
                return _find_and_calculate(c2, x, y + h * r, w, h - h * r, group_depth)
            elif current_node["split_type"] == "vertical":
                found = _find_and_calculate(c1, x, y, w * r, h, group_depth)
                if found: return found
                return _find_and_calculate(c2, x + w * r, y, w - w * r, h, group_depth)
        elif node_type == "group":
            current_title_height = section_title_h * (decay_factor ** group_depth)
            return _find_and_calculate(
                current_node['child_tree'],
                x, y + current_title_height, w, h - current_title_height,
                group_depth + 1
            )
        return None

    return _find_and_calculate(root_node, initial_x, initial_y, initial_w, initial_h)


# ---- Step 6-3: Layout Ratio Adjustment --------------------------------------
def update_internal_node_ratio(root_node, target_id, new_ratio):
    """Find internal node by ID and update its split_ratio in-place. Returns True on success."""
    if not (0.0 < new_ratio < 1.0):
        print(f"Warning: new_ratio {new_ratio} must be between 0 and 1.")
        return False

    def _find_and_update(node):
        if not node:
            return False
        node_type = node.get("type")
        if node_type == target_id:
            current_ratio = node['split_ratio']
            node['split_ratio'] = new_ratio
            print(f"   split_ratio {current_ratio:.4f} → {new_ratio:.4f}  [{target_id}]")
            return True
        if "internal" in node_type:
            for child in node.get("children", []):
                if _find_and_update(child):
                    return True
        elif node_type == "group":
            if node.get('child_tree'):
                if _find_and_update(node['child_tree']):
                    return True
        return False

    if not _find_and_update(root_node):
        print(f"Warning: Internal Node '{target_id}' not found.")
        return False
    return True





def _mp_normalize_title(s):
    """Lowercase and strip section numbers/punctuation for fuzzy title matching."""
    s = s.lower()
    s = re.sub(r'^\d+[\.\d\s]*', '', s)
    s = re.sub(r'[^a-z\s]', '', s)
    return s.strip()


def _text_to_bullets(text, max_n, font_size, font_name):
    """Convert raw text to a bullet list without LLM.

    Used to fill sparse panels from the original section content stored in the
    poster tree, avoiding an extra LLM round-trip.
    """
    text = text.strip()
    # Try sentence-boundary split first
    parts = re.split(r'(?<=[.!?])\s+', text)
    sentences = [p.strip() for p in parts if len(p.strip()) > 15][:max_n]
    if len(sentences) < 2:
        # Fallback: split on newlines / semicolons
        parts = re.split(r'[\n;]', text)
        sentences = [p.strip() for p in parts if len(p.strip()) > 15][:max_n]
    return [
        {"alignment": "left", "bullet": True, "level": 0,
         "font_name": font_name, "font_size": font_size,
         "runs": [{"text": s.rstrip('.') + '.', "bold": False}]}
        for s in sentences
    ]


def _mp_find_raw_section_text(panel_name, raw_sections):
    """Match panel_name to raw_content section for richer action-template context.

    Threshold lowered to 0.10 and containment bonus added (same logic as
    _find_best_section_text in step_05) so subsection names such as
    "Multi-Head Attention" still match numbered paper section titles.
    """
    norm_target = _mp_normalize_title(panel_name)
    target_words = set(norm_target.split())
    if not target_words or not raw_sections:
        return None
    best_score, best_content = 0.0, None
    for sec in raw_sections:
        norm_title = _mp_normalize_title(sec.get('title', ''))
        title_words = set(norm_title.split())
        if not title_words:
            continue
        overlap = len(target_words & title_words)
        union = len(target_words | title_words)
        score = overlap / union if union else 0
        # Containment bonus: all target words present in title → reliable match
        if target_words and target_words.issubset(title_words):
            score = max(score, 0.25)
        if score > best_score:
            best_score = score
            best_content = sec.get('content', '')
    return best_content if best_score > 0.10 else None


def _item_in_result(item, result_json, threshold=40):
    """Return True if item's text already appears in result_json (dedup check)."""
    item_text = (item.get('runs') or [{}])[0].get('text', '').lower()[:threshold]
    if not item_text:
        return False
    return any(
        (b.get('runs') or [{}])[0].get('text', '').lower()[:threshold] == item_text
        for b in result_json
    )


def _deterministic_fit(
        result_json, available_height, occupied_height, base_font_size,
        min_fs=30, target_fill=0.80, bank_out=None, min_items=2
    ):
    """Truncate content to reach target_fill of available_height (no font size changes).

    target_fill controls the desired fill ratio (default 0.80 = 80%).
    min_items: minimum bullets to keep (default 2; pass 1 for Phase-4 last-resort).

    bank_out: if a list is passed, removed items are appended to it so the caller
    can store them in _content_bank for later fill recovery.
    """
    if not result_json:
        return result_json
    _fill = occupied_height / max(available_height, 1.0)
    if _fill <= target_fill:
        return result_json

    result_json = list(result_json)
    original_n = len(result_json)

    # Proportional truncation — preserve font size
    target_n = max(min_items, int(original_n * target_fill / _fill))
    _removed = result_json[target_n:]
    result_json = result_json[:target_n]
    if bank_out is not None and _removed:
        bank_out.extend(_removed)

    return result_json


# ---- Step 6 [Entry]: Content + Layout → Overflow-corrected Arrangements ----
# Calls: 6-1 render_panel  →  6-2 render_textbox  (overflow simulation)
#        6-3 update_internal_node_ratio  (ratio fix on overflow)
#        generate_complete_layout  (step_04, re-layout after ratio change)
#        generate_poster_code      (step_07, interim render for VLM feedback)
def step_06_modification_planning(
        args, LLM_config, VLM_config,
        max_attempt=3, bullet_content_path=None, tree_split_path=None, raw_content_path=None
    ):
    """Iterative VLM→LLM feedback loop: render poster panels, critique with VLM, revise layout/content with LLM."""
    total_input_token_t, total_output_token_t = 0, 0
    total_input_token_v, total_output_token_v = 0, 0

    # Use provided paths or fallback to static paths for backward compatibility
    if bullet_content_path is None:
        bullet_content_path = f'contents/<{args.model_name_t}_{args.model_name_v}>_{args.poster_name}_bullet_point_content_{args.index}_.json'
    if tree_split_path is None:
        tree_split_path = f'tree_splits/<{args.model_name_t}_{args.model_name_v}>_{args.poster_name}_tree_split_{args.index}.json'

    # information about content
    with open(bullet_content_path, 'r') as f:
        intial_bullet_content = json.load(f)

    # information about layout
    with open(tree_split_path, 'r') as f:
        tree_split_results = json.load(f)

    panels = tree_split_results['panels']
    panel_arrangement = tree_split_results['panel_arrangement']
    figure_arrangement = tree_split_results['figure_arrangement']
    text_arrangement = tree_split_results['text_arrangement']
    content_panel_tree = tree_split_results['content_panel_tree']
    poster_width = tree_split_results['poster_width']
    poster_height = tree_split_results['poster_height']
    title_h = tree_split_results['title_h']
    section_title_h = tree_split_results['section_title_h']
    subsection_title_h = tree_split_results['subsection_title_h']
    units_per_inch = tree_split_results['units_per_inch']
    shrink_margin = tree_split_results['shrink_margin']
    font_name = tree_split_results['font_name']
    font_size = tree_split_results['cont_font_size']
    title_panel = tree_split_results['title_panel']
    subsec_font_size = tree_split_results['subsection_title_font_size']
    sec_font_size = tree_split_results['section_title_font_size']

    sibling_content_map = {}
    parent_content_map = {}

    def find_sibling_content(panel_list):
        """Build sibling_content_map by walking the panel tree for each node's siblings."""
        if not panel_list:
            return

        total_content = []
        for i, current_panel in enumerate(panel_list):
            current_panel_id = current_panel.get("panel_id")
            siblings_content = []

            if "content" in current_panel:
                total_content.append(current_panel["content"])
                for j, sibling_panel in enumerate(panel_list):
                    sibling_panel_id = sibling_panel.get("panel_id")
                    if i != j:
                        if sibling_panel['children'] == [] and "content" in sibling_panel:
                            siblings_content.append(sibling_panel["content"])
                        else:
                            if sibling_panel_id not in parent_content_map:
                                parent_content_map[sibling_panel_id] = find_sibling_content(sibling_panel["children"])
                            siblings_content.append(parent_content_map[sibling_panel_id])

            if "children" in current_panel and current_panel["children"] != []:
                if current_panel_id not in parent_content_map:
                    parent_content_map[current_panel_id] = find_sibling_content(current_panel["children"])

            if siblings_content != []:
                sibling_content_map[current_panel_id] = siblings_content

        return total_content if total_content else None


    find_sibling_content(panels)

    layout_agent_name = '_layout_agent_single'
    action_agent_name = '_action_agent'
    critic_agent_name = '_critic_agent'

    with open(f"utils/prompt_templates/{layout_agent_name}.yaml", "r", encoding='utf-8') as f:
        layout_config = yaml.safe_load(f)

    with open(f"utils/prompt_templates/{action_agent_name}.yaml", "r", encoding='utf-8') as f:
        action_config = yaml.safe_load(f)

    with open(f"utils/prompt_templates/{critic_agent_name}.yaml", "r", encoding='utf-8') as f:
        critic_config = yaml.safe_load(f)

    layout_model = make_model(VLM_config)
    action_model = make_model(LLM_config)
    critic_model = make_model(VLM_config)

    jinja_env = Environment(undefined=StrictUndefined)

    layout_template = jinja_env.from_string(layout_config["template"])

    inc_action_template = jinja_env.from_string(action_config["inc_template"])
    dec_action_template = jinja_env.from_string(action_config["dec_template"])
    tweak_action_template = jinja_env.from_string(action_config["tweak_template"])

    critic_template = jinja_env.from_string(critic_config["template"])

    layout_sys_msg = layout_config['system_prompt']
    action_sys_msg = action_config['system_prompt']
    critic_sys_msg = critic_config['system_prompt']


    layout_agent = ChatAgent(
        system_message=layout_sys_msg,
        model=layout_model,
        message_window_size=5
    )

    action_agent = ChatAgent(
        system_message=action_sys_msg,
        model=action_model,
        message_window_size=1
    )

    critic_agent = ChatAgent(
        system_message=critic_sys_msg,
        model=critic_model,
        message_window_size=1
    )


    # Load raw_content sections for richer action-template context
    raw_sections = []
    if raw_content_path and os.path.exists(raw_content_path):
        try:
            _rc = json.load(open(raw_content_path, 'r'))
            raw_sections = _rc.get('sections', [])
            print(f"   📄 Loaded {len(raw_sections)} raw sections for action context")
        except Exception as e:
            print(f"   ⚠️ Could not load raw_content for modification planning: {e}")

    bullet_content = copy.deepcopy(intial_bullet_content)

    # Activity log (written incrementally; read by viewer for live progress)
    _act_log = {"events": [], "running": True}
    _act_log_path = os.path.join(args.tmp_dir, "step6_activity_log.json")

    def _log_event(evt: dict):
        evt["ts"] = round(time.time(), 2)
        _act_log["events"].append(evt)
        try:
            with open(_act_log_path, "w") as _lf:
                json.dump(_act_log, _lf)
        except Exception:
            pass


    theme_title_text_color = (255, 255, 255)
    theme_title_fill_color = (63, 86, 147)
    theme_subsec_title_text_color = (32, 56, 100)  # dark navy font for subsection titles

    theme = {
        'panel_visible': True,
        'textbox_visible': False,
        'figure_visible': False,
        'panel_theme': {
            'color': (255, 255, 255),
            # 'color': theme_title_fill_color,
            'thickness': 5,
            'line_style': 'solid',
        },
        'split_theme': {
            'color':(255, 0, 0),
            'thickness': 30,
            'line_style': 'solid',
        },
        'textbox_theme': None,
        'figure_theme': None,
    }


    num_attempts = 0
    LAYOUT_MODIFICATION = True
    CONTENT_MODIFICATION = True
    final_arrangements = {}

    # Stores (width, height) per textbox from the previous attempt's layout.
    # Used by Option-C selective reset: only panels whose dimensions changed
    # are reset to initial_bullet_content; stable panels keep their content.
    _prev_text_dims = {}   # key: (panel_id, textbox_name) → (width, height)
    _content_bank = {}    # key: (str(panel_id), textbox_name) → removed bullet items for fill recovery

    # Progressive proactive boost for main/overview/architecture figures
    # Applied before the loop AND at the start of each attempt so figures grow
    # progressively: attempt 0 → 65%, attempt 1 → 69%, attempt 2+ → 73% max.
    #
    # Bug fix (old code): `_node.get('split_ratio', 0.5)` always returned 0.5
    # because get_internal_nodes_with_panels_bfs() doesn't include split_ratio.
    # Now we use get_internal_split_coords() to read the ACTUAL current ratio.
    def _is_main_panel(panel_list):
        for p in (panel_list or []):
            if p and p.get('asset_path'):
                if p.get('is_main_figure'):
                    return True
        return False

    def _has_any_figure(panel_list):
        return any((p or {}).get('asset_path') or (p or {}).get('gp', 0) > 0 for p in (panel_list or []))

    def _apply_figure_boost(tree, target_main=0.65, target_fig=0.45):
        """Boost figure-panel split ratios toward target_main/target_fig.

        Uses actual current split ratios (via get_internal_split_coords) so the
        boost is proportional to how much growth is still needed each iteration.
        Returns True if any ratio was changed.
        """
        changed = False
        _nodes = get_internal_nodes_with_panels_bfs(tree)
        for _node in _nodes:
            _iid   = _node.get('internal_node_id')
            _left  = _node.get('left_panels', [])
            _right = _node.get('right_panels', [])
            _lm, _rm = _is_main_panel(_left), _is_main_panel(_right)
            _lf, _rf = _has_any_figure(_left), _has_any_figure(_right)

            # Read actual current split ratio
            _split_data = get_internal_split_coords(
                tree, _iid,
                initial_x=0, initial_y=title_h,
                initial_w=poster_width, initial_h=poster_height - title_h,
                section_title_h=section_title_h,
                decay_factor=subsection_title_h / section_title_h
            )
            _ratio = _split_data[3] if _split_data else 0.5

            if _lm and not _rm and _ratio < target_main:
                _nr = min(target_main + 0.07, _ratio + 0.15)
                _nr = max(0.15, min(0.85, _nr))
                print(f"   ✨ Main-fig boost: {_iid}  {_ratio:.2f} → {_nr:.2f} (left, target={target_main:.2f})")
                update_internal_node_ratio(tree, _iid, _nr)
                changed = True
            elif _rm and not _lm and (1.0 - _ratio) < target_main:
                _nr = max(1.0 - target_main - 0.07, _ratio - 0.15)
                _nr = max(0.15, min(0.85, _nr))
                print(f"   ✨ Main-fig boost: {_iid}  {_ratio:.2f} → {_nr:.2f} (right, target={target_main:.2f})")
                update_internal_node_ratio(tree, _iid, _nr)
                changed = True
            elif _lf and not _rf and not _lm and _ratio < target_fig:
                _nr = min(target_fig + 0.05, _ratio + 0.10)
                _nr = max(0.15, min(0.85, _nr))
                print(f"   🖼️  Fig boost: {_iid}  {_ratio:.2f} → {_nr:.2f} (left)")
                update_internal_node_ratio(tree, _iid, _nr)
                changed = True
            elif _rf and not _lf and not _rm and (1.0 - _ratio) < target_fig:
                _nr = max(1.0 - target_fig - 0.05, _ratio - 0.10)
                _nr = max(0.15, min(0.85, _nr))
                print(f"   🖼️  Fig boost: {_iid}  {_ratio:.2f} → {_nr:.2f} (right)")
                update_internal_node_ratio(tree, _iid, _nr)
                changed = True
        return changed

    print("\n🔍 Proactive figure panel boost (pre-loop, attempt 0)...")
    # target_main=0.65: main figures get at least 65% of their split.
    # target_fig=0.45: regular figures get at least 45% (near equal-split with text).
    if _apply_figure_boost(content_panel_tree, target_main=0.65, target_fig=0.45):
        panel_arrangement, figure_arrangement, text_arrangement = generate_complete_layout(
            title_panel, content_panel_tree, poster_width, poster_height,
            title_h, section_title_h, subsection_title_h, shrink_margin
        )


    # Build a (panel_id, textbox_id) → content_for_ppt lookup from bullet_content.
    # Using a dict instead of positional indexing makes syncing robust to layout
    # changes that add or remove text boxes (e.g. a tiny panel growing large enough
    # to need a text box after update_internal_node_ratio).
    def _bc_key(entry):
        # Key by (panel_id, textbox_name).
        # textbox_name alone is NOT unique — all body panels share 'p<Content>_body'.
        # (panel_id, textbox_id) is also broken: split_textbox() shallow-copies and
        # leaves both title rows with textbox_id=0.
        # (panel_id, textbox_name) is always unique because panel_ids are distinct and
        # textbox_name differentiates split title rows ('_t0' vs '_t1').
        return (str(entry.get('panel_id', '')), entry.get('textbox_name', ''))

    def _build_content_lookup(bc):
        lookup = {}
        for entry in bc:
            key = _bc_key(entry)
            if key[1]:  # textbox_name must be non-empty
                lookup[key] = entry.get('content_for_ppt', [])
        return lookup

    _bullet_lookup = _build_content_lookup(bullet_content)

    def _sync_text_arrangement(text_arr, bc_lookup, bc_list):
        """Populate content_for_ppt for every entry in text_arr.

        Prefer dict lookup (panel_id, textbox_name) so that text boxes added or
        removed after a layout change don't cause an IndexError.  Falls back to the
        existing content_for_ppt already on the entry (from a previous sync),
        then to a raw-content placeholder.
        """
        for entry in text_arr:
            key = _bc_key(entry)
            if key in bc_lookup:
                entry['content_for_ppt'] = copy.deepcopy(bc_lookup[key])
            elif 'content_for_ppt' not in entry or not entry['content_for_ppt']:
                # New text box introduced by a layout change — use raw content
                raw = entry.get('content', '')
                if raw:
                    entry['content_for_ppt'] = [{
                        "alignment": "left", "bullet": True, "level": 0,
                        "font_size": 40, "runs": [{"text": str(raw)[:300]}]
                    }]
                else:
                    entry['content_for_ppt'] = []

    def _enforce_title_format(text_arr):
        """Title textboxes must never carry bullet formatting; strip leading section numbers."""
        for entry in text_arr:
            if 'title' in entry.get('textbox_name', '').lower():
                for item in entry.get('content_for_ppt', []):
                    item['bullet'] = False
                    for run in item.get('runs', []):
                        if run.get('text'):
                            run['text'] = strip_section_numbering(run['text'])

    while True:

        print(f" ================ Attempt {num_attempts} ================ ")
        _log_event({"type": "attempt_start", "attempt": num_attempts})

        # Progressive figure boost removed: boosting figure panels on overflow attempts
        # is counterproductive — figure panels share splits with text siblings, so growing
        # the figure side SHRINKS the text side, which makes text overflow WORSE.
        # The pre-loop boost already sets a conservative figure allocation; the layout
        # agent handles per-split fine-tuning via VLM feedback each iteration.
        if num_attempts > 0:
            pass  # no progressive boost; rely on layout agent for corrections

        # Rebuild lookup from current bullet_content so that corrections from previous
        # attempts (or from within this attempt's content loop) are not overwritten.
        _bullet_lookup = _build_content_lookup(bullet_content)

        _sync_text_arrangement(text_arrangement, _bullet_lookup, bullet_content)
        _enforce_title_format(text_arrangement)
        # Keep the legacy new_content list for code below that still references it
        new_content = [t['content_for_ppt'] for t in text_arrangement]

        text_arrangement_index = 2 # Skip the poster title and author

        _apply_title_styles(text_arrangement, theme_title_text_color, theme_title_fill_color,
                            subsec_text_color=theme_subsec_title_text_color)

        stem = f"step6_attempt_{num_attempts:02d}_start"
        _render_to_jpg(poster_width, poster_height,
                       panel_arrangement, figure_arrangement, text_arrangement,
                       stem, args.tmp_dir, visible=False, theme=theme)


        if LAYOUT_MODIFICATION:

            print(" --- Layout Modification ---")
            internal_nodes = get_internal_nodes_with_panels_bfs(content_panel_tree)

            for iteration, internal_node in enumerate(internal_nodes):

                internal_id = internal_node["internal_node_id"]
                left_panels = internal_node["left_panels"]
                right_panels = internal_node["right_panels"]

                print(f" === Internal Node: {internal_id}")
                print(' [1] render split')

                theme_visible = copy.deepcopy(theme)
                theme_visible['figure_visible'] = True
                theme_title_fill_color_ = (147, 186, 69)
                theme_visible['panel_theme']['color'] = theme_title_fill_color_
                theme_visible['textbox_visible'] = True
                theme_visible['textbox_theme'] = {'color': theme_title_fill_color_, 'thickness': 5, 'line_style': 'solid'}
                theme_visible['figure_theme'] = {'color': (0, 0, 255), 'thickness': 10, 'line_style': 'solid'}

                test_panel_arrangement = copy.deepcopy(panel_arrangement)
                test_figure_arrangement = copy.deepcopy(figure_arrangement)

                test_text_arrangement_ = copy.deepcopy(text_arrangement)
                test_text_arrangement = []
                test_text_arrangement_idx = []
                for idx, t in enumerate(test_text_arrangement_):
                        test_text_arrangement.append(t)
                        test_text_arrangement_idx.append(idx)

                left_box_info, right_box_info, split_type, current_split_ratio = get_internal_split_coords(
                    content_panel_tree,
                    internal_id,
                    initial_x=0,
                    initial_y=title_h,
                    initial_w=poster_width,
                    initial_h=poster_height - title_h,
                    section_title_h=section_title_h,
                    decay_factor=subsection_title_h/section_title_h
                )


                _sync_text_arrangement(test_text_arrangement, _bullet_lookup, bullet_content)
                _enforce_title_format(test_text_arrangement)

                left_box_info['content_for_ppt'] = [
                    {
                        "alignment": "center", "bullet": False, "level": 0,  "font_size": 60,
                        "runs": [{"text": " ", "bold": False}]
                    }
                ]
                right_box_info['content_for_ppt'] = [
                    {
                        "alignment": "center", "bullet": False, "level": 0,  "font_size": 60,
                        "runs": [{"text": " ", "bold": False}]
                    }
                ]

                test_text_arrangement.append(left_box_info)
                test_text_arrangement.append(right_box_info)

                # Section titles use visible-theme fill_color_; subsection titles keep real
                # content (no subsec_text_color override) so VLM sees actual fill state.
                _apply_title_styles(test_text_arrangement, theme_title_text_color, theme_title_fill_color_)

                stem = f"step6_attempt_{num_attempts:02d}_iter_{iteration:02d}_{internal_id}"
                output, err = _render_to_jpg(
                    poster_width, poster_height,
                    test_panel_arrangement, test_figure_arrangement, test_text_arrangement,
                    stem, args.tmp_dir, visible=True, theme=theme_visible,
                )
                img = Image.open(f'{args.tmp_dir}/{stem}.jpg')

                pad = 0
                x = min(left_box_info['x'], right_box_info['x']) - pad
                y = min(left_box_info['y'], right_box_info['y']) - pad
                if split_type == 'horizontal':
                    if left_box_info['width'] != right_box_info['width']:
                        print(f"Warning: left_box_info['width'] ({left_box_info['width']}) != right_box_info['width'] ({right_box_info['width']})")
                    w = left_box_info['width']
                    h = (left_box_info['height'] + right_box_info['height'])
                elif split_type == 'vertical':
                    if left_box_info['height'] != right_box_info['height']:
                        print(f"Warning: left_box_info['height'] ({left_box_info['height']}) != right_box_info['height'] ({right_box_info['height']})")
                    w = (left_box_info['width'] + right_box_info['width'])
                    h = left_box_info['height']
                w += pad*2; h += pad*2

                rescale_factor = img.size[0] / poster_width
                x = int(x * rescale_factor)
                y = int(y * rescale_factor)
                w = int(w * rescale_factor)
                h = int(h * rescale_factor)
                box = (x, y, x + w, y + h)
                cropped_ = img.crop(box)
                cropped = ImageOps.expand(cropped_, border=100, fill='white')
                os.makedirs(args.tmp_dir, exist_ok=True)
                cropped.save(f'{args.tmp_dir}/{stem}_cropped.jpg')
                cropped = Image.open(f'{args.tmp_dir}/{stem}_cropped.jpg')


                print(' [2] layout agent')

                # Compute text fill for each side so the layout agent has concrete numbers.
                _left_ids  = {str((p or {}).get('panel_id', '')) for p in left_panels}
                _right_ids = {str((p or {}).get('panel_id', '')) for p in right_panels}
                _body_txts = test_text_arrangement[:-2]  # exclude split-marker overlays

                def _fill_for_side(txts, figs, pid_set):
                    # Compute per-panel text fill and return the MAX across all panels on this side.
                    # Using max (not sum/average) prevents one spacious sibling panel from diluting
                    # a severely overflowing neighbor.
                    #
                    # Fill signal = text_fill only (txt_occup / txt_avail).
                    # Figures are fixed-size assets; the panel needs more space only when TEXT
                    # overflows, not when a large figure inflates a combined formula.
                    # Old formula max(combined_fill, text_fill) caused false "cramped" signals:
                    # when fig_avail >> txt_avail, combined_fill → ~100% regardless of text sparsity,
                    # so a panel with a large table and a half-empty text box appeared 94% full.
                    #
                    # Figure-only panels (no text box): return 85% — panel has content but is not
                    # a candidate for shrinking or growing purely on text grounds.
                    panel_txt_avail = {}
                    panel_txt_occup = {}
                    for _t in txts:
                        pid = str(_t.get('panel_id', ''))
                        if pid not in pid_set:
                            continue
                        if 'title' in _t.get('textbox_name', '').lower():
                            continue
                        _c = _t.get('content_for_ppt', [])
                        if not _c:
                            continue
                        _r = render_textbox(_t, copy.deepcopy(_c), args.tmp_dir,
                                            units_per_inch=units_per_inch,
                                            num_attempt=99, simulation_only=True)
                        if _r and _r[2] > 0:
                            panel_txt_avail[pid] = panel_txt_avail.get(pid, 0.0) + _r[2]
                            panel_txt_occup[pid] = panel_txt_occup.get(pid, 0.0) + _r[3]

                    panel_has_fig = set()
                    for _f in figs:
                        pid = str(_f.get('panel_id', ''))
                        if pid not in pid_set:
                            continue
                        if _f.get('height', 0) > 0:
                            panel_has_fig.add(pid)

                    all_pids = set(panel_txt_avail.keys()) | panel_has_fig
                    if not all_pids:
                        return None

                    per_panel_fills = []
                    for pid in all_pids:
                        ta = panel_txt_avail.get(pid, 0.0)
                        to = panel_txt_occup.get(pid, 0.0)
                        if ta > 0:
                            per_panel_fills.append(to / ta)
                        else:
                            per_panel_fills.append(0.85)  # figure-only panel

                    return int(100 * min(max(per_panel_fills), 1.5))

                c1_fill_pct = _fill_for_side(_body_txts, test_figure_arrangement, _left_ids)
                c2_fill_pct = _fill_for_side(_body_txts, test_figure_arrangement, _right_ids)
                _c1_label = 'TOP' if split_type == 'horizontal' else 'LEFT'
                _c2_label = 'BOTTOM' if split_type == 'horizontal' else 'RIGHT'
                print(f"   📊 Side fill: {_c1_label}={c1_fill_pct}%, {_c2_label}={c2_fill_pct}%")

                # Pre-compute required direction from fill numbers so the LLM
                # cannot hallucinate fill values in its reasoning.
                _c1_dir_label = _c1_label.lower()
                _c2_dir_label = _c2_label.lower()
                if c1_fill_pct is not None and c2_fill_pct is not None:
                    if c1_fill_pct > 90 and c2_fill_pct <= 90:
                        _req_dir = _c1_label
                        _req_delta = 4 if c1_fill_pct >= 140 else 3 if c1_fill_pct >= 110 else 2
                        _req_note = f"{_c1_label} cramped ({c1_fill_pct}%), {_c2_label} has space ({c2_fill_pct}%)"
                    elif c2_fill_pct > 90 and c1_fill_pct <= 90:
                        _req_dir = _c2_label
                        _req_delta = 4 if c2_fill_pct >= 140 else 3 if c2_fill_pct >= 110 else 2
                        _req_note = f"{_c2_label} cramped ({c2_fill_pct}%), {_c1_label} has space ({c1_fill_pct}%)"
                    elif c1_fill_pct > 90 and c2_fill_pct > 90:
                        _gap = abs(c1_fill_pct - c2_fill_pct)
                        if _gap >= 15:
                            _req_dir = _c1_label if c1_fill_pct >= c2_fill_pct else _c2_label
                            _req_delta = 2
                            _req_note = f"both cramped; {_req_dir} worse ({max(c1_fill_pct, c2_fill_pct)}% vs {min(c1_fill_pct, c2_fill_pct)}%)"
                        else:
                            _req_dir = 'none'
                            _req_delta = 0
                            _req_note = f"both cramped, gap={_gap}pp < 15 — cannot redistribute"
                    else:
                        _req_dir = 'none'
                        _req_delta = 0
                        _req_note = f"both in 40–90% range ({c1_fill_pct}%, {c2_fill_pct}%)"
                else:
                    _req_dir = 'none'
                    _req_delta = 0
                    _req_note = 'fill data unavailable'

                jinja_args = {
                    'split_type': split_type,
                    'split_ratio': current_split_ratio,
                    'c1_label': _c1_label,
                    'c2_label': _c2_label,
                    'c1_fill_pct': c1_fill_pct,
                    'c2_fill_pct': c2_fill_pct,
                    'required_direction': _req_dir,
                    'required_delta': _req_delta,
                    'required_note': _req_note,
                }
                prompt = layout_template.render(**jinja_args)
                layout_msg = BaseMessage.make_user_message(
                    role_name="User",
                    content=prompt,
                    image_list=[cropped],
                )

                layout_agent.reset()
                layout_response = agent_step_with_timeout(layout_agent, layout_msg)
                input_token, output_token = account_token(layout_response)
                total_input_token_v += input_token
                total_output_token_v += output_token
                final_action = get_json_from_response(layout_response.msgs[0].content)

                print(f"   Layout agent response: {layout_response.msgs[0].content}")
                print(' [2] action execution')

                # Detect figure panels on each side of this split.
                # Figure width = full panel width, so shrinking a figure-containing
                # panel makes the figure smaller — block that direction.
                left_has_figure = any(
                    (p or {}).get('asset_path') or (p or {}).get('gp', 0) > 0
                    for p in left_panels
                )
                right_has_figure = any(
                    (p or {}).get('asset_path') or (p or {}).get('gp', 0) > 0
                    for p in right_panels
                )

                # Guard against missing keys in layout agent response
                raw_delta = final_action.get('delta', 0) if final_action else 0
                raw_dir = (final_action.get('direction', 'none') or 'none').lower() if final_action else 'none'

                # Fill-based override: correct model errors when fill data unambiguously
                # indicates the right direction.  The VLM can mis-read the image when one
                # side has a single dense textbox and the other has many small sub-panels
                # that each look less packed individually.
                #
                # Case A — both sides cramped (>90%), gap < 15pp:
                #   Redistribution hurts the victim as much as it helps the winner.
                #   The total space is simply insufficient; content modification is the fix.
                #   → force 'none' regardless of what the VLM said.
                # Case B — both sides cramped (>90%), gap ≥ 15pp:
                #   Redirect toward the more cramped side.  Delta is capped below to avoid
                #   pushing the victim over 100%.
                # Case C — exactly one side cramped and model chose the OTHER side:
                #   e.g. c1=150%, c2=70% → model said "bottom/right" → override to "top/left"
                _c1_dir = _c1_label.lower()   # 'top' or 'left'
                _c2_dir = _c2_label.lower()   # 'bottom' or 'right'
                # Pre-compute _both_cramped outside the if block so the delta cap below can reuse it.
                _both_cramped = (c1_fill_pct is not None and c2_fill_pct is not None
                                 and c1_fill_pct > 90 and c2_fill_pct > 90)
                _fill_gap = abs((c1_fill_pct or 0) - (c2_fill_pct or 0)) if _both_cramped else 0

                # Proactive gentle correction when VLM said 'none' but fill clearly shows
                # one-sided overflow.  Applied BEFORE the main override block so figure
                # protection and delta caps still apply afterward.
                # Uses a small fixed delta (raw_delta≈2 equivalent = 0.06) to avoid
                # over-correction — the goal is a nudge, not a large redistribution.
                if raw_dir == 'none' and c1_fill_pct is not None and c2_fill_pct is not None:
                    _GENTLE_DELTA = (2.0 / 5) * 0.15  # ≈ 0.06, equivalent to raw_delta=2
                    if c2_fill_pct > 100 and c1_fill_pct <= 90:
                        raw_dir = _c2_dir
                        delta = _GENTLE_DELTA
                        print(f"   🔁 Gentle override (none→{raw_dir}): "
                              f"c2={c2_fill_pct}% overflow, c1={c1_fill_pct}% has space "
                              f"— nudging delta={_GENTLE_DELTA:.3f}")
                    elif c1_fill_pct > 100 and c2_fill_pct <= 90:
                        raw_dir = _c1_dir
                        delta = _GENTLE_DELTA
                        print(f"   🔁 Gentle override (none→{raw_dir}): "
                              f"c1={c1_fill_pct}% overflow, c2={c2_fill_pct}% has space "
                              f"— nudging delta={_GENTLE_DELTA:.3f}")

                # Figure recovery: when raw_dir is still 'none' (both fills acceptable)
                # but the figure is below its natural rendered size, gently grow it back.
                #
                # Horizontal split: figure visual fill = rendered_fig_h / panel_h.
                #   Below 0.70 means the figure is leaving > 30% of the panel empty.
                # Vertical split: compare rendered_fig_w vs panel's natural_fig_w.
                #   If the figure is width-compressed (below 90% of natural), recover.
                #
                # Recovery is blocked when the opposing text side overflows (> 85%)
                # because that means the figure shrinkage was intentional and necessary.
                if raw_dir == 'none' and c1_fill_pct is not None and c2_fill_pct is not None:
                    _RECOVERY_DELTA = (2.0 / 5) * 0.10  # ≈ 0.04, same order as gentle override
                    if left_has_figure and not right_has_figure:
                        if split_type == 'horizontal' and _nat_fig_h_L > 0:
                            _vis_fill_L = _nat_fig_h_L / left_box_info['height'] if left_box_info.get('height', 0) > 0 else 0.85
                            if _vis_fill_L < 0.70 and c2_fill_pct <= 85:
                                raw_dir = _c1_dir
                                delta = _RECOVERY_DELTA
                                print(f"   📈 Figure recovery (top): vis_fill={int(100*_vis_fill_L)}%, "
                                      f"text={c2_fill_pct}% → grow figure (delta={_RECOVERY_DELTA:.3f})")
                        elif split_type == 'vertical' and _nat_fig_w_L > 0:
                            _natural_fig_w_L = max((float(p.get('natural_fig_w', 0) or 0) for p in left_panels), default=0.0)
                            if _natural_fig_w_L > 0:
                                _shrink_ratio_L = _nat_fig_w_L / _natural_fig_w_L
                                if _shrink_ratio_L < 0.90 and c2_fill_pct <= 85:
                                    raw_dir = _c1_dir
                                    delta = _RECOVERY_DELTA
                                    print(f"   📈 Figure recovery (left): shrink={_shrink_ratio_L:.0%}, "
                                          f"text={c2_fill_pct}% → grow figure (delta={_RECOVERY_DELTA:.3f})")
                    elif right_has_figure and not left_has_figure:
                        if split_type == 'horizontal' and _nat_fig_h_R > 0:
                            _vis_fill_R = _nat_fig_h_R / right_box_info['height'] if right_box_info.get('height', 0) > 0 else 0.85
                            if _vis_fill_R < 0.70 and c1_fill_pct <= 85:
                                raw_dir = _c2_dir
                                delta = _RECOVERY_DELTA
                                print(f"   📈 Figure recovery (bottom): vis_fill={int(100*_vis_fill_R)}%, "
                                      f"text={c1_fill_pct}% → grow figure (delta={_RECOVERY_DELTA:.3f})")
                        elif split_type == 'vertical' and _nat_fig_w_R > 0:
                            _natural_fig_w_R = max((float(p.get('natural_fig_w', 0) or 0) for p in right_panels), default=0.0)
                            if _natural_fig_w_R > 0:
                                _shrink_ratio_R = _nat_fig_w_R / _natural_fig_w_R
                                if _shrink_ratio_R < 0.90 and c1_fill_pct <= 85:
                                    raw_dir = _c2_dir
                                    delta = _RECOVERY_DELTA
                                    print(f"   📈 Figure recovery (right): shrink={_shrink_ratio_R:.0%}, "
                                          f"text={c1_fill_pct}% → grow figure (delta={_RECOVERY_DELTA:.3f})")

                if raw_dir != 'none' and c1_fill_pct is not None and c2_fill_pct is not None:
                    if _both_cramped:
                        if _fill_gap < 15:
                            # Gap too small: any redistribution hurts the victim as much as it
                            # helps the winner.  Content modification is the only real fix here.
                            print(f"   🔒 Fill override (both cramped, gap={_fill_gap:.0f}pp < 15): "
                                  f"{raw_dir} → none  (c1={c1_fill_pct}%, c2={c2_fill_pct}%)")
                            raw_dir = 'none'
                        else:
                            # Large gap: redirect toward the more cramped side.
                            # Delta will be capped after this block to protect the victim.
                            if c1_fill_pct - c2_fill_pct >= 15 and raw_dir != _c1_dir:
                                print(f"   🔒 Fill override (both cramped): {raw_dir} → {_c1_dir} "
                                      f"(c1={c1_fill_pct}% >> c2={c2_fill_pct}%)")
                                raw_dir = _c1_dir
                            elif c2_fill_pct - c1_fill_pct >= 15 and raw_dir != _c2_dir:
                                print(f"   🔒 Fill override (both cramped): {raw_dir} → {_c2_dir} "
                                      f"(c2={c2_fill_pct}% >> c1={c1_fill_pct}%)")
                                raw_dir = _c2_dir
                    else:
                        # At most one side is cramped.
                        if c1_fill_pct > 90 and c2_fill_pct <= 90 and raw_dir not in [_c1_dir, 'none']:
                            print(f"   🔒 Fill override (c1 cramped): {raw_dir} → {_c1_dir} "
                                  f"(c1={c1_fill_pct}%, c2={c2_fill_pct}%)")
                            raw_dir = _c1_dir
                        elif c2_fill_pct > 90 and c1_fill_pct <= 90 and raw_dir not in [_c2_dir, 'none']:
                            print(f"   🔒 Fill override (c2 cramped): {raw_dir} → {_c2_dir} "
                                  f"(c2={c2_fill_pct}%, c1={c1_fill_pct}%)")
                            raw_dir = _c2_dir

                # Protect figure panels from being shrunk BELOW a minimum ratio.
                # Main figures (is_main_figure=True) get a higher minimum (0.55) so the
                # layout agent cannot undo the _apply_figure_boost target of 0.65.
                # When BOTH sides have figures, the minimum is derived from the PDF size
                # ratio (within 1.5× tolerance) to preserve relative figure proportions.
                # 'left'/'top'  → ratio increases → RIGHT/BOTTOM shrinks
                # 'right'/'bottom' → ratio decreases → LEFT/TOP shrinks
                _left_is_main  = _is_main_panel(left_panels)
                _right_is_main = _is_main_panel(right_panels)
                _lfs = max((float((p or {}).get('figure_size', 0) or 0) for p in left_panels), default=0.0)
                _rfs = max((float((p or {}).get('figure_size', 0) or 0) for p in right_panels), default=0.0)

                # Dynamic figure compression-floor
                # For each side, find the max rendered figure height/width from the
                # current figure arrangement.  The split ratio cannot shrink below
                # _dyn_min without compressing the rendered figure asset.
                #
                # Horizontal split → height is the critical dimension:
                #   min_panel_h = rendered_fig_h + stacked_overhead + min_text_h
                # Vertical split → width is the critical dimension:
                #   min_panel_w = rendered_fig_w + 2×shrink_margin
                _STACKED_OH = 30   # top_margin(6) + 2×shrink(12) + content_bottom_extra(12)
                _MIN_TXT_H  = 40   # MIN_BODY_TEXT_H
                _W_OVERHEAD = 12   # 2 × SHRINK_MARGIN

                _nat_fig_h_L = max((float(_f.get('height', 0)) for _f in test_figure_arrangement
                                    if str(_f.get('panel_id', '')) in _left_ids), default=0.0)
                _nat_fig_h_R = max((float(_f.get('height', 0)) for _f in test_figure_arrangement
                                    if str(_f.get('panel_id', '')) in _right_ids), default=0.0)
                _nat_fig_w_L = max((float(_f.get('width', 0)) for _f in test_figure_arrangement
                                    if str(_f.get('panel_id', '')) in _left_ids), default=0.0)
                _nat_fig_w_R = max((float(_f.get('width', 0)) for _f in test_figure_arrangement
                                    if str(_f.get('panel_id', '')) in _right_ids), default=0.0)

                # Fix 1: overhead differs by whether the panel has a text box.
                # Figure-only panels (tp=0, empty content) have no text box →
                # compression floor = fig_h + _STACKED_OH (30) only, not + _MIN_TXT_H (40).
                _has_text_L = any(
                    str(_t.get('panel_id', '')) in _left_ids
                    and 'title' not in _t.get('textbox_name', '').lower()
                    and _t.get('content_for_ppt')
                    for _t in _body_txts
                )
                _has_text_R = any(
                    str(_t.get('panel_id', '')) in _right_ids
                    and 'title' not in _t.get('textbox_name', '').lower()
                    and _t.get('content_for_ppt')
                    for _t in _body_txts
                )
                _OH_L = _STACKED_OH + (_MIN_TXT_H if _has_text_L else 0)
                _OH_R = _STACKED_OH + (_MIN_TXT_H if _has_text_R else 0)

                if split_type == 'horizontal':
                    _total_dim = left_box_info['height'] + right_box_info['height']
                    _dyn_min_L = max(0.15, (_nat_fig_h_L + _OH_L) / _total_dim) if _nat_fig_h_L > 0 and _total_dim > 0 else 0.15
                    _dyn_min_R = max(0.15, (_nat_fig_h_R + _OH_R) / _total_dim) if _nat_fig_h_R > 0 and _total_dim > 0 else 0.15
                else:
                    _total_dim = left_box_info['width'] + right_box_info['width']
                    _dyn_min_L = max(0.15, (_nat_fig_w_L + _W_OVERHEAD) / _total_dim) if _nat_fig_w_L > 0 and _total_dim > 0 else 0.15
                    _dyn_min_R = max(0.15, (_nat_fig_w_R + _W_OVERHEAD) / _total_dim) if _nat_fig_w_R > 0 and _total_dim > 0 else 0.15


                # Fix 2: visual fill — does the figure actually fill its panel?
                # horizontal: figure_h / panel_h  |  vertical: figure_w / available_w
                # If vis_fill < 0.70 → panel has whitespace the figure doesn't use →
                #   safe to allow the panel to shrink to _dyn_min (donate whitespace).
                # If vis_fill ≥ 0.70 → figure fills the panel → keep 0.35 floor.
                _WHITESPACE_THRESH = 0.70
                if split_type == 'horizontal':
                    _panel_h_L = left_box_info.get('height', 0)
                    _panel_h_R = right_box_info.get('height', 0)
                    _vis_fill_L = (_nat_fig_h_L / _panel_h_L) if _nat_fig_h_L > 0 and _panel_h_L > 0 else 0.85
                    _vis_fill_R = (_nat_fig_h_R / _panel_h_R) if _nat_fig_h_R > 0 and _panel_h_R > 0 else 0.85
                else:
                    _avail_w_L = max(1, left_box_info.get('width', 0) - _W_OVERHEAD)
                    _avail_w_R = max(1, right_box_info.get('width', 0) - _W_OVERHEAD)
                    _vis_fill_L = min(1.0, _nat_fig_w_L / _avail_w_L) if _nat_fig_w_L > 0 else 0.85
                    _vis_fill_R = min(1.0, _nat_fig_w_R / _avail_w_R) if _nat_fig_w_R > 0 else 0.85
                _left_has_whitespace  = _vis_fill_L < _WHITESPACE_THRESH
                _right_has_whitespace = _vis_fill_R < _WHITESPACE_THRESH


                if _lfs > 0 and _rfs > 0 and left_has_figure and right_has_figure:
                    # Natural split ratio from PDF sizes; enforce within 1.5× tolerance.
                    # Fix 3: main figures with panel whitespace bypass the proportion
                    # formula and use _dyn_min instead — donating unused vertical space.
                    _natural = _lfs / (_lfs + _rfs)
                    _prop_min_L = max(0.20, _natural / 1.5)
                    _prop_min_R = max(0.20, (1.0 - _natural) / 1.5)
                    MIN_LEFT_RATIO  = max(_dyn_min_L, 0.20) if (_left_is_main  and _left_has_whitespace)  else _prop_min_L
                    MIN_RIGHT_RATIO = max(_dyn_min_R, 0.20) if (_right_is_main and _right_has_whitespace) else _prop_min_R
                elif left_has_figure and not right_has_figure:
                    # Fill-based minimum floored by dynamic compression point.
                    _BASE_MIN_L = 0.22
                    _FULL_MIN_L = 0.40
                    if c1_fill_pct is not None:
                        _t_l = max(0.0, min(1.0, (c1_fill_pct - 50) / 50))
                        MIN_LEFT_RATIO = _BASE_MIN_L + _t_l * (_FULL_MIN_L - _BASE_MIN_L)
                    else:
                        MIN_LEFT_RATIO = _FULL_MIN_L
                    MIN_LEFT_RATIO = max(_dyn_min_L, MIN_LEFT_RATIO)
                    MIN_RIGHT_RATIO = 0.18
                elif right_has_figure and not left_has_figure:
                    _BASE_MIN_R = 0.22
                    _FULL_MIN_R = 0.40
                    if c2_fill_pct is not None:
                        _t_r = max(0.0, min(1.0, (c2_fill_pct - 50) / 50))
                        MIN_RIGHT_RATIO = _BASE_MIN_R + _t_r * (_FULL_MIN_R - _BASE_MIN_R)
                    else:
                        MIN_RIGHT_RATIO = _FULL_MIN_R
                    MIN_RIGHT_RATIO = max(_dyn_min_R, MIN_RIGHT_RATIO)
                    MIN_LEFT_RATIO = 0.18
                else:
                    MIN_LEFT_RATIO = MIN_RIGHT_RATIO = 0.20

                # Main figure floor: low (_dyn_min) when panel has whitespace,
                # otherwise keep the old 0.35 floor to protect well-used figure panels.
                if _left_is_main:
                    _main_floor_L = _dyn_min_L if _left_has_whitespace else max(_dyn_min_L, 0.35)
                    MIN_LEFT_RATIO = max(_main_floor_L, MIN_LEFT_RATIO)
                    print(f"   📐 Main-fig floor (left):  dyn={_dyn_min_L:.3f} vis={_vis_fill_L:.0%}"
                          f" ws={'Y' if _left_has_whitespace else 'N'} → final={MIN_LEFT_RATIO:.3f}")
                if _right_is_main:
                    _main_floor_R = _dyn_min_R if _right_has_whitespace else max(_dyn_min_R, 0.35)
                    MIN_RIGHT_RATIO = max(_main_floor_R, MIN_RIGHT_RATIO)
                    print(f"   📐 Main-fig floor (right): dyn={_dyn_min_R:.3f} vis={_vis_fill_R:.0%}"
                          f" ws={'Y' if _right_has_whitespace else 'N'} → final={MIN_RIGHT_RATIO:.3f}")

                delta = (float(raw_delta) / 5) * 0.10

                # When both sides are cramped and redistribution still proceeds (large gap),
                # cap delta so the victim's fill doesn't exceed 100 %.
                # Approximation: text fill ∝ 1/height, so fill_after = fill_before × h_before/h_after.
                #   TOP/LEFT direction  → victim is RIGHT/BOTTOM, fraction = (1 − ratio)
                #     max_delta = (1−ratio) × (100 − victim_fill) / 100
                #   RIGHT/BOTTOM direction → victim is LEFT/TOP, fraction = ratio
                #     max_delta = ratio × (100 − victim_fill) / 100
                if _both_cramped and raw_dir in ['left', 'top']:
                    _victim_fill = c2_fill_pct
                    _max_delta_fill = (1.0 - current_split_ratio) * max(0.0, 100.0 - _victim_fill) / 100.0
                    if delta > _max_delta_fill:
                        print(f"   📉 Both-cramped delta cap: {delta:.3f} → {_max_delta_fill:.3f} "
                              f"(victim {_c2_label} at {_victim_fill}%)")
                        delta = max(0.0, _max_delta_fill)
                elif _both_cramped and raw_dir in ['right', 'bottom']:
                    _victim_fill = c1_fill_pct
                    _max_delta_fill = current_split_ratio * max(0.0, 100.0 - _victim_fill) / 100.0
                    if delta > _max_delta_fill:
                        print(f"   📉 Both-cramped delta cap: {delta:.3f} → {_max_delta_fill:.3f} "
                              f"(victim {_c1_label} at {_victim_fill}%)")
                        delta = max(0.0, _max_delta_fill)

                if raw_dir in ['left', 'top']:
                    candidate_ratio = current_split_ratio + delta
                    if right_has_figure and (1.0 - candidate_ratio) < MIN_RIGHT_RATIO:
                        _opp_fill = c1_fill_pct if c1_fill_pct is not None else 0
                        _my_fill  = c2_fill_pct if c2_fill_pct is not None else 100
                        if _opp_fill > 120 and _my_fill < 70:
                            effective_min = max(0.30, MIN_RIGHT_RATIO * 0.65)
                            print(f"   🔓 Relaxing figure min ({MIN_RIGHT_RATIO:.2f} → {effective_min:.2f}): "
                                  f"opposing overflow {_opp_fill}%, this side {_my_fill}%")
                        else:
                            effective_min = MIN_RIGHT_RATIO
                        # Partial correction: clamp delta to the maximum allowed before hitting effective_min.
                        # Avoids a complete skip when the figure side still has some room to shrink.
                        max_delta = (1.0 - current_split_ratio) - effective_min
                        if max_delta > 0.005:
                            delta = max_delta
                            print(f"   🖼️  Clamping '{raw_dir}': figure side limited to {effective_min:.2f}, "
                                  f"applying partial delta {delta:.3f}")
                        else:
                            print(f"   🖼️  Blocking '{raw_dir}': figure side already at minimum "
                                  f"{(1.0 - current_split_ratio):.2f} ≤ {effective_min:.2f} — overriding to 'none'")
                            raw_dir = 'none'
                elif raw_dir in ['right', 'bottom']:
                    candidate_ratio = current_split_ratio - delta
                    if left_has_figure and candidate_ratio < MIN_LEFT_RATIO:
                        # When the opposing side severely overflows (>120%) and this side has
                        # empty space (<70%), the figure min is overly strict: the empty space
                        # comes from the text box, not the figure itself.  Relax the minimum
                        # to allow space redistribution without hardcoding it away entirely.
                        _opp_fill = c2_fill_pct if c2_fill_pct is not None else 0
                        _my_fill  = c1_fill_pct if c1_fill_pct is not None else 100
                        if _opp_fill > 120 and _my_fill < 70:
                            effective_min = max(0.30, MIN_LEFT_RATIO * 0.65)
                            print(f"   🔓 Relaxing figure min ({MIN_LEFT_RATIO:.2f} → {effective_min:.2f}): "
                                  f"opposing overflow {_opp_fill}%, this side {_my_fill}%")
                        else:
                            effective_min = MIN_LEFT_RATIO
                        # Partial correction: clamp delta to the maximum allowed before hitting effective_min.
                        # Avoids a complete skip when the figure side still has some room to shrink.
                        max_delta = current_split_ratio - effective_min
                        if max_delta > 0.005:
                            delta = max_delta
                            print(f"   🖼️  Clamping '{raw_dir}': figure side limited to {effective_min:.2f}, "
                                  f"applying partial delta {delta:.3f}")
                        else:
                            print(f"   🖼️  Blocking '{raw_dir}': figure side already at minimum "
                                  f"{current_split_ratio:.2f} ≤ {effective_min:.2f} — overriding to 'none'")
                            raw_dir = 'none'

                if raw_dir in ['left', 'top']:
                    new_split_ratio = current_split_ratio + delta
                elif raw_dir in ['right', 'bottom']:
                    new_split_ratio = current_split_ratio - delta
                else:
                    new_split_ratio = current_split_ratio  # 'none' or unknown → no change

                # Clamp split_ratio so neither side of the split is too narrow
                new_split_ratio = max(0.15, min(0.85, new_split_ratio))

                # For vertical splits, also enforce a minimum absolute column width
                # (~6 inches = 150 layout units) so text is never squeezed unreadable.
                # Increased from 100 to 150 to prevent Conclusion-like sections from
                # becoming too narrow to show a single-word title on one line.
                if split_type == 'vertical':
                    total_w = left_box_info['width'] + right_box_info['width']
                    MIN_COL_W = 150  # ~6 inches
                    left_new_w  = total_w * new_split_ratio
                    right_new_w = total_w * (1.0 - new_split_ratio)
                    if left_new_w < MIN_COL_W:
                        new_split_ratio = MIN_COL_W / total_w
                        print(f"   📏 Width guard: left column too narrow, clamped ratio to {new_split_ratio:.3f}")
                    elif right_new_w < MIN_COL_W:
                        new_split_ratio = 1.0 - MIN_COL_W / total_w
                        print(f"   📏 Width guard: right column too narrow, clamped ratio to {new_split_ratio:.3f}")

                _log_event({
                    "type": "layout_iter",
                    "attempt": num_attempts,
                    "iter": iteration,
                    "internal_id": internal_id,
                    "split_type": split_type,
                    "before": round(current_split_ratio, 4),
                    "after": round(new_split_ratio, 4),
                    "direction": raw_dir,
                    "delta": int(raw_delta) if raw_delta else 0,
                })

                if new_split_ratio != current_split_ratio:

                    update_internal_node_ratio(content_panel_tree, internal_id, new_split_ratio)

                    panel_arrangement, figure_arrangement, text_arrangement = generate_complete_layout(
                        title_panel,
                        content_panel_tree,
                        poster_width,
                        poster_height,
                        title_h,
                        section_title_h,
                        subsection_title_h,
                        shrink_margin
                    )


                layout_changed = new_split_ratio != current_split_ratio
                print(' [3] post-action render')

                # Always sync text_arrangement from bullet_content so subsequent iters see latest content
                _sync_text_arrangement(text_arrangement, _bullet_lookup, bullet_content)
                new_content = [t['content_for_ppt'] for t in text_arrangement]

                # [3] viz: section titles only (no subsec override) — preserving existing behaviour
                _apply_title_styles(text_arrangement, theme_title_text_color, theme_title_fill_color)

                # Only render _applied poster if layout actually changed (avoid duplicate slides in viewer)
                if layout_changed:
                    stem = f"step6_attempt_{num_attempts:02d}_iter_{iteration:02d}_applied"
                    _render_to_jpg(
                        poster_width, poster_height,
                        panel_arrangement, figure_arrangement, text_arrangement,
                        stem, args.tmp_dir, visible=False, theme=theme
                    )

                    # After-viz: same 연두 panel + red split-box as before-viz but new ratio.
                    try:
                        _av_panel  = copy.deepcopy(panel_arrangement)
                        _av_figure = copy.deepcopy(figure_arrangement)
                        _av_text   = [copy.deepcopy(_t) for _t in text_arrangement]

                        _av_left, _av_right, _, _ = get_internal_split_coords(
                            content_panel_tree, internal_id,
                            initial_x=0, initial_y=title_h,
                            initial_w=poster_width, initial_h=poster_height - title_h,
                            section_title_h=section_title_h,
                            decay_factor=subsection_title_h / section_title_h
                        )
                        _sync_text_arrangement(_av_text, _bullet_lookup, bullet_content)
                        _enforce_title_format(_av_text)
                        _av_left['content_for_ppt']  = [{"alignment": "center", "bullet": False, "level": 0, "font_size": 60, "runs": [{"text": " ", "bold": False}]}]
                        _av_right['content_for_ppt'] = [{"alignment": "center", "bullet": False, "level": 0, "font_size": 60, "runs": [{"text": " ", "bold": False}]}]
                        _av_text.append(_av_left)
                        _av_text.append(_av_right)

                        _apply_title_styles(_av_text, theme_title_text_color, theme_title_fill_color_)

                        _av_stem = f"step6_attempt_{num_attempts:02d}_iter_{iteration:02d}_{internal_id}_after_viz"
                        _render_to_jpg(poster_width, poster_height,
                                       _av_panel, _av_figure, _av_text,
                                       _av_stem, args.tmp_dir, visible=True, theme=theme_visible)
                        print(f"   ✅ After-viz saved: {_av_stem}.jpg")
                    except Exception as _e:
                        print(f"   ⚠️ Could not save after_viz: {_e}")
                else:
                    print(f"   ↔️  No layout change at iter {iteration}, skipping _applied render")



        if CONTENT_MODIFICATION:
            print(" --- Content Modification ---")

            theme_textbox_text_color = (0, 0, 0)
            theme_textbox_fill_color = (255, 255, 153)

            # Option-C selective reset: only reset panels whose dimensions changed.
            #
            # Thresholds in layout units (25 units = 1 inch):
            #   _SAME_H_THRESH = 5  — float-only drift < 0.01 u; smallest real layout
            #     change at Level-2 depth ≈ 3-4 u, so threshold cleanly separates
            #     float noise ("same") from meaningful shrink/grow.
            #   _SAME_W_THRESH = 10 — text wrapping is less sensitive to width than
            #     height, so a wider tolerance avoids unnecessary resets.
            #
            # Fix-1 exception for shrank+trimmed panels:
            #   When a box shrank AND the previous attempt had already trimmed content
            #   (fewer bullets than initial), keep the previous trimmed result rather
            #   than resetting to initial.  Phase 1 will re-verify fit in the new dims;
            #   if it's still FIT (or only marginally over), the LLM-trimmed quality is
            #   preserved and Phase 5 can fill any remaining gap.  This prevents the
            #   pattern where reset→extreme-overflow→deterministic-fit discards the
            #   LLM's semantically aware choices.
            #
            # Panels that fully reset also reset their content bank (no lost items).
            # Panels kept via Fix-1 retain their bank so Phase 5 can still recover.
            _SAME_H_THRESH = 5.0
            _SAME_W_THRESH = 10.0
            _init_lookup = _build_content_lookup(intial_bullet_content)
            _bc_by_key   = {_bc_key(b): b for b in bullet_content}

            _reset_count = _keep_count = _fix1_count = 0
            for _t in text_arrangement:
                _key  = _bc_key(_t)
                _prev = _prev_text_dims.get(_key)
                _curr_w = _t.get('width',  0.0)
                _curr_h = _t.get('height', 0.0)

                if _prev is None:
                    _changed = True   # first attempt or newly created textbox
                    _box_shrank = False
                else:
                    _pw, _ph = _prev
                    _changed = (abs(_curr_h - _ph) > _SAME_H_THRESH or
                                abs(_curr_w - _pw) > _SAME_W_THRESH)
                    _box_shrank = _curr_h < _ph - _SAME_H_THRESH

                if _changed:
                    _prev_content = _bc_by_key.get(_key, {}).get('content_for_ppt', []) if _key in _bc_by_key else []
                    _init_content = _init_lookup.get(_key, [])
                    _prev_was_trimmed = len(_prev_content) < len(_init_content)

                    if _box_shrank and _prev_was_trimmed:
                        # Fix-1: keep previous LLM-trimmed result; bank also preserved.
                        _fix1_count += 1
                    else:
                        # Standard reset: restore initial content and clear bank.
                        if _key in _init_lookup and _key in _bc_by_key:
                            _bc_by_key[_key]['content_for_ppt'] = copy.deepcopy(_init_lookup[_key])
                        _content_bank.pop(_key, None)
                        _reset_count += 1
                else:
                    _keep_count += 1

            print(f"   🔄 Selective reset: {_reset_count} panels reset to initial, "
                  f"{_keep_count} panels kept (dim unchanged), "
                  f"{_fix1_count} panels kept via Fix-1 (shrank+trimmed)")

            _bullet_lookup = _build_content_lookup(bullet_content)
            _sync_text_arrangement(text_arrangement, _bullet_lookup, bullet_content)
            _enforce_title_format(text_arrangement)

            test_text_arrangement = copy.deepcopy(text_arrangement)

            # Iterate over ALL textboxes directly (not panel-by-panel) so that
            # both title and content textboxes of each subsection are handled.
            # Avoids the old bug where only the FIRST textbox per panel_id was
            # processed (typically the title), leaving content textboxes uncorrected.
            for t_idx in range(2, len(test_text_arrangement)):  # skip poster title (0) and author (1)
                text_dict = test_text_arrangement[t_idx]
                panel_id = text_dict.get('panel_id')
                tname = text_dict.get('textbox_name', '')

                # Skip top-level section containers (single-digit panel_ids)
                if _pid_is_section(panel_id):
                    continue

                # Get corresponding bullet_content entry by (panel_id, textbox_name)
                bc_entry = None
                for _b in bullet_content:
                    if (_b.get('textbox_name', '') == tname and
                            str(_b.get('panel_id', '')) == str(panel_id)):
                        bc_entry = _b
                        break
                if bc_entry is None:
                    continue

                # Sync latest content from bullet_content
                text_dict['content_for_ppt'] = copy.deepcopy(bc_entry.get('content_for_ppt', []))
                is_title = 'title' in tname.lower()
                result_json = list(text_dict.get('content_for_ppt', []))

                # Reset body content to canonical font_size every attempt so font never
                # drifts small from a previous attempt's _deterministic_fit shrink.
                if not is_title:
                    for item in result_json:
                        item['font_name'] = font_name
                        item['font_size'] = font_size

                _panel_match = re.search(r'p<(.+?)>', tname)
                _panel_name = _panel_match.group(1) if _panel_match else ''

                # Title textboxes: fix overflow only; never expand underflow
                # Section and subsection title font sizes must remain CONSISTENT across
                # the entire poster. We truncate long titles rather than shrink the font,
                # so all titles maintain the same visual weight.
                if is_title:
                    _t_check = render_textbox(
                        text_dict, copy.deepcopy(result_json), args.tmp_dir,
                        units_per_inch=units_per_inch, num_attempt=0, simulation_only=True
                    )
                    if _t_check is None or _t_check[0] != 'OVERFLOW':
                        continue  # FIT or UNDERFLOW → leave title as-is
                    _, _, _t_avail, _t_occup = _t_check
                    _base_fs = sec_font_size if _pid_is_section(panel_id) else subsec_font_size
                    # min_fs = _base_fs: no font shrink allowed — truncate only.
                    # Font consistency across all titles is more important than
                    # fitting a single long title at a smaller size.
                    result_json = _deterministic_fit(result_json, _t_avail, _t_occup, _base_fs, min_fs=_base_fs)
                    for item in result_json:
                        item['font_name'] = font_name
                        item['font_size'] = _base_fs   # enforce canonical title font size
                    _set_content(bc_entry, panel_id, tname, result_json, _bullet_lookup)
                    _t_vol = int(100 * _t_occup / max(_t_avail, 1))
                    print(f"   ✂️ Title overflow fixed (truncation only, fs={_base_fs}): "
                          f"panel {panel_id} ({_t_vol}%)")
                    continue

                # Body textboxes
                # Phase 1: initial simulation — save as num_attempt=0 ("before" in viewer).
                # NOTE: Phase 2 uses a different attempt number so it never overwrites
                # this initial snapshot; the viewer always shows the true initial state.
                style_result_json = copy.deepcopy(result_json)
                style_bullet_content(style_result_json, theme_textbox_text_color, theme_textbox_fill_color)
                render_result = render_textbox(
                    text_dict, style_result_json, args.tmp_dir,
                    units_per_inch=units_per_inch, num_attempt=0, simulation_only=True
                )
                if render_result is None:
                    continue
                status, p_idx, available_height, occupied_height = render_result
                text_volume = int(100 * occupied_height / available_height)
                print(f" === panel {panel_id} ({_panel_name})")
                print(f"✅ [Status] {status} (Occupied: {occupied_height:.2f}, Available: {available_height:.2f}, Volume: {text_volume}%)")

                # Phase 2: deterministic pre-correction for severe overflow (>120%).
                # Bypasses the LLM for cases the LLM can't reliably fix in a few shots.
                # min_fs=font_size: truncation only, no font shrink (keeps sizes uniform).
                # Uses num_attempt=1 so it never overwrites the Phase-1 "before" snapshot.
                if text_volume > 120:
                    print(f"   ⚡ Extreme overflow ({text_volume}%) — deterministic fit for panel {panel_id}")
                    _removed_p2 = []
                    result_json = _deterministic_fit(result_json, available_height, occupied_height, font_size, min_fs=font_size, bank_out=_removed_p2)
                    _bank_push(_content_bank, panel_id, tname, _removed_p2)
                    for item in result_json:
                        item['font_name'] = font_name
                        item['font_size'] = font_size  # enforce canonical — truncation already resolved overflow
                    _set_content(bc_entry, panel_id, tname, result_json, _bullet_lookup)
                    style_result_json = copy.deepcopy(result_json)
                    style_bullet_content(style_result_json, theme_textbox_text_color, theme_textbox_fill_color)
                    render_result = render_textbox(
                        text_dict, style_result_json, args.tmp_dir,
                        units_per_inch=units_per_inch, num_attempt=1, simulation_only=True
                    )
                    if render_result is None:
                        continue
                    status, p_idx, available_height, occupied_height = render_result
                    text_volume = int(100 * occupied_height / available_height)
                    print(f"   📐 After deterministic fit: {status} ({text_volume}%)")

                # Phase 3: LLM-assisted loop for moderate over/underflow (65–150%)
                # Use the tree node content stored in text_dict — it is already unique per panel.
                # _mp_find_raw_section_text(_panel_name, raw_sections) always returns None for
                # body panels because _panel_name is "Content" (a generic label), so falling
                # back to raw_sections was leaving current_content_for_action = '' for every
                # single panel, making the action agent blind to section-specific material.
                _raw_text = text_dict.get('content') or ''
                if not _raw_text:
                    # Last resort: try Jaccard match by panel name (works for named subsections)
                    _raw_text = _mp_find_raw_section_text(_panel_name, raw_sections) or ''
                current_content_for_action = _raw_text
                if not current_content_for_action:
                    print(f"   ⚠️ No source content for action agent on panel '{_panel_name}'")

                # Start at 2: attempt 0 is reserved for Phase-1 "before" snapshot,
                # attempt 1 for Phase-2 deterministic correction, so Phase-3 LLM
                # iterations start at 2 and never overwrite either snapshot.
                text_mod_attempts = 2
                last_text_action = None  # 'dec' | 'inc' — prevents dec↔inc oscillation

                while True:
                    # Cap LLM attempts: overflow gets 3 shots before Phase 4 deterministic fit;
                    # underflow/other gets 4 (expansion is less time-critical).
                    # text_mod_attempts starts at 2, so limits are 2+3=5 and 2+4=6.
                    _max_iters = (2 + 3) if text_volume > 85 else (2 + 4)
                    if text_mod_attempts >= _max_iters:
                        break

                    if status == 'FIT':
                        print("Text fits well. No modification needed.")
                        break

                    _log_event({
                        "type": "content_check",
                        "attempt": num_attempts,
                        "panel_id": str(panel_id),
                        "name": _panel_name or str(panel_id),
                        "iter": text_mod_attempts,
                        "status": status,
                        "volume": text_volume,
                    })

                    jinja_args = {
                        'previous_response': result_json,
                        'current_content': current_content_for_action,
                        'sibling_contents': sibling_content_map.get(panel_id, []),
                        'current_volume_ratio': text_volume
                    }

                    if text_volume > 85:  # Overflow → LLM shorten, then deterministic fallback
                        if last_text_action == 'inc':
                            break
                        _vol_before_dec = text_volume
                        prompt = dec_action_template.render(**jinja_args)
                        response = agent_step_with_timeout(action_agent, prompt)
                        input_token, output_token = account_token(response)
                        total_input_token_t += input_token
                        total_output_token_t += output_token
                        _new_json = get_json_from_response(response.msgs[0].content)
                        if _new_json:
                            # Protect panels with ≤2 level-1 bullets: if LLM would remove one,
                            # ignore the change — a slightly cramped 2-bullet panel is better
                            # than a sparse 1-bullet panel. Phase 4 safety net handles font shrink.
                            _orig_l1 = sum(1 for _b in result_json
                                           if _b.get('level', 0) == 0 and _b.get('bullet', False))
                            _new_l1 = sum(1 for _b in _new_json
                                          if _b.get('level', 0) == 0 and _b.get('bullet', False))
                            if _orig_l1 <= 2 and _new_l1 < _orig_l1:
                                print(f"   🛡️ LLM dec would drop level-1 bullets "
                                      f"({_orig_l1}→{_new_l1}); ignoring — panel at minimum")
                                break
                            _pre_llm_json = list(result_json)  # save before LLM overwrites
                            result_json = _new_json
                            last_text_action = 'dec'
                            # Quick re-check: if LLM barely helped (<5% improvement), apply
                            # deterministic fit immediately rather than burning more LLM calls.
                            _q_json = copy.deepcopy(result_json)
                            for _qi in _q_json:
                                _qi['font_name'] = font_name
                                _qi['font_size'] = font_size
                            _qs = copy.deepcopy(_q_json)
                            style_bullet_content(_qs, theme_textbox_text_color, theme_textbox_fill_color)
                            _qr = render_textbox(text_dict, _qs, args.tmp_dir,
                                                 units_per_inch=units_per_inch, num_attempt=text_mod_attempts,
                                                 simulation_only=True)
                            if _qr is not None:
                                _q_vol = int(100 * _qr[3] / max(_qr[2], 1))
                                if _q_vol > 85 and (_vol_before_dec - _q_vol) < 5:
                                    # If LLM made overflow WORSE, revert to pre-LLM content
                                    # before applying deterministic fit — don't truncate from
                                    # a longer-than-original starting point.
                                    _fit_base = _pre_llm_json if _q_vol > _vol_before_dec else result_json
                                    if _q_vol > _vol_before_dec:
                                        print(f"   ⚡ LLM dec made overflow WORSE ({_vol_before_dec}%→{_q_vol}%): reverting + deterministic fit")
                                    else:
                                        print(f"   ⚡ LLM dec made little progress ({_vol_before_dec}%→{_q_vol}%): deterministic fit")
                                    _removed_lp = []
                                    _lp_min_items = 1 if _q_vol > 130 else 2
                                    result_json = _deterministic_fit(_fit_base, _qr[2], _qr[3], font_size, target_fill=0.78, min_fs=font_size, bank_out=_removed_lp, min_items=_lp_min_items)
                                    _bank_push(_content_bank, panel_id, tname, _removed_lp)
                                    for item in result_json:
                                        item['font_name'] = font_name
                                        item['font_size'] = font_size
                                    _set_content(bc_entry, panel_id, tname, result_json, _bullet_lookup)
                                    break
                        else:
                            print(f"   ⚠️ dec_action returned empty JSON for '{_panel_name}': deterministic fit")
                            _removed_ed = []
                            _ed_min_items = 1 if text_volume > 130 else 2
                            result_json = _deterministic_fit(result_json, available_height, occupied_height, font_size, target_fill=0.78, min_fs=font_size, bank_out=_removed_ed, min_items=_ed_min_items)
                            _bank_push(_content_bank, panel_id, tname, _removed_ed)
                            for item in result_json:
                                item['font_name'] = font_name
                                item['font_size'] = font_size
                            _set_content(bc_entry, panel_id, tname, result_json, _bullet_lookup)
                            break

                    elif text_volume < 55:  # Underflow → try to expand panel content
                        # 55–85%: natural breathing room for a poster; only trigger LLM
                        # expansion below 55% to recover over-truncated panels (raised from 40%
                        # to catch content removed by aggressive Phase-2/4 truncation).
                        # Dynamic cap: taller panels can hold more bullets; 30px ≈ one bullet
                        # at 20pt/1.4x-spacing at 72.5 DPI simulation. Floor=4, ceiling=7.
                        MAX_BULLETS = max(4, min(7, int(available_height / 30)))
                        if len(result_json) >= MAX_BULLETS:
                            print(f"   📌 Panel {panel_id}: already at MAX_BULLETS ({MAX_BULLETS}), skip expansion")
                            break

                        _fill_ratio = occupied_height / max(available_height, 1.0)
                        # Direct injection only when panel is critically empty (< 20% fill).
                        if _fill_ratio < 0.20 and available_height > 80 and last_text_action != 'inc' and _raw_text:
                            _cur_fs = result_json[0].get('font_size', font_size) if result_json else font_size
                            _lh = _cur_fs * 1.25 / 72.0
                            _ph = text_dict.get('height', 200) / units_per_inch
                            _cap = min(MAX_BULLETS, max(1, int(_ph * 0.70 / (_lh * 1.6))))
                            _can_add = _cap - len(result_json)
                            if _can_add > 0:
                                _extra = _text_to_bullets(_raw_text, _can_add, _cur_fs, font_name)
                                _existing = {b['runs'][0]['text'].lower()[:40] for b in result_json}
                                _added = 0
                                for _eb in _extra:
                                    if _added >= 1:  # add at most 1 bullet per injection pass
                                        break
                                    if len(result_json) >= MAX_BULLETS:
                                        break
                                    _t = _eb['runs'][0]['text'].lower()[:40]
                                    if _t not in _existing:
                                        result_json.append(_eb)
                                        _existing.add(_t)
                                        _added += 1
                                if _added:
                                    _set_content(bc_entry, panel_id, tname, result_json, _bullet_lookup)
                                    last_text_action = 'inc'
                                    print(f"   ➕ Panel {panel_id}: injected {_added} bullet from raw text")
                                    text_mod_attempts += 1
                                    style_result_json = copy.deepcopy(result_json)
                                    style_bullet_content(style_result_json, theme_textbox_text_color, theme_textbox_fill_color)
                                    render_result = render_textbox(
                                        text_dict, style_result_json, args.tmp_dir,
                                        units_per_inch=units_per_inch, num_attempt=text_mod_attempts, simulation_only=True
                                    )
                                    if render_result is None:
                                        break
                                    status, p_idx, available_height, occupied_height = render_result
                                    text_volume = int(100 * occupied_height / available_height)
                                    print(f"✅ [Status] {status} (Volume: {text_volume}%)")
                                    continue

                        # LLM expansion: for under-filled panels (< 55%); accept anything ≥ 55%
                        allow_inc = (last_text_action != 'dec' and text_volume < 55)
                        if allow_inc and current_content_for_action:
                            prompt = inc_action_template.render(**jinja_args)
                            response = agent_step_with_timeout(action_agent, prompt)
                            input_token, output_token = account_token(response)
                            total_input_token_t += input_token
                            total_output_token_t += output_token
                            _new_json = get_json_from_response(response.msgs[0].content)
                            if _new_json:
                                # Cap per-call growth to +2 bullets so Introduction/Conclusion
                                # sections with rich source text don't balloon in a single LLM
                                # call.  Multiple passes (up to _max_iters) recover gradually.
                                _inc_cap = min(len(result_json) + 2, MAX_BULLETS)
                                result_json = _new_json[:_inc_cap]
                                last_text_action = 'inc'
                            else:
                                print(f"   ⚠️ inc_action returned empty JSON for '{_panel_name}', keeping previous")
                                break
                        else:
                            break  # sparse but not critical, or no source content → accept as-is
                    else:
                        break  # 40–85%: treat as acceptable fill — leave it alone

                    # Enforce font_name/font_size after every LLM modification
                    for item in result_json:
                        item['font_name'] = font_name
                        item['font_size'] = font_size

                    text_mod_attempts += 1

                    # Re-simulate for next iteration
                    style_result_json = copy.deepcopy(result_json)
                    style_bullet_content(style_result_json, theme_textbox_text_color, theme_textbox_fill_color)
                    render_result = render_textbox(
                        text_dict, style_result_json, args.tmp_dir,
                        units_per_inch=units_per_inch, num_attempt=text_mod_attempts, simulation_only=True
                    )
                    if render_result is None:
                        break
                    status, p_idx, available_height, occupied_height = render_result
                    text_volume = int(100 * occupied_height / available_height)
                    print(f"✅ [Status] {status} (Occupied: {occupied_height:.2f}, Available: {available_height:.2f}, Volume: {text_volume}%)")

                # Phase 4: Guaranteed safety net — force fit if still overflowing or cramped.
                # Triggers on both genuine overflow AND panels crammed above 87% fill,
                # since cramped text (90-99%) looks bad even without literal overflow.
                style_result_json = copy.deepcopy(result_json)
                style_bullet_content(style_result_json, theme_textbox_text_color, theme_textbox_fill_color)
                final_render_result = render_textbox(
                    text_dict, style_result_json, args.tmp_dir,
                    units_per_inch=units_per_inch, num_attempt=99, simulation_only=True
                )
                final_vol = text_volume  # fallback if render fails
                if final_render_result is not None:
                    final_status, _, final_avail, final_occup = final_render_result
                    final_vol = int(100 * final_occup / max(final_avail, 1))
                    print(f"✅ [Final] {final_status} (Volume: {final_vol}%)")
                    if final_status == 'OVERFLOW' or final_vol > 105:
                        print(f"   🔒 Safety net: trimming panel {panel_id} ({final_vol}% → ~80%)")
                        _removed_sn = []
                        # Allow 1 bullet only for severe overflow (>130%); mild cases keep 2.
                        _sn_min_items = 1 if final_vol > 130 else 2
                        result_json = _deterministic_fit(result_json, final_avail, final_occup, font_size, target_fill=0.80, min_fs=font_size, bank_out=_removed_sn, min_items=_sn_min_items)
                        _bank_push(_content_bank, panel_id, tname, _removed_sn)
                        for item in result_json:
                            item['font_name'] = font_name
                            item['font_size'] = font_size  # enforce canonical — no font shrink allowed
                        # Update final_vol after safety-net trim
                        _sn_style = copy.deepcopy(result_json)
                        style_bullet_content(_sn_style, theme_textbox_text_color, theme_textbox_fill_color)
                        _sn_r = render_textbox(text_dict, _sn_style, args.tmp_dir,
                                               units_per_inch=units_per_inch, num_attempt=99, simulation_only=True)
                        if _sn_r is not None:
                            final_vol = int(100 * _sn_r[3] / max(_sn_r[2], 1))

                _set_content(bc_entry, panel_id, tname, result_json, _bullet_lookup)

                # Phase 5: Fill — recover over-shrunk panels from content bank.
                # Runs after Phase 4 with no LLM; adds bank items one at a time until
                # volume reaches 75-85% target zone or candidates are exhausted.
                # Source priority: bank (recently removed items) → initial bullets not
                # currently in result (fallback so first attempt also benefits).
                _FILL_MIN, _FILL_MAX = 75, 85
                if 'title' not in tname.lower() and final_vol < _FILL_MIN:
                    _bk = (str(panel_id), tname)
                    _bank_items = list(_content_bank.get(_bk, []))
                    _init_items = _init_lookup.get(_bk, [])
                    _fallback = [x for x in _init_items if not _item_in_result(x, result_json)]
                    # Merge: bank first; add fallback items not already in bank
                    _bank_set_texts = {(b.get('runs') or [{}])[0].get('text', '')[:40] for b in _bank_items}
                    _candidates = _bank_items + [x for x in _fallback
                                                 if (x.get('runs') or [{}])[0].get('text', '')[:40] not in _bank_set_texts]

                    _n_bank = len(_bank_items)
                    _used_bank_indices = set()
                    _filled = 0

                    for _ci, _cand in enumerate(_candidates):
                        if _item_in_result(_cand, result_json):
                            continue
                        _tj = result_json + [copy.deepcopy(_cand)]
                        for _ti in _tj:
                            _ti['font_name'] = font_name
                            _ti['font_size'] = font_size
                        _ts = copy.deepcopy(_tj)
                        style_bullet_content(_ts, theme_textbox_text_color, theme_textbox_fill_color)
                        _tr = render_textbox(text_dict, _ts, args.tmp_dir,
                                             units_per_inch=units_per_inch, num_attempt=98, simulation_only=True)
                        if _tr is None:
                            continue
                        _tv = int(100 * _tr[3] / max(_tr[2], 1))
                        if _tv <= _FILL_MAX:
                            result_json = _tj
                            final_vol = _tv
                            if _ci < _n_bank:
                                _used_bank_indices.add(_ci)
                            _filled += 1
                            print(f"   📥 Phase 5: recovered bullet into panel {panel_id} (vol {final_vol}%)")
                            if final_vol >= _FILL_MIN:
                                break  # target zone reached

                    if _filled:
                        # Update content bank: remove consumed items
                        _content_bank[_bk] = [x for i, x in enumerate(_bank_items)
                                               if i not in _used_bank_indices]
                        _set_content(bc_entry, panel_id, tname, result_json, _bullet_lookup)

        # Snapshot textbox dimensions for next attempt's selective-reset comparison.
        # Captured here (after layout mod + content mod) so the next attempt
        # compares against the FINAL layout dimensions of this attempt.
        _prev_text_dims = {
            _bc_key(t): (t.get('width', 0.0), t.get('height', 0.0))
            for t in text_arrangement
        }

        # =====================================================================
        # End of attempt: save results, render end poster, decide to continue
        # =====================================================================

        num_attempts += 1

        # Always save final arrangements before any break so the caller always
        # receives the latest layout+content regardless of which attempt ended.
        # Use name-based sync (not positional) because after layout changes
        # text_arrangement and bullet_content may have different lengths or ordering.
        _final_bc_lookup = _build_content_lookup(bullet_content)
        _sync_text_arrangement(text_arrangement, _final_bc_lookup, bullet_content)
        _enforce_title_format(text_arrangement)
        final_arrangements = {
            'panel_arrangement': panel_arrangement,
            'figure_arrangement': figure_arrangement,
            'text_arrangement': text_arrangement,
        }

        _apply_title_styles(text_arrangement, theme_title_text_color, theme_title_fill_color,
                            subsec_text_color=theme_subsec_title_text_color)

        stem = f"step6_attempt_{num_attempts:02d}_end"
        _render_to_jpg(poster_width, poster_height,
                       panel_arrangement, figure_arrangement, text_arrangement,
                       stem, args.tmp_dir, visible=False, theme=theme)

        # Stop when we have completed max_attempt full passes (layout + content each).
        if num_attempts >= max_attempt:
            print(f'Completed {max_attempt} attempt(s), stopping.')
            break

        # Critic evaluation: check if another attempt is worthwhile.
        # Both layout AND content will always re-run on the next attempt.
        print(" ================ Critic evaluation ================ ")
        poster_img = Image.open(f'{args.tmp_dir}/{stem}.jpg')
        prompt = critic_template.render()
        critic_msg = BaseMessage.make_user_message(role_name="User", content=prompt, image_list=[poster_img])
        critic_agent.reset()
        critic_response = agent_step_with_timeout(critic_agent, critic_msg)
        input_token, output_token = account_token(critic_response)
        total_input_token_v += input_token
        total_output_token_v += output_token
        critic_agent_response = get_json_from_response(critic_response.msgs[0].content)
        print(f"Critic: {critic_agent_response}")
        if not critic_agent_response or 'label' not in critic_agent_response:
            critic_agent_response = {'label': 'none'}
        _log_event({"type": "critic", "attempt": num_attempts, "label": critic_agent_response.get('label', 'none')})
        critic_label = critic_agent_response['label'].lower()
        if 'none' in critic_label:
            print("✅ Critic: poster looks good, stopping early.")
            break
        print(f"Critic: '{critic_label}' → running another full attempt (layout + content).")


    _act_log["running"] = False
    _log_event({"type": "done"})

    return total_input_token_t, total_output_token_t, total_input_token_v, total_output_token_v, final_arrangements


if __name__ == '__main__':

    # PYTHONPATH=$PYTHONPATH:. python PosterForest/step_06_modification_planning.py

    parser = argparse.ArgumentParser()
    parser.add_argument('--poster_name', type=str, default='VAR')
    parser.add_argument('--model_name_t', type=str, default='vllm_qwen3')
    parser.add_argument('--model_name_v', type=str, default='vllm_qwen3_vl')
    # parser.add_argument('--model_name_t', type=str, default='4o')
    # parser.add_argument('--model_name_v', type=str, default='4o')
    parser.add_argument('--poster_tree_path', type=str, default=None)
    parser.add_argument('--poster_bullet_path', type=str, default=None)
    parser.add_argument('--tmp_dir', type=str, default='tmp_debug')
    parser.add_argument('--index', type=int, default=1)
    parser.add_argument('--max_attempt', type=int, default=1)
    args = parser.parse_args()

    import shutil
    import os

    if os.path.exists(f'{args.tmp_dir}'):
        shutil.rmtree(f'{args.tmp_dir}')

    os.makedirs(args.tmp_dir, exist_ok=True)


    agent_config_t = get_agent_config(args.model_name_t)
    agent_config_v = get_agent_config(args.model_name_v)
    input_token_t, output_token_t, input_token_v, output_token_v, final_arrangements = step_06_modification_planning(args, agent_config_t, agent_config_v, args.max_attempt, args.poster_bullet_path, args.poster_tree_path)


