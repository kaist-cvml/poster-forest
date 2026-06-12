"""
STEP 7: PPTX Code Generation

Assembles a Python code string that uses python-pptx to construct the poster
slide — panels (background rectangles), textboxes, and figure images — then
executes that code to produce the final .pptx file.

Input:  panel/text/figure arrangement dicts (Step 4/5/6, inch units)
Output: Executed PPTX code → saved .pptx file
"""

import re
import json
import os

def sanitize_for_var(name):
    """Replace non-identifier characters with underscore for use as a Python variable name."""
    return re.sub(r'[^0-9a-zA-Z_]+', '_', name)


# ---- Step 7-1: Presentation Initialization ----------------------------------
def initialize_poster_code(width, height, slide_object_name, presentation_object_name, utils_functions):
    """Return Python code string that creates the presentation and a blank slide."""
    code = utils_functions
    code += fr'''
# Poster: {presentation_object_name}
{presentation_object_name} = create_poster(width_inch={width}, height_inch={height})

# Slide: {slide_object_name}
{slide_object_name} = add_blank_slide({presentation_object_name})
'''

    return code

# ---- Step 7-5: Presentation Save --------------------------------------------
def save_poster_code(output_file, utils_functions, presentation_object_name):
    """Return Python code string that saves the presentation to output_file."""
    code = utils_functions
    code = fr'''
# Save the presentation
save_presentation({presentation_object_name}, file_name="{output_file}")
'''
    return code

# ---- Step 7-2: Panel (Background Rectangle) Code ---------------------------
def generate_panel_code(panel_dict, utils_functions, slide_object_name, visible=False, theme=None):
    """Return Python code string that draws a panel (background rectangle) on the slide."""
    code = utils_functions
    if 'section_name' in panel_dict:
        raw_name = panel_dict["section_name"]
    else:
        raw_name = panel_dict["panel_name"]
    var_name = 'var_' + sanitize_for_var(raw_name)

    code += fr'''
# Panel: {raw_name}
{var_name} = add_textbox(
    {slide_object_name},
    '{var_name}',
    {panel_dict['x']},
    {panel_dict['y']},
    {panel_dict['width']},
    {panel_dict['height']},
    text="",
    word_wrap=True,
    font_size=40,
    bold=False,
    italic=False,
    alignment="left",
    fill_color=None,
    font_name="Arial"
)
'''

    if visible:
        if theme is None:
            code += fr'''
# Make border visible
style_shape_border({var_name}, color=(0, 0, 0), thickness=5, line_style="solid")
'''
        else:
            code += fr'''
# Make border visible
style_shape_border({var_name}, color={theme['color']}, thickness={theme['thickness']}, line_style="{theme['line_style']}")
'''

    return code

# ---- Step 7-3: Textbox Code -------------------------------------------------
def generate_textbox_code(
    text_dict,
    utils_functions,
    slide_object_name,
    visible=False,
    content=None,
    theme=None,
    tmp_dir='tmp',
):
    """Return Python code string that places a textbox with formatted content on the slide."""
    code = utils_functions
    raw_name = text_dict["textbox_name"]
    var_name = sanitize_for_var(raw_name)

    is_title = text_dict.get('is_title', None)

    code += fr'''
# Textbox: {raw_name}
{var_name} = add_textbox(
    {slide_object_name}, 
    '{var_name}', 
    {text_dict['x']}, 
    {text_dict['y']}, 
    {text_dict['width']}, 
    {text_dict['height']}, 
    text="", 
    word_wrap=True,
    font_size=40,
    bold=False,
    italic=False,
    alignment="left",
    fill_color=None,
    font_name="Arial",
    is_title={is_title}
)
'''
    if visible:
        if theme is None:
            code += fr'''
# Make border visible
style_shape_border({var_name}, color=(255, 0, 0), thickness=5, line_style="solid")
'''
        else:
            code += fr'''
# Make border visible
style_shape_border({var_name}, color={theme['color']}, thickness={theme['thickness']}, line_style="{theme['line_style']}")
'''

    if content is not None:
        import os as _os
        _os.makedirs(tmp_dir, exist_ok=True)
        tmp_name = f'{tmp_dir}/{text_dict["panel_id"]}_{var_name}_content.json'
        json.dump(content, open(tmp_name, 'w'), indent=4)
        code += fr'''
fill_textframe({var_name}, json.load(open('{tmp_name}', 'r')))
'''

    return code

# ---- Step 7-4: Figure (Image) Code ------------------------------------------
def generate_figure_code(figure_dict, utils_functions, slide_object_name, img_path, visible=False, theme=None):
    """Return Python code string that inserts an image at the figure's position on the slide."""
    code = utils_functions
    raw_name = figure_dict["figure_name"]
    var_name = sanitize_for_var(raw_name)

    # Resolve to absolute path so generated code works regardless of working directory
    abs_img_path = os.path.abspath(img_path) if img_path and not os.path.isabs(img_path) else (img_path or "")

    code += fr'''
# Figure: {raw_name}
{var_name} = add_image(
    {slide_object_name},
    '{var_name}',
    {figure_dict['x']},
    {figure_dict['y']},
    {figure_dict['width']},
    {figure_dict['height']},
    image_path="{abs_img_path}"
)
'''

    if visible:
        if theme is None:
            code += fr'''
# Make border visible
style_shape_border({var_name}, color=(0, 0, 255), thickness=5, line_style="long_dash_dot")
'''
        else:
            code += fr'''
# Make border visible
style_shape_border({var_name}, color={theme['color']}, thickness={theme['thickness']}, line_style="{theme['line_style']}")
'''

    return code


# ---- Step 7 [Entry]: Arrangements → python-pptx Code String ----------------
# Calls: 7-1 initialize_poster_code
#        7-2 generate_panel_code      (per panel)
#        7-3 generate_textbox_code    (per textbox)
#        7-4 generate_figure_code     (per figure)
#        7-5 save_poster_code
def generate_poster_code(
    panel_arrangement_list,
    text_arrangement_list,
    figure_arrangement_list,
    presentation_object_name,
    slide_object_name,
    utils_functions,
    slide_width,
    slide_height,
    img_path,
    save_path,
    visible=False,
    content=None,
    only_textbox=False,
    test_rendering=False,
    theme=None,
    tmp_dir='tmp',
):
    """Assemble and execute python-pptx code for all panels, textboxes, and figures; save the .pptx file."""
    code = ''
    code += initialize_poster_code(slide_width, slide_height, slide_object_name, presentation_object_name, utils_functions)

    if theme is None:
        panel_visible = visible
        textbox_visible = visible
        figure_visible = visible

        panel_theme, textbox_theme, figure_theme = None, None, None
    else:
        panel_visible = theme['panel_visible']
        textbox_visible = theme['textbox_visible']
        figure_visible = theme['figure_visible']
        panel_theme = theme['panel_theme']
        textbox_theme = theme['textbox_theme']
        figure_theme = theme['figure_theme']

    for p in panel_arrangement_list:
        try:
            panel_visible = int(p['panel_id']) < 10  # section panels are visible
        except (ValueError, TypeError):
            panel_visible = True   # non-numeric id (e.g. 'title') → show
        if test_rendering:
            panel_visible = True
        code += generate_panel_code(p, '', slide_object_name, panel_visible, panel_theme)

    if only_textbox:
        t = text_arrangement_list[0]
        code += generate_textbox_code(t, '', slide_object_name, textbox_visible, content, textbox_theme, tmp_dir)
    else:

        for i in range(len(text_arrangement_list)):
            t = text_arrangement_list[i]
            textbox_content = content[i]['content_for_ppt']

            try:
                is_split = int(t['panel_id']) < 0
            except (ValueError, TypeError):
                is_split = False
            if is_split:
                textbox_visible_ = True
                code += generate_textbox_code(t, '', slide_object_name, textbox_visible_, textbox_content, theme['split_theme'], tmp_dir)
            else:
                textbox_visible_ = textbox_visible
                if 'visible' in t.keys():
                    textbox_visible_ = t['visible']
                code += generate_textbox_code(t, '', slide_object_name, textbox_visible_, textbox_content, textbox_theme, tmp_dir)

    for f in figure_arrangement_list:
        if img_path is None:
            code += generate_figure_code(f, '', slide_object_name, f['figure_path'], figure_visible, figure_theme)
        else:
            code += generate_figure_code(f, '', slide_object_name, img_path, figure_visible, figure_theme)

    code += save_poster_code(save_path, '', presentation_object_name)

    return code


def generate_poster_code_ablation(
    panel_arrangement_list,
    text_arrangement_list,
    figure_arrangement_list,
    presentation_object_name,
    slide_object_name,
    utils_functions,
    slide_width,
    slide_height,
    img_path,
    save_path,
    visible=False,
    content=None,
    only_textbox=False,
    test_rendering=False,
    theme=None,
    tmp_dir='tmp',
):
    """Ablation variant of generate_poster_code: flat layout without hierarchical tree structure."""
    code = ''
    code += initialize_poster_code(slide_width, slide_height, slide_object_name, presentation_object_name, utils_functions)

    if theme is None:
        panel_visible = visible
        textbox_visible = visible
        figure_visible = visible

        panel_theme, textbox_theme, figure_theme = None, None, None
    else:
        panel_visible = theme['panel_visible']
        textbox_visible = theme['textbox_visible']
        figure_visible = theme['figure_visible']
        panel_theme = theme['panel_theme']
        textbox_theme = theme['textbox_theme']
        figure_theme = theme['figure_theme']

        type_fill_color = dict()
        type_fill_color['section'] = (84, 130, 53)
        type_fill_color['subsection'] = (169, 209, 142)
        type_fill_color['content'] = (226, 240, 217)

    for p in panel_arrangement_list:
        try:
            panel_visible = int(p['panel_id']) < 10
        except (ValueError, TypeError):
            panel_visible = True
        if test_rendering:
            panel_visible = True
        code += generate_panel_code(p, '', slide_object_name, panel_visible, panel_theme)

    if only_textbox:
        t = text_arrangement_list[0]
        code += generate_textbox_code(t, '', slide_object_name, textbox_visible, content, textbox_theme, tmp_dir)
    else:

        t = text_arrangement_list[0]; textbox_content = content[0]['content_for_ppt']
        code += generate_textbox_code(t, '', slide_object_name, False, textbox_content, textbox_theme, tmp_dir)
        t = text_arrangement_list[1]; textbox_content = content[1]['content_for_ppt']
        code += generate_textbox_code(t, '', slide_object_name, False, textbox_content, textbox_theme, tmp_dir)

        for i in range(2, len(text_arrangement_list)):
            t = text_arrangement_list[i]
            fcolor = type_fill_color[t['node_type']]
            textbox_content = [
                    {
                        "alignment": "left",
                        "bullet": False,
                        "level": 0,
                        "font_size": 50,
                        "runs": [
                            {
                                "text": " ",
                                "bold": True,
                            }
                        ]
                    }
                ]
            textbox_content[0]['runs'][0]['fill_color'] = fcolor

            textbox_visible_ = True
            code += generate_textbox_code(t, '', slide_object_name, textbox_visible_, textbox_content, textbox_theme, tmp_dir)


    for f in figure_arrangement_list:
        if img_path is None:
            code += generate_figure_code(f, '', slide_object_name, f['figure_path'], figure_visible, figure_theme)
        else:
            code += generate_figure_code(f, '', slide_object_name, img_path, figure_visible, figure_theme)

    code += save_poster_code(save_path, '', presentation_object_name)

    return code