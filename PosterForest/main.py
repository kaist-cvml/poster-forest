from PosterForest.pipeline import (
    parse_hierarchical_document, refine_document_tree, generate_panel_layout,
    extract_figures_and_tables, print_tree_unified, save_tree_with_metadata, count_total_nodes  # Step 1, 2, 3
)
from utils.wei_utils import get_agent_config, utils_functions, run_code, style_bullet_content, scale_to_target_area, char_capacity  # General utilities
from PosterForest.step_04_layout_hierarchy import layout_initialization, get_arrangments_in_inches, to_inches, generate_complete_layout  # Step 4 - Layout generation
from PosterForest.step_07_pptx_generation import generate_poster_code  # Step 7 - PowerPoint generation
from utils.src.utils import ppt_to_images  # Step 8 - Image conversion
from PosterForest.step_05_content_generation import gen_bullet_point_content  # Step 5 - Content generation
from utils.ablation_utils import no_tree_get_layout  # Step 4 - Alternative layout (ablation)
from PosterForest.step_06_modification_planning import step_06_modification_planning # Step 6 - Feedback loop

import argparse
import json
import os
import time
import shutil
from datetime import datetime

from PosterForest.poster_config import (
    UNITS_PER_INCH,
    TITLE_HEIGHT_RATIO, SECTION_TITLE_HEIGHT, SUBSECTION_TITLE_HEIGHT,
    TITLE_FONT_SIZE, TITLE_FONT_SIZE_2,
    SECTION_TITLE_FONT_SIZE, SUBSECTION_TITLE_FONT_SIZE, CONT_FONT_SIZE,
    SHRINK_MARGIN, FONT_NAME, MAX_ATTEMPT, BOTTOM_MARGIN,
    THEME_TITLE_TEXT_COLOR, THEME_TITLE_FILL_COLOR, THEME_SUBSEC_TITLE_TEXT_COLOR,
)

units_per_inch = UNITS_PER_INCH

# Theme settings - colors and styles used in poster design
theme_title_text_color = THEME_TITLE_TEXT_COLOR
theme_title_fill_color = THEME_TITLE_FILL_COLOR
theme = {
    'panel_visible': True,
    'textbox_visible': False,
    'figure_visible': False,
    'panel_theme': {
        'color': (255, 255, 255),
        'thickness': 5,
        'line_style': 'solid',
    },
    'textbox_theme': None,
    'figure_theme': None,
}

def cleanup_global_folders(poster_name, args, keep_traditional=True):
    """
    Clean up global folders after pipeline completion

    Args:
        poster_name: Poster name
        args: Command-line arguments
        keep_traditional: Whether to keep the traditional folder structure for backward compatibility
    """
    if not keep_traditional:
        # Remove global contents folder files for this poster
        contents_file = f'contents/<{args.model_name_t}_{args.model_name_v}>_{args.poster_name}_raw_content_{args.index}.json'
        if os.path.exists(contents_file):
            os.remove(contents_file)
            print(f"🧹 Cleaned up: {contents_file}")

        bullet_file = f'contents/<{args.model_name_t}_{args.model_name_v}>_{args.poster_name}_bullet_point_content_{args.index}.json'
        if os.path.exists(bullet_file):
            os.remove(bullet_file)
            print(f"🧹 Cleaned up: {bullet_file}")

        final_arrangements = f'final_arrangements/<{args.model_name_t}_{args.model_name_v}>_{args.poster_name}_final_arrangement_{args.index}.json'
        if os.path.exists(final_arrangements):
            os.remove(final_arrangements)
            print(f"🧹 Cleaned up: {final_arrangements}")

        # Remove tree_splits file for this poster
        tree_split_file = f'tree_splits/<{args.model_name_t}_{args.model_name_v}>_{args.poster_name}_tree_split_{args.index}.json'
        if os.path.exists(tree_split_file):
            os.remove(tree_split_file)
            print(f"🧹 Cleaned up: {tree_split_file}")

        # Remove outline file for this poster
        outline_file = f'outlines/<{args.model_name_t}_{args.model_name_v}>_{args.poster_name}_outline_{args.index}.json'
        if os.path.exists(outline_file):
            os.remove(outline_file)
            print(f"🧹 Cleaned up: {outline_file}")

        # Remove images_and_tables files for this poster
        images_json_file = f'<{args.model_name_t}_{args.model_name_v}>_images_and_tables/{args.poster_name}_images.json'
        if os.path.exists(images_json_file):
            os.remove(images_json_file)
            print(f"🧹 Cleaned up: {images_json_file}")

        tables_json_file = f'<{args.model_name_t}_{args.model_name_v}>_images_and_tables/{args.poster_name}_tables.json'
        if os.path.exists(tables_json_file):
            os.remove(tables_json_file)
            print(f"🧹 Cleaned up: {tables_json_file}")

        # Remove images_and_tables directory for this poster
        images_dir = f'<{args.model_name_t}_{args.model_name_v}>_images_and_tables/{args.poster_name}'
        if os.path.exists(images_dir):
            shutil.rmtree(images_dir)
            print(f"🧹 Cleaned up: {images_dir}/")
    else:
        print("🔄 Keeping traditional folder structure for backward compatibility")

STEP_NAMES = [
    "01_parse_raw_poster",
    "02_filter_images_tables",
    "03_generate_outline",
    "04_generate_layout",
    "05_generate_content",
    "06_step_06_modification_planning",
    "07_generate_powerpoint",
    "08_finalize_output",
]


class LazyStepDirs:
    """Returns per-step output paths, creating the actual directory on first access."""

    def __init__(self, output_base):
        self._output_base = output_base
        self._created = set()

    def __getitem__(self, step_name):
        path = os.path.join(self._output_base, step_name)
        if step_name not in self._created:
            os.makedirs(path, exist_ok=True)
            self._created.add(step_name)
        return path


def create_output_structure(poster_name, model_name_t, model_name_v):
    """Creates only the base output directory; per-step folders are created on demand."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = f"outputs/{timestamp}_{model_name_t}_{model_name_v}_{poster_name}"
    os.makedirs(output_base, exist_ok=True)
    return output_base, LazyStepDirs(output_base)

def print_step_header(step_num, step_name, description=""):
    """Print per-step header"""
    print("\n" + "="*80)
    print(f"📋 STEP {step_num:02d}: {step_name}")
    if description:
        print(f"   {description}")
    print("="*80)

def print_step_result(step_num, input_tokens=None, output_tokens=None, files_saved=None, duration=None):
    """Print step completion summary"""
    print(f"\n✅ STEP {step_num:02d} COMPLETED")
    if duration:
        print(f"   ⏱️  Duration: {duration:.2f}s")
    if input_tokens and output_tokens:
        print(f"   🔢 Tokens: {input_tokens} → {output_tokens}")
    if files_saved:
        print(f"   💾 Saved: {', '.join(files_saved)}")
    print("-"*80)

def save_step_metadata(step_dir, step_name, duration, **kwargs):
    """Save step metadata to JSON file"""
    metadata = {
        "step_name": step_name,
        "timestamp": datetime.now().isoformat(),
        "duration": duration,
        **kwargs
    }
    metadata_path = os.path.join(step_dir, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    return "metadata.json"

def copy_file_if_exists(src_path, dest_dir, dest_filename=None):
    """Copy file if it exists and return the filename"""
    if os.path.exists(src_path):
        if dest_filename is None:
            dest_filename = os.path.basename(src_path)
        dest_path = os.path.join(dest_dir, dest_filename)
        shutil.copy2(src_path, dest_path)
        return dest_filename
    return None

def copy_directory_if_exists(src_dir, dest_dir, dest_dirname=None):
    """Copy directory if it exists and return the directory name"""
    if os.path.exists(src_dir):
        if dest_dirname is None:
            dest_dirname = os.path.basename(src_dir)
        dest_path = os.path.join(dest_dir, dest_dirname)
        shutil.copytree(src_dir, dest_path, dirs_exist_ok=True)
        return f"{dest_dirname}/ (directory)"
    return None

def save_json_data(data, step_dir, filename):
    """Helper function to save JSON data"""
    filepath = os.path.join(step_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return filename

def handle_step_results(step_dir, step_name, duration, file_operations=None, data_saves=None, **metadata_kwargs):
    """
    Handle step results with file operations and metadata saving

    Args:
        step_dir: Step directory path
        step_name: Name of the step
        duration: Step duration
        file_operations: List of tuples [(src_path, dest_filename), ...]
        data_saves: List of tuples [(data, filename), ...]
        **metadata_kwargs: Additional metadata to save

    Returns:
        List of saved files
    """
    saved_files = []

    # Handle file operations (copying existing files)
    if file_operations:
        for operation in file_operations:
            if len(operation) == 2:
                src_path, dest_filename = operation
                result = copy_file_if_exists(src_path, step_dir, dest_filename)
            elif len(operation) == 3 and operation[2] == 'directory':
                src_path, dest_dirname = operation[0], operation[1]
                result = copy_directory_if_exists(src_path, step_dir, dest_dirname)
            else:
                continue

            if result:
                saved_files.append(result)

    # Handle data saves (saving new JSON data)
    if data_saves:
        for data, filename in data_saves:
            result = save_json_data(data, step_dir, filename)
            saved_files.append(result)

    # Save metadata
    metadata_file = save_step_metadata(step_dir, step_name, duration, **metadata_kwargs)
    saved_files.append(metadata_file)

    return saved_files

def count_total_nodes(node):
    """Recursively count the total number of nodes in the tree"""
    count = 1 # count the current node
    for child in node.get('children', []):
        count += count_total_nodes(child)
    return count

def execute_step(step_num, step_name, step_func):
    """
    Helper function to execute a pipeline step

    Args:
        step_num: Step number
        step_name: Step name
        step_func: Function to execute

    Returns:
        tuple: (result, execution_time)
    """
    step_start = time.time()
    print_step_header(step_num, step_name)

    # Execute the step
    result = step_func()

    step_duration = time.time() - step_start

    # Each step already prints its own tree output, so skip duplicate printing

    return result, step_duration

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Poster Generation Pipeline')
    parser.add_argument('--paper_path', type=str)
    parser.add_argument('--model_name_t', type=str, default='4o')
    parser.add_argument('--model_name_v', type=str, default='4o')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--poster_name', type=str, default=None)
    parser.add_argument('--tmp_dir', type=str, default='tmp')
    parser.add_argument('--poster_width_inches', type=int, default=None)
    parser.add_argument('--poster_height_inches', type=int, default=None)
    parser.add_argument('--no_blank_detection', action='store_true', help='When overflow is severe, try this option.')
    parser.add_argument('--ablation_no_tree_layout', action='store_true', help='Ablation study: no tree layout')
    parser.add_argument('--ablation_no_commenter', action='store_true', help='Ablation study: no commenter')
    parser.add_argument('--ablation_no_example', action='store_true', help='Ablation study: no example')
    parser.add_argument('--ablation_flat_tree', action='store_true', help='Ablation study: flat tree structure (no section-subsection hierarchy)')
    parser.add_argument('--use_caption_based_parsing', action='store_true', default=True, help='Use caption-based parsing (default)')

    args = parser.parse_args()

    start_time = time.time()

    # Initial setup
    detail_log = {}
    agent_config_t = get_agent_config(args.model_name_t)
    agent_config_v = get_agent_config(args.model_name_v)
    poster_name = args.paper_path.split('/')[-2].replace(' ', '_')
    if args.poster_name is None:
        args.poster_name = poster_name
    else:
        poster_name = args.poster_name

    # Create structured output directory (including tmp_dir)
    output_base, step_dirs = create_output_structure(poster_name, args.model_name_t, args.model_name_v)

    # Redirect tmp_dir to the per-run output folder
    args.tmp_dir = step_dirs["tmp"]

    print("\n" + "🎯 PAPER-TO-POSTER GENERATION PIPELINE (HIERARCHICAL)" + " "*15)
    print("="*80)
    print(f"📄 Input Paper: {args.paper_path}")
    print(f"🤖 Text Model: {args.model_name_t}, Vision Model: {args.model_name_v}")
    print(f"📁 Output Directory: {output_base}")
    print(f"🗂️ Temporary Directory: {args.tmp_dir}")
    print("="*80)

    # Poster size configuration
    meta_json_path = args.paper_path.replace('paper.pdf', 'meta.json')
    if args.poster_width_inches is not None and args.poster_height_inches is not None:
        poster_width = args.poster_width_inches * units_per_inch
        poster_height = args.poster_height_inches * units_per_inch
    elif os.path.exists(meta_json_path):
        meta_json = json.load(open(meta_json_path, 'r'))
        poster_width = meta_json['width']
        poster_height = meta_json['height']
    else:
        poster_width = 48 * units_per_inch
        poster_height = 36 * units_per_inch

    poster_width, poster_height = scale_to_target_area(poster_width, poster_height)
    poster_width_inches = to_inches(poster_width, units_per_inch)
    poster_height_inches = to_inches(poster_height, units_per_inch)

    if poster_width_inches > 56 or poster_height_inches > 56:
        if poster_width_inches >= poster_height_inches:
            scale_factor = 56 / poster_width_inches
        else:
            scale_factor = 56 / poster_height_inches
        poster_width_inches  *= scale_factor
        poster_height_inches *= scale_factor
        poster_width  = poster_width_inches  * units_per_inch
        poster_height = poster_height_inches * units_per_inch

    print(f'📏 Poster Size: {poster_width_inches:.1f} x {poster_height_inches:.1f} inches')

    total_input_tokens_t, total_output_tokens_t = 0, 0
    total_input_tokens_v, total_output_tokens_v = 0, 0

    # -------------------------------------------------------------------------
    # Step 1: Hierarchical document parsing - build tree structure from PDF
    # Functions used: parse_hierarchical_document, extract_figures_and_tables
    # -------------------------------------------------------------------------
    def step1_parse_document():
        """Step 1 execution function - Document parsing and structure extraction"""
        print("🔍 Parsing document structure and extracting content...")

        # Use caption-based parsing (now the standard approach)
        input_token, output_token, document_tree, raw_result, raw_content_path, figures, tables, text_content = \
            parse_hierarchical_document(
                args, agent_config_t, step_dirs["01_parse_raw_poster"]
            )

        # Save results
        saved_filename = save_tree_with_metadata(
            document_tree,
            step_dirs["01_parse_raw_poster"],
            f'{poster_name}_raw_tree.json',
            tree_type="document",
            parsing_method="caption_based",
            figures_found=len(figures),
            tables_found=len(tables)
        )

        return {
            'tree': document_tree,
            'figures': figures,
            'tables': tables,
            'text_content': text_content,
            'input_tokens': input_token,
            'output_tokens': output_token,
            'saved_files': [saved_filename, f'{poster_name}_raw_content.json', f'{poster_name}_images.json', f'{poster_name}_tables.json']
        }

    result1, duration1 = execute_step(1, "HIERARCHICAL DOCUMENT PARSING", step1_parse_document)

    total_input_tokens_t += result1['input_tokens']
    total_output_tokens_t += result1['output_tokens']
    detail_log['hierarchical_parse_in_t'] = result1['input_tokens']
    detail_log['hierarchical_parse_out_t'] = result1['output_tokens']

    print_step_result(1, result1['input_tokens'], result1['output_tokens'], result1['saved_files'], duration1)

    # -------------------------------------------------------------------------
    # Step 2: Tree refinement - structural improvement for poster optimization (depth-2 constraint)
    # Functions used: refine_document_tree
    # -------------------------------------------------------------------------
    def step2_refine_tree():
        """Step 2 execution function - Tree refinement for poster optimization"""
        if getattr(args, 'ablation_flat_tree', False):
            print("🌳 Refining tree structure with FLAT LAYOUT (ablation study)...")
        else:
            print("🌳 Refining tree structure for poster layout optimization...")

        input_token, output_token, refined_tree = refine_document_tree(
            args, agent_config_t, result1['tree'], result1['figures'], result1['tables'],
            text_content=result1.get('text_content')
        )

        # Save results
        main_sections = len(refined_tree.get('children', []))
        if getattr(args, 'ablation_flat_tree', False):
            # In flat mode, all children are at the same level (no subsections)
            total_subsections = 0
            refinement_operations = "flatten_hierarchy_ablation"
        else:
            total_subsections = sum(len(child.get('children', [])) for child in refined_tree.get('children', []))
            refinement_operations = "merge_and_prune"

        saved_filename = save_tree_with_metadata(
            refined_tree,
            step_dirs["02_filter_images_tables"],
            f'{poster_name}_refined_tree.json',
            tree_type="refined" if not getattr(args, 'ablation_flat_tree', False) else "flat_ablation",
            main_sections=main_sections,
            total_subsections=total_subsections,
            refinement_operations=refinement_operations,
            ablation_flat_tree=getattr(args, 'ablation_flat_tree', False)
        )

        return {
            'tree': refined_tree,
            'input_tokens': input_token,
            'output_tokens': output_token,
            'saved_files': [saved_filename]
        }

    result2, duration2 = execute_step(2, "TREE REFINEMENT", step2_refine_tree)

    total_input_tokens_t += result2['input_tokens']
    total_output_tokens_t += result2['output_tokens']
    detail_log['tree_refinement_in_t'] = result2['input_tokens']
    detail_log['tree_refinement_out_t'] = result2['output_tokens']

    print_step_result(2, result2['input_tokens'], result2['output_tokens'], result2['saved_files'], duration2)

    # -------------------------------------------------------------------------
    # Step 3: Panel layout generation - convert refined tree to poster panels (depth-2 constraint)
    # Functions used: generate_panel_layout
    # -------------------------------------------------------------------------
    def step3_generate_panels():
        """Step 3 execution function - Panel layout generation from refined tree"""
        print("📐 Converting refined tree to poster panels...")

        input_token, output_token, poster_tree, figures = generate_panel_layout(
            args, agent_config_t, result2['tree'], result1['figures'], result1['tables']
        )

        # Save results
        total_panels = count_total_nodes(poster_tree)
        figures_mapped = len(figures)

        poster_tree_file = save_tree_with_metadata(
            poster_tree,
            step_dirs["03_generate_outline"],
            f'{poster_name}_poster_panels.json',
            tree_type="poster",
            total_panels=total_panels,
            figures_mapped=figures_mapped
        )

        # Also save the figures mapping
        figures_file = save_json_data(figures, step_dirs["03_generate_outline"], f'{poster_name}_figures.json')

        return {
            'tree': poster_tree,
            'figures': figures,
            'input_tokens': input_token,
            'output_tokens': output_token,
            'saved_files': [poster_tree_file, figures_file]
        }

    result3, duration3 = execute_step(3, "PANEL LAYOUT GENERATION", step3_generate_panels)

    total_input_tokens_t += result3['input_tokens']
    total_output_tokens_t += result3['output_tokens']
    detail_log['panel_generation_in_t'] = result3['input_tokens']
    detail_log['panel_generation_out_t'] = result3['output_tokens']

    print_step_result(3, result3['input_tokens'], result3['output_tokens'], result3['saved_files'], duration3)

    # -------------------------------------------------------------------------
    # Hierarchical parsing complete - final summary
    # -------------------------------------------------------------------------
    print("\n" + "🎉 HIERARCHICAL PARSING COMPLETED!" + " "*37)
    print("="*80)
    print(f"⏱️  Steps 1-3 Time: {time.time() - start_time:.2f}s")
    print(f"🔢 Total Tokens: {total_input_tokens_t} → {total_output_tokens_t}")
    print(f"📁 All Results Saved: {output_base}")
    print("="*80)
    print("\n📊 Generated Outputs:")
    print(f"   📋 Raw Tree: {step_dirs['01_parse_raw_poster']}/{poster_name}_raw_tree.json")
    print(f"   🌳 Refined Tree: {step_dirs['02_filter_images_tables']}/{poster_name}_refined_tree.json")
    print(f"   📐 Panel Layout: {step_dirs['03_generate_outline']}/{poster_name}_poster_panels.json")
    print(f"   🖼️ Figure Mapping: {step_dirs['03_generate_outline']}/{poster_name}_figures.json")
    print("="*80)
    print("\n💡 Next steps: Run steps 4-8 for complete poster generation")
    print("   These steps remain unchanged from the original pipeline")
    print("   Note: All outputs now conform to depth 2 constraint (sections → subsections only)")

    # Save intermediate results
    interim_results = {
        'poster_tree': result3['tree'],
        'filtered_figures': result3['figures'],
        'figures': result1['figures'],
        'tables': result1['tables'],
        'poster_dimensions': {
            'width': poster_width,
            'height': poster_height,
            'width_inches': poster_width_inches,
            'height_inches': poster_height_inches
        },
        'token_usage': {
            'total_input': total_input_tokens_t,
            'total_output': total_output_tokens_t,
            'detail_log': detail_log
        }
    }

    interim_file = os.path.join(output_base, 'pipeline_results.json')
    with open(interim_file, 'w', encoding='utf-8') as f:
        json.dump(interim_results, f, indent=4, ensure_ascii=False)

    print(f"\n📄 Complete results saved: {interim_file}")



    # -------------------------------------------------------------------------
    # Step 4: Layout Generation
    # -------------------------------------------------------------------------

    print('=== Step 4: Layout Generation ===')

    step_start = time.time()


    # Typography and layout constants imported from PosterForest.poster_config

    # Remove static file dependency: read from step_dirs instead
    with open(os.path.join(step_dirs["03_generate_outline"], f'{poster_name}_poster_panels.json'), 'r') as f:
        poster_tree = json.load(f)
    panels = poster_tree['tree_data']['children']
    panels = panels[1:]

    for p in panels:
        if 'abstract' in p['section_name'].lower():
            panels.remove(p)
            break

    # Layout uses a slightly reduced height so panels end BOTTOM_MARGIN units
    # above the slide bottom, leaving clean empty space at the bottom edge.
    effective_poster_height = poster_height - BOTTOM_MARGIN
    title_panel, content_panel_tree = layout_initialization(
        panels,
        poster_width,
        effective_poster_height,
        title_h = effective_poster_height * TITLE_HEIGHT_RATIO,
    )

    panel_arrangement, figure_arrangement, text_arrangement = generate_complete_layout(
        title_panel,
        content_panel_tree,
        poster_width=poster_width,
        poster_height=effective_poster_height,
        title_h=effective_poster_height * TITLE_HEIGHT_RATIO,
        section_title_h=SECTION_TITLE_HEIGHT,
        subsection_title_h=SUBSECTION_TITLE_HEIGHT,
        shrink_margin=SHRINK_MARGIN
    )

    # Save layout results
    # poster_height is the actual full slide height (used for PPTX slide size).
    # Panels are laid out within effective_poster_height, leaving BOTTOM_MARGIN
    # of empty space at the bottom.
    tree_split_results = {
        'poster_width': poster_width,
        'poster_height': poster_height,
        'panels': panels,
        'panel_arrangement': panel_arrangement,
        'figure_arrangement': figure_arrangement,
        'text_arrangement': text_arrangement,

        'title_panel': title_panel,
        'content_panel_tree': content_panel_tree,
        'title_h': effective_poster_height * TITLE_HEIGHT_RATIO,

        # 'matching_figures': matching_figures,
        'section_title_h': SECTION_TITLE_HEIGHT,  # Use 32 for section titles
        'subsection_title_h': SUBSECTION_TITLE_HEIGHT,  # Use 25 for subsection titles

        'title_font_size': TITLE_FONT_SIZE,  # Example font size for title
        'title_font_size_2': TITLE_FONT_SIZE_2,  # Example font size for secondary title
        'section_title_font_size': SECTION_TITLE_FONT_SIZE,  # Example font size for section titles
        'subsection_title_font_size': SUBSECTION_TITLE_FONT_SIZE,  # Example font size for subsection titles
        'cont_font_size': CONT_FONT_SIZE,  # Example font size
        'shrink_margin': SHRINK_MARGIN,
        'units_per_inch': UNITS_PER_INCH,  # Define units per inch for conversion
        'font_name': FONT_NAME,
    }

    # Save to step_dirs (removing static dependency)
    save_json_data(tree_split_results, step_dirs["04_generate_layout"], f'{poster_name}_tree_split_results.json')


    input_token, output_token = 0, 0  # Tree layout doesn't use tokens

    # Save Step 4 results
    step_data = {
        'step_info.json': {
            'step': 4,
            'name': 'layout_generation',
            'input_tokens': input_token,
            'output_tokens': output_token,
            'num_panels': len(panel_arrangement) if panel_arrangement else 0,
            'num_figures': len(figure_arrangement) if figure_arrangement else 0,
            'num_text_boxes': len(text_arrangement) if text_arrangement else 0,
        },
        'layout_results.json': tree_split_results,
    }

    save_json_data(step_data, step_dirs["04_generate_layout"], f'{poster_name}_step_data.json')


    ''' ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ '''



    # Step 5: Generate content
    print('=== Step 5: Content Generation ===')

    # Set required file paths from step_dirs
    tree_split_path = os.path.join(step_dirs["04_generate_layout"], f'{poster_name}_tree_split_results.json')
    raw_content_path = os.path.join(step_dirs["01_parse_raw_poster"], f'{poster_name}_raw_content.json')

    input_token_t, output_token_t, input_token_v, output_token_v, bullet_point_contents = gen_bullet_point_content(args, agent_config_t, tree_split_path=tree_split_path, raw_content_path=raw_content_path)
    total_input_tokens_t += input_token_t
    total_output_tokens_t += output_token_t
    total_input_tokens_v += input_token_v
    total_output_tokens_v += output_token_v

    print(f'Content generation token consumption T: {input_token_t} -> {output_token_t}')
    print(f'Content generation token consumption V: {input_token_v} -> {output_token_v}')

    detail_log['content_in_t'] = input_token_t
    detail_log['content_out_t'] = output_token_t
    detail_log['content_in_v'] = input_token_v
    detail_log['content_out_v'] = output_token_v

    # Save to step_dirs (removing static dependency)
    save_json_data(bullet_point_contents, step_dirs["05_generate_content"], f'{poster_name}_bullet_contents.json')

    step_data = {
        'step_info.json': {
            'step': 5,
            'name': 'content_generation',
            'description': 'Poster content generation',
            'input_tokens_text': input_token_t,
            'output_tokens_text': output_token_t,
            'input_tokens_vision': input_token_v,
            'output_tokens_vision': output_token_v,
        }
    }

    save_json_data(step_data, step_dirs["05_generate_content"], f'{poster_name}_step_data.json')



    # Step 6: Feedback Loop for layout and content optimization
    print('=== Step 6: Feedback Loop ===')

    # Set required file paths from step_dirs
    bullet_content_path = os.path.join(step_dirs["05_generate_content"], f'{poster_name}_bullet_contents.json')
    tree_split_path = os.path.join(step_dirs["04_generate_layout"], f'{poster_name}_tree_split_results.json')

    input_token_t, output_token_t, input_token_v, output_token_v, final_arrangements = step_06_modification_planning(
        args, agent_config_t, agent_config_v, MAX_ATTEMPT,
        bullet_content_path=bullet_content_path,
        tree_split_path=tree_split_path,
        raw_content_path=raw_content_path,
    )
    total_input_tokens_t += input_token_t
    total_output_tokens_t += output_token_t
    total_input_tokens_v += input_token_v
    total_output_tokens_v += output_token_v
    print(f'Content generation token consumption T: {input_token_t} -> {output_token_t}')
    print(f'Content generation token consumption V: {input_token_v} -> {output_token_v}')


    detail_log['feedback_in_t'] = input_token_t
    detail_log['feedback_out_t'] = output_token_t
    detail_log['feedback_in_v'] = input_token_v
    detail_log['feedback_out_v'] = output_token_v

    # Save to step_dirs (removing static dependency)
    save_json_data(final_arrangements, step_dirs["06_step_06_modification_planning"], f'{poster_name}_final_arrangements.json')

    step_data = {
        'step_info.json': {
            'step': 6,
            'name': 'feedback_loop',
            'description': 'Feedback loop',
            'input_tokens_text': input_token_t,
            'output_tokens_text': output_token_t,
            'input_tokens_vision': input_token_v,
            'output_tokens_vision': output_token_v,
        }
    }

    save_json_data(step_data, step_dirs["06_step_06_modification_planning"], f'{poster_name}_step_data.json')



    ''' ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ '''



    # Step 7: Generate the PowerPoint
    print('=== Step 7: PowerPoint Generation ===')

    panel_arrangement = final_arrangements['panel_arrangement']
    figure_arrangement = final_arrangements['figure_arrangement']
    text_arrangement = final_arrangements['text_arrangement']


    # Apply styles to the top title box
    style_bullet_content(text_arrangement[0]['content_for_ppt'], theme_title_text_color, theme_title_fill_color)
    style_bullet_content(text_arrangement[1]['content_for_ppt'], theme_title_text_color, theme_title_fill_color)

    for i in range(2, len(text_arrangement)): # Apply styles to section titles
        curr_content = text_arrangement[i]
        if 'title' in curr_content['textbox_name'].lower():
            try:
                _is_sec = int(curr_content['panel_id']) < 10
            except (ValueError, TypeError):
                _is_sec = False
            if _is_sec: # section title
                style_bullet_content(curr_content['content_for_ppt'], theme_title_text_color, theme_title_fill_color)
            # else: # subsection title
            #     style_bullet_content(curr_content['content_for_ppt'], theme_subtitle_text_color, theme_subtitle_fill_color)


    width_inch, height_inch, panel_arrangement_inches, figure_arrangement_inches, text_arrangement_inches = get_arrangments_in_inches(
        poster_width, poster_height, panel_arrangement, figure_arrangement, text_arrangement, UNITS_PER_INCH
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
        save_path=f'{args.tmp_dir}/poster.pptx',
        visible=False,
        content=text_arrangement,
        theme=theme,
        tmp_dir=args.tmp_dir,
    )

    # Save the poster code first so step 7 dir is always non-empty even if execution fails
    with open(os.path.join(step_dirs["07_generate_powerpoint"], 'poster_code.py'), 'w', encoding='utf-8') as f:
        f.write(poster_code)

    output, err = run_code(poster_code)
    if err is not None:
        raise RuntimeError(f'Error in generating PowerPoint: {err}')

    # Save step 7 metadata so viewer marks it as complete
    save_json_data(
        {'step': 7, 'name': 'powerpoint_generation', 'status': 'complete'},
        step_dirs["07_generate_powerpoint"],
        f'{poster_name}_step_data.json',
    )


    ''' ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ '''



    # Step 8: Convert the PowerPoint to figures
    print('=== Step 8: Image Conversion and Final Save ===')
    pptx_path = os.path.join(args.tmp_dir, 'poster.pptx')
    try:
        ppt_to_images(pptx_path, args.tmp_dir, output_type='jpg')
    except Exception as e:
        print(f'⚠️  Image conversion failed: {e}  (PPTX still saved)')

    end_time = time.time()
    time_taken = end_time - start_time

    # Final log data
    final_log_data = {
        'input_tokens_t': total_input_tokens_t,
        'output_tokens_t': total_output_tokens_t,
        'input_tokens_v': total_input_tokens_v,
        'output_tokens_v': total_output_tokens_v,
        'time_taken': time_taken,
        'poster_name': poster_name,
        'paper_path': args.paper_path,
        'poster_size': {
            'width_inches': poster_width_inches,
            'height_inches': poster_height_inches,
        },
        'parameters': {
            'model_name_t': args.model_name_t,
            'model_name_v': args.model_name_v,
            'ablation_no_tree_layout': args.ablation_no_tree_layout,
            'ablation_no_commenter': args.ablation_no_commenter,
            'ablation_no_example': args.ablation_no_example,
            'ablation_flat_tree': getattr(args, 'ablation_flat_tree', False),
        }
    }

    # Final outputs go in the root; intermediate debug renders go under debug_renders/
    final_dir = step_dirs['08_finalize_output']
    debug_dir = os.path.join(final_dir, 'debug_renders')
    os.makedirs(debug_dir, exist_ok=True)

    shutil.copy2(pptx_path, os.path.join(final_dir, 'poster_final.pptx'))
    final_image_src = os.path.join(args.tmp_dir, 'poster.jpg')
    if os.path.exists(final_image_src):
        shutil.copy2(final_image_src, os.path.join(final_dir, 'poster_final.jpg'))

    for img_file in os.listdir(args.tmp_dir):
        if not img_file.endswith(('.png', '.jpg', '.jpeg')):
            continue
        if img_file in ('poster.jpg', 'poster.png'):
            continue  # already copied as poster_final.jpg
        shutil.copy2(os.path.join(args.tmp_dir, img_file), os.path.join(debug_dir, img_file))


    # shutil.rmtree(args.tmp_dir)
    # print(f"🧹 Cleaned up: {args.tmp_dir}")

    step_data = {
        'step_info.json': {
            'step': 8,
            'name': 'final_results',
            'description': 'Final poster and output artifacts saved',
            'total_time_taken': time_taken,
            'total_tokens_used': {
                'text_model': {
                    'input': total_input_tokens_t,
                    'output': total_output_tokens_t,
                },
                'vision_model': {
                    'input': total_input_tokens_v,
                    'output': total_output_tokens_v,
                }
            }
        },
        'final_log.json': final_log_data,
        'detail_log.json': detail_log,
    }

    # Save step results and get the directory path
    save_json_data(step_data, step_dirs["08_finalize_output"], f'{poster_name}_step_data.json')



    print(f'Poster generation complete!')
    print(f'Total time elapsed: {time_taken:.2f}s')
    print(f'Output saved to: {output_base}')
    print(f'Text model token usage: {total_input_tokens_t} -> {total_output_tokens_t}')
    print(f'Vision model token usage: {total_input_tokens_v} -> {total_output_tokens_v}') 