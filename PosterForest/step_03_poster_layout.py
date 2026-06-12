"""
STEP 3: Poster Panel Layout Generation

Converts the refined tree from Step 2 into a poster panel tree.
Each leaf becomes a content node with at most one figure/table asset;
tp/gp proportions are computed and normalized for downstream layout.

Input:  Refined tree + filtered figure/table dicts from Step 2
Output: Poster panel tree + figure-path mapping for Step 4
"""


# ---- Step 3-1: Layout Parameter Calculation ---------------------------------
def count_total_nodes(tree):
    """Recursively count all nodes in the tree"""
    if not tree:
        return 0

    count = 1  # current node
    children = tree.get('children', [])
    for child in children:
        count += count_total_nodes(child)

    return count


def calculate_poster_layout_parameters(content, assets, figures, tables):
    """Calculate tp, gp, figure_size, figure_aspect, is_main_figure, is_table for poster content nodes."""
    import math as _math
    text_len = len(content) if content else 0
    is_main_figure = False  # set True for overview/architecture figures
    is_table = False        # set True when a table (not figure) is the primary asset

    # Proportional tp: sqrt-scaled char count.  Cap at 1200 chars to allow genuine
    # long sections (Method, Experiments) to get proportionally more panel space while
    # still dampening extreme outliers.  Range: 100 chars→90, 400→180, 800→255, 1200+→312.
    # Ratio 3.5:1 (was 2:1 at cap=400) so short Conclusion panels don't steal space
    # from dense Method sections, reducing large empty areas in small sections.
    if text_len == 0:
        tp = 0
    else:
        _POSTER_TEXT_CEIL = 1200  # raised from 400: allows ~3.5× ratio between short/long sections
        tp = max(80, int(_math.sqrt(min(text_len, _POSTER_TEXT_CEIL)) * 9))

    # Graphics proportion calculation and asset info extraction
    gp = 0
    asset_path = None
    figure_size = 0
    figure_aspect = 1.0

    if assets.get("figures") and figures:
        figure_id = assets["figures"][0]
        figure_data = None

        if str(figure_id) in figures:
            figure_data = figures[str(figure_id)]
            asset_path = figure_data.get('figure_path')
        elif figure_id in figures:
            figure_data = figures[figure_id]
            asset_path = figure_data.get('figure_path')

        if figure_data:
            figure_size = figure_data.get('figure_size', 0)
            figure_aspect = figure_data.get('figure_aspect', 1.0)

            # Detect main/overview/architecture figures that deserve more poster space.
            # A figure is "main" when its caption describes the overall method,
            # architecture, pipeline, or system — these must be large on a poster.
            _MAIN_KW = {
                # structural / architecture
                'overview', 'architecture', 'framework', 'pipeline', 'workflow',
                'diagram', 'schematic', 'illustration',
                # method-level descriptions
                'proposed method', 'proposed approach', 'our method', 'our approach',
                'our framework', 'our model', 'our pipeline',
                'method overview', 'model overview', 'system overview',
                # compound high-signal phrases only (bare 'overall'/'proposed' removed
                # to avoid false positives like "overall class distribution")
                'overall architecture', 'overall framework',
                'network architecture', 'model architecture',
            }
            cap = (figure_data.get('caption') or '').lower()
            cap_num = figure_data.get('caption_number')
            _is_early = False
            try:
                # Figures 1–3 are typically the most prominent method/overview figures
                _is_early = cap_num is not None and int(str(cap_num)) <= 3
            except (TypeError, ValueError):
                pass
            _is_main = any(kw in cap for kw in _MAIN_KW)

            # Teaser/motivation figures (limitations, prior work, problem statement) must NOT
            # be treated as main figures even when early and landscape — they are comparison
            # or motivation figures, not the paper's primary method overview.
            _TEASER_KW = {'limitation', 'existing method', 'prior work', 'problem', 'challenge', 'motivation'}
            _is_teaser = any(kw in cap for kw in _TEASER_KW)

            # Main figure boost: architecture/overview figures get the largest gp.
            # Early landscape heuristic is suppressed for teaser/motivation figures.
            if _is_main or (_is_early and figure_aspect >= 1.2 and not _is_teaser):
                gp = 0.40 if text_len > 0 else 0.35   # main/overview figure
                is_main_figure = True
            elif figure_aspect >= 1.5:
                gp = 0.30 if text_len > 0 else 0.25   # landscape but not flagged main
            else:
                gp = 0.25 if text_len > 0 else 0.20   # normal figure

    elif assets.get("tables") and tables:
        table_id = assets["tables"][0]
        table_data = None

        if str(table_id) in tables:
            table_data = tables[str(table_id)]
            asset_path = table_data.get('table_path')
        elif table_id in tables:
            table_data = tables[table_id]
            asset_path = table_data.get('table_path')

        if table_data:
            gp = 0.20 if text_len > 0 else 0.15
            figure_size = table_data.get('figure_size', 0)
            figure_aspect = table_data.get('figure_aspect', 1.0)
            is_table = True

    return tp, gp, text_len, asset_path, figure_size, figure_aspect, is_main_figure, is_table


# ---- Step 3-2: Content Node Construction ------------------------------------
def create_content_nodes_from_assets(content, assets, figures, tables, base_panel_id):
    """Create content nodes ensuring each has at most one asset"""
    content_nodes = []

    # Separate asset ID lists (distinct from `figures`/`tables` dicts passed as params)
    figure_ids = assets.get('figures', [])
    table_ids = assets.get('tables', [])
    all_assets = [(fig, 'figure') for fig in figure_ids] + \
        [(tbl, 'table') for tbl in table_ids]

    if not all_assets:
        # No assets - create single content node
        tp, gp, text_len, asset_path, figure_size, figure_aspect, is_main_figure, is_table = \
            calculate_poster_layout_parameters(content, {}, figures, tables)

        # 🔧 Generate concatenated numeric Panel ID
        if base_panel_id == "0":
            content_panel_id = "0"  # Keep Title as "0"
        else:
            # Append 1 to all content: "1" → "11", "2" → "21"
            content_panel_id = f"{base_panel_id}1"

        content_nodes.append({
            'panel_id': content_panel_id,
            'section_name': 'Content',
            'node_type': 'content',
            'tp': tp,
            'gp': gp,
            'text_len': text_len,
            'figure_size': figure_size,
            'figure_aspect': figure_aspect,
            'is_main_figure': is_main_figure,
            'is_table': is_table,
            'content': content,
            'assets': {'figures': [], 'tables': [], 'references': []},
            'asset_path': asset_path,
            'children': []
        })
    else:
        # Split content among assets
        content_parts = split_content_for_assets(content, len(all_assets))

        for i, ((asset_id, asset_type), content_part) in enumerate(zip(all_assets, content_parts), 1):
            asset_dict = {
                'figures': [asset_id] if asset_type == 'figure' else [],
                'tables': [asset_id] if asset_type == 'table' else [],
                'references': []
            }

            tp, gp, text_len, asset_path, figure_size, figure_aspect, is_main_figure, is_table = \
                calculate_poster_layout_parameters(content_part, asset_dict, figures, tables)

            # 🔧 Generate concatenated numeric Panel ID
            if base_panel_id == "0":
                # Multiple assets for Title: "01", "02"
                content_panel_id = f"0{i}"
            else:
                # General content: "11", "12" or "111", "112"
                content_panel_id = f"{base_panel_id}{i}"

            content_nodes.append({
                'panel_id': content_panel_id,
                'section_name': f'Content [{i}/{len(all_assets)}]' if len(all_assets) > 1 else 'Content',
                'node_type': 'content',
                'tp': tp,
                'gp': gp,
                'text_len': text_len,
                'figure_size': figure_size,
                'figure_aspect': figure_aspect,
                'is_main_figure': is_main_figure,
                'is_table': is_table,
                'content': content_part,
                'assets': asset_dict,
                'asset_path': asset_path,
                'children': []
            })

    return content_nodes


def split_content_evenly(content: str, num_parts: int) -> list[str]:
    """Split *content* into *num_parts* roughly equal sentence-level chunks."""
    if num_parts <= 1 or not content:
        return [content] * max(1, num_parts)
    sentences = [s.strip() for s in content.split('.') if s.strip()]
    if len(sentences) <= num_parts:
        parts = [sentences[i] + '.' if i < len(sentences) else '' for i in range(num_parts)]
        return parts
    spp = len(sentences) // num_parts
    parts = []
    for i in range(num_parts):
        sl = sentences[i * spp : (len(sentences) if i == num_parts - 1 else (i + 1) * spp)]
        parts.append('. '.join(sl) + '.' if sl else '')
    return parts


# Backward-compat aliases
split_content_for_assets = split_content_evenly


# ---- Step 3-3: Proportion Normalization -------------------------------------
def normalize_poster_tree_proportions(poster_tree):
    """Normalize tp and gp values so their sums equal 1.0 across all content nodes"""

    # Collect all content nodes
    content_nodes = []

    def collect_content_nodes(node):
        if node.get('node_type') == 'content':
            content_nodes.append(node)
        for child in node.get('children', []):
            collect_content_nodes(child)

    collect_content_nodes(poster_tree)

    if not content_nodes:
        return

    # Calculate current sums
    total_tp = sum(node.get('tp', 0) for node in content_nodes)
    total_gp = sum(node.get('gp', 0) for node in content_nodes)

    print(f"🔢 Normalizing proportions:")
    print(f"   📝 Total tp before normalization: {total_tp:.3f}")
    print(f"   🖼️ Total gp before normalization: {total_gp:.3f}")

    # Normalize tp values
    if total_tp > 0:
        for node in content_nodes:
            old_tp = node.get('tp', 0)
            new_tp = old_tp / total_tp
            node['tp'] = new_tp

    # Normalize gp values
    if total_gp > 0:
        for node in content_nodes:
            old_gp = node.get('gp', 0)
            new_gp = old_gp / total_gp
            node['gp'] = new_gp

    # Verify normalization
    new_total_tp = sum(node.get('tp', 0) for node in content_nodes)
    new_total_gp = sum(node.get('gp', 0) for node in content_nodes)

    print(f"   ✅ Total tp after normalization: {new_total_tp:.3f}")
    print(f"   ✅ Total gp after normalization: {new_total_gp:.3f}")

    return poster_tree


# ---- Step 3 [Entry]: Refined Tree → Poster Panel Tree -----------------------
# Calls: 3-4 create_poster_tree
#             └- 3-4 process_section
#                     └- 3-4 process_subsection
#                             └- 3-4 create_content_nodes
#        3-3 normalize_poster_tree_proportions
#        3-5 create_figures_mapping_from_poster_tree
def generate_panel_layout(args, agent_config, refined_tree, figures=None, tables=None):
    """
    Step 3: Generate poster tree from refined tree
    Creates proper poster panel structure with content nodes as leaves
    """
    print("🎯 Step 3: Converting refined tree to poster tree...")

    if figures is None:
        figures = {}
    if tables is None:
        tables = {}

    # Create poster tree
    poster_tree = create_poster_tree(refined_tree, figures, tables)

    # Normalize tp and gp proportions
    poster_tree = normalize_poster_tree_proportions(poster_tree)

    # Create figures mapping for compatibility
    figures = create_figures_mapping_from_poster_tree(poster_tree)

    print("✅ Poster tree generated successfully")

    # 📊 Step 3 complete: Print Poster Tree
    print(f"\n📋 STEP 3 COMPLETE: Poster Panel Tree Structure")
    print("="*80)
    from .tree_visualization import print_tree_unified
    print_tree_unified(poster_tree, tree_type="poster", show_details=True)
    print("="*80)
    total_panels = count_total_nodes(poster_tree)
    figures_mapped = len(figures)
    print(f"📊 Poster result: {total_panels} panels, {figures_mapped} figure mappings")

    return 0, 0, poster_tree, figures


# ---- Step 3-4: Poster Tree Assembly -----------------------------------------
def create_poster_tree(refined_tree, figures, tables):
    """
    Create poster tree following the structure:
    root - section - [subsection] - content (leaf)
    """
    print("🏗️  Creating poster tree structure...")

    from .step_02b_asset_management import asset_tracker
    asset_tracker.reset()

    # Root node
    poster_tree = {
        'panel_id': 'root',
        'section_name': refined_tree.get('section_name', 'Research Paper'),
        'node_type': 'root',
        'children': []
    }

    # Process title content (not abstract - that's handled as separate section)
    title_content = refined_tree.get('content', '').strip()
    if title_content:
        title_assets = refined_tree.get('assets', {})
        # Ensure title panels don't get any major assets (figures/tables)
        safe_title_assets = {
            'figures': [],  # No figures for title
            'tables': [],   # No tables for title
            # Keep references if any
            'references': title_assets.get('references', [])
        }
        content_nodes = create_content_nodes(
            title_content, safe_title_assets, figures, tables, 'title')
        poster_tree['children'].extend(content_nodes)

    # Process sections
    section_counter = 1
    for section in refined_tree.get('children', []):
        section_node = process_section(
            section, section_counter, figures, tables)
        if section_node:
            poster_tree['children'].append(section_node)
            section_counter += 1

    return poster_tree


def _min_fig_num_of_node(node):
    """Return minimum figure number (int) across a content or subsection node.

    Nodes without figures return 0 so they sort before any figure-bearing node,
    preserving text-intro panels at the front of their section.
    """
    if node.get('node_type') == 'content':
        figs = node.get('assets', {}).get('figures', [])
    else:
        figs = []
        for child in node.get('children', []):
            figs.extend(child.get('assets', {}).get('figures', []))
    nums = []
    for f in figs:
        try:
            nums.append(int(str(f)))
        except (ValueError, TypeError):
            pass
    return min(nums) if nums else 0


# ---- Section Node Processing ------------------------------------------------
def process_section(section, section_counter, figures, tables):
    """Process a section from refined tree into poster section"""
    section_name = section.get('section_name', f'Section {section_counter}')
    section_content = section.get('content', '').strip()
    section_assets = section.get('assets', {})
    section_children = section.get('children', [])

    # Create section node
    section_node = {
        'panel_id': str(section_counter),
        'section_name': section_name,
        'node_type': 'section',
        'children': []
    }

    # Add section's direct content if any
    # section_level_asset=True: assets directly under a section (not a subsection)
    # are treated as main figures to ensure they render larger
    if section_content or section_assets.get('figures') or section_assets.get('tables'):
        content_nodes = create_content_nodes(
            section_content, section_assets, figures, tables, str(section_counter),
            section_level_asset=True)
        section_node['children'].extend(content_nodes)

    # Process subsections if any
    if section_children:
        subsection_counter = 1
        for subsection in section_children:
            subsection_node = process_subsection(
                subsection, section_counter, subsection_counter, figures, tables)
            if subsection_node:
                section_node['children'].append(subsection_node)
                subsection_counter += 1

    # Sort ONLY section-level content nodes among themselves by figure number.
    # Subsection nodes are never reordered — their sequence reflects the paper's
    # narrative structure. Section-level content (with key=0 for text-only or
    # key=figure-number) is placed before all subsections by process_section's
    # design, so sorting them internally preserves figure order without risk of
    # swapping subsection positions.
    if len(section_node['children']) > 1:
        sec_content = [c for c in section_node['children'] if c.get('node_type') == 'content']
        subsec_nodes = [c for c in section_node['children'] if c.get('node_type') != 'content']
        sec_content.sort(key=_min_fig_num_of_node)
        section_node['children'] = sec_content + subsec_nodes

    return section_node if section_node['children'] else None


# ---- Subsection Node Processing ---------------------------------------------
def process_subsection(subsection, section_counter, subsection_counter, figures, tables):
    """Process a subsection from refined tree into poster subsection"""
    subsection_name = subsection.get(
        'section_name', f'Subsection {section_counter}.{subsection_counter}')
    subsection_content = subsection.get('content', '').strip()
    subsection_assets = subsection.get('assets', {})

    # Create subsection node
    subsection_node = {
        'panel_id': f"{section_counter}{subsection_counter}",
        'section_name': subsection_name,
        'node_type': 'subsection',
        'children': []
    }

    # Add subsection content
    if subsection_content or subsection_assets.get('figures') or subsection_assets.get('tables'):
        content_nodes = create_content_nodes(
            subsection_content, subsection_assets, figures, tables, f"{section_counter}{subsection_counter}")
        subsection_node['children'].extend(content_nodes)

    return subsection_node if subsection_node['children'] else None


# ---- Content Leaf Node Creation ---------------------------------------------
def create_content_nodes(content, assets, figures, tables, parent_id, section_level_asset=False):
    """
    Create content nodes from content and assets
    Each content node can have at most one asset
    section_level_asset: when True, figures directly under a section (not subsection) are
    promoted to is_main_figure so they render at a larger size on the poster.
    """
    content_nodes = []

    # Separate available assets (use distinct local names to avoid shadowing the dict params)
    asset_figure_ids = [f for f in assets.get('figures', []) if f in figures]
    asset_table_ids = [t for t in assets.get('tables', []) if t in tables]

    total_assets = len(asset_figure_ids) + len(asset_table_ids)

    if total_assets == 0:
        if content:
            content_node = create_single_content_node(
                content, {}, figures, tables, parent_id, 1)
            content_nodes.append(content_node)

    elif total_assets == 1:
        if asset_figure_ids:
            asset_dict = {'figures': [asset_figure_ids[0]], 'tables': [], 'references': []}
        else:
            asset_dict = {'figures': [], 'tables': [asset_table_ids[0]], 'references': []}

        content_node = create_single_content_node(
            content, asset_dict, figures, tables, parent_id, 1,
            section_level_asset=section_level_asset)
        content_nodes.append(content_node)

    else:
        all_assets = [(f, 'figure') for f in asset_figure_ids] + \
                     [(t, 'table') for t in asset_table_ids]
        content_parts = split_content_for_multiple_assets(content, len(all_assets))

        for i, ((asset_id, asset_type), content_part) in enumerate(zip(all_assets, content_parts), 1):
            if asset_type == 'figure':
                asset_dict = {'figures': [asset_id], 'tables': [], 'references': []}
            else:
                asset_dict = {'figures': [], 'tables': [asset_id], 'references': []}

            content_node = create_single_content_node(
                content_part, asset_dict, figures, tables, parent_id, i,
                section_level_asset=section_level_asset)
            content_nodes.append(content_node)

    return content_nodes


def create_single_content_node(content, assets, figures, tables, parent_id, content_index,
                                section_level_asset=False):
    """Create a single content node with layout parameters"""

    # Generate panel_id based on parent structure
    if parent_id == 'title':
        panel_id = 'title'
    elif parent_id == 'abstract':  # Keep for backward compatibility
        panel_id = 'abstract'
    else:
        # 🔧 Use concatenated numeric format: "1" → "11", "11" → "111", "12" → "121"
        panel_id = f"{parent_id}{content_index}"

    # Calculate layout parameters
    tp, gp, text_len, asset_path, figure_size, figure_aspect, is_main_figure, is_table = \
        calculate_poster_layout_parameters(content, assets, figures, tables)

    # Figures directly attached to a section (not a subsection) act as the section's
    # representative visual — promote them to main figure so they render larger.
    if section_level_asset and asset_path and not is_table and not is_main_figure:
        is_main_figure = True
        gp = 0.40 if text_len > 0 else 0.35

    # Create content node
    content_node = {
        'panel_id': panel_id,
        'section_name': f'Content [{content_index}]' if content_index > 1 else 'Content',
        'node_type': 'content',
        'tp': tp,
        'gp': gp,
        'text_len': text_len,
        'figure_size': figure_size,
        'figure_aspect': figure_aspect,
        'is_main_figure': is_main_figure,
        'is_table': is_table,
        'content': content,
        'assets': assets,
        'asset_path': asset_path,
        'children': []
    }

    return content_node


split_content_for_multiple_assets = split_content_evenly


# ---- Step 3-5: Figure Mapping -----------------------------------------------
def create_figures_mapping_from_poster_tree(poster_tree):
    """Create figures mapping from poster tree for compatibility"""
    figures = {}

    def extract_assets(node):
        if node.get('node_type') == 'content':
            assets = node.get('assets', {})
            section_name = node.get('section_name', '')

            for figure_id in assets.get('figures', []):
                figures[f"{section_name}_fig_{figure_id}"] = {
                    "image": figure_id,
                    "path": node.get('asset_path')
                }

            for table_id in assets.get('tables', []):
                figures[f"{section_name}_table_{table_id}"] = {
                    "table": table_id,
                    "path": node.get('asset_path')
                }

        for child in node.get('children', []):
            extract_assets(child)

    extract_assets(poster_tree)
    return figures
