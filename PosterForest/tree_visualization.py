"""
Tree Visualization and Storage Utilities

Shared across all pipeline stages.

Input:  Any tree structure (document / refined / poster)
Output: Console-formatted tree output; JSON-serialized metadata files
"""

import json
import os
from datetime import datetime


# ---- Unified Tree Printer ---------------------------------------------------
def print_tree_unified(node, tree_type="document", prefix="", is_last=True, max_content_length=100, show_details=True, figures=None, tables=None):
    """Unified tree printer for document, refined, and poster tree types."""
    icons = {
        "document": {"section": "📂", "content": "📝", "default": "📄"},
        "refined": {"section": "🗂️", "subsection": "📁", "content": "📝", "default": "📋"},
        "poster": {"title": "📄", "section": "📁", "content": "📝", "default": "📐"}
    }
    
    current_prefix = prefix + ("└-- " if is_last else "├-- ")
    next_prefix = prefix + ("    " if is_last else "│   ")

    title = node.get('title', node.get('section_name', 'Unnamed'))
    node_type = node.get('node_type', 'unknown')
    panel_id = node.get('panel_id')
    content = node.get('content', '')
    children = node.get('children', [])
    section_id = node.get('id', '')

    icon_set = icons.get(tree_type, icons["document"])
    if tree_type == "poster" and panel_id == "0":
        icon = icon_set["title"]
    elif node_type in icon_set:
        icon = icon_set[node_type]
    elif children:
        icon = icon_set.get("section", icon_set["default"])
    else:
        icon = icon_set.get("content", icon_set["default"])

    if tree_type == "document" and section_id:
        node_info = f"[{section_id}] {title}"
    elif tree_type == "poster":
        # For poster tree, display cleanly with panel_id and title only
        if panel_id is not None:
            if node_type == "root" or panel_id == "0":
                node_info = f"{title}"  # Root shows title only
            elif node_type == "content":
                # Content node: display in compact form
                tp = node.get('tp', 0)
                gp = node.get('gp', 0)
                figure_size = node.get('figure_size', 0)
                node_info = f"{title} (tp={tp:.3f}, gp={gp:.3f}, fig_size={figure_size})"
            else:
                # Section, subsection nodes: title only
                node_info = f"{title}"
        else:
            node_info = f"{title}"
    else:
        node_info = f"{title}"

    if panel_id is not None and tree_type != "poster":
        node_info = f"Panel {panel_id}: {node_info}"

    if tree_type == "poster" and panel_id is not None:
        indexed_node_info = f"[{panel_id}] {node_info}"
    else:
        indexed_node_info = node_info
    
    print(f"{current_prefix}{icon} {indexed_node_info}")
    
    if show_details:
        detail_prefix = next_prefix + "    "

        if content.strip():
            word_count = len(content.split())
            preview_length = 50 if tree_type == "poster" else 40
            content_preview = content.replace('\n', ' ')[:preview_length]
            if len(content_preview) == preview_length:
                content_preview += "..."
            print(f"{detail_prefix}💬 Text: {word_count} words - {content_preview}")

        assets = node.get('assets', {})

        if assets.get('figures'):
            for fig_id in assets['figures']:
                if tree_type == "document" and figures and fig_id in figures:
                    caption = figures[fig_id].get('caption', 'No caption')
                    words = caption.split()
                    caption_preview = (' '.join(words[:8]) + '...') if len(words) > 8 else caption
                    if len(caption_preview) > 50:
                        caption_preview = caption_preview[:47] + '...'
                    print(f"{detail_prefix}🖼️  Figure {fig_id}: {caption_preview}")
                else:
                    print(f"{detail_prefix}🖼️  Figure: {fig_id}")

        if assets.get('tables'):
            for table_id in assets['tables']:
                if tree_type == "document" and tables and table_id in tables:
                    caption = tables[table_id].get('caption', 'No caption')
                    words = caption.split()
                    caption_preview = (' '.join(words[:8]) + '...') if len(words) > 8 else caption
                    if len(caption_preview) > 50:
                        caption_preview = caption_preview[:47] + '...'
                    print(f"{detail_prefix}📊 Table {table_id}: {caption_preview}")
                else:
                    print(f"{detail_prefix}📊 Table: {table_id}")

        if tree_type == "poster":
            asset_path = node.get('asset_path')
            if asset_path:
                print(f"{detail_prefix}📎 Asset Path: {asset_path}")

        if assets.get('references'):
            print(f"{detail_prefix}🔗 References: {len(assets['references'])} items")

    for i, child in enumerate(children):
        is_child_last = (i == len(children) - 1)
        print_tree_unified(child, tree_type, next_prefix, is_child_last, max_content_length, show_details, figures, tables)


# ---- Backward-compatible Aliases --------------------------------------------
def print_tree(node, prefix="", is_last=True, max_content_length=100, figures=None, tables=None):
    """Backward-compatible alias for document tree printing."""
    print_tree_unified(node, "document", prefix, is_last, max_content_length, show_details=True, figures=figures, tables=tables)


def print_refined_tree(node, prefix="", is_last=True, max_content_length=100):
    """Backward-compatible alias for refined tree printing."""
    print_tree_unified(node, "refined", prefix, is_last, max_content_length, show_details=True)


def print_poster_tree(node, prefix="", is_last=True, max_content_length=50):
    """Backward-compatible alias for poster tree printing."""
    print_tree_unified(node, "poster", prefix, is_last, max_content_length, show_details=True)


# ---- Tree Metadata Serialization --------------------------------------------
def save_tree_with_metadata(tree, step_dir, filename, tree_type="document", **metadata):
    """Save a tree dict to step_dir/filename with basic metadata. Returns filename."""
    node_count = len(str(tree).split('"children"')) if tree else 0
    save_data = {
        'tree_type': tree_type,
        'timestamp': datetime.now().isoformat(),
        'tree_data': tree,
        'basic_stats': {'estimated_nodes': node_count},
        'metadata': metadata,
    }
    filepath = os.path.join(step_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=4, ensure_ascii=False)
    return filename