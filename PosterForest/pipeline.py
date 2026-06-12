"""
PosterForest Pipeline — Steps 1-3 Facade.

PosterForest/
├-- pipeline.py                      # This file: Steps 1-3 integration facade
├-- step_01a_document_parsing.py     # Step 1a: PDF → hierarchical document tree + asset pre-assignment
├-- step_01b_asset_extraction.py     # Step 1b: figure/table extraction from docling output
├-- step_02a_tree_refinement.py      # Step 2a: tree refinement + content summarization
├-- step_02b_asset_management.py     # Step 2b: asset tracking, filtering, deduplication
├-- step_03_poster_layout.py         # Step 3: poster panel layout generation
├-- step_04_layout_hierarchy.py      # Step 4: binary-split layout → inch coordinates
├-- step_05_content_generation.py    # Step 5: LLM bullet-point content generation
├-- step_06_modification_planning.py # Step 6: VLM/LLM feedback loop + overflow detection (merged)
├-- step_07_pptx_generation.py       # Step 7: PPTX file generation
└-- tree_visualization.py            # Tree debug printing and JSON export
"""

# Import configuration and document converter setup
from dotenv import load_dotenv
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
from docling.document_converter import DocumentConverter, PdfFormatOption

from .poster_config import IMAGE_RESOLUTION_SCALE

# Import specialized modules
from .tree_visualization import (
    print_tree_unified, print_tree, print_refined_tree, print_poster_tree,
    save_tree_with_metadata
)

from .step_01a_document_parsing import (
    generate_raw_content_from_tree, parse_hierarchical_document
)

from .step_01b_asset_extraction import extract_figures_and_tables

from .step_02a_tree_refinement import (
    collect_content_for_summarization, batch_summarize_content, summarize_text_content
)

from .step_02b_asset_management import (
    AssetTracker, asset_tracker, filter_assets_for_poster,
    apply_asset_assignment_for_poster, ensure_all_assets_assigned
)

from .step_02a_tree_refinement import (
    refine_document_tree, calculate_layout_parameters, is_poster_suitable_section,
    create_poster_tree_3_level_with_summaries
)

from .step_03_poster_layout import (
    calculate_poster_layout_parameters, create_content_nodes_from_assets,
    split_content_for_assets, normalize_poster_tree_proportions,
    generate_panel_layout, create_poster_tree, process_section, process_subsection,
    create_content_nodes, create_single_content_node, split_content_for_multiple_assets,
    create_figures_mapping_from_poster_tree, count_total_nodes
)

# Configuration setup
load_dotenv()

pipeline_options = PdfPipelineOptions()
pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
pipeline_options.generate_page_images = True
pipeline_options.generate_picture_images = True
pipeline_options.accelerator_options = AcceleratorOptions(
    num_threads=4, device=AcceleratorDevice.CPU
)

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

# Export main functions - organized by functionality
__all__ = [
    # === MAIN PIPELINE FUNCTIONS (Step 1-3) ===
    # Step 1: Document parsing and asset extraction
    'parse_hierarchical_document',           # Core parsing function
    'extract_figures_and_tables',            # Asset extraction
    'generate_raw_content_from_tree',       # Backward compatibility

    # Step 2: Tree refinement
    'refine_document_tree',                 # Main refinement function
    'filter_assets_for_poster',             # Asset filtering

    # Step 3: Panel layout generation
    'generate_panel_layout',                # Main layout function
    'create_poster_tree',                   # Poster tree creation

    # === UTILITY FUNCTIONS ===
    # Tree visualization and validation
    'print_tree_unified', 'print_tree', 'print_refined_tree', 'print_poster_tree',
    'save_tree_with_metadata',

    # Content processing
    'collect_content_for_summarization', 'batch_summarize_content', 'summarize_text_content',

    # Asset management
    'AssetTracker', 'asset_tracker', 'apply_asset_assignment_for_poster', 'ensure_all_assets_assigned',

    # Tree processing utilities
    'calculate_layout_parameters', 'is_poster_suitable_section',
    'create_poster_tree_3_level_with_summaries',

    # Layout utilities
    'calculate_poster_layout_parameters', 'create_content_nodes_from_assets',
    'split_content_for_assets', 'normalize_poster_tree_proportions',
    'process_section', 'process_subsection', 'create_content_nodes',
    'create_single_content_node', 'split_content_for_multiple_assets',
    'create_figures_mapping_from_poster_tree', 'count_total_nodes',

    # === CONFIGURATION ===
    'doc_converter', 'IMAGE_RESOLUTION_SCALE'
]
