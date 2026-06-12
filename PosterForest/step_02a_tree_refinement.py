"""
STEP 2: Document Tree Refinement for Poster Optimization

Restructures the raw document tree from Step 1 into a depth-2 poster tree
(section → subsection), filters unsuitable sections, summarizes content via LLM,
and assigns figures/tables to the most relevant panels.

Input:  Document tree + figure/table dicts from Step 1
Output: Poster-optimized refined tree with asset assignments
"""

import re
from jinja2 import Template
from camel.agents import ChatAgent
from utils.wei_utils import account_token, make_model, agent_step_with_timeout
from utils.src.utils import get_json_from_response

from .step_02b_asset_management import (
    asset_tracker, filter_assets_for_poster, apply_asset_assignment_for_poster,
    ensure_all_assets_assigned
)

_CAPTION_PREFIX_RE = re.compile(
    r'^(?:figure|table|fig\.?)\s*\d+[\.:]\s*', re.IGNORECASE
)


def _clean_section_title(title: str) -> str:
    """Strip figure/table caption prefixes from section titles.

    e.g. 'Table 3: Variations on the Transformer' → 'Variations on the Transformer'
    """
    if not title:
        return title
    cleaned = _CAPTION_PREFIX_RE.sub('', title).strip()
    return cleaned if cleaned else title


# ---- Step 2-2: Content Collection for Summarization -------------------------
def collect_content_for_summarization(node, content_list, path=""):
    """Collect all content that needs summarization for batch processing."""
    node_id = node.get('id', 'unknown')
    current_path = f"{path}/{node_id}" if path else node_id

    content = node.get('content', '').strip()
    if content and len(content) > 100:
        content_list.append({
            'id': node_id,
            'path': current_path,
            'title': node.get('title', ''),
            'type': node.get('type', ''),
            'content': content
        })

    for child in node.get('children', []):
        collect_content_for_summarization(child, content_list, current_path)


# ---- Step 2-3: LLM Batch Summarization --------------------------------------
def batch_summarize_content(content_list, args, agent_config):
    """Batch summarize content using LLM."""
    if not content_list:
        return 0, 0, {}

    print(f"\n🤖 Starting LLM-based content summarization (batch processing)...")
    print(f"📊 Total content sections to summarize: {len(content_list)}")

    summarization_template = Template("""
You are an expert at summarizing academic content for poster presentations.

Your task is to summarize multiple sections of an academic paper into concise, poster-friendly content. Each section should be summarized to 4-7 sentences that capture the key information while maintaining academic rigor.

## Summarization Guidelines:
- **Length**: Each summary MUST be 4-7 sentences and at least 60 words. Do NOT produce one-sentence or two-sentence summaries.
- **Content**: Preserve key technical details, findings, metrics, and insights
- **Clarity**: Use clear, precise language suitable for poster display
- **Completeness**: Ensure main points, methods, and results are not lost in summarization
- **Context**: Maintain enough context for standalone understanding

## Content to Summarize:
{% for item in content_batch %}

**Section {{ loop.index }}: {{ item.title }}** ({{ item.type }})
Path: {{ item.path }}
Content:
{{ item.content }}

---
{% endfor %}

## Output Format:
Provide the summaries in JSON format with the exact section IDs:

```json
{
  "summaries": {
{% for item in content_batch %}
    "{{ item.id }}": "Your 4-7 sentence summary for {{ item.title }}..."{% if not loop.last %},{% endif %}
{% endfor %}
  }
}
```

Generate concise, poster-appropriate summaries for each section:
""")

    print(f"🔧 Setting up LLM model: {agent_config['model_type']}")
    model = make_model(agent_config)

    agent = ChatAgent(
        system_message="You are an expert academic content summarizer for poster presentations.",
        model=model,
        message_window_size=5
    )

    total_chars = sum(len(item['content']) for item in content_list)
    # vllm_qwen: max_model_len=16384 → small batches to stay within context.
    # For cloud/GPT models: single batch when small, 5-section batches when large.
    if args.model_name_t.startswith('vllm_qwen'):
        batch_size = 3
        print(f"📄 vllm_qwen model — using small batch size ({batch_size}) to fit context window")
    elif total_chars < 50000:
        batch_size = len(content_list)
        print(f"📄 Small document ({total_chars:,} chars) - processing all sections in single batch")
    else:
        batch_size = 5
        print(f"📚 Large document ({total_chars:,} chars) - using batch processing (size: {batch_size})")

    total_input_tokens = 0
    total_output_tokens = 0
    all_summaries = {}

    for i in range(0, len(content_list), batch_size):
        batch = content_list[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(content_list) + batch_size - 1) // batch_size

        print(f"\n📤 Processing batch {batch_num}/{total_batches} ({len(batch)} sections):")
        for item in batch:
            print(f"   📝 {item['id']}: {item['title']} ({len(item['content'])} chars)")

        prompt = summarization_template.render(content_batch=batch)

        print(f"🚀 Sending API request for batch {batch_num}...")
        agent.reset()
        response = agent_step_with_timeout(agent, prompt)

        input_token, output_token = account_token(response)
        total_input_tokens += input_token
        total_output_tokens += output_token

        print(f"✅ Batch {batch_num} done — tokens: {input_token} → {output_token}")

        try:
            batch_result = get_json_from_response(response.msgs[0].content)
            if 'summaries' in batch_result:
                all_summaries.update(batch_result['summaries'])
                print(f"   ✅ Summarized {len(batch_result['summaries'])} sections")
        except Exception as e:
            print(f"   ❌ Error parsing batch {batch_num}: {e}")
            for item in batch:
                all_summaries[item['id']] = summarize_text_content(item['content'], target_sentences=4)

    print(f"\n🎯 Summarization complete: {len(all_summaries)}/{len(content_list)} sections, "
          f"tokens: {total_input_tokens} → {total_output_tokens}")
    return total_input_tokens, total_output_tokens, all_summaries


# ---- Step 2-3 Fallback: Local Sentence Extraction ---------------------------
def summarize_text_content(content, target_sentences=5):
    """Fallback: extract key sentences without LLM."""
    if not content or not content.strip():
        return ""

    sentences = re.split(r'[.!?]+', content)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= target_sentences:
        return content.strip()

    if target_sentences == 3:
        indices = [0, len(sentences)//2, len(sentences)-1]
    elif target_sentences == 4:
        indices = [0, len(sentences)//3, 2*len(sentences)//3, len(sentences)-1]
    elif target_sentences == 5:
        indices = [0, len(sentences)//4, len(sentences)//2, 3*len(sentences)//4, len(sentences)-1]
    else:
        indices = [0, len(sentences)//5, 2*len(sentences)//5, 3*len(sentences)//5, 4*len(sentences)//5, len(sentences)-1]

    indices = sorted(set(indices))
    return '. '.join(sentences[i] for i in indices if i < len(sentences)) + '.'


# ---- Step 2 [Entry]: Document Tree → Refined Poster Tree --------------------
# Calls: 2-1 filter_assets_for_poster         (step_02b_asset_management.py)
#        2-2 collect_content_for_summarization
#        2-3 batch_summarize_content
#        2-4 create_poster_tree_3_level_with_summaries
#             └- reassign_assets_by_caption_relevance
#             └- apply_asset_assignment_for_poster    (step_02b_asset_management.py)
#             └- ensure_all_assets_assigned           (step_02b_asset_management.py)
def refine_document_tree(args, agent_config, document_tree, figures, tables, text_content=None):
    """
    Optimize the document tree for poster layout
    - Adjust tree structure to satisfy the depth-2 constraint
    - Merge and remove low-importance sections

    Args:
        args: Command-line arguments
        agent_config: Agent configuration
        document_tree: Original document tree
        figures: Figure dictionary
        tables: Table dictionary
        text_content: Raw markdown text from Step 1 (pre-LLM-truncation). Used by
            reassign_assets_by_caption_relevance as the primary source for "Figure N"
            citation detection.  Optional — falls back to LLM-summarised content if None.

    Returns:
        tuple: (input_tokens, output_tokens, refined_tree)
    """
    print("🎯 Pre-filtering assets for poster suitability...")
    filtered_figures, filtered_tables = filter_assets_for_poster(figures, tables)

    print(f"📊 Asset filtering results:")
    print(f"   🖼️  Figures: {len(figures)} → {len(filtered_figures)} (filtered)")
    print(f"   📊 Tables: {len(tables)} → {len(filtered_tables)} (filtered)")

    print("🛠️  Creating refined 3-level poster tree with LLM summarization...")

    # Collect all content that needs summarization for batch processing
    content_to_summarize = []
    collect_content_for_summarization(document_tree, content_to_summarize)

    # Batch summarize content using LLM
    input_tokens, output_tokens, summarized_content = batch_summarize_content(
        content_to_summarize, args, agent_config
    )

    # Create refined tree with summarized content
    if getattr(args, 'ablation_flat_tree', False):
        print("🔄 ABLATION MODE: Using flat tree structure (no hierarchy)")
        refined_tree = create_flat_tree_with_summaries(
            document_tree, filtered_figures, filtered_tables, summarized_content, args, agent_config
        )
    else:
        refined_tree = create_poster_tree_3_level_with_summaries(
            document_tree, filtered_figures, filtered_tables, summarized_content, args, agent_config,
            text_content=text_content
        )

    print("✅ 3-level poster tree created successfully with LLM summarization")

    # 📊 Step 2 complete: Print Refined Tree
    print(f"\n📋 STEP 2 COMPLETE: Refined Tree Structure")
    print("="*80)
    from .tree_visualization import print_tree_unified
    print_tree_unified(refined_tree, tree_type="refined", show_details=True)
    print("="*80)

    # Refinement result statistics
    main_sections = len(refined_tree.get('children', []))
    total_subsections = sum(len(child.get('children', [])) for child in refined_tree.get('children', []))
    print(f"📊 Refinement result: {main_sections} main sections, {total_subsections} subsections")

    return input_tokens, output_tokens, refined_tree


# ---- Step 2-1: Asset and Section Filtering Helpers --------------------------
def calculate_layout_parameters(content, assets, figures, tables):
    """
    Calculate basic parameters for refined tree (Step 2)
    Step 3 will calculate tp/gp values for poster tree
    Returns: text_len, asset_path
    """
    text_len = len(content) if content else 0
    asset_path = None

    # Find asset path for reference
    if assets.get("figures") and figures:
        figure_id = assets["figures"][0]
        if figure_id in figures:
            asset_path = figures[figure_id].get('figure_path')
    elif assets.get("tables") and tables:
        table_id = assets["tables"][0]
        if table_id in tables:
            asset_path = tables[table_id].get('table_path')

    return text_len, asset_path


def extract_paper_keywords(document_tree):
    """
    Extract key technical terms from paper title and abstract for dynamic section filtering
    """
    keywords = []

    # Extract from paper title
    title = document_tree.get("title", "")
    if title:
        # Remove common words and extract meaningful terms
        title_words = title.lower().replace(":", "").split()
        # Filter out common words
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'what', 'matters', 'emergent'}
        meaningful_words = [word for word in title_words if len(word) > 3 and word not in stopwords]
        keywords.extend(meaningful_words)

    # Extract from abstract (first child is usually abstract)
    children = document_tree.get("children", [])
    if children:
        abstract = children[0]  # First section is typically abstract
        abstract_content = abstract.get("content", "")
        if abstract_content:
            # Extract technical terms (capitalized words, acronyms, compound terms)
            import re
            # Find technical terms like "3DGM", "RGB", etc.
            technical_terms = re.findall(r'\b[A-Z]{2,}\w*\b|\b\d+D\w*\b', abstract_content)
            # Find compound technical terms
            compound_terms = re.findall(r'\b[a-z]+\s+[a-z]+ing\b|\b[a-z]+\s+mapping\b|\b[a-z]+\s+splatting\b', abstract_content.lower())

            keywords.extend([term.lower() for term in technical_terms])
            keywords.extend([term.replace(' ', '') for term in compound_terms])

    # Remove duplicates and return
    unique_keywords = list(set(keywords))
    print(f"   🔑 Extracted paper keywords: {unique_keywords}")
    return unique_keywords


def calculate_asset_importance(asset_id, asset_type, filtered_figures, filtered_tables, context_content="", section_name=""):
    """
    Calculate importance score for an asset based on various factors
    Higher score = more important
    Used for asset limiting (when too many assets exist in a section)

    Priority: Method assets > Experiment tables > Other assets
    """
    score = 0

    # Get asset data for caption analysis
    caption = ""
    if asset_type == 'figure' and asset_id in filtered_figures:
        figure_data = filtered_figures[asset_id]
        caption = figure_data.get('caption', '').lower()
    elif asset_type == 'table' and asset_id in filtered_tables:
        table_data = filtered_tables[asset_id]
        caption = table_data.get('caption', '').lower()

    # 🎯 HIGHEST PRIORITY: Method-related assets (MUST NOT be removed)
    method_keywords = [
        'method', 'approach', 'algorithm', 'technique', 'strategy',
        'architecture', 'framework', 'model', 'system', 'design', 
        'pipeline', 'workflow', 'process', 'procedure', 'scheme',
        'structure', 'network', 'component', 'module', 'flow',
        'overview', 'diagram', 'illustration', 'schematic'
    ]

    section_lower = section_name.lower()
    is_method_section = any(keyword in section_lower for keyword in [
        'method', 'approach', 'algorithm', 'technique', 'model', 'architecture',
        'framework', 'design', 'implementation', 'proposed'
    ])

    # Method asset detection
    method_score = 0
    for keyword in method_keywords:
        if keyword in caption:
            method_score += 10  # Very high score for method assets

    if method_score > 0 or is_method_section:
        score += 100  # CRITICAL: Method assets get maximum priority
        print(f"      🎯 CRITICAL METHOD ASSET: {asset_type} {asset_id} (score: +100)")

    # 🔬 HIGH PRIORITY: Experiment tables and result figures
    experiment_keywords = [
        'result', 'performance', 'comparison', 'evaluation', 'experiment',
        'analysis', 'finding', 'benchmark', 'accuracy', 'error',
        'precision', 'recall', 'f1', 'score', 'metric', 'test'
    ]

    is_experiment_section = any(keyword in section_lower for keyword in [
        'experiment', 'result', 'evaluation', 'analysis', 'finding',
        'performance', 'comparison', 'benchmark'
    ])

    experiment_score = 0
    for keyword in experiment_keywords:
        if keyword in caption:
            experiment_score += 5

    if (experiment_score > 0 or is_experiment_section) and asset_type == 'table':
        score += 50  # Tables in experiment sections are crucial
        print(f"      🔬 CRITICAL EXPERIMENT TABLE: {asset_id} (score: +50)")
    elif experiment_score > 0:
        score += 25  # Other experiment assets
        print(f"      📊 Important experiment asset: {asset_type} {asset_id} (score: +25)")

    # Standard scoring for non-critical assets (figures and tables share the same fields)
    asset_data = None
    if asset_type == 'figure' and asset_id in filtered_figures:
        asset_data = filtered_figures[asset_id]
    elif asset_type == 'table' and asset_id in filtered_tables:
        asset_data = filtered_tables[asset_id]
        score += 2  # tables are generally important for results

    if asset_data is not None:
        # Factor 1: Asset size (larger assets are often more important)
        size = asset_data.get('figure_size', 0)
        if size > 500000:
            score += 3
        elif size > 200000:
            score += 2
        else:
            score += 1

        # Factor 2: Caption length (more detailed captions suggest importance)
        if len(caption) > 100:
            score += 2
        elif len(caption) > 50:
            score += 1

        # Factor 3: Aspect ratio (some ratios are better for posters)
        aspect = asset_data.get('figure_aspect', 1.0)
        if 1.2 <= aspect <= 2.5:
            score += 1

    # Factor 4: Content relevance (if context is provided)
    if context_content and caption:
        content_words = set(context_content.lower().split())
        caption_words = set(caption.split())

        # Keyword overlap bonus
        overlap = len(content_words & caption_words)
        if overlap > 3:
            score += 2
        elif overlap > 1:
            score += 1

    return score


def _is_experiment_asset_caption(caption):
    """Return True when the figure/table caption very specifically suggests an experiment result.
    Intentionally narrow: 'challenges' and motivation figures should stay in Introduction."""
    caption_l = caption.lower()
    # Only flag when caption contains explicit result metrics or ground-truth evidence
    strong_signals = [
        'ground-truth', 'ground truth', 'ablation study', 'ablation table',
        'miou', 'map', 'f1 score', 'top-1', 'top-5', 'bleu', 'fid score',
    ]
    return any(kw in caption_l for kw in strong_signals)


def limit_section_assets(section_panel, filtered_figures, filtered_tables, max_assets=4):
    """
    Limit total assets in a section to max_assets by removing least important ones

    Note: All sections have the same 4-asset limit, but method/experiment assets
    get priority through importance scoring (method: +100, experiment tables: +50)
    Special rule: Introduction-like sections get at most 1 asset (teaser figure only).
    """
    section_name = section_panel.get('section_name', '')
    section_lower = section_name.lower()

    # Introduction should have at most 2 figures (teaser + key comparison).
    # Experiment/qualitative result figures must not appear in Introduction.
    _INTRO_LIKE = {'introduction', 'intro', 'motivation', 'overview', 'abstract'}
    if any(kw in section_lower for kw in _INTRO_LIKE):
        max_assets = 2

    # 🎯 SECTION TYPE DETECTION: For logging and context
    is_method_section = any(keyword in section_lower for keyword in [
        'method', 'approach', 'algorithm', 'technique', 'model', 'architecture',
        'framework', 'design', 'implementation', 'proposed'
    ])

    is_experiment_section = any(keyword in section_lower for keyword in [
        'experiment', 'result', 'evaluation', 'analysis', 'finding',
        'performance', 'comparison', 'benchmark'
    ])

    # Log section type for transparency
    if is_method_section:
        print(f"   🎯 METHOD SECTION: '{section_name}' - critical assets get +100 priority")
    elif is_experiment_section:
        print(f"   🔬 EXPERIMENT SECTION: '{section_name}' - tables get +50 priority")

    # All sections use the same limit, but importance scoring ensures critical assets survive

    # Collect all assets from subsections
    all_assets = []

    def collect_assets_from_node(node, parent_content=""):
        node_assets = node.get('assets', {})
        node_content = node.get('content', '')
        context = parent_content + " " + node_content

        # Add figures
        for fig_id in node_assets.get('figures', []):
            importance = calculate_asset_importance(fig_id, 'figure', filtered_figures, filtered_tables, context, section_name)
            all_assets.append({
                'id': fig_id,
                'type': 'figure',
                'importance': importance,
                'node': node,
                'asset_key': 'figures'
            })

        # Add tables  
        for table_id in node_assets.get('tables', []):
            importance = calculate_asset_importance(table_id, 'table', filtered_figures, filtered_tables, context, section_name)
            all_assets.append({
                'id': table_id,
                'type': 'table', 
                'importance': importance,
                'node': node,
                'asset_key': 'tables'
            })

        # Recursively collect from children
        for child in node.get('children', []):
            collect_assets_from_node(child, context)

    # Collect all assets in this section
    collect_assets_from_node(section_panel)

    # If within limit, no need to filter
    if len(all_assets) <= max_assets:
        print(f"   ✅ Section '{section_name}': {len(all_assets)} assets (within {max_assets} limit)")
        return

    # Sort by importance (descending) and keep only top max_assets assets
    # Critical assets (method: +100, experiment tables: +50) will be at the top
    all_assets.sort(key=lambda x: x['importance'], reverse=True)
    assets_to_keep = all_assets[:max_assets]
    assets_to_remove = all_assets[max_assets:]

    print(f"   🔄 Section '{section_name}': Limiting {len(all_assets)} → {max_assets} assets (keeping highest priority)")

    # Remove less important assets
    for asset_info in assets_to_remove:
        node = asset_info['node']
        asset_list = node['assets'][asset_info['asset_key']]
        if asset_info['id'] in asset_list:
            asset_list.remove(asset_info['id'])
            print(f"      🗑️ Removed {asset_info['type']} {asset_info['id']} (importance: {asset_info['importance']})")

    # Log kept assets
    for asset_info in assets_to_keep:
        print(f"      ✅ Kept {asset_info['type']} {asset_info['id']} (importance: {asset_info['importance']})")


# ---- Section Classification -------------------------------------------------
# Unified replacement for the old trio:
#   is_poster_unsuitable_section  +  is_poster_suitable_section  +  is_method_related_section
#
# Returns ('exclude', reason) or ('method', reason) or ('include', reason).
# Callers choose what to do with each tier; the decision logic stays here.

_SECTION_ALWAYS_EXCLUDE = {
    # Hard excludes — never appear on a poster regardless of assets
    'exclude': [
        # Post-matter
        'reference', 'bibliography', 'citation', 'work cited', 'literature cited',
        'acknowledgment', 'acknowledgement', 'thank', 'funding', 'grant',
        'appendix', 'appendices', 'supplement', 'additional detail', 'supplementary material',
        # Admin
        'author contribution', 'conflict of interest', 'ethics statement',
        'data availability', 'code availability', 'impact statement',
        'author', 'affiliation', 'correspondence',
        # Technical boilerplate
        'implementation detail', 'technical detail', 'code detail',
        'derivation', 'proof', 'mathematical detail',
    ],
    # Related-work / background: include only when assets are present (decided by caller)
    'related': [
        'related work', 'literature review', 'prior work', 'previous work',
        'background', 'survey',
    ],
    # Method: always include
    'method': [
        'method', 'approach', 'algorithm', 'model', 'architecture', 'framework',
        'design', 'technique', 'pipeline', 'workflow', 'network', 'system',
        'proposed', 'our model', 'our approach',
    ],
    # Standard poster content
    'include': [
        'introduction', 'intro', 'motivation', 'problem', 'contribution',
        'experiment', 'result', 'evaluation', 'analysis', 'finding',
        'conclusion', 'summary', 'discussion',
        'overview', 'dataset', 'data', 'setup',
        'comparison', 'baseline', 'benchmark', 'ablation', 'performance',
    ],
}


def classify_section(title: str, content: str = '',
                     paper_keywords=None) -> tuple[str, str]:
    """Classify a section for poster inclusion.

    Returns (decision, reason) where decision is one of:
      'exclude'  – never include (references, appendix, admin, …)
      'related'  – related-work / background: include only if assets present
      'method'   – high-priority method section: always include
      'include'  – standard poster content
    """
    tl = title.lower().strip()

    # Hard excludes (exact 'abstract' match or keyword in title)
    if tl == 'abstract':
        return 'exclude', 'abstract'
    # Short content that reads like an abstract (even if not titled 'abstract')
    if content and len(content.split()) < 100:
        cl = content.lower()
        abs_signals = sum(1 for p in ('we present', 'we propose', 'this paper',
                                      'in this work', 'we introduce') if p in cl)
        if abs_signals >= 3 and 'intro' not in tl:
            return 'exclude', 'abstract-like'

    for kw in _SECTION_ALWAYS_EXCLUDE['exclude']:
        if kw in tl:
            return 'exclude', kw

    for kw in _SECTION_ALWAYS_EXCLUDE['related']:
        if kw in tl:
            return 'related', kw

    # High-citation prose → likely a related-work/survey section even without the title keyword
    if content:
        cit = len(re.findall(r'\[\d+\]', content))
        words = len(content.split())
        if words > 0 and cit / words > 0.15:
            return 'related', 'high-citation-density'

    for kw in _SECTION_ALWAYS_EXCLUDE['method']:
        if kw in tl:
            return 'method', kw

    for kw in _SECTION_ALWAYS_EXCLUDE['include']:
        if kw in tl:
            return 'include', kw

    if paper_keywords:
        for kw in paper_keywords:
            if kw.lower() in tl:
                return 'include', f'paper-keyword:{kw}'

    if content and len(content.strip()) > 20:
        return 'include', 'has-content'
    return 'exclude', 'no-content'


# Keep thin compatibility shims so callers in create_flat_tree_with_summaries
# and create_poster_tree_3_level_with_summaries can migrate incrementally.

def is_poster_unsuitable_section(title, content=""):
    decision, reason = classify_section(title, content)
    if decision == 'exclude':
        return True, reason
    return False, None


def is_method_related_section(title, content="", position=None, total=None):
    decision, _ = classify_section(title, content)
    return decision == 'method'


def is_poster_suitable_section(title, content, section_id="",
                                paper_keywords=None, position=None, total=None):
    decision, reason = classify_section(title, content, paper_keywords)
    if decision == 'exclude':
        print(f"   🚫 Excluding '{title}': {reason}")
        return False
    if decision in ('method', 'include'):
        print(f"   ✅ Including '{title}': {reason}")
        return True
    # 'related' → caller decides based on assets; return False here (conservative)
    return False


# ---- Step 2-5 (Ablation): Flat Tree Construction ----------------------------
def create_flat_tree_with_summaries(document_tree, filtered_figures, filtered_tables, summaries, args, agent_config):
    """
    Create flat poster tree structure for ablation study (no section-subsection hierarchy)
    All sections and subsections are promoted to the same level
    """
    print("🛠️  Creating FLAT poster tree structure (ablation: no hierarchy)...")

    # Reset global asset tracking and enforce uniqueness constraint for posters
    asset_tracker.reset()
    print(f"   🔒 Asset uniqueness constraint: Each asset can only be assigned once")

    # Title panel (root level)
    title_content = document_tree.get("content", "")
    title_id = document_tree.get("id", "root")

    # Use LLM summary if available, otherwise use original or fallback summarization
    if title_id in summaries:
        title_summary = summaries[title_id]
        print(f"   📝 Using LLM summary for title")
    else:
        title_summary = summarize_text_content(title_content, target_sentences=4)
        print(f"   📝 Using fallback summary for title")

    text_len, asset_path = calculate_layout_parameters(
        title_summary, {"figures": [], "tables": []}, filtered_figures, filtered_tables
    )

    refined_tree = {
        "panel_id": "0",
        "section_name": document_tree.get("title", "Title"),
        "tp": 0.08,  # Fixed for title
        "text_len": text_len,
        "gp": 0,  # No graphics in title
        "figure_size": 0,
        "figure_aspect": 1,
        "content": title_summary,
        "importance": 1.0,  # Default importance for title
        "assets": {"figures": [], "tables": [], "references": []},
        "asset_path": asset_path,
        "children": []
    }

    # Extract paper-specific keywords for dynamic section filtering
    paper_keywords = extract_paper_keywords(document_tree)

    # Collect all sections and subsections into a flat list
    print(f"\n🔗 Flattening hierarchy: collecting all sections and subsections...")
    flat_sections = []
    panel_id_counter = 1

    original_children = document_tree.get("children", [])
    total_sections = len(original_children)

    for i, child in enumerate(original_children):
        section_title = _clean_section_title(child.get("title", f"Section {i + 1}"))
        section_content = child.get("content", "")
        section_id = child.get("id", str(i + 1))
        section_assets = child.get("assets", {"figures": [], "tables": [], "references": []})
        section_position = i + 1

        has_significant_assets = (len(section_assets.get('figures', [])) > 0 or
                                   len(section_assets.get('tables', [])) > 0)
        decision, reason = classify_section(section_title, section_content, paper_keywords)
        if decision == 'exclude':
            print(f"   🚫 Excluding '{section_title}': {reason}")
            continue
        if decision == 'related' and not has_significant_assets:
            print(f"   🚫 Excluding related-work '{section_title}': no assets")
            continue

        if decision in ('method', 'include') or has_significant_assets:
            flat_sections.append({
                'type': 'section',
                'title': section_title,
                'content': section_content,
                'id': section_id,
                'assets': section_assets,
                'original_index': i,
                'position': section_position,
                'has_assets': has_significant_assets
            })
            print(f"   ✅ Added section '{section_title}' to flat list")

        # Add all suitable subsections to the same level
        original_subsections = child.get("children", [])
        for j, subsection in enumerate(original_subsections):
            subsection_title = _clean_section_title(subsection.get("title", f"Subsection {j + 1}"))
            subsection_content = subsection.get("content", "")
            subsection_id = subsection.get("id", f"{section_id}.{j + 1}")
            subsection_assets = subsection.get("assets", {"figures": [], "tables": [], "references": []})

            has_subsection_assets = (len(subsection_assets.get('figures', [])) > 0 or
                                      len(subsection_assets.get('tables', [])) > 0)
            sub_decision, sub_reason = classify_section(subsection_title, subsection_content, paper_keywords)
            if sub_decision == 'exclude':
                print(f"   🚫 Excluding subsection '{subsection_title}': {sub_reason}")
                continue
            if sub_decision == 'related' and not has_subsection_assets:
                continue
            is_suitable_subsection = sub_decision in ('method', 'include') or has_subsection_assets

            # Merge deeper content into subsection
            merged_content = subsection_content
            merged_assets = {
                'figures': list(subsection_assets.get('figures', [])),
                'tables': list(subsection_assets.get('tables', [])),
                'references': list(subsection_assets.get('references', [])),
            }

            # Merge sub-subsections into content (flatten completely)
            subsection_children = subsection.get("children", [])
            for subsubsection in subsection_children:
                sub_title = subsubsection.get("title", "")
                sub_content = subsubsection.get("content", "")

                sub_dec, _ = classify_section(sub_title, sub_content, paper_keywords)
                if sub_dec == 'exclude':
                    continue

                if sub_dec in ('method', 'include'):
                    if sub_content.strip():
                        merged_content += "\n\n" + sub_content

                    # Merge assets
                    sub_assets = subsubsection.get("assets", {"figures": [], "tables": [], "references": []})
                    for asset_type in ['figures', 'tables', 'references']:
                        merged_assets[asset_type].extend(sub_assets.get(asset_type, []))

            # Include subsection if suitable or has assets
            if (is_suitable_subsection or has_subsection_assets) and merged_content.strip():
                flat_sections.append({
                    'type': 'subsection',
                    'title': subsection_title,
                    'content': merged_content,
                    'id': subsection_id,
                    'assets': merged_assets,
                    'original_index': (i, j),
                    'parent_section': section_title,
                    'has_assets': has_subsection_assets or len(merged_assets.get('figures', [])) > 0 or len(merged_assets.get('tables', [])) > 0
                })
                print(f"   ✅ Added subsection '{subsection_title}' to flat list (parent: {section_title})")

    print(f"\n📊 Flat structure created: {len(flat_sections)} total panels (no hierarchy)")

    # Process all flat sections as peer-level panels
    for idx, section_info in enumerate(flat_sections):
        section_title = _clean_section_title(section_info['title'])
        section_content = section_info['content']
        section_id = section_info['id']
        section_assets = section_info['assets']
        section_type = section_info['type']

        # Use LLM summary if available
        if section_id in summaries:
            content_summary = summaries[section_id]
            print(f"   📝 Using LLM summary for {section_id}: {section_title}")
        else:
            content_summary = summarize_text_content(section_content, target_sentences=5)
            print(f"   📝 Using fallback summary for {section_id}: {section_title}")

        # Apply poster-optimized asset assignment with uniqueness constraint
        filtered_section_assets = apply_asset_assignment_for_poster(
            section_assets, filtered_figures, filtered_tables, section_title
        )

        # Calculate layout parameters
        text_len, asset_path = calculate_layout_parameters(
            content_summary, filtered_section_assets, filtered_figures, filtered_tables
        )

        # Method/architecture sections get higher gp so the layout allocates more space to figures
        _is_method_section = is_method_related_section(section_title)
        _has_figure = bool(filtered_section_assets.get('figures'))
        _has_table  = bool(filtered_section_assets.get('tables'))
        if _has_figure and _is_method_section:
            _gp = 0.5   # method + figure → generous space for architecture diagram
        elif _has_figure or _has_table:
            _gp = 0.35  # any other section with an asset
        else:
            _gp = 0.0

        # Create section panel (no children - completely flat)
        section_panel = {
            "panel_id": str(panel_id_counter),
            "section_name": section_title,
            "tp": min(0.45, max(0.15, len(content_summary) / 1000)),  # Adaptive text proportion
            "text_len": text_len,
            "gp": _gp,
            "figure_size": 0,
            "figure_aspect": 1,
            "content": content_summary,
            "importance": 0.9 if _is_method_section else 0.8,
            "assets": filtered_section_assets,
            "asset_path": asset_path,
            "children": []  # No children in flat structure
        }

        # Set figure_size/figure_aspect from the first available asset (figure or table)
        if filtered_section_assets.get('figures') and filtered_figures:
            figure_id = filtered_section_assets['figures'][0]
            if figure_id in filtered_figures:
                figure_data = filtered_figures[figure_id]
                section_panel["figure_size"] = figure_data.get('figure_size', 0)
                section_panel["figure_aspect"] = figure_data.get('figure_aspect', 1.0)
        elif filtered_section_assets.get('tables') and filtered_tables:
            table_id = filtered_section_assets['tables'][0]
            if table_id in filtered_tables:
                table_data = filtered_tables[table_id]
                section_panel["figure_size"] = table_data.get('figure_size', 0)
                section_panel["figure_aspect"] = table_data.get('figure_aspect', 1.0)

        # Add to tree
        refined_tree["children"].append(section_panel)
        panel_id_counter += 1

        type_indicator = "🗂️ SECTION" if section_type == 'section' else "📄 SUBSECTION"
        asset_info = f" ({len(filtered_section_assets.get('figures', []))} figs, {len(filtered_section_assets.get('tables', []))} tables)" if filtered_section_assets.get('figures') or filtered_section_assets.get('tables') else ""
        print(f"   ✅ Created flat panel {panel_id_counter-1}: {type_indicator} '{section_title}'{asset_info}")

    print(f"\n🎯 FLAT TREE STRUCTURE COMPLETED")
    print(f"   📊 Total panels: {len(refined_tree['children'])} (all at same level)")
    print(f"   🔗 Hierarchy removed: sections and subsections are now peers")

    return refined_tree


# ---- Step 2-4: Asset Reassignment by Caption Relevance ----------------------
def reassign_assets_by_caption_relevance(document_tree, filtered_figures, filtered_tables, agent_config=None, text_content=None):
    """
    Ensure each asset is in exactly one section using a hybrid strategy:

    Phase 1 — Regex citation scan (deterministic, no LLM):
        Search every section's full text for explicit "Figure N" / "Table N" references.
        Primary source: raw markdown text from text_content (if provided) — this
        preserves all "Figure N" citations as written in the paper.  Fallback: the
        LLM-summarised content stored in tree nodes (which often omits figure refs).
        • 0 citations found → LLM fallback (or score-based if no LLM)
        • 1 citation found → assign directly, no disambiguation needed
        • 2+ citations found → LLM selects the PRIMARY section (where it's analysed,
          not just mentioned in passing), or score-based if no LLM

    Phase 2 — Caption-section relevance scoring (existing, LLM-free):
        Used when citation scan is inconclusive or when LLM is unavailable.

    Also handles assets that are NOT in any section yet (unassigned after Step 1 LLM),
    running the same regex + LLM pipeline for them after the main loop.
    """
    # Build raw section text index from markdown (primary citation source)
    # The LLM-summarised content stored in tree nodes rarely preserves "Figure N"
    # citations verbatim.  The original markdown from Docling does.  Building a
    # heading→body index lets _all_section_content() search both sources.
    _raw_section_texts: dict = {}
    if text_content:
        _HDG_RAW_RE = re.compile(r'^#{1,4}\s+(.+?)$', re.MULTILINE)
        _NUM_PREFIX_RAW_RE = re.compile(r'^[\d]+(?:\.[\d]+)*\.?\s*')
        def _norm_raw(t: str) -> str:
            return _NUM_PREFIX_RAW_RE.sub('', t).strip().lower()
        headings_raw = list(_HDG_RAW_RE.finditer(text_content))
        for idx, h in enumerate(headings_raw):
            raw_title = h.group(1).strip()
            start = h.end()
            end = headings_raw[idx + 1].start() if idx + 1 < len(headings_raw) else len(text_content)
            body = text_content[start:end]
            for key in (_norm_raw(raw_title), raw_title.lower()):
                _raw_section_texts[key] = _raw_section_texts.get(key, '') + ' ' + body
        print(f"   📄 Raw section index: {len(_raw_section_texts)} keys from markdown "
              f"({len(text_content):,} chars)")
    else:
        print(f"   ⚠️  No raw text_content provided — citation scan uses only LLM-summarised content")

    def _llm_select_section(caption, asset_id, asset_type, citing_nodes, agent_config):
        """Ask the LLM which of the citing_nodes is the PRIMARY section for this asset.

        citing_nodes may be non-empty (multiple citations) or empty (no citation found).
        When empty the LLM assigns based on caption alone.
        """
        if not agent_config:
            return None
        try:
            candidates_for_llm = citing_nodes if citing_nodes else []
            # Build a numbered list with title + brief content snippet
            items = []
            for i, node in enumerate(candidates_for_llm, 1):
                title   = node.get('title', f'Section {i}')
                snippet = (_all_section_content(node) or '')[:200].replace('\n', ' ')
                items.append(f'{i}. "{title}": {snippet}')
            sections_text = '\n'.join(items)

            if citing_nodes:
                context = (
                    f'The following sections all explicitly cite this {asset_type}:\n'
                    f'{sections_text}\n\n'
                    f'Which section is the PRIMARY location where this {asset_type} '
                    f'is INTRODUCED or ANALYSED in depth (not just briefly mentioned)?'
                )
            else:
                # No citation found — show all candidate sections
                all_cands = [s for s in candidates if not is_generic(s)][:8]
                for i, node in enumerate(all_cands, 1):
                    title   = node.get('title', f'Section {i}')
                    snippet = (_all_section_content(node) or '')[:150].replace('\n', ' ')
                    items.append(f'{i}. "{title}": {snippet}')
                sections_text = '\n'.join(items)
                context = (
                    f'This {asset_type} is not explicitly cited in any section.\n'
                    f'Candidate sections:\n{sections_text}\n\n'
                    f'Based on the caption, which section is the most appropriate home?'
                )
                candidates_for_llm = all_cands

            prompt = (
                f'Academic paper asset:\n'
                f'Caption: "{caption}"\n\n'
                f'{context}\n\n'
                f'Reply with JSON only: {{"section_number": <integer 1–{len(candidates_for_llm)}>}}'
            )

            model = make_model(agent_config)
            agent = ChatAgent(
                system_message=(
                    'You are a research paper analyst. '
                    'Identify which section of an academic paper primarily introduces '
                    'or analyses a given figure or table.'
                ),
                model=model,
                message_window_size=1,
            )
            from camel.messages import BaseMessage
            _msg = BaseMessage.make_user_message(role_name='User', content=prompt)
            resp = agent_step_with_timeout(agent, _msg)
            result = get_json_from_response(resp.msgs[0].content)
            if result and 'section_number' in result:
                idx = int(result['section_number']) - 1
                if 0 <= idx < len(candidates_for_llm):
                    chosen = candidates_for_llm[idx]
                    print(f'   🤖 LLM selected: "{chosen.get("title", "?")}"')
                    return chosen
        except Exception as e:
            print(f'   ⚠️  LLM selection failed: {e}')
        return None
    _GENERIC = {
        'introduction', 'abstract', 'background', 'related work', 'related works',
        'motivation', 'overview', 'preliminary', 'preliminaries', 'problem statement',
    }
    _INTRO_TITLES = {'introduction', 'intro', 'motivation', 'overview', 'abstract'}

    # Caption keywords that strongly indicate METHOD figures
    _METHOD_CAP_KW = {
        'architecture', 'framework', 'pipeline', 'workflow', 'overview',
        'structure', 'module', 'component', 'network', 'layer',
        'illustration', 'schematic', 'diagram', 'design', 'proposed',
    }
    # Caption keywords that strongly indicate EXPERIMENT figures.
    # NOTE: 'visualization' and 'visual' intentionally excluded — they describe the
    # *format* of the figure, not its scientific role.  Many method-explanation figures
    # (e.g. "CLIP similarity visualization") use these words but belong in Method sections.
    _EXPERIMENT_CAP_KW = {
        'result', 'results', 'comparison', 'compared', 'versus', 'baseline',
        'qualitative', 'quantitative', 'ablation', 'accuracy', 'performance',
        'evaluation', 'benchmark', 'state-of-the-art', 'ground-truth', 'prediction',
        'example',
    }
    # Section title keywords for Method and Experiment sections
    _METHOD_SEC_KW = {
        'method', 'approach', 'model', 'architecture', 'framework', 'algorithm',
        'network', 'design', 'system', 'technique', 'proposed',
    }
    _EXPERIMENT_SEC_KW = {
        'experiment', 'result', 'evaluation', 'analysis', 'performance',
        'comparison', 'ablation', 'benchmark',
    }

    def is_generic(node):
        t = node.get('title', '').lower().strip()
        return t in _GENERIC or any(t.startswith(g) for g in _GENERIC)

    def _is_intro_section(node):
        t = node.get('title', '').lower().strip()
        return t in _INTRO_TITLES or any(t.startswith(k) for k in _INTRO_TITLES)

    def _section_type(node):
        """Return 'method', 'experiment', or 'other' based on section title."""
        t = node.get('title', '').lower()
        if any(kw in t for kw in _METHOD_SEC_KW):
            return 'method'
        if any(kw in t for kw in _EXPERIMENT_SEC_KW):
            return 'experiment'
        return 'other'

    def _figure_type(caption):
        """Infer figure type from caption: 'method', 'experiment', 'intro', or 'unknown'."""
        cap = caption.lower()
        # Strip "Figure N:" prefix
        cap = re.sub(r'^(figure|table|fig\.?)\s*\d+[:\.\s]*', '', cap)
        method_score = sum(1 for kw in _METHOD_CAP_KW if kw in cap)
        exp_score    = sum(1 for kw in _EXPERIMENT_CAP_KW if kw in cap)
        if method_score > exp_score:
            return 'method'
        if exp_score > method_score:
            return 'experiment'
        return 'unknown'

    # Multi-word phrases that conclusively identify a method-overview figure
    # (the main architecture diagram, not a component-level figure).
    # Kept conservative (multi-gram, "overall/overview + method noun") to avoid
    # false-positives on component figures that merely say "network layer".
    _OVERVIEW_CAP_PHRASES = [
        'overall architecture', 'overall framework', 'overall pipeline',
        'overall approach', 'overall system', 'overall structure',
        'overall design', 'overall workflow',
        'overview of our', 'overview of the proposed', 'overview of proposed',
        'our overall', 'proposed architecture', 'proposed framework',
        'proposed pipeline', 'proposed method overview',
    ]

    def _is_overview_fig(caption_text):
        """True when caption conclusively indicates a method-overview / main-architecture figure.

        These figures represent the entire method, not a single component.  They belong
        in the top-level method section (L1), not in a specific subsection (L2) that
        happens to cite them.
        """
        cap_l = re.sub(r'^(figure|table|fig\.?)\s*\d+[:\.\s]*', '', caption_text.lower())
        return any(phrase in cap_l for phrase in _OVERVIEW_CAP_PHRASES)

    _NUM_PREFIX_CONTENT_RE = re.compile(r'^[\d]+(?:\.[\d]+)*\.?\s*')

    def _all_section_content(sec):
        """Return all searchable text for a section node.

        Combines two sources:
          1. Raw markdown body text (from text_content via _raw_section_texts) —
             the primary source because it preserves every "Figure N" citation.
          2. LLM-summarised content stored in tree nodes — used as fallback when
             no raw text mapping is available or when text_content was not passed.

        Recursive collection over children ensures "Figure N" references in nested
        subsections are also found (important when Docling distributes text across
        child nodes and for the parent/child de-duplication logic).
        """
        parts = [sec.get('content', '')]
        # Include raw markdown text for this section if the index was built
        if _raw_section_texts:
            title = sec.get('title', '')
            norm = _NUM_PREFIX_CONTENT_RE.sub('', title).strip().lower()
            for key in (norm, title.lower()):
                if key in _raw_section_texts:
                    parts.append(_raw_section_texts[key])
                    break
        for ch in sec.get('children', []):
            parts.append(_all_section_content(ch))
        return ' '.join(filter(None, parts))

    def section_score(cap_words, section, fig_type, cite_re=None):
        """
        Score a section as a home for this figure.

        Tier 0 (Strongest): Explicit "Figure N" / "Fig. N" citation found anywhere in
            the section tree (direct content + ALL subsection content recursively).
            +500 bonus — strongly dominates all other signals so that a figure mentioned
            only in Introduction is ALWAYS placed in Introduction even if its caption
            contains method-like keywords.  If the figure is cited in multiple sections,
            each gets the bonus and similarity scoring breaks the tie.

        Tier 1 (Title overlap): Caption word ∩ section title × 5.
        Tier 2 (Content overlap): Caption word ∩ section body × 1.
        Tier 3 (Weak type hint): Section type matches inferred figure type → +20.
            Kept very weak so it never overrides a citation from a different section.
        """
        title   = section.get('title', '').lower()
        content = section.get('content', '').lower()

        title_words   = {w.strip('.,;:()') for w in title.split()   if len(w.strip('.,;:()')) > 3}
        content_words = {w.strip('.,;:()') for w in content.split() if len(w.strip('.,;:()')) > 3}

        # Tier 0: explicit citation anywhere in the section tree (recursive)
        # Reduced from 500 → 200 so that type+title matching can compete when multiple
        # sections all cite the figure (e.g. Method introduces it, Experiments re-references
        # it — the introducing section should win, not whichever one cited it last).
        if cite_re is not None:
            full_text = _all_section_content(section).lower()
            citation_bonus = 200 if cite_re.search(full_text) else 0
        else:
            citation_bonus = 0

        score  = citation_bonus
        score += 5 * len(cap_words & title_words)                        # Tier 1
        score += 1 * len(cap_words & (content_words - title_words))      # Tier 2

        # Tier 3: type-section alignment bonus
        sec_type = _section_type(section)
        if fig_type == sec_type and fig_type in ('method', 'experiment'):
            score += 50  # raised from 20 so type match matters alongside citations
        elif fig_type == 'unknown' and sec_type == 'experiment':
            # Figures with unrecognised captions (e.g. "Effect of X on Y", ablation
            # patterns not in keyword list) are more likely experiment/ablation than
            # method. Give a soft preference toward experiment sections without
            # requiring individual keywords to be hard-coded.
            score += 15

        # Penalty: method figures should not be placed in experiment sections unless they are
        # the ONLY citing section.
        if fig_type == 'method' and sec_type == 'experiment':
            score -= 80

        # Penalty: method figures cited in intro sections are forward references, not
        # the primary presentation site.  Mirror the experiment penalty so that a method
        # section with the type-alignment bonus (+50) reliably outscores an intro section
        # that merely previews the figure.
        if fig_type == 'method' and _is_intro_section(section):
            score -= 100

        return score

    def depth_of(node):
        for c in document_tree.get('children', []):
            if node is c:
                return 1
            if node in c.get('children', []):
                return 2
        return 0

    # Candidate sections: L1 + L2, not poster-unsuitable.
    # L2 children of 'related' sections are excluded — Related Works subsections are
    # always pruned from the poster tree so any figure assigned there is silently lost.
    candidates = []
    for child in document_tree.get('children', []):
        if not is_poster_unsuitable_section(child.get('title', ''), child.get('content', ''))[0]:
            candidates.append(child)
            _child_is_relwork = (
                classify_section(child.get('title', ''), child.get('content', ''))[0] == 'related'
            )
            if not _child_is_relwork:
                for sub in child.get('children', []):
                    if not is_poster_unsuitable_section(sub.get('title', ''), sub.get('content', ''))[0]:
                        candidates.append(sub)

    if not candidates:
        return

    # Collect all nodes holding each asset
    asset_locations: dict = {}
    def _collect(node):
        for fid in node.get('assets', {}).get('figures', []):
            if fid in filtered_figures:
                asset_locations.setdefault((fid, 'figure'), []).append(node)
        for tid in node.get('assets', {}).get('tables', []):
            if tid in filtered_tables:
                asset_locations.setdefault((tid, 'table'), []).append(node)
        for ch in node.get('children', []):
            _collect(ch)
    _collect(document_tree)

    _CAPTION_STOPWORDS = {
        'the', 'and', 'for', 'with', 'from', 'this', 'that', 'are', 'its',
        'each', 'both', 'using', 'used', 'our', 'show', 'shows', 'shown',
        'left', 'right', 'top', 'bottom',
    }

    for (asset_id, asset_type), nodes in asset_locations.items():
        asset_key  = 'figures' if asset_type == 'figure' else 'tables'
        asset_data = (filtered_figures if asset_type == 'figure' else filtered_tables).get(asset_id, {})
        caption    = asset_data.get('caption', '')
        fig_type   = _figure_type(caption)

        # Build cap_words from cleaned caption (strip number prefix, stopwords, short tokens)
        cap_clean = re.sub(r'^(figure|table|fig\.?)\s*\d+[:\.\s]*', '', caption.lower())
        cap_words = {
            w.strip('.,;:()')
            for w in cap_clean.split()
            if len(w.strip('.,;:()')) > 3
            and w.strip('.,;:()').isalpha()
            and w.strip('.,;:()') not in _CAPTION_STOPWORDS
        }

        # Build citation-detection regex.
        # For figures: matches "Figure N", "Fig. N", "fig N" etc.
        # For tables:  matches "Table N", "Tab. N", "tab N" etc.
        # Use caption_number (e.g. "1", "2") which is the actual label in the paper.
        _cap_num = asset_data.get('caption_number') or str(asset_id)
        num = re.escape(str(_cap_num))
        if asset_type == 'figure':
            cite_re = re.compile(
                rf'\b(?:fig(?:ure)?\.?\s*{num}|fig\.\s*{num})\b',
                re.IGNORECASE,
            )
        else:  # table
            cite_re = re.compile(
                rf'\b(?:tab(?:le)?\.?\s*{num})\b',
                re.IGNORECASE,
            )

        # Score helper bound to this asset
        def _score(section, _cw=cap_words, _ft=fig_type, _cr=cite_re):
            return section_score(_cw, section, _ft, cite_re=_cr)

        # Citation-first strategy: if ANY candidate has an explicit "Figure N" / "Table N"
        # citation, use citations as the primary signal regardless of generic/non-generic
        # status.  This ensures e.g. teaser figures (Introduction) and result tables
        # (Section 4) are placed correctly rather than drifting to Introduction.
        if cite_re is not None:
            _cited = [s for s in candidates
                      if cite_re.search(_all_section_content(s).lower())]
            if len(_cited) == 1:
                # Unambiguous: exactly one section cites this figure → assign there.
                target = _cited[0]
                # Exception: a method-type figure cited only in a Related Work /
                # Background section should NOT be locked there.  These sections cite
                # method figures to contrast with prior work, but the figure belongs in
                # the method section.  Fall through to score-based assignment instead.
                _RELWORK_KW = {'related', 'background', 'prior', 'literature', 'review'}
                _target_title_l = target.get('title', '').lower()
                _is_relwork = any(kw in _target_title_l for kw in _RELWORK_KW)
                if fig_type == 'method' and _is_relwork:
                    print(f"   ⚠️  {asset_type} {asset_id} ({fig_type}): sole citation is in related-work context ('{target.get('title', '')}') — falling back to score-based assignment")
                    # Score all non-related-work candidates; method sections get a type bonus.
                    _non_rw = [s for s in candidates
                                if not any(kw in s.get('title', '').lower() for kw in _RELWORK_KW)]
                    if _non_rw:
                        _rw_scores = {id(s): _score(s) for s in _non_rw}
                        _rw_best = max(_rw_scores.values())
                        if _rw_best > 0:
                            _rw_top = [s for s in _non_rw if _rw_scores[id(s)] == _rw_best]
                            method_target = max(_rw_top, key=depth_of)
                        else:
                            _method_secs = [s for s in _non_rw if _section_type(s) == 'method']
                            method_target = _method_secs[0] if _method_secs else _non_rw[0]
                        for node in nodes:
                            lst = node.get('assets', {}).get(asset_key, [])
                            if asset_id in lst:
                                lst.remove(asset_id)
                                print(f"   🔄 {asset_type} {asset_id} ({fig_type}): removed from '{node.get('title', '')}' [method-override]")
                        method_target.setdefault('assets', {'figures': [], 'tables': [], 'references': []})
                        method_target['assets'].setdefault(asset_key, [])
                        if asset_id not in method_target['assets'][asset_key]:
                            method_target['assets'][asset_key].append(asset_id)
                        print(f"   🔄 {asset_type} {asset_id} ({fig_type}): assigned to '{method_target.get('title', '')}' [method-override]")
                        continue
                # Remove from all non-target holding nodes
                for node in nodes:
                    if node is not target:
                        lst = node.get('assets', {}).get(asset_key, [])
                        if asset_id in lst:
                            lst.remove(asset_id)
                            print(f"   🔄 {asset_type} {asset_id} ({fig_type}): removed from '{node.get('title', '')}' [citation-first]")
                if target not in nodes:
                    target.setdefault('assets', {'figures': [], 'tables': [], 'references': []})
                    target['assets'].setdefault(asset_key, [])
                    if asset_id not in target['assets'][asset_key]:
                        target['assets'][asset_key].append(asset_id)
                        print(f"   🔄 {asset_type} {asset_id} ({fig_type}): added to '{target.get('title', '')}' [citation-first]")
                else:
                    print(f"   🔄 {asset_type} {asset_id} ({fig_type}): kept in '{target.get('title', '')}' [citation-first, removed from {len(nodes)-1} other(s)]")
                continue  # Skip the generic/score-based fallback below
            elif len(_cited) > 1:
                # Multiple sections cite this figure.
                #
                # Pre-filter: remove Related Works / Background sections from candidates.
                # These sections are always pruned from the poster tree, so assigning
                # a figure there causes it to be silently dropped.  Early figures (e.g.
                # a comparison-paradigm diagram in section 2.2 "Open-Vocabulary Object
                # Detection") should stay in Introduction rather than drift to a prunable
                # Related Works subsection.
                _RELWORK_PRUNE_KW = {'related', 'background', 'prior', 'literature', 'review'}
                _non_relwork = [s for s in _cited
                                if not any(kw in s.get('title', '').lower()
                                           for kw in _RELWORK_PRUNE_KW)]
                if _non_relwork:
                    _cited = _non_relwork
                #
                # Step 1: De-duplicate. _all_section_content() is RECURSIVE, so a
                # mention in a child subsection (L2) also makes the parent section (L1)
                # appear in _cited even when the parent's OWN body text does not cite it.
                #
                # OLD (bug): remove the parent whenever any child is also in _cited.
                #   → This always picked subsection 3.3 over section 3, even when
                #     Figure 3 was the method overview introduced in 3.1/3.2 and 3.3
                #     was just making a back-reference.
                #
                # NEW (fix): a parent is removed only when at least one of its DIRECT
                # children directly cites the figure in the child's OWN body text
                # (non-recursive). If the child's citation was inherited recursively
                # from its OWN children, the parent is NOT removed — scoring decides.
                def _child_directly_cites(child_node, cr):
                    """True iff the child's own (non-recursive) text directly cites."""
                    own_text = (child_node.get('content', '') or '').lower()
                    # Also include raw markdown body for this specific node
                    if _raw_section_texts:
                        title = child_node.get('title', '')
                        norm = _NUM_PREFIX_CONTENT_RE.sub('', title).strip().lower()
                        for key in (norm, title.lower()):
                            if key in _raw_section_texts:
                                own_text += ' ' + _raw_section_texts[key]
                                break
                    return bool(cr.search(own_text))

                _cited_ids = {id(s) for s in _cited}

                # Overview method figures (e.g. "Overall architecture of our method")
                # represent the ENTIRE method section, not a single subsection.
                # Standard dedup removes the L1 method parent when an L2 subsection
                # directly cites the figure — but that is exactly backwards for these
                # figures: the subsection only back-references what Section 3 introduces.
                # Detect this case and reverse the dedup direction:
                #   standard:  keep deepest citing node  (L2 child wins over L1 parent)
                #   overview:  keep shallowest method node (L1 parent wins over L2 child)
                _is_overview = (fig_type == 'method') and _is_overview_fig(caption)
                _l1_method_ids = {
                    id(s) for s in _cited
                    if depth_of(s) == 1 and _section_type(s) == 'method'
                }

                if _is_overview and _l1_method_ids:
                    # Remove L2 subsections whose L1 method parent is already in _cited.
                    # This keeps Section 3 "PIIPN" and drops Branch Merging 3.3 so that
                    # Section 3 can win over Introduction via the +50 type-alignment bonus.
                    _cited_deduped = [
                        s for s in _cited
                        if depth_of(s) != 2 or not any(
                            id(parent) in _l1_method_ids
                            for parent in document_tree.get('children', [])
                            if s in parent.get('children', [])
                        )
                    ]
                    if _cited_deduped:
                        _cited = _cited_deduped
                        print(f"   🏗️  {asset_type} {asset_id}: overview-fig dedup — "
                              f"kept L1 method parent(s), dropped L2 subsection(s)")
                else:
                    # Standard dedup: remove L1 parent when its L2 child directly cites.
                    _cited_deduped = [
                        s for s in _cited
                        if not any(
                            id(ch) in _cited_ids and _child_directly_cites(ch, cite_re)
                            for ch in s.get('children', [])
                        )
                    ]
                    if _cited_deduped:
                        _cited = _cited_deduped

                if len(_cited) == 1:
                    # De-dup resolved to a single section — assign directly, no LLM needed.
                    _sole = _cited[0]
                    for node in nodes:
                        if node is not _sole:
                            lst = node.get('assets', {}).get(asset_key, [])
                            if asset_id in lst:
                                lst.remove(asset_id)
                                print(f"   🔄 {asset_type} {asset_id}: removed from '{node.get('title', '')}' [dedup-cite]")
                    _sole.setdefault('assets', {'figures': [], 'tables': [], 'references': []})
                    _sole['assets'].setdefault(asset_key, [])
                    if asset_id not in _sole['assets'][asset_key]:
                        _sole['assets'][asset_key].append(asset_id)
                    print(f"   🔄 {asset_type} {asset_id}: assigned to '{_sole.get('title', '')}' [dedup-cite]")
                    continue

                # Step 2: Score-based ranking among the (still-multiple) citing sections.
                # section_score() applies type-alignment bonuses (+50 for matching type)
                # and penalties (-80 for method/unknown figures in experiment sections).
                # If one section wins by a clear margin, use it without an LLM call.
                _cited_scored = sorted(_cited, key=lambda s: _score(s), reverse=True)
                _best_score  = _score(_cited_scored[0])
                _second_score = _score(_cited_scored[1])
                _score_diff  = _best_score - _second_score

                if _score_diff >= 40:
                    # Clear winner by score — no LLM needed.
                    llm_target = _cited_scored[0]
                    _tag = 'score-multi-cite'
                else:
                    # Scores are close → ask LLM to break the tie.
                    llm_target = _llm_select_section(caption, asset_id, asset_type, _cited, agent_config)
                    _tag = 'LLM-multi-cite'

                if llm_target is not None:
                    # Enforce chosen assignment
                    for node in nodes:
                        if node is not llm_target:
                            lst = node.get('assets', {}).get(asset_key, [])
                            if asset_id in lst:
                                lst.remove(asset_id)
                                print(f"   🔄 {asset_type} {asset_id}: removed from '{node.get('title', '')}' [{_tag}]")
                    llm_target.setdefault('assets', {'figures': [], 'tables': [], 'references': []})
                    llm_target['assets'].setdefault(asset_key, [])
                    if asset_id not in llm_target['assets'][asset_key]:
                        llm_target['assets'][asset_key].append(asset_id)
                    print(f"   🔄 {asset_type} {asset_id}: assigned to '{llm_target.get('title', '')}' [{_tag}]")
                    continue  # skip score-based fallback

        # Already in exactly one specific (non-generic) section → trust docling's assignment.
        if len(nodes) == 1 and not is_generic(nodes[0]) and nodes[0] is not document_tree:
            continue

        specific = [n for n in nodes if n is not document_tree and not is_generic(n)
                    and not is_poster_unsuitable_section(n.get('title', ''), n.get('content', ''))[0]]

        if specific:
            max_d   = max(depth_of(n) for n in specific)
            deepest = [n for n in specific if depth_of(n) == max_d]
            target  = max(deepest, key=_score)
        else:
            # All holdings are generic/root — use full candidate set
            if not cap_words and fig_type == 'unknown' and cite_re is None:
                continue
            all_scores = {id(s): _score(s) for s in candidates}
            best_score = max(all_scores.values())
            if best_score <= 0:
                continue
            top_cands = [s for s in candidates if all_scores[id(s)] == best_score]
            # Among equal-score candidates, prefer the deepest (more specific) node
            target = max(top_cands, key=depth_of)

        # Remove from all non-target nodes
        for node in nodes:
            if node is not target:
                lst = node.get('assets', {}).get(asset_key, [])
                if asset_id in lst:
                    lst.remove(asset_id)
                    print(f"   🔄 {asset_type} {asset_id} ({fig_type}): removed from '{node.get('title', '')}'")

        # Add to target if it wasn't a holding node
        if target not in nodes:
            target.setdefault('assets', {'figures': [], 'tables': [], 'references': []})
            target['assets'].setdefault(asset_key, [])
            if asset_id not in target['assets'][asset_key]:
                target['assets'][asset_key].append(asset_id)
                print(f"   🔄 {asset_type} {asset_id} ({fig_type}): added to '{target.get('title', '')}'")
        else:
            print(f"   🔄 {asset_type} {asset_id} ({fig_type}): kept in '{target.get('title', '')}'  "
                  f"(removed from {len(nodes)-1} other node(s))")

    # Handle assets completely unassigned after Step-1 LLM
    # These are figures/tables that appear in filtered_figures/tables but were
    # never placed in any section node by the Step-1 LLM.
    # Strategy: regex citation scan → if 1 hit assign directly; otherwise LLM.
    print(f"\n🔍 Scanning for completely unassigned assets...")

    placed_fids = {fid for (fid, t), _ in asset_locations.items() if t == 'figure'}
    placed_tids = {tid for (tid, t), _ in asset_locations.items() if t == 'table'}
    unplaced_figs = set(filtered_figures.keys()) - placed_fids
    unplaced_tbls = set(filtered_tables.keys()) - placed_tids

    if not (unplaced_figs or unplaced_tbls):
        print("   ✅ No unassigned assets found.")
    else:
        print(f"   📋 Unassigned — figures: {list(unplaced_figs)}, tables: {list(unplaced_tbls)}")

    def _assign_unplaced(asset_id, asset_data, asset_key_name):
        cap_num = asset_data.get('caption_number')
        caption = asset_data.get('caption', '')
        is_tbl  = (asset_key_name == 'tables')

        # Build citation regex
        _cr = None
        _cited_unplaced = []
        if cap_num is not None:
            _num = re.escape(str(cap_num))
            if is_tbl:
                _cr = re.compile(rf'\b(?:tab(?:le)?\.?\s*{_num})\b', re.IGNORECASE)
            else:
                _cr = re.compile(rf'\b(?:fig(?:ure)?\.?\s*{_num}|fig\.\s*{_num})\b', re.IGNORECASE)
            _cited_unplaced = [s for s in candidates if _cr.search(_all_section_content(s).lower())]

        if len(_cited_unplaced) == 1:
            # Unambiguous single citation — assign directly, no LLM needed
            target = _cited_unplaced[0]
            print(f"   ✅ [Regex] {asset_key_name[:-1]} {asset_id} (cap#{cap_num}) → '{target.get('title', '?')}'")
        else:
            # 0 or 2+ matches → ask LLM to decide
            llm_target = _llm_select_section(
                caption, asset_id, asset_key_name[:-1], _cited_unplaced, agent_config
            )

            if llm_target is not None:
                target = llm_target
                tag = 'LLM-multi' if _cited_unplaced else 'LLM-nocite'
                print(f"   ✅ [{tag}] {asset_key_name[:-1]} {asset_id} (cap#{cap_num}) → '{target.get('title', '?')}'")
            elif _cited_unplaced:
                # LLM unavailable but citations exist → pick highest scoring cited section
                cap_words_fb = {w for w in caption.lower().split()
                                if len(w) > 3 and w not in _CAPTION_STOPWORDS}
                target = max(_cited_unplaced, key=lambda s: section_score(
                    cap_words_fb, s, 'unknown', cite_re=_cr
                ))
                print(f"   ✅ [Score] {asset_key_name[:-1]} {asset_id} (cap#{cap_num}) → '{target.get('title', '?')}'")
            else:
                print(f"   ⚠️  {asset_key_name[:-1]} {asset_id}: no citation found and LLM unavailable — skipping")
                return

        target.setdefault('assets', {'figures': [], 'tables': [], 'references': []})
        target['assets'].setdefault(asset_key_name, [])
        if asset_id not in target['assets'][asset_key_name]:
            target['assets'][asset_key_name].append(asset_id)

    for fid in unplaced_figs:
        _assign_unplaced(fid, filtered_figures.get(fid, {}), 'figures')
    for tid in unplaced_tbls:
        _assign_unplaced(tid, filtered_tables.get(tid, {}), 'tables')


# ---- Step 2-4: 3-Level Poster Tree Construction -----------------------------
def create_poster_tree_3_level_with_summaries(document_tree, filtered_figures, filtered_tables, summaries, args, agent_config, text_content=None):
    """
    Create 3-level poster tree using LLM-generated summaries (no importance scoring)
    """
    print("🛠️  Creating 3-level poster tree structure with LLM summaries...")

    # Reset global asset tracking and enforce uniqueness constraint for posters
    asset_tracker.reset()
    print(f"   🔒 Asset uniqueness constraint: Each asset can only be assigned once")

    # Normalize all asset IDs to strings so that integer IDs from LLM-parsed JSON
    # match the string keys in filtered_figures / filtered_tables throughout step 2.
    def _normalize_ids(node):
        if 'assets' in node:
            for k in ('figures', 'tables'):
                if k in node['assets']:
                    node['assets'][k] = [str(id_) for id_ in node['assets'][k]]
        for ch in node.get('children', []):
            _normalize_ids(ch)
    _normalize_ids(document_tree)
    print(f"   🔧 Asset IDs normalized to strings")

    # Reassign figures/tables to sections using regex citation scan first,
    # then LLM when the scan is ambiguous (0 or 2+ matches).
    print(f"\n🔄 Reassigning assets by citation scan + LLM fallback...")
    reassign_assets_by_caption_relevance(document_tree, filtered_figures, filtered_tables,
                                          agent_config=agent_config, text_content=text_content)

    # Title panel (root level)
    title_content = document_tree.get("content", "")
    title_id = document_tree.get("id", "root")

    # Use LLM summary if available, otherwise use original or fallback summarization
    if title_id in summaries:
        title_summary = summaries[title_id]
        print(f"   📝 Using LLM summary for title")
    else:
        title_summary = summarize_text_content(title_content, target_sentences=4)
        print(f"   📝 Using fallback summary for title")

    text_len, asset_path = calculate_layout_parameters(
        title_summary, {"figures": [], "tables": []}, filtered_figures, filtered_tables
    )

    refined_tree = {
        "panel_id": "0",
        "section_name": document_tree.get("title", "Title"),
        "tp": 0.08,  # Fixed for title
        "text_len": text_len,
        "gp": 0,  # No graphics in title
        "figure_size": 0,
        "figure_aspect": 1,
        "content": title_summary,
        "importance": 1.0,  # Default importance for title
        "assets": {"figures": [], "tables": [], "references": []},
        "asset_path": asset_path,
        "children": []
    }

    # Extract paper-specific keywords for dynamic section filtering
    paper_keywords = extract_paper_keywords(document_tree)

    # Process main sections with poster suitability filtering
    print(f"\n🎯 Filtering sections for poster suitability...")
    original_children = document_tree.get("children", [])
    panel_id_counter = 1
    total_sections = len(original_children)

    for i, child in enumerate(original_children):
        section_title = _clean_section_title(child.get("title", f"Section {i + 1}"))
        section_content = child.get("content", "")
        section_id = child.get("id", str(i + 1))
        original_subsections = child.get("children", [])
        section_assets = child.get("assets", {"figures": [], "tables": [], "references": []})

        has_significant_assets = (
            len(section_assets.get('figures', [])) > 0 or
            len(section_assets.get('tables', [])) > 0
        )
        decision, reason = classify_section(section_title, section_content, paper_keywords)
        is_method_section = (decision == 'method')

        if decision == 'exclude':
            print(f"   🚫 Excluding '{section_title}': {reason}")
            continue
        if decision == 'related' and not has_significant_assets:
            print(f"   🚫 Excluding related-work '{section_title}': no assets")
            continue
        # 'method', 'include', or 'related' with assets → proceed
        print(f"   ✅ Including '{section_title}' ({decision}: {reason})")

        # Second filter: basic preservation criteria (no importance-based filtering)
        should_preserve = (
            len(section_content) > 50 or
            len(original_subsections) > 0 or
            any(child.get("assets", {}).get(asset_type) for asset_type in ['figures', 'tables'])
        )

        if not should_preserve:
            print(f"   🚫 Excluding section '{section_title}': insufficient content")
            continue

        # Create section panel
        section_panel = {
            "panel_id": str(panel_id_counter),
            "section_name": section_title,
            "tp": 0,  # Sections act as containers
            "text_len": 0,
            "gp": 0,
            "figure_size": 0,
            "figure_aspect": 1,
            "content": "",
            "importance": 0.8,  # Default importance for sections
            "assets": {"figures": [], "tables": [], "references": []},
            "asset_path": None,
            "children": []
        }

        # Process subsections with poster suitability filtering
        if original_subsections:
            subsection_id_counter = 1

            # Collect section-level assets that need to be distributed to subsections
            # section_assets is already defined above
            print(f"   📦 Section '{section_title}' has {len(section_assets.get('figures', []))} figures and {len(section_assets.get('tables', []))} tables at section level")

            # First pass: collect all suitable subsections with enhanced method detection
            suitable_subsections = []
            total_subsections = len(original_subsections)

            for j, subsection in enumerate(original_subsections):
                subsection_title = _clean_section_title(subsection.get("title", f"Subsection {j + 1}"))
                subsection_content = subsection.get("content", "")
                subsection_id = subsection.get("id", f"{section_id}.{j + 1}")
                subsection_assets = subsection.get("assets", {"figures": [], "tables": [], "references": []})

                has_subsection_assets = (
                    len(subsection_assets.get('figures', [])) > 0 or
                    len(subsection_assets.get('tables', [])) > 0
                )
                sub_decision, sub_reason = classify_section(
                    subsection_title, subsection_content, paper_keywords)
                is_method_subsection = (sub_decision == 'method')

                # Decide whether to include this subsection
                if sub_decision == 'exclude':
                    print(f"   🚫 Excluding subsection '{subsection_title}': {sub_reason}")
                    continue
                if sub_decision == 'related' and not has_subsection_assets:
                    if not is_method_section:
                        print(f"   🚫 Excluding related subsection '{subsection_title}': no assets")
                        continue
                # In method sections, accept all non-excluded subsections
                if not (sub_decision in ('method', 'include') or has_subsection_assets or is_method_section):
                    continue

                if True:  # always-include block (replaces old `if should_include_subsection:`)
                    suitable_subsections.append({
                        'index': j,
                        'subsection': subsection,
                        'title': subsection_title,
                        'content': subsection_content,
                        'id': subsection_id,
                        'is_method': is_method_subsection,
                        'has_assets': has_subsection_assets
                    })

            # Distribute section-level assets to the best-matching subsection
            # (scored by caption-word overlap with subsection title + content)
            if section_assets and suitable_subsections:
                _STOPWORDS = {
                    'the', 'and', 'for', 'with', 'from', 'this', 'that', 'are',
                    'its', 'each', 'both', 'using', 'based', 'used', 'via', 'per',
                    'set', 'our', 'can', 'into', 'also', 'such', 'left', 'right',
                    'two', 'all', 'show', 'shows', 'shown',
                }

                def _best_subsection_idx(caption, cap_number=None, is_table=False):
                    """
                    Find the best subsection for a figure or table.

                    Tier 0 (Strongest): explicit "Figure N" / "Table N" citation in the
                      subsection's own body text (non-recursive). This mirrors the
                      citation-first strategy in reassign_assets_by_caption_relevance,
                      preventing a back-reference in a late subsection from winning over
                      the subsection that actually INTRODUCES the asset.
                      • Exactly 1 direct cite → assign there immediately.
                      • 2+ direct cites → add citation bonus to score, let scoring decide.

                    Tier 1: asset-type bonus (+50) when section type matches
                    Tier 2: caption → subsection TITLE overlap (5×)
                    Tier 3: caption → subsection CONTENT overlap (1×)
                    """
                    cap_clean = re.sub(r'^(figure|table|fig\.?)\s*\d+[:\.\s]*', '', caption.lower())
                    cap_words = {
                        w.strip('.,;:()')
                        for w in cap_clean.split()
                        if len(w.strip('.,;:()')) > 3
                        and w.strip('.,;:()').isalpha()
                        and w.strip('.,;:()') not in _STOPWORDS
                    }

                    # Build citation regex: figure → "fig N", table → "tab N"
                    _cite_re_sub = None
                    if cap_number is not None:
                        _num = re.escape(str(cap_number))
                        if is_table:
                            _cite_re_sub = re.compile(
                                rf'\b(?:tab(?:le)?\.?\s*{_num})\b',
                                re.IGNORECASE,
                            )
                        else:
                            _cite_re_sub = re.compile(
                                rf'\b(?:fig(?:ure)?\.?\s*{_num}|fig\.\s*{_num})\b',
                                re.IGNORECASE,
                            )

                    # Figure type from caption
                    _M_KW = {'architecture', 'framework', 'pipeline', 'overview', 'structure',
                              'module', 'component', 'network', 'proposed', 'diagram', 'design'}
                    _E_KW = {'result', 'results', 'comparison', 'qualitative', 'ablation',
                              'accuracy', 'performance', 'evaluation', 'visualization', 'prediction'}
                    cap_lower = cap_clean
                    m_score = sum(1 for kw in _M_KW if kw in cap_lower)
                    e_score = sum(1 for kw in _E_KW if kw in cap_lower)
                    fig_type = 'method' if m_score > e_score else ('experiment' if e_score > m_score else 'unknown')

                    _M_SEC = {'method', 'approach', 'model', 'architecture', 'framework',
                               'algorithm', 'network', 'design', 'system', 'proposed'}
                    _E_SEC = {'experiment', 'result', 'evaluation', 'analysis',
                               'ablation', 'benchmark', 'comparison'}

                    # Tier 0: check explicit "Figure N" citation in each subsection's OWN text.
                    # This is the ONLY reliable signal — keyword overlap alone is misleading
                    # because caption words of an overview figure (e.g. "attention", "part",
                    # "segmentation") may happen to match a specific later subsection title
                    # ("Attention Control for Ambiguity and Omission") even when that figure
                    # is NOT primarily presented there.
                    _direct_idxs: list = []
                    if _cite_re_sub is not None:
                        for _si, sub in enumerate(suitable_subsections):
                            own_text = (sub.get('content') or '').lower()
                            if _cite_re_sub.search(own_text):
                                _direct_idxs.append(_si)

                    if len(_direct_idxs) == 1:
                        # Unambiguous direct citation → assign there immediately.
                        _asset_label = "Table" if is_table else "Figure"
                        print(f"   🔍 [direct-cite] {_asset_label} cap#{cap_number} → "
                              f"'{suitable_subsections[_direct_idxs[0]]['title']}'")
                        return _direct_idxs[0]

                    def sub_score(sub_idx, sub):
                        title_l   = sub['title'].lower()
                        content_l = sub['content'].lower()
                        t_words = {w.strip('.,;:()') for w in title_l.split()   if len(w.strip('.,;:()')) > 3}
                        c_words = {w.strip('.,;:()') for w in content_l.split() if len(w.strip('.,;:()')) > 3}
                        score  = 5 * len(cap_words & t_words)
                        score += 1 * len(cap_words & (c_words - t_words))
                        # Type-match bonus
                        if fig_type == 'method'     and any(kw in title_l for kw in _M_SEC):
                            score += 50
                        if fig_type == 'experiment' and any(kw in title_l for kw in _E_SEC):
                            score += 50
                        # Citation bonus dominates keyword overlap
                        if sub_idx in _direct_idxs:
                            score += 200
                        return score

                    if not _direct_idxs:
                        # No subsection directly cites this figure.
                        # Use keyword scoring (title ×5 + content ×1) to find the best
                        # matching subsection.  If there is a clear title-word match the
                        # figure belongs there; otherwise fall back to the first subsection
                        # (safe default for overview figures that introduce a whole section).
                        if cap_words:
                            _nc_scores = [sub_score(si, sub) for si, sub in enumerate(suitable_subsections)]
                            _nc_best = max(_nc_scores)
                            if _nc_best > 0:
                                _nc_idx = _nc_scores.index(_nc_best)
                                _asset_label = "Table" if is_table else "Figure"
                                print(f"   🔍 [no-cite → score({_nc_best})] {_asset_label} cap#{cap_number} → "
                                      f"'{suitable_subsections[_nc_idx]['title']}'")
                                return _nc_idx
                        _asset_label = "Table" if is_table else "Figure"
                        print(f"   🔍 [no-cite → first] {_asset_label} cap#{cap_number} → "
                              f"'{suitable_subsections[0]['title']}' (no direct subsection cite, no keyword match)")
                        return 0

                    # 2+ direct cites: use scoring with a strong citation bonus to break ties.
                    if not cap_words:
                        return _direct_idxs[0]

                    scores = [sub_score(si, sub) for si, sub in enumerate(suitable_subsections)]
                    return scores.index(max(scores)) if scores else 0

                for figure_id in section_assets.get("figures", []):
                    if figure_id not in filtered_figures:
                        continue
                    fig_data = filtered_figures[figure_id]
                    cap = fig_data.get('caption', '')
                    cap_num = fig_data.get('caption_number')
                    # If this figure's number is lower than every figure already in any
                    # subsection, keep it at section level. process_section renders
                    # section-level content before all subsections, so this guarantees
                    # it appears in the poster before those higher-numbered figures.
                    # Figures that are NOT lower-than-all are distributed normally —
                    # the score algorithm places them in the semantically best subsection,
                    # and since their numbers are higher than earlier subsection figures,
                    # the poster order remains correct.
                    _keep_at_section = False
                    if cap_num is not None:
                        try:
                            _cur_num = int(str(cap_num))
                            _sub_all = []
                            for _si in suitable_subsections:
                                _sub_all.extend(
                                    _si['subsection'].get('assets', {}).get('figures', [])
                                )
                            if _sub_all:
                                _min_sub = min(
                                    int(str(f)) for f in _sub_all if str(f).isdigit()
                                )
                                if _cur_num < _min_sub:
                                    _keep_at_section = True
                                    print(f"   📌 Figure {figure_id} (cap#{cap_num}) kept at "
                                          f"section level (#{_cur_num} < min subsection "
                                          f"fig #{_min_sub})")
                        except (ValueError, TypeError):
                            pass
                    if _keep_at_section:
                        continue
                    idx = _best_subsection_idx(cap, cap_number=cap_num)
                    tgt = suitable_subsections[idx]
                    tgt.setdefault('assigned_figures', []).append(figure_id)
                    print(f"   ✅ Figure {figure_id} (cap#{cap_num}) → '{tgt['title']}'")

                for table_id in section_assets.get("tables", []):
                    if table_id not in filtered_tables:
                        continue
                    tbl_data = filtered_tables[table_id]
                    cap = tbl_data.get('caption', '')
                    cap_num = tbl_data.get('caption_number')
                    idx = _best_subsection_idx(cap, cap_number=cap_num, is_table=True)
                    tgt = suitable_subsections[idx]
                    tgt.setdefault('assigned_tables', []).append(table_id)
                    print(f"   📊 Table {table_id} → '{tgt['title']}'")

                # Register section-level kept figures in section_panel["assets"] so that
                # ensure_all_assets_assigned can find them when traversing the refined tree.
                # Figures distributed to subsections are NOT added here — they're tracked
                # in sub_info["assigned_figures"] and merged into subsection panels below.
                _distributed = set()
                for _si in suitable_subsections:
                    _distributed.update(_si.get('assigned_figures', []))
                _kept = [f for f in section_assets.get('figures', []) if f not in _distributed]
                if _kept:
                    section_panel["assets"]["figures"].extend(_kept)

            # Second pass: create subsection panels with distributed assets
            for sub_info in suitable_subsections:
                j = sub_info['index']
                subsection = sub_info['subsection']
                subsection_title = _clean_section_title(sub_info['title'])
                subsection_content = sub_info['content']
                subsection_id = sub_info['id']
                subsection_assets = subsection.get("assets", {"figures": [], "tables": [], "references": []})

                # Merge deeper levels into subsection (depth constraint for poster)
                merged_content = subsection_content
                merged_assets = {
                    'figures': list(subsection_assets.get('figures', [])),
                    'tables': list(subsection_assets.get('tables', [])),
                    'references': list(subsection_assets.get('references', [])),
                }

                # Add distributed section-level assets
                if 'assigned_figures' in sub_info:
                    merged_assets['figures'].extend(sub_info['assigned_figures'])
                if 'assigned_tables' in sub_info:
                    merged_assets['tables'].extend(sub_info['assigned_tables'])

                subsection_children = subsection.get("children", [])
                if subsection_children:
                    for subsubsection in subsection_children:
                        sub_content = subsubsection.get("content", "")
                        sub_title = subsubsection.get("title", "")

                        # Priority 1: Always exclude unsuitable sub-subsections
                        is_unsuitable, unsuitable_type = is_poster_unsuitable_section(sub_title, sub_content)
                        if is_unsuitable:
                            print(f"     🚫 Excluding {unsuitable_type} sub-subsection '{sub_title}'")
                            continue

                        # Filter sub-subsection content for poster suitability
                        if is_poster_suitable_section(sub_title, sub_content, "", paper_keywords):
                            if sub_content.strip():
                                merged_content += "\n\n" + sub_content

                            # Merge assets from deeper levels
                            sub_assets = subsubsection.get("assets", {"figures": [], "tables": [], "references": []})
                            for asset_type in ['figures', 'tables', 'references']:
                                merged_assets[asset_type].extend(sub_assets.get(asset_type, []))

                # Skip if no content after filtering
                if not merged_content.strip():
                    continue

                # Use LLM summary if available
                if subsection_id in summaries:
                    content_summary = summaries[subsection_id]
                    print(f"   📝 Using LLM summary for {subsection_id}: {subsection_title}")
                else:
                    content_summary = summarize_text_content(merged_content, target_sentences=5)
                    print(f"   📝 Using fallback summary for {subsection_id}: {subsection_title}")

                # Apply poster-optimized asset assignment with uniqueness constraint
                filtered_subsection_assets = apply_asset_assignment_for_poster(
                    merged_assets, filtered_figures, filtered_tables, subsection_title
                )

                # Calculate layout parameters
                text_len, asset_path = calculate_layout_parameters(
                    content_summary, filtered_subsection_assets, filtered_figures, filtered_tables
                )

                # Create subsection panel (leaf node)
                subsection_panel = {
                    "panel_id": f"{panel_id_counter}{subsection_id_counter}",  # 🔧 Concatenated numeric format: "11", "12", "21", "22"
                    "section_name": subsection_title,
                    "tp": 0, # Leaf nodes have no tp/gp
                    "text_len": text_len,
                    "gp": 0,
                    "figure_size": 0,
                    "figure_aspect": 1,
                    "content": content_summary,
                    "importance": 0.7,  # Default importance for subsections
                    "assets": filtered_subsection_assets,
                    "asset_path": asset_path,
                    "children": []  # Leaf node
                }

                section_panel["children"].append(subsection_panel)
                subsection_id_counter += 1
        else:
            # Section has no subsections - section becomes a content node directly
            # Double-check: exclude unsuitable sections even at this level
            is_unsuitable, unsuitable_type = is_poster_unsuitable_section(section_title, section_content)
            if is_unsuitable:
                print(f"   🚫 Excluding direct {unsuitable_type} section '{section_title}'")
                continue

            # Use LLM summary if available
            if section_id in summaries:
                section_content_summary = summaries[section_id]
                print(f"   📝 Using LLM summary for {section_id}: {section_title}")
            else:
                section_content_summary = summarize_text_content(section_content, target_sentences=4)
                print(f"   📝 Using fallback summary for {section_id}: {section_title}")

            section_assets = child.get("assets", {"figures": [], "tables": [], "references": []})
            filtered_section_assets = apply_asset_assignment_for_poster(
                section_assets, filtered_figures, filtered_tables, section_title
            )

            # Calculate layout parameters
            text_len, asset_path = calculate_layout_parameters(
                section_content_summary, filtered_section_assets, filtered_figures, filtered_tables
            )

            # Update section panel to be a content node directly (no subsections)
            section_panel.update({
                "node_type": "content",  # Mark as content node
                "tp": 0,  # Leaf nodes have no tp/gp
                "text_len": text_len,
                "gp": 0,
                "figure_size": 0,
                "figure_aspect": 1,
                "content": section_content_summary,
                "assets": filtered_section_assets,
                "asset_path": asset_path,
                "children": []  # No children as it's a content node
            })

            print(f"   ✅ Section '{section_title}' converted to direct content node (no subsections)")

        # Add section if it has subsections (children) OR if it has content directly
        has_children = len(section_panel["children"]) > 0
        has_content = section_panel.get("content", "").strip() != ""

        if has_children or has_content:
            refined_tree["children"].append(section_panel)
            panel_id_counter += 1
        else:
            print(f"   🚫 Excluding section '{section_title}': no content or subsections after filtering")

    # Ensure all important assets are assigned with uniqueness constraint
    ensure_all_assets_assigned(refined_tree, filtered_figures, filtered_tables)

    # Apply section-level asset limits with priority scoring
    print(f"\n🔧 Applying section-level asset limits...")
    print(f"   📏 Uniform limit: 4 assets per section")
    print(f"   🎯 Method assets: Protected with highest priority (+100 score)")
    print(f"   🔬 Experiment tables: Protected with high priority (+50 score)")
    for section in refined_tree.get('children', []):
        limit_section_assets(section, filtered_figures, filtered_tables, max_assets=4)

    print(f"\n✅ 3-level tree structure created with LLM summaries and poster filtering:")
    print(f"   📄 Root: 1 title panel")
    print(f"   📁 Sections: {len(refined_tree['children'])}")
    print(f"   📝 Subsections (leaf): {sum(len(section.get('children', [])) for section in refined_tree['children'])}")

    # Asset assignment summary (after limiting)
    def count_all_assets(node):
        """Recursively count all assets in a tree"""
        figures = len(node.get('assets', {}).get('figures', []))
        tables = len(node.get('assets', {}).get('tables', []))
        for child in node.get('children', []):
            child_figures, child_tables = count_all_assets(child)
            figures += child_figures
            tables += child_tables
        return figures, tables

    total_figures, total_tables = count_all_assets(refined_tree)

    print(f"   🖼️ Total figures assigned: {total_figures}")
    print(f"   📊 Total tables assigned: {total_tables}")
    print(f"   🔒 Asset uniqueness: Each asset appears exactly once")
    print(f"   📏 Asset limit: Maximum 4 assets per section (uniform)")
    print(f"   🎯 Method assets: Protected with highest priority (score +100)")
    print(f"   🔬 Experiment tables: Protected with high priority (score +50)")

    return refined_tree 