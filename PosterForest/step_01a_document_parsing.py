"""
STEP 1: PDF Document Parsing and Hierarchical Structure Extraction

Input: PDF file path
Output: Document tree structure + extracted assets (figure, table)
"""

import json
import os
import re
from pathlib import Path
import PIL
import torch

from dotenv import load_dotenv
from utils.src.utils import get_json_from_response
from utils.src.model_utils import parse_pdf

from camel.agents import ChatAgent
from tenacity import retry, stop_after_attempt
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from marker.models import create_model_dict
from utils.wei_utils import account_token, make_model, agent_step_with_timeout

from jinja2 import Template

from PosterForest.poster_config import IMAGE_RESOLUTION_SCALE

load_dotenv()


# ---- Step 1-1: Post-conclusion Boundary Detection ---------------------------
def detect_post_conclusion_page(docling_result):
    """Returns (page_no, boundary_seq) of the earliest post-conclusion section header.
    boundary_seq is element-precise: content before it is kept even when it shares a page with References.
    Returns (None, None) if not found.
    """
    _STOP_RE = re.compile(
        r'^\s*(references|bibliography|appendix|supplementary|supplemental'
        r'|additional\s+material|extended\s+data|acknowledgement|acknowledgment'
        r'|broader\s+impact|ethics|limitations|future\s+work)',
        re.IGNORECASE
    )
    best_page = None
    best_seq = None
    try:
        from docling_core.types.doc import SectionHeaderItem
        for seq_idx, (element, _level) in enumerate(docling_result.document.iterate_items()):
            label = getattr(element, 'label', None)
            if label is None:
                label = getattr(element, '__class__', type(element)).__name__
            text = getattr(element, 'text', None) or ''
            prov = getattr(element, 'prov', None)
            if not prov or not text:
                continue
            is_header = (
                'section' in str(label).lower()
                or 'heading' in str(label).lower()
                or isinstance(element, SectionHeaderItem)
            )
            if is_header and _STOP_RE.match(text.strip()):
                page_no = prov[0].page_no if hasattr(prov[0], 'page_no') else None
                if page_no is not None and (best_page is None or page_no < best_page):
                    best_page = page_no
                    best_seq = seq_idx
                    print(f"🛑 Post-conclusion boundary: page {page_no}, seq {seq_idx} ('{text.strip()[:60]}')")
    except Exception as e:
        print(f"   ⚠️  detect_post_conclusion_page failed: {e}")
    return best_page, best_seq


# ---- Step 1-2: Element-order Text Reconstruction ----------------------------
def rebuild_text_before_page(docling_result, boundary_seq):
    """Rebuild text up to boundary_seq (element-precise), falling back to _cut_text_by_regex."""
    _CLEAN_RE = re.compile(r"<!--[\s\S]*?-->")

    if boundary_seq is not None:
        try:
            from docling_core.types.doc import SectionHeaderItem, PictureItem, TableItem

            text_parts = []
            for seq_idx, (element, _level) in enumerate(docling_result.document.iterate_items()):
                if seq_idx >= boundary_seq:
                    break

                text = (getattr(element, 'text', None) or '').strip()
                if not text or isinstance(element, (PictureItem, TableItem)):
                    continue

                if isinstance(element, SectionHeaderItem):
                    h = getattr(element, 'level', 2)
                    try:
                        h = max(1, min(int(h), 6))
                    except (TypeError, ValueError):
                        h = 2
                    text_parts.append(f'\n{"#" * h} {text}\n')
                else:
                    text_parts.append(text)

            rebuilt = '\n'.join(text_parts)
            if len(rebuilt.strip()) > 300:
                print(f"   ✂️  Text up to seq {boundary_seq}: {len(rebuilt):,} chars")
                return rebuilt
            print(f"   ⚠️  Seq-filtered text too short ({len(rebuilt.strip())} chars); falling back to regex cut")

        except Exception as e:
            print(f"   ⚠️  rebuild_text_before_page failed ({e}); falling back to regex cut")

    # Fallback: export full markdown, cut by regex
    full_markdown = _CLEAN_RE.sub("", docling_result.document.export_to_markdown())
    return _cut_text_by_regex(full_markdown)


def _cut_text_by_regex(text_content):
    """Regex-based fallback: find the earliest post-conclusion heading in markdown and cut there."""
    _PATTERNS = [
        (r'\n#+\s*References\s*\n',            'References'),
        (r'\n#+\s*REFERENCES\s*\n',            'References'),
        (r'\n#+\s*Bibliography\s*\n',          'Bibliography'),
        (r'\n\*\*References\*\*\s*\n',         'References'),
        (r'\nReferences\s*\n=+\s*\n',          'References'),
        (r'\n\s*References\s*\n',              'References'),
        (r'\n#+\s*Appendix\b',                 'Appendix'),
        (r'\n#+\s*Supplementary\b',            'Supplementary'),
        (r'\n#+\s*Supplemental\b',             'Supplemental'),
        (r'\n\*\*Appendix\b',                  'Appendix'),
        (r'\nAppendix\s*\n[-=]+\s*\n',         'Appendix'),
        (r'\n#+\s*Acknowledgements?\s*\n',     'Acknowledgements'),
        (r'\n\*\*Acknowledgements?\*\*\s*\n',  'Acknowledgements'),
        (r'\n#+\s*Broader\s+Impact\b',         'Broader Impact'),
        (r'\n#+\s*Ethics\b',                   'Ethics'),
        (r'\n#+\s*Limitations\b',              'Limitations'),
        (r'\n#+\s*Future\s+Work\b',            'Future Work'),
        (r'\n#+\s*Additional\s+Material\b',    'Additional Material'),
        (r'\n#+\s*Extended\s+Data\b',          'Extended Data'),
    ]
    cut_pos, found_label = len(text_content), None
    for pattern, label in _PATTERNS:
        m = re.search(pattern, text_content, re.IGNORECASE)
        if m and m.start() < cut_pos:
            cut_pos, found_label = m.start(), label

    if found_label:
        print(f"   ✂️  Regex cut at '{found_label}': {len(text_content):,} → {cut_pos:,} chars")
        return text_content[:cut_pos]
    print(f"   ℹ️  No post-conclusion boundary found in text ({len(text_content):,} chars)")
    return text_content


pipeline_options = PdfPipelineOptions()
pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
pipeline_options.generate_page_images = True
pipeline_options.generate_picture_images = True

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)


# ---- Step 1-4: Citation-based Asset Reassignment ----------------------------
def _correct_asset_assignments_from_citations(document_tree, body_text, figures, tables):
    """Move assets from generic sections (Introduction, Abstract) to the most specific
    section that explicitly cites them ("Figure N", "Table N") in the raw body text.
    Called right after LLM builds document_tree, while full body_text is available.
    """
    # Background/Preliminary/Motivation are valid destinations; only intro/abstract are too vague.
    _SOURCE_GENERIC_RE = re.compile(
        r'^(introduction|abstract)',
        re.IGNORECASE,
    )
    # Related Works is excluded: it's always pruned from the poster tree, so assets there would be lost.
    _DEST_EXCLUDE_RE = re.compile(
        r'^(introduction|abstract|related.?works?|background)',
        re.IGNORECASE,
    )

    _INTRO_CAPTION_KW = {
        'challenge', 'problem', 'motivation', 'teaser', 'limitation',
        'failure', 'insight', 'illustrate', 'example', 'comparison',
        'disadvantage', 'shortcoming', 'observation', 'gap', 'difficult',
        'issue', 'highlight', 'inspire', 'demonstrate',
    }

    def is_source_generic(title: str) -> bool:
        return title == '' or bool(_SOURCE_GENERIC_RE.match(title.strip()))

    def is_valid_dest(title: str) -> bool:
        return not bool(_DEST_EXCLUDE_RE.match(title.strip()))

    _HDG_RE = re.compile(r'^#{1,4}\s+(.+?)$', re.MULTILINE)
    _NUM_PREFIX_RE = re.compile(r'^[\d]+(?:\.[\d]+)*\.?\s*')

    def _norm_title(t: str) -> str:
        return _NUM_PREFIX_RE.sub('', t).strip().lower()

    headings = list(_HDG_RE.finditer(body_text))
    section_texts: dict = {}
    for idx, h in enumerate(headings):
        heading_text = h.group(1).strip()
        start = h.end()
        end = headings[idx + 1].start() if idx + 1 < len(headings) else len(body_text)
        body = body_text[start:end]
        section_texts[_norm_title(heading_text)] = body
        if heading_text.lower() != _norm_title(heading_text):
            section_texts[heading_text.lower()] = body

    _FIG_RE = re.compile(r'\bfig(?:ure)?s?\.?\s*(\d+)\b', re.IGNORECASE)
    _TBL_RE = re.compile(r'\btab(?:le)?s?\.?\s*(\d+)\b', re.IGNORECASE)

    def sections_citing(fig_num, asset_type: str):
        if not fig_num:
            return []
        num = str(fig_num)
        pat = _FIG_RE if asset_type == 'figure' else _TBL_RE
        return [t for t, txt in section_texts.items()
                if any(m.group(1) == num for m in pat.finditer(txt))]

    def find_holding_node(node, asset_id, key):
        if asset_id in node.get('assets', {}).get(key, []):
            return node
        for ch in node.get('children', []):
            r = find_holding_node(ch, asset_id, key)
            if r:
                return r
        return None

    def find_node_by_title(node, title_lower: str):
        node_raw  = node.get('title', '').lower().strip()
        node_norm = _norm_title(node.get('title', ''))
        if node_raw == title_lower or node_norm == title_lower:
            return node
        for ch in node.get('children', []):
            r = find_node_by_title(ch, title_lower)
            if r:
                return r
        return None

    def depth_of(root, target, d=0):
        if root is target:
            return d
        for ch in root.get('children', []):
            r = depth_of(ch, target, d + 1)
            if r is not None:
                return r
        return None

    # DFS order estimates each section's relative position (used for position affinity scoring).
    _dfs_order: dict = {}
    _counter = [0]
    def _compute_dfs_order(node):
        _dfs_order[id(node)] = _counter[0]
        _counter[0] += 1
        for ch in node.get('children', []):
            _compute_dfs_order(ch)
    _compute_dfs_order(document_tree)
    _total_nodes = max(_counter[0] - 1, 1)

    total_assets_count = len(figures) + len(tables)

    all_assets = (
        [(fid, fd, 'figures', 'figure') for fid, fd in figures.items()] +
        [(tid, td, 'tables',  'table')  for tid, td in tables.items()]
    )

    for asset_id, asset_data, asset_key, asset_type in all_assets:
        fig_num = asset_data.get('caption_number')

        current_node = find_holding_node(document_tree, asset_id, asset_key)
        if current_node is None:
            continue
        current_section = current_node.get('title', '').lower().strip()
        if current_node is not document_tree and not is_source_generic(current_section):
            continue  # Already in a specific section - trust the LLM

        # Keep early/motivational figures in Introduction; but method/architecture figures
        # should still be moved even if they appear early (e.g. Transformer Figure 1).
        caption_text = asset_data.get('caption', '').lower()
        is_motivational = any(kw in caption_text for kw in _INTRO_CAPTION_KW)
        try:
            is_early = fig_num is not None and int(fig_num) <= 3
        except (TypeError, ValueError):
            is_early = False
        is_in_intro = 'introduction' in current_section

        _METHOD_CAPTION_KW = {
            'architecture', 'framework', 'pipeline', 'workflow', 'overview',
            'model', 'network', 'structure', 'diagram', 'system', 'design',
            'approach', 'method', 'algorithm', 'scheme', 'module',
        }
        # Check only first two sentences to avoid casual method mentions in comparison captions.
        caption_sents = caption_text.split('.')
        caption_head = ' '.join(caption_sents[:2]).strip() if len(caption_sents) >= 2 else caption_text
        is_method_figure = any(kw in caption_head for kw in _METHOD_CAPTION_KW)

        if is_in_intro and (is_early or is_motivational) and not is_method_figure:
            print(f"   📌 [{asset_type} {asset_id}] Keeping in Introduction "
                  f"(fig_num={fig_num}, early={is_early}, motivational={is_motivational})")
            continue

        citing = sections_citing(fig_num, asset_type)
        valid_dests = [t for t in citing if is_valid_dest(t)]
        if not valid_dests:
            continue

        try:
            asset_pos = (int(fig_num) - 1) / max(total_assets_count - 1, 1) if fig_num else 0.5
        except (TypeError, ValueError):
            asset_pos = 0.5

        # Score by depth (specificity) + position affinity (earlier figures prefer earlier sections)
        best_node, best_score = None, -float('inf')
        for title_lower in valid_dests:
            node = find_node_by_title(document_tree, title_lower)
            if node is None or node is document_tree:
                continue
            d = depth_of(document_tree, node) or 0
            node_pos = _dfs_order.get(id(node), 0) / _total_nodes
            pos_affinity = max(0.0, 1.0 - abs(asset_pos - node_pos) * 2.0)
            score = d * 2 + pos_affinity
            if score > best_score:
                best_score, best_node = score, node

        if best_node is None or best_node is current_node:
            continue

        current_node['assets'][asset_key].remove(asset_id)
        best_node.setdefault('assets', {'figures': [], 'tables': [], 'references': []})
        best_node['assets'].setdefault(asset_key, [])
        if asset_id not in best_node['assets'][asset_key]:
            best_node['assets'][asset_key].append(asset_id)
        print(f"   📌 [{asset_type} {asset_id}] "
              f"'{current_node.get('title', '')}' → '{best_node.get('title', '')}' "
              f"(citation+position, fig_num={fig_num})")


# ---- Step 1-5: Raw Content JSON Generation ----------------------------------
@retry(stop=stop_after_attempt(5))
def generate_raw_content_from_tree(document_tree, body_text, args, agent_config):
    """Generate sections JSON from document_tree using gen_poster_raw_content_v2 template."""
    import random

    template = Template(open("utils/prompts/gen_poster_raw_content_v2.txt").read())

    model = make_model(agent_config)
    sys_msg = 'You are the author of the paper, and you will create a poster for the paper.'
    agent = ChatAgent(
        system_message=sys_msg,
        model=model,
        message_window_size=10,
        token_limit=agent_config.get('token_limit', None)
    )

    while True:
        prompt = template.render(
            markdown_document=body_text,
        )
        agent.reset()
        response = agent_step_with_timeout(agent, prompt)
        in_tok, out_tok = account_token(response)

        sections_json = get_json_from_response(response.msgs[0].content)

        if len(sections_json) > 0 and 'sections' in sections_json:
            break
        print('Error: Empty response from gen_poster_raw_content_v2 template, retrying...')
        if args.model_name_t.startswith('vllm_qwen'):
            body_text = body_text[:max(4000, int(len(body_text) * 0.7))]

    if len(sections_json['sections']) > 9:
        # cap at 9: first 2 + random 5 from middle + last 2
        selected_sections = sections_json['sections'][:2] + random.sample(sections_json['sections'][2:-2], 5) + sections_json['sections'][-2:]
        sections_json['sections'] = selected_sections

    has_title = False
    for section in sections_json['sections']:
        if type(section) != dict or not 'title' in section or not 'content' in section:
            print(f"Ouch! The response is invalid, the LLM is not following the format :(")
            print('Trying again...')
            raise Exception("Invalid section format")
        if 'title' in section['title'].lower():
            has_title = True

    if not has_title:
        print('Ouch! The response is invalid, the LLM is not following the format :(')
        raise Exception("No title section found")

    return sections_json, in_tok, out_tok


# ---- Step 1 [Entry]: PDF → Document Tree ------------------------------------
# 1-1  detect_post_conclusion_page      → boundary (page_no, seq)
# 1-2  rebuild_text_before_page         → clean body text (fallback: _cut_text_by_regex)
# 1-3  extract_figures_and_tables       → figure/table dicts
# 1-4  _correct_asset_assignments_from_citations
# 1-5  generate_raw_content_from_tree
@retry(stop=stop_after_attempt(5))
def parse_hierarchical_document(args, agent_config, step_dir=None):
    """Caption-guided hierarchical document parsing.
    Returns (in_tok, out_tok, document_tree, docling_result, sections_json_path, figures, tables, full_body_text).
    """

    # ---- 1-1: parse PDF -----------------------------------------------------
    pdf_path = args.paper_path
    docling_result = doc_converter.convert(pdf_path)
    post_conclusion_page, boundary_seq = detect_post_conclusion_page(docling_result)
    if boundary_seq is not None:
        print(f"   📌 Boundary: seq {boundary_seq} (page {post_conclusion_page}) - content before this point kept")
    else:
        print(f"   📌 No post-conclusion boundary found; regex fallback will apply to text")

    # ---- 1-2: build body text -----------------------------------------------
    doc_markdown = docling_result.document.export_to_markdown()
    html_comment_re = re.compile(r"<!--[\s\S]*?-->")
    body_text = html_comment_re.sub("", doc_markdown)

    if len(body_text) < 500:
        print('\n⚠️ Parsing with docling failed, using marker instead\n')
        marker_model = create_model_dict(device='cuda', dtype=torch.float16)
        body_text, _ = parse_pdf(pdf_path, model_lst=marker_model, save_file=False)
        body_text = _cut_text_by_regex(body_text)
    else:
        body_text = rebuild_text_before_page(docling_result, boundary_seq)

    # ---- 1-3: extract figures & tables --------------------------------------
    from .step_01b_asset_extraction import extract_figures_and_tables

    figures, tables, _, _, _ = extract_figures_and_tables(
        args, docling_result, step_dir, boundary_seq=boundary_seq
    )

    figures_info = {}
    tables_info = {}

    print(f"\n📝 Preparing caption information for hierarchical parsing prompt:")

    for fig_id, fig_data in figures.items():
        if fig_data.get('caption'):
            figures_info[fig_id] = {
                'caption': fig_data['caption'],
                'path': fig_data['figure_path'],
                'caption_number': fig_data.get('caption_number'),
                'original_index': fig_data.get('original_index')
            }
            caption_source = f"(caption-based)" if fig_data.get('caption_number') else f"(fallback)"
            print(f"   🖼️ Figure {fig_id} {caption_source}: {fig_data['caption'][:60]}...")

    for table_id, table_data in tables.items():
        if table_data.get('caption'):
            tables_info[table_id] = {
                'caption': table_data['caption'],
                'path': table_data['table_path'],
                'caption_number': table_data.get('caption_number'),
                'original_index': table_data.get('original_index')
            }
            caption_source = f"(caption-based)" if table_data.get('caption_number') else f"(fallback)"
            print(f"   📊 Table {table_id} {caption_source}: {table_data['caption'][:60]}...")

    print(f"\n📊 Ready for caption-guided parsing: {len(figures_info)} figures, {len(tables_info)} tables")

    # ---- build document tree (LLM) ------------------------------------------
    template = Template(open("utils/prompts/hierarchical_parsing_caption_based.txt").read())

    model = make_model(agent_config)
    sys_msg = 'You are a document structure analysis expert who creates hierarchical tree representations of academic papers using figure and table captions as structural guidance.'
    agent = ChatAgent(
        system_message=sys_msg,
        model=model,
        message_window_size=10,
        token_limit=agent_config.get('token_limit', None)
    )

    print(f"\n🤖 Sending document to LLM for hierarchical structure analysis...")
    print(f"   📄 Document length: {len(body_text):,} characters")
    print(f"   🖼️ Figures to analyze: {len(figures_info)}")
    print(f"   📊 Tables to analyze: {len(tables_info)}")
    print(f"   ⏱️ Expected duration: 30-120 seconds (depends on document complexity & API response)")
    print(f"   💡 Large documents require more processing time for accurate structure analysis")

    retry_count = 0
    max_retries = 3

    if len(body_text) > 100000:
        print(f"   ⚠️  Warning: Very large document ({len(body_text):,} chars) - processing may take longer")
    if len(body_text) > 200000:
        print(f"   🚀 Tip: Consider using summarized content for faster processing if quality allows")

    # Keep full text for Step 2 citation detection; truncate only the LLM input.
    # vllm_qwen limit: ~40k chars (16384 token ctx, ~3k for output, ~2.5k for template).
    full_body_text = body_text

    original_len = len(body_text)
    if args.model_name_t.startswith('vllm_qwen') and len(body_text) > 40000:
        print(f"   ✂️  vllm_qwen context limit: truncating {original_len:,} → 40,000 chars")
        body_text = body_text[:40000]
        print(f"   📏 Truncated document size: {len(body_text):,} characters")
    elif len(body_text) > 60000:
        print(f"   ✂️  Document too large for optimal processing ({original_len:,} chars)")
        print(f"   🔄 Truncating to 60,000 characters to avoid token limits...")
        body_text = body_text[:60000]
        print(f"   📏 Truncated document size: {len(body_text):,} characters")

    while retry_count < max_retries:
        retry_count += 1
        print(f"   🔄 Attempt {retry_count}/{max_retries}: Calling LLM API (model: {args.model_name_t})...")

        prompt = template.render(
            markdown_document=body_text,
            figures_info=figures_info,
            tables_info=tables_info
        )
        prompt_len = len(prompt)
        print(f"   🔄 Calling LLM API (model: {args.model_name_t})...")
        print(f"      📏 Prompt size: {prompt_len:,} characters (~{prompt_len//1000}KB)")

        agent.reset()
        response = agent_step_with_timeout(agent, prompt)
        in_tok, out_tok = account_token(response)
        print(f"   ✅ LLM response received (tokens: {in_tok} → {out_tok})")

        response_str = response.msgs[0].content
        print(f"   📝 Response length: {len(response_str):,} characters")

        if out_tok >= 16000:
            print(f"   ⚠️  Warning: Response may be truncated (near token limit: {out_tok} tokens)")

        try:
            tree_json = get_json_from_response(response_str)
        except Exception as e:
            print(f"   ❌ JSON parsing error: {str(e)}")
            print(f"   🔍 Response ends with: ...{response_str[-100:]}")
            tree_json = {}

        if len(tree_json) > 0 and 'tree' in tree_json:
            print(f"   ✅ Valid hierarchical tree structure received!")
            break

        print(f'   ❌ Attempt {retry_count} failed: Invalid hierarchical parsing response')

        if retry_count < max_retries:
            if args.model_name_t.startswith('vllm_qwen'):
                truncated_len = max(8000, int(len(body_text) * 0.70))
                print(f"   🔄 vllm_qwen retry: reducing {len(body_text):,} → {truncated_len:,} chars")
                body_text = body_text[:truncated_len]
            elif out_tok >= 16000:
                truncated_len = int(len(body_text) * 0.8)
                print(f"   ✂️  Token limit reached, reducing document size to {truncated_len:,} chars")
                body_text = body_text[:truncated_len]
            print(f"   🔁 Retrying in 3 seconds...")
            import time
            time.sleep(3)
        else:
            print(f"   🚫 Maximum retries ({max_retries}) reached!")
            print(f"   💡 Suggestion: Try with a shorter document or different model")
            raise RuntimeError("Failed to get valid hierarchical parsing response after maximum retries")

    document_tree = tree_json['tree']

    # ---- 1-4: reassign assets by citation -----------------------------------
    print(f"\n📌 Post-processing asset assignments from raw citations...")
    _correct_asset_assignments_from_citations(document_tree, body_text, figures, tables)

    # Remove bare caption-titled sections ("Table 3", "Figure 1") and strip caption
    # prefixes from section titles ("Figure 2: Overview" → "Overview").
    _FAKE_BARE_RE = re.compile(
        r'^(table|figure|fig\.?|appendix|supplementary|supplement)\s*\d*\.?\s*$',
        re.IGNORECASE,
    )
    _FAKE_CAPTION_RE = re.compile(
        r'^(?:table|figure|fig\.?)\s*\d+[\.:]\s*(.+)$',
        re.IGNORECASE,
    )

    def _remove_fake_sections(node):
        """Recursively clean Table/Figure-titled sections."""
        children = node.get('children', [])
        new_children = []
        for child in children:
            title = (child.get('title') or child.get('section_name') or '').strip()
            m_caption = _FAKE_CAPTION_RE.match(title)
            if _FAKE_BARE_RE.match(title):
                for key in ('figures', 'tables', 'references'):
                    for aid in child.get('assets', {}).get(key, []):
                        node.setdefault('assets', {'figures': [], 'tables': [], 'references': []})
                        node['assets'].setdefault(key, [])
                        if aid not in node['assets'][key]:
                            node['assets'][key].append(aid)
                for gc in child.get('children', []):
                    _remove_fake_sections(gc)
                    new_children.append(gc)
                print(f"   🗑️  Removed fake section '{title}' (assets promoted to parent)")
            elif m_caption:
                clean_title = m_caption.group(1).strip()
                child['title'] = clean_title
                if 'section_name' in child:
                    child['section_name'] = clean_title
                print(f"   ✏️  Renamed caption-title '{title}' → '{clean_title}'")
                _remove_fake_sections(child)
                new_children.append(child)
            else:
                _remove_fake_sections(child)
                new_children.append(child)
        node['children'] = new_children

    _remove_fake_sections(document_tree)
    print(f"   ✅ Fake section cleanup complete")

    print(f"\n📋 STEP 1 COMPLETE: Hierarchical Document Tree Structure")
    print("="*80)
    from .tree_visualization import print_tree_unified
    print_tree_unified(document_tree, tree_type="document", show_details=True, figures=figures, tables=tables)
    print("="*80)
    print(f"📊 Parsing result: {len(figures)} figures, {len(tables)} tables extracted")
    if figures:
        print(f"🔍 Figure IDs: {list(figures.keys())[:5]}{'...' if len(figures) > 5 else ''}")
    if tables:
        print(f"🔍 Table IDs: {list(tables.keys())[:5]}{'...' if len(tables) > 5 else ''}")

    # ---- 1-5: generate sections JSON (backward compatibility) ---------------
    sections_in_tok, sections_out_tok = 0, 0
    try:
        sections_json, sections_in_tok, sections_out_tok = generate_raw_content_from_tree(document_tree, body_text, args, agent_config)
    except Exception as e:
        print(f"Warning: Failed to generate sections JSON using template: {e}")
        sections_json = {"sections": []}

        title_section = {
            "title": document_tree.get("title", "Title"),
            "content": document_tree.get("content", "")[:500]
        }
        sections_json["sections"].append(title_section)

        children = document_tree.get("children", [])
        for child in children:
            section = {
                "title": child.get("title", "Section"),
                "content": child.get("content", "")
            }
            if "children" in child and child["children"]:
                for subsection in child["children"]:
                    section["content"] += "\n\n" + subsection.get("content", "")
            sections_json["sections"].append(section)

    if step_dir:
        sections_json_path = os.path.join(step_dir, f'{args.poster_name}_raw_content.json')
    else:
        os.makedirs('contents', exist_ok=True)
        sections_json_path = f'contents/<{args.model_name_t}_{args.model_name_v}>_{args.poster_name}_raw_content_{args.index}.json'

    with open(sections_json_path, 'w', encoding='utf-8') as f:
        json.dump(sections_json, f, indent=4, ensure_ascii=False)

    # Save pre-truncation text so Step 2 citation detection can search all "Figure N" mentions.
    if step_dir:
        body_text_path = os.path.join(step_dir, f'{args.poster_name}_text_content.txt')
        with open(body_text_path, 'w', encoding='utf-8') as f:
            f.write(full_body_text)

    total_in_tok = in_tok + sections_in_tok
    total_out_tok = out_tok + sections_out_tok

    return total_in_tok, total_out_tok, document_tree, docling_result, sections_json_path, figures, tables, full_body_text
