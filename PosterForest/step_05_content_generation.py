"""
STEP 5: Bullet-point Content Generation

Reads the tree-split JSON produced by Step 4 and generates formatted
bullet-point text for every panel via LLM.  Title and author panels are
handled by a dedicated agent; body panels are processed in parallel threads.

Input:  tree_split JSON (Step 4) + raw_content JSON (Step 1) for fallback text
Output: text_arrangement list with 'content_for_ppt' populated per textbox
"""

from dotenv import load_dotenv
from utils.src.utils import get_json_from_response
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import re

from camel.agents import ChatAgent
from PIL import Image

from utils.wei_utils import *

from utils.pptx_utils import *
from utils.critic_utils import *
import yaml
from jinja2 import Environment, StrictUndefined
load_dotenv()


def _normalize_title(s):
    """Lowercase, strip leading section numbers and punctuation for fuzzy matching."""
    s = s.lower()
    s = re.sub(r'^\d+[\.\d\s]*', '', s)   # strip "2.", "2.1 ", etc.
    s = re.sub(r'[^a-z\s]', '', s)
    return s.strip()


def _find_best_section_text(panel_name, raw_sections):
    """
    Return the raw_content section text that best matches panel_name.
    raw_sections is a list of {"title": ..., "content": ...} dicts.
    Returns None if no reasonable match found.

    Uses Jaccard overlap on normalized words.  Threshold lowered to 0.10
    (from 0.15) so subsection names like "Multi-Head Attention" still match
    raw paper section titles even when word overlap is partial.
    Falls back to partial-word containment when full-word Jaccard is weak,
    which helps with numbered titles like "3.2 Multi-Head Attention".
    """
    norm_target = _normalize_title(panel_name)
    target_words = set(norm_target.split())
    if not target_words:
        return None

    best_score, best_content = 0.0, None
    for sec in raw_sections:
        norm_title = _normalize_title(sec.get('title', ''))
        title_words = set(norm_title.split())
        if not title_words:
            continue
        overlap = len(target_words & title_words)
        union = len(target_words | title_words)
        score = overlap / union if union else 0
        # Containment bonus: if ALL target words appear in title, boost score.
        # Handles "Attention" matching "Multi-Head Attention" where Jaccard is low.
        if target_words and target_words.issubset(title_words):
            score = max(score, 0.25)
        if score > best_score:
            best_score = score
            best_content = sec.get('content', '')

    return best_content if best_score > 0.10 else None


def _sentences_to_bullets(text, max_sentences=10):
    """Split raw text into bullet-point JSON entries.

    Tries three progressively coarser strategies so we always get content:
    1. Split on sentence-ending punctuation (handles ". ", "! ", "? ").
    2. Split on newlines / semicolons.
    3. Chunk into ~120-char word groups.
    """
    text = text.strip()
    # Strategy 1: sentence boundaries
    parts = re.split(r'(?<=[.!?])\s+', text)
    sentences = [p.strip() for p in parts if len(p.strip()) > 12][:max_sentences]

    if len(sentences) < 3:
        # Strategy 2: newlines or semicolons
        parts = re.split(r'[\n;]', text)
        sentences = [p.strip() for p in parts if len(p.strip()) > 12][:max_sentences]

    if len(sentences) < 3:
        # Strategy 3: word chunks ~120 chars
        words = text.split()
        sentences, chunk = [], []
        for w in words:
            chunk.append(w)
            if len(' '.join(chunk)) >= 120:
                sentences.append(' '.join(chunk))
                chunk = []
        if chunk:
            sentences.append(' '.join(chunk))
        sentences = sentences[:max_sentences]

    return [
        {"alignment": "left", "bullet": True, "level": 0,
         "runs": [{"text": s.rstrip('.') + '.', "bold": False}]}
        for s in sentences
    ]


# ---- Step 5-3: Single Section Bullet Generation ----------------------------
def gen_content_process_section(
    section_name, 
    outline, 
    raw_content, 
    raw_outline, 
    template, 
    create_actor_agent, 
    MAX_ATTEMPT
):
    """
    Process a single section in its own thread or process.
    Returns (section_name, result_json, total_input_token, total_output_token).
    """
    # Create a fresh ActorAgent instance for each parallel call
    actor_agent = create_actor_agent()

    section_outline = ''
    num_attempts = 0
    total_input_token = 0
    total_output_token = 0
    result_json = None

    while True:
        print(f"[Thread] Generating content for section: {section_name}")

        if len(section_outline) == 0:
            # Initialize the section outline
            section_outline = json.dumps(outline[section_name], indent=4)

        # Render prompt using Jinja template
        jinja_args = {
            'json_outline': section_outline,
            'json_content': raw_content,
        }
        prompt = template.render(**jinja_args)

        # Step the actor_agent and track tokens
        response = agent_step_with_timeout(actor_agent, prompt)
        input_token, output_token = account_token(response)
        total_input_token += input_token
        total_output_token += output_token

        # Parse JSON and possibly adjust text length
        result_json = get_json_from_response(response.msgs[0].content)
        new_section_outline, suggested = generate_length_suggestions(
            result_json,
            json.dumps(outline[section_name]),
            raw_outline[section_name]
        )
        section_outline = json.dumps(new_section_outline, indent=4)

        if not suggested:
            # No more adjustments needed
            break

        print(f"[Thread] Adjusting text length for section: {section_name}...")

        num_attempts += 1
        if num_attempts >= MAX_ATTEMPT:
            break

    return section_name, result_json, total_input_token, total_output_token


# ---- Step 5-2: Parallel Section Processing ----------------------------------
def gen_content_parallel_process_sections(
    sections,
    outline,
    raw_content,
    raw_outline,
    template,
    create_actor_agent,
    MAX_ATTEMPT=3
):
    """
    Parallelize the section processing using ThreadPoolExecutor.
    """
    poster_content = {}
    total_input_token = 0
    total_output_token = 0

    # Create a pool of worker threads (or processes)
    with ThreadPoolExecutor() as executor:
        futures = []

        # Submit each section to be processed in parallel
        for section_name in sections:
            futures.append(
                executor.submit(
                    gen_content_process_section, 
                    section_name,
                    outline,
                    raw_content,
                    raw_outline,
                    template,
                    create_actor_agent,
                    MAX_ATTEMPT
                )
            )

        # Collect results as they complete
        for future in as_completed(futures):
            section_name, result_json, sec_input_token, sec_output_token = future.result()
            poster_content[section_name] = result_json
            total_input_token += sec_input_token
            total_output_token += sec_output_token

    return poster_content, total_input_token, total_output_token


# ---- Step 5-1: Poster Title & Author Generation ----------------------------
def gen_poster_title_content(args, actor_config, raw_content_path=None):
    """Generate LLM-based title content for the poster title panel."""
    total_input_token, total_output_token = 0, 0

    # Use provided path or fallback to static path for backward compatibility
    if raw_content_path is None:
        raw_content_path = f'contents/<{args.model_name_t}_{args.model_name_v}>_{args.poster_name}_raw_content_{args.index}.json'

    raw_content = json.load(open(raw_content_path, 'r'))
    actor_agent_name = 'poster_title_agent'

    title_string = raw_content.get('meta', {})

    with open(f'utils/prompt_templates/{actor_agent_name}.yaml', "r", encoding='utf-8') as f:
        content_config = yaml.safe_load(f)
    jinja_env = Environment(undefined=StrictUndefined)
    template = jinja_env.from_string(content_config["template"])

    actor_model = make_model(actor_config)

    actor_sys_msg = content_config['system_prompt']
    actor_agent = ChatAgent(
        system_message=actor_sys_msg,
        model=actor_model,
        message_window_size=30
    )

    jinja_args = {
        'title_string': title_string,
    }
    prompt = template.render(**jinja_args)
    # Step the actor_agent and track tokens
    actor_agent.reset()
    response = agent_step_with_timeout(actor_agent, prompt)
    input_token, output_token = account_token(response)
    total_input_token += input_token
    total_output_token += output_token
    result_json = get_json_from_response(response.msgs[0].content)

    return result_json, total_input_token, total_output_token


# ---- Step 5 [Entry]: Layout + Raw Content → Bullet-point Content ------------
# Calls: 5-1 gen_poster_title_content
#        5-2 gen_content_parallel_process_sections
#             └- 5-3 gen_content_process_section  (per section, in threads)
def gen_bullet_point_content(args, actor_config, tree_split_path=None, raw_content_path=None):
    """Generate bullet-point content for all panels in parallel using LLM."""
    total_input_token_t, total_output_token_t = 0, 0
    total_input_token_v, total_output_token_v = 0, 0
    actor_agent_name = 'bullet_point_agent'

    # Use provided path or fallback to static path for backward compatibility
    if tree_split_path is None:
        tree_split_path = f'tree_splits/<{args.model_name_t}_{args.model_name_v}>_{args.poster_name}_tree_split_{args.index}.json'

    with open(tree_split_path, 'r') as f:
        tree_split_results = json.load(f)

    with open(f"utils/prompt_templates/{actor_agent_name}.yaml", "r", encoding='utf-8') as f:
        content_config = yaml.safe_load(f)

    jinja_env = Environment(undefined=StrictUndefined)
    template = jinja_env.from_string(content_config["template"])

    actor_model = make_model(actor_config)

    actor_sys_msg = content_config['system_prompt']

    actor_agent = ChatAgent(
        system_message=actor_sys_msg,
        model=actor_model,
        message_window_size=30
    )

    # Load raw_content sections for richer text lookup (avoid using 2-3 sentence tree summaries)
    raw_sections = []
    if raw_content_path and os.path.exists(raw_content_path):
        try:
            raw_content_data = json.load(open(raw_content_path, 'r'))
            raw_sections = raw_content_data.get('sections', [])
        except Exception as e:
            print(f"   ⚠️ Could not load raw_content for bullet generation: {e}")

    text_arrangement_list = tree_split_results['text_arrangement']
    text_arrangement_index = 2 # Skip the poster title and author and the first section title
    section_title_font_size = tree_split_results['section_title_font_size']
    subsection_title_font_size = tree_split_results['subsection_title_font_size']
    title_font_size = tree_split_results['title_font_size']
    title_font_size_2 = tree_split_results['title_font_size_2']
    cont_font_size = tree_split_results.get('cont_font_size', 40)
    font_name = tree_split_results.get('font_name', 'Arial')
    units_per_inch = tree_split_results.get('units_per_inch', 25)


    title_json, title_input_token, title_output_token = gen_poster_title_content(args, actor_config, raw_content_path)
    total_input_token_t += title_input_token
    total_output_token_t += title_output_token

    title_json['title'][0]['font_name'] = font_name
    title_json['title'][0]['font_size'] = title_font_size

    for item in range(len(title_json['textbox1'])):
        title_json['textbox1'][item]['font_name'] = font_name
        title_json['textbox1'][item]['font_size'] = title_font_size_2

    text_arrangement_list[0]['content_for_ppt'] = title_json['title']
    text_arrangement_list[1]['content_for_ppt'] = title_json['textbox1']


    for i in range(text_arrangement_index, len(text_arrangement_list)):
        t = text_arrangement_list[i]

        textbox_name = t['textbox_name']
        print(f'Generating bullet point content for section {textbox_name}...')

        if 'title' in t['textbox_name'].lower():
            try:
                _is_sec_title = int(t['panel_id']) < 10
            except (ValueError, TypeError):
                _is_sec_title = False
            base_fs    = section_title_font_size if _is_sec_title else subsection_title_font_size
            title_text = str(t.get('content') or '')
            # Font size is FIXED at base_fs — never shrink for long titles.
            # Instead, truncate the TEXT to fit within 2 lines at the allocated
            # box width so the title never spills into a 3rd line.
            title_w_in = t.get('width', 300) / units_per_inch
            # Approx chars per line: 0.50 × (pt / 72) in per char for bold Arial
            _char_w_in = max(0.01, 0.50 * base_fs / 72.0)
            _chars_per_line = max(8, int((title_w_in - 0.2) / _char_w_in))
            _max_chars = _chars_per_line * 2          # 2 lines maximum
            if len(title_text) > _max_chars:
                title_text = title_text[:_max_chars - 1].rstrip() + '…'
                print(f"   ✂️  Title truncated to {_max_chars} chars for '{title_text[:40]}'")
            t['content_for_ppt'] = [{"alignment": "left", "bullet": False, "level": 0,
                                      "font_name": font_name, "font_size": base_fs,
                                      "runs": [{"text": title_text, "bold": True}]}]
        else:
            # Use tree node content directly — it is already unique per panel and semantically
            # correct.  The raw_content.json sections use LLM-generated poster-level names that
            # don't align with the poster panel names produced by the tree layout, so Jaccard
            # matching between last_section_title and raw_content section titles frequently maps
            # MULTIPLE different panels to the SAME raw section → identical LLM output for all.
            tree_content = t.get('content') or ''
            content_for_bullets = tree_content

            # Supplement with raw_content only when tree content is absent or very short
            # (e.g. a title-only node whose body text was not captured by the tree).
            panel_name_match = re.search(r'p<(.+?)>', t.get('textbox_name', ''))
            panel_name = panel_name_match.group(1) if panel_name_match else ''
            if len(tree_content.strip()) < 80 and raw_sections and not panel_name.lower().startswith('content') and panel_name.lower() not in ('body', ''):
                raw_text = _find_best_section_text(panel_name, raw_sections)
                if raw_text:
                    content_for_bullets = raw_text
                    print(f"   ✅ Tree content short; supplementing from raw section '{panel_name}' ({len(raw_text)} chars)")
                else:
                    print(f"   ⚠️ Short tree content and no raw section match for '{panel_name}'")
            else:
                if tree_content:
                    print(f"   ✅ Using tree content for '{panel_name}' ({len(tree_content)} chars)")
                else:
                    print(f"   ⚠️ Empty tree content for '{panel_name}'")

            jinja_args = {
                'content_of_section': content_for_bullets,
            }

            prompt = template.render(**jinja_args)

            # Step the actor_agent and track tokens
            actor_agent.reset()
            response = actor_agent.step(prompt)
            input_token, output_token = account_token(response)
            total_input_token_t += input_token
            total_output_token_t += output_token

            result_json = get_json_from_response(response.msgs[0].content)

            # Fallback: if LLM returned empty bullets, generate from raw content directly.
            if not result_json and content_for_bullets.strip():
                result_json = _sentences_to_bullets(content_for_bullets)
                print(f"   ⚠️ Empty bullets for '{t.get('textbox_name','')}' — using content fallback ({len(result_json)} lines)")

            panel_w_in = t.get('width', 600) / units_per_inch
            panel_h_in = t.get('height', 100) / units_per_inch
            panel_font_size = cont_font_size  # uniform across all content panels

            # Bullet cap: how many bullets fit given adaptive font and panel height.
            # line_h ≈ font_pt × 1.25 / 72 inches (calibrated to match actual Arial rendering);
            # avg 1.6 lines/bullet (wrap estimate).
            line_h_in = panel_font_size * 1.25 / 72.0
            lines_per_bullet = 1.6  # conservative wrap estimate
            max_bullets = max(2, int(panel_h_in * 0.85 / (line_h_in * lines_per_bullet)))
            max_bullets = max(3, min(6, max_bullets))  # poster panels: always 3–6 bullets

            # Supplement sparse LLM output up to 3/4 of max_bullets.
            target_supplement = max(4, max_bullets * 3 // 4)
            if result_json and len(result_json) < target_supplement and content_for_bullets.strip():
                extra = _sentences_to_bullets(content_for_bullets, max_sentences=max_bullets)
                existing_texts = {b['runs'][0]['text'].lower()[:40] for b in result_json}
                for eb in extra:
                    et = eb['runs'][0]['text'].lower()[:40]
                    if et not in existing_texts:
                        result_json.append(eb)
                        existing_texts.add(et)
                    if len(result_json) >= max_bullets:
                        break
                print(f"   ➕ Supplemented sparse bullets → {len(result_json)} total")

            # Hard-cap to panel capacity.
            if result_json and len(result_json) > max_bullets:
                result_json = result_json[:max_bullets]
                print(f"   ✂️  Capped to {max_bullets} bullets (w={panel_w_in:.1f}in h={panel_h_in:.1f}in font={panel_font_size}pt)")

            for item in range(len(result_json)):
                result_json[item]['font_name'] = font_name
                result_json[item]['font_size'] = panel_font_size

            t['content_for_ppt'] = result_json

    return total_input_token_t, total_output_token_t, total_input_token_v, total_output_token_v, text_arrangement_list


