"""
STEP 2-3 Support: Asset Filtering and Deduplication Management

Filters figures/tables for poster suitability, deduplicates asset assignments,
and ensures every retained asset is placed in exactly one panel.

Input:  Raw figure/table dicts from Step 1
Output: Filtered asset dicts + deduplicated panel assignments
"""


# ---- Asset Tracker -----------------------------------------------------------

# Global asset deduplication tracker
class AssetTracker:
    def __init__(self):
        self.used_assets = set()

    def reset(self):
        self.used_assets.clear()

    def _get_namespaced_id(self, asset_id, asset_type):
        """Create a namespaced ID to avoid conflicts between figures and tables"""
        return f"{asset_type}:{asset_id}"

    def resolve_asset_id(self, asset_id, asset_type, filtered_figures, filtered_tables):
        """Resolve an asset ID via direct match, string coercion, or numeric extraction."""
        target_dict = filtered_figures if asset_type == 'figure' else filtered_tables

        if asset_id in target_dict:
            return asset_id

        str_id = str(asset_id)
        if str_id in target_dict:
            return str_id

        import re
        if isinstance(asset_id, str):
            for num in re.findall(r'\d+', asset_id):
                if num in target_dict:
                    return num

        return None

    def try_assign_asset(self, asset_id, asset_type, filtered_figures, filtered_tables):
        """Try to assign asset, checking for duplicates"""
        resolved_id = self.resolve_asset_id(asset_id, asset_type, filtered_figures, filtered_tables)

        if not resolved_id:
            return None

        # Use namespaced ID to avoid figure/table conflicts
        namespaced_id = self._get_namespaced_id(resolved_id, asset_type)
        if namespaced_id in self.used_assets:
            return None

        # Don't add to used_assets here - let the caller do it when actually assigned
        return resolved_id

    def mark_asset_as_used(self, asset_id, asset_type):
        """Mark an asset as used after successful assignment."""
        self.used_assets.add(self._get_namespaced_id(asset_id, asset_type))

    def is_asset_used(self, asset_id, asset_type):
        """Check if an asset is already used"""
        namespaced_id = self._get_namespaced_id(asset_id, asset_type)
        return namespaced_id in self.used_assets


# Global tracker instance
asset_tracker = AssetTracker()


# ---- Step 2 [Entry]: Raw Assets → Filtered + Assigned Assets -----------------
# 2-1 filter_assets_for_poster          → scored and capped figure/table dicts
# 2-2 apply_asset_assignment_for_poster → per-section dedup assignment
# 2-3 ensure_all_assets_assigned        → final unassigned-asset recovery

# ---- Step 2-1: Asset Pre-filtering ------------------------------------------
def filter_assets_for_poster(figures, tables):
    """Filter figures and tables for poster suitability with method figure priority"""
    print("🎯 Filtering assets for poster generation with method figure priority...")

    filtered_figures = {}
    filtered_tables = {}

    # Filter figures with method priority
    scored_images = []
    for img_id, img_data in figures.items():
        caption = img_data.get('caption', '').lower()
        figure_size = img_data.get('figure_size', 0)
        aspect_ratio = img_data.get('figure_aspect', 1.0)

        score = 0
        reasoning = []

        # HIGHEST PRIORITY: Method-related figures (must include almost all)
        # Note: format-descriptive words (overview, diagram, illustration, schematic)
        # are moved to a lower-priority bucket so teaser/intro figures don't get
        # incorrectly boosted into method sections.
        method_keywords = [
            'method', 'approach', 'algorithm', 'technique', 'strategy',
            'architecture', 'framework', 'model', 'system', 'design',
            'pipeline', 'workflow', 'process', 'procedure', 'scheme',
            'structure', 'network', 'component', 'module', 'flow',
        ]
        # Format-descriptive keywords: moderate priority, don't trigger method boost
        format_keywords = ['overview', 'diagram', 'illustration', 'schematic']

        method_score = 0
        for keyword in method_keywords:
            if keyword in caption:
                method_score += 5  # Very high score for method-related content
                reasoning.append(f"METHOD: {keyword}")

        # Format keywords: moderate boost, do NOT trigger method-figure extra boost
        for keyword in format_keywords:
            if keyword in caption:
                score += 3
                reasoning.append(f"FORMAT: {keyword}")

        # If this is clearly a method figure, give it maximum priority
        if method_score > 0:
            score += method_score + 10  # Extra boost for method figures
            reasoning.append("METHOD FIGURE - HIGH PRIORITY")

        # High value content (second priority)
        high_value_keywords = [
            'result', 'performance', 'comparison', 'finding',
            'evaluation', 'experiment', 'analysis', 'benchmark'
        ]

        for keyword in high_value_keywords:
            if keyword in caption:
                score += 3
                reasoning.append(f"High-value: {keyword}")

        # General research content (third priority)
        research_keywords = [
            'training', 'dataset', 'learning', 'generation',
            'validation', 'test', 'metric', 'accuracy'
        ]

        for keyword in research_keywords:
            if keyword in caption:
                score += 2
                reasoning.append(f"Research: {keyword}")

        # Visual quality bonus
        if 0.3 <= aspect_ratio <= 4.0:
            score += 1
            reasoning.append("Good aspect ratio")

        if figure_size > 50000:
            score += 1
            reasoning.append("Good resolution")

        scored_images.append({
            'id': img_id,
            'data': img_data,
            'score': score,
            'reasoning': reasoning,
            'is_method': method_score > 0
        })

    # Sort by score (method figures will be at the top)
    scored_images.sort(key=lambda x: x['score'], reverse=True)

    # Separate method figures and other figures
    method_figures = [img for img in scored_images if img['is_method']]
    other_figures = [img for img in scored_images if not img['is_method']]

    print(f"   🔍 Found {len(method_figures)} method-related figures out of {len(scored_images)} total")

    # Include ALL method figures (up to a reasonable limit)
    method_limit = min(len(method_figures), 15)  # Allow more method figures
    other_limit = max(10 - method_limit, 3)      # Reserve space for other important figures

    # Add all high-scoring method figures first
    for img in method_figures[:method_limit]:
        filtered_figures[img['id']] = img['data']
        print(f"   ✅ Method Figure {img['id']}: score={img['score']} - {', '.join(img['reasoning'][:2])}")

    # Add other high-scoring figures
    for img in other_figures:
        if img['score'] >= 1 and len(filtered_figures) < (method_limit + other_limit):
            filtered_figures[img['id']] = img['data']
            print(f"   ✅ Other Figure {img['id']}: score={img['score']} - {', '.join(img['reasoning'][:2])}")

    print(f"   📊 Total figures selected: {len(filtered_figures)} (Method: {len([img for img in method_figures[:method_limit]])}, Other: {len(filtered_figures) - len([img for img in method_figures[:method_limit]])})")

    # Filter tables (existing logic)
    scored_tables = []
    for table_id, table_data in tables.items():
        caption = table_data.get('caption', '').lower()

        score = 0
        reasoning = []

        # 🔬 HIGHEST PRIORITY: Experiment and result tables
        experiment_keywords = [
            'result', 'performance', 'comparison', 'benchmark', 'evaluation',
            'experiment', 'analysis', 'finding', 'accuracy', 'metric', 
            'precision', 'recall', 'f1', 'score', 'error', 'test',
            'ablation', 'baseline', 'state of the art', 'sota'
        ]

        experiment_score = 0
        for keyword in experiment_keywords:
            if keyword in caption:
                experiment_score += 5  # High priority for experiment tables
                reasoning.append(f"EXPERIMENT: {keyword}")

        if experiment_score > 0:
            score += experiment_score + 5  # Extra boost for experiment tables
            reasoning.append("EXPERIMENT TABLE - HIGH PRIORITY")

        # Secondary priority: Method and technical tables
        method_table_keywords = [
            'method', 'algorithm', 'approach', 'parameter', 'setting',
            'configuration', 'hyperparameter', 'detail', 'specification'
        ]

        for keyword in method_table_keywords:
            if keyword in caption:
                score += 3
                reasoning.append(f"Method: {keyword}")

        # Quality indicators (mirrors figure scoring)
        if len(caption) >= 20:  # Detailed captions are often important
            score += 2
            reasoning.append("Detailed caption")
        elif len(caption) >= 10:
            score += 1
            reasoning.append("Good caption")

        table_size = table_data.get('figure_size', 0)
        if table_size > 50000:
            score += 1
            reasoning.append("Good resolution")

        table_aspect = table_data.get('figure_aspect', 1.0)
        if 0.3 <= table_aspect <= 4.0:
            score += 1
            reasoning.append("Good aspect ratio")

        scored_tables.append({
            'id': table_id,
            'data': table_data,
            'score': score,
            'reasoning': reasoning,
            'is_experiment': experiment_score > 0
        })

    # Sort by score (experiment tables will be at the top)
    scored_tables.sort(key=lambda x: x['score'], reverse=True)

    # Separate experiment and other tables
    experiment_tables = [tbl for tbl in scored_tables if tbl['is_experiment']]
    other_tables = [tbl for tbl in scored_tables if not tbl['is_experiment']]

    print(f"   🔬 Found {len(experiment_tables)} experiment-related tables out of {len(scored_tables)} total")

    # Include ALL experiment tables (up to a reasonable limit)
    experiment_limit = min(len(experiment_tables), 6)  # Allow more experiment tables
    other_limit = max(8 - experiment_limit, 2)         # Reserve some space for other tables

    # Add all high-scoring experiment tables first
    for tbl in experiment_tables[:experiment_limit]:
        filtered_tables[tbl['id']] = tbl['data']
        print(f"   ✅ Experiment Table {tbl['id']}: score={tbl['score']} - {', '.join(tbl['reasoning'][:2])}")

    # Add other important tables
    for tbl in other_tables:
        if tbl['score'] >= 1 and len(filtered_tables) < (experiment_limit + other_limit):
            filtered_tables[tbl['id']] = tbl['data']
            print(f"   ✅ Other Table {tbl['id']}: score={tbl['score']} - {', '.join(tbl['reasoning'][:2])}")

    print(f"   📊 Total tables selected: {len(filtered_tables)} (Experiment: {len([tbl for tbl in experiment_tables[:experiment_limit]])}, Other: {len(filtered_tables) - len([tbl for tbl in experiment_tables[:experiment_limit]])})")

    return filtered_figures, filtered_tables


# ---- Step 2-4 Helper: Section Asset Assignment -------------------------------
def apply_asset_assignment_for_poster(assets, filtered_figures, filtered_tables, section_title=""):
    """
    Apply poster-optimized asset assignment with duplicate prevention
    Ensures all important assets are preserved but each asset appears only once
    """
    filtered_assets = {"figures": [], "tables": [], "references": assets.get("references", [])}

    print(f"      🔍 Processing assets for section '{section_title}':")
    print(f"         📝 Input figures: {assets.get('figures', [])}")
    print(f"         📝 Input tables: {assets.get('tables', [])}")

    # Assign all available figures (preserve all for poster, but no duplicates)
    for figure_id in assets.get("figures", []):
        print(f"      🖼️ Processing figure {figure_id}...")
        resolved_id = asset_tracker.try_assign_asset(
            figure_id, 'figure', filtered_figures, filtered_tables
        )
        if resolved_id:
            filtered_assets["figures"].append(resolved_id)
            asset_tracker.mark_asset_as_used(resolved_id, 'figure') # Mark as used after successful assignment
            print(f"      ✅ Assigned figure {resolved_id} to '{section_title}'")
        else:
            # Check why assignment failed
            check_id = asset_tracker.resolve_asset_id(figure_id, 'figure', filtered_figures, filtered_tables)
            if check_id:
                if asset_tracker.is_asset_used(check_id, 'figure'):
                    print(f"      ⚠️ Figure {check_id} already assigned to another section (skipping for '{section_title}')")
                else:
                    print(f"      ❌ Unexpected: Figure {check_id} resolved but not in used_assets")
            else:
                print(f"      ❌ Figure {figure_id} could not be resolved (not found in filtered assets)")

    # Assign all available tables (preserve all for poster, but no duplicates)
    for table_id in assets.get("tables", []):
        print(f"      📊 Processing table {table_id}...")
        resolved_id = asset_tracker.try_assign_asset(
            table_id, 'table', filtered_figures, filtered_tables
        )
        if resolved_id:
            filtered_assets["tables"].append(resolved_id)
            asset_tracker.mark_asset_as_used(resolved_id, 'table') # Mark as used after successful assignment
            print(f"      ✅ Assigned table {resolved_id} to '{section_title}'")
        else:
            # Check why assignment failed
            check_id = asset_tracker.resolve_asset_id(table_id, 'table', filtered_figures, filtered_tables)
            if check_id:
                if asset_tracker.is_asset_used(check_id, 'table'):
                    print(f"      ⚠️ Table {check_id} already assigned to another section (skipping for '{section_title}')")
                else:
                    print(f"      ❌ Unexpected: Table {check_id} resolved but not in used_assets")
            else:
                print(f"      ❌ Table {table_id} could not be resolved (not found in filtered assets)")

    # Sort figures by numeric ID so they appear in paper order (Figure 2 before Figure 3, etc.)
    filtered_assets["figures"].sort(
        key=lambda fid: int(str(fid)) if str(fid).isdigit() else float('inf')
    )
    print(f"      📋 Final assets for '{section_title}': figures={filtered_assets['figures']}, tables={filtered_assets['tables']}")

    return filtered_assets


# ---- Step 2-4 Finalizer: Unassigned Asset Recovery --------------------------
def ensure_all_assets_assigned(refined_tree, filtered_figures, filtered_tables):
    """
    Ensure all filtered assets are assigned to appropriate sections
    This prevents important assets from being lost while maintaining uniqueness constraint
    """
    print(f"\n🔄 Ensuring all important assets are assigned (no duplicates)...")

    # Collect all currently assigned assets (normalize IDs to str for safe comparison)
    assigned_figures = set()
    assigned_tables = set()

    def collect_assigned_assets(node):
        assets = node.get("assets", {})
        assigned_figures.update(str(x) for x in assets.get("figures", []))
        assigned_tables.update(str(x) for x in assets.get("tables", []))
        for child in node.get("children", []):
            collect_assigned_assets(child)

    collect_assigned_assets(refined_tree)

    # Find unassigned assets: in filtered dicts but not yet in any panel
    unassigned_figures = set(str(k) for k in filtered_figures.keys()) - assigned_figures
    unassigned_tables = set(str(k) for k in filtered_tables.keys()) - assigned_tables

    print(f"   📊 Currently assigned - Figures: {len(assigned_figures)}, Tables: {len(assigned_tables)}")
    print(f"   🚨 Unassigned - Figures: {len(unassigned_figures)}, Tables: {len(unassigned_tables)}")

    if unassigned_figures:
        print(f"   🖼️ Unassigned figures: {list(unassigned_figures)}")
    if unassigned_tables:
        print(f"   📊 Unassigned tables: {list(unassigned_tables)}")

    # Assign unassigned assets to the best-matching section
    if unassigned_figures or unassigned_tables:
        print(f"   🔧 Assigning unassigned assets by caption+position similarity...")

        # Collect all leaf panels (non-title), preserving tree order for position affinity
        all_panels = []
        def _collect_panels(node):
            if node.get("panel_id", "0") == "0":
                for ch in node.get("children", []):
                    _collect_panels(ch)
                return
            if not node.get("children"):
                all_panels.append(node)
            for ch in node.get("children", []):
                _collect_panels(ch)
        _collect_panels(refined_tree)

        _STOP_WORDS = {
            'the', 'a', 'an', 'in', 'of', 'to', 'and', 'or', 'is', 'are',
            'was', 'for', 'with', 'on', 'at', 'by', 'we', 'our', 'its',
            'this', 'that', 'which', 'from', 'as', 'be', 'not', 'are',
        }

        # Keywords for figure-type and section-type inference.
        # NOTE: 'overview' and 'diagram' intentionally excluded from _M_CAP — they are
        # format-descriptive words that often appear in teaser/introduction figures
        # (e.g. "An overview of our challenges") and should NOT force a method classification.
        _M_CAP = {'architecture', 'framework', 'pipeline', 'structure',
                  'module', 'component', 'network', 'layer', 'design'}
        _E_CAP = {'result', 'results', 'comparison', 'qualitative', 'ablation',
                  'accuracy', 'performance', 'evaluation', 'visualization', 'prediction'}
        # Teaser/intro keywords: figure likely belongs in Introduction / Motivation
        _T_CAP = {'teaser', 'motivat', 'challeng', 'problem', 'limitation',
                  'existing method', 'prior work', 'drawback', 'difficul'}
        _M_SEC = {'method', 'approach', 'model', 'architecture', 'framework',
                  'algorithm', 'network', 'design', 'system', 'proposed'}
        _E_SEC = {'experiment', 'result', 'evaluation', 'analysis',
                  'ablation', 'benchmark', 'comparison'}
        _INTRO_TITLES = {'introduction', 'intro', 'motivation', 'overview', 'abstract',
                         'related work', 'background', 'preliminary'}

        def _infer_fig_type(cap, cap_number=None, is_table=False):
            """Classify asset as 'teaser', 'method', 'experiment', or 'unknown'.

            Tables are NEVER classified as 'teaser' — they are always analytical assets
            that belong where they are cited (method/experiment/result sections), never
            in Introduction as a teaser image.

            Early figures (cap ≤ 2) with weak method/experiment signal are treated as
            teasers that belong in the introduction, not in method/experiment sections.
            """
            c = cap.lower()
            ms = sum(1 for k in _M_CAP if k in c)
            es = sum(1 for k in _E_CAP if k in c)
            ts = sum(1 for k in _T_CAP if k in c)

            # Tables: skip teaser classification entirely
            if is_table:
                return 'method' if ms > es else ('experiment' if es > ms else 'unknown')

            try:
                cap_num_int = int(cap_number) if cap_number is not None else 999
            except (TypeError, ValueError):
                cap_num_int = 999
            # Figures 1–2 with no dominant method/experiment signal are likely teasers
            is_very_early = cap_num_int <= 2
            is_early = cap_num_int <= 3
            # Treat as teaser if: explicit teaser keywords present (and not more method),
            # OR figure is among the first two and method signal is not dominant
            if ts > 0 and ts >= ms:
                return 'teaser'
            if is_very_early and ms <= 1 and es == 0:
                return 'teaser'
            if is_early and ms == 0 and es == 0:
                return 'teaser'
            return 'method' if ms > es else ('experiment' if es > ms else 'unknown')

        def _infer_sec_type(panel):
            t = panel.get('section_name', '').lower()
            if any(k in t for k in _M_SEC): return 'method'
            if any(k in t for k in _E_SEC): return 'experiment'
            return 'other'

        def _is_intro_panel(panel):
            t = panel.get('section_name', '').lower().strip()
            return t in _INTRO_TITLES or any(t.startswith(k) for k in _INTRO_TITLES)

        def _panel_score(panel, caption, panel_idx, total_panels,
                         cap_number, total_assets):
            sec_name  = panel.get("section_name", "").lower()
            content   = panel.get("content", "").lower()

            cap_clean = caption.lower()
            cap_words = {w for w in cap_clean.split()
                         if len(w) > 3 and w not in _STOP_WORDS}

            # Tier 0: explicit "Figure N" / "Table N" citation in section content.
            # This mirrors the citation-first logic in reassign_assets_by_caption_relevance.
            _cite_bonus = 0
            if cap_number is not None:
                import re as _re_cite
                _num = _re_cite.escape(str(cap_number))
                # Detect whether this is a table or a figure from the caption prefix.
                _is_table_asset = (
                    cap_clean.startswith('table') or
                    cap_clean[:20].lstrip().startswith('tab.')
                )
                if _is_table_asset:
                    _cite_re = _re_cite.compile(
                        rf'\b(?:tab(?:le)?\.?\s*{_num})\b',
                        _re_cite.IGNORECASE
                    )
                else:
                    _cite_re = _re_cite.compile(
                        rf'\b(?:fig(?:ure)?\.?\s*{_num}|fig\.\s*{_num})\b',
                        _re_cite.IGNORECASE
                    )
                if _cite_re.search(content):
                    _cite_bonus = 200  # strong but beatable by type+section match

            # Tier 1: section NAME match (5×) — more discriminative than body
            name_words    = {w for w in sec_name.split() if len(w) > 3 and w not in _STOP_WORDS}
            content_words = {w for w in content.split()  if len(w) > 3 and w not in _STOP_WORDS}
            score  = _cite_bonus
            score += 5 * len(cap_words & name_words)
            score += 1 * len(cap_words & (content_words - name_words))

            try:
                cap_num_int = int(cap_number) if cap_number is not None else None
            except (TypeError, ValueError):
                cap_num_int = None

            # Tier 2: figure-type / section-type bonus
            fig_type = _infer_fig_type(caption, cap_number, is_table=_is_table_asset)
            sec_type = _infer_sec_type(panel)

            if fig_type == 'teaser':
                # Teaser figures strongly prefer Introduction/Motivation panels
                if _is_intro_panel(panel):
                    score += 100
                # No penalty for going elsewhere — but intro is preferred
            elif fig_type == sec_type and fig_type in ('method', 'experiment'):
                score += 100

            # Hard block: method/experiment figures must not go to Introduction.
            # Teaser figures are explicitly exempt from this block.
            if _is_intro_panel(panel) and fig_type in ('method', 'experiment'):
                return -1000

            # Soft block: teaser and method figures should NOT go to experiment panels.
            # Citations in experiment sections are usually back-references ("as shown in Fig. N"),
            # not the primary presentation site. Apply a penalty so the method/intro section
            # wins even when experiment also cites the figure.
            if fig_type in ('teaser', 'method') and sec_type == 'experiment':
                score -= 150  # strong penalty but not an absolute block

            # Tier 3: position affinity — strengthened for early figures
            # Early figures (≤3) appear near the start of a paper and typically belong
            # in the introduction/motivation section; give them a stronger positional pull.
            if cap_num_int and total_assets > 0 and total_panels > 0:
                fig_pos = (cap_num_int - 1) / max(total_assets - 1, 1)
                pan_pos = panel_idx / max(total_panels - 1, 1)
                # 3× weight for early figures; 0.5× for others
                pos_weight = 3.0 if (cap_num_int is not None and cap_num_int <= 3) else 0.5
                score += max(0.0, 1.0 - abs(fig_pos - pan_pos) * 2.5) * pos_weight
            return score

        total_assets_count = len(filtered_figures) + len(filtered_tables)
        total_panels = max(len(all_panels), 1)

        for figure_id in list(unassigned_figures):
            # Skip only if there are no panels to assign to.
            # Do NOT check asset_tracker.is_asset_used: a figure can be in
            # unassigned_figures (not in any panel) yet still be marked "used"
            # in the tracker if its original section was dropped — that stale
            # tracker state must not prevent us from placing it here.
            if not all_panels:
                continue
            fig_data = filtered_figures.get(figure_id, {})
            caption = fig_data.get('caption', '')
            cap_number = fig_data.get('caption_number')
            scores = [
                _panel_score(p, caption, i, total_panels,
                             cap_number, total_assets_count)
                for i, p in enumerate(all_panels)
            ]
            best = all_panels[scores.index(max(scores))]
            best["assets"]["figures"].append(figure_id)
            asset_tracker.mark_asset_as_used(figure_id, 'figure')
            print(f"      ➕ Figure {figure_id} (cap#{cap_number}) → "
                  f"'{best['section_name']}' (score={max(scores):.2f})")

        for table_id in list(unassigned_tables):
            # Same reasoning as for figures: ignore stale tracker state.
            if not all_panels:
                continue
            tbl_data = filtered_tables.get(table_id, {})
            caption = tbl_data.get('caption', '')
            cap_number = tbl_data.get('caption_number')
            scores = [
                _panel_score(p, caption, i, total_panels,
                             cap_number, total_assets_count)
                for i, p in enumerate(all_panels)
            ]
            best = all_panels[scores.index(max(scores))]
            best["assets"]["tables"].append(table_id)
            asset_tracker.mark_asset_as_used(table_id, 'table')
            print(f"      ➕ Table {table_id} (cap#{cap_number}) → "
                  f"'{best['section_name']}' (score={max(scores):.2f})")

    # Final validation: check for duplicates
    print(f"\n🔍 Final validation - checking for asset duplicates...")
    all_assigned_figures = []
    all_assigned_tables = []

    def collect_all_assets(node):
        assets = node.get("assets", {})
        all_assigned_figures.extend(assets.get("figures", []))
        all_assigned_tables.extend(assets.get("tables", []))
        for child in node.get("children", []):
            collect_all_assets(child)

    collect_all_assets(refined_tree)

    # Check for duplicates
    figures_duplicates = set([x for x in all_assigned_figures if all_assigned_figures.count(x) > 1])
    tables_duplicates = set([x for x in all_assigned_tables if all_assigned_tables.count(x) > 1])

    if figures_duplicates:
        print(f"   ❌ DUPLICATE FIGURES DETECTED: {list(figures_duplicates)}")
    else:
        print(f"   ✅ No duplicate figures found")

    if tables_duplicates:
        print(f"   ❌ DUPLICATE TABLES DETECTED: {list(tables_duplicates)}")
    else:
        print(f"   ✅ No duplicate tables found")

    print(f"   ✅ Asset assignment completed - all important assets preserved with uniqueness") 