"""
Asset extraction: figures and tables from Docling-parsed PDFs.

Public API
----------
extract_figures_and_tables(args, raw_result, step_dir, boundary_seq)
    → figures, tables, figures_json_path, tables_json_path, output_dir

extract_figure_number(caption)   → str | None
extract_table_number(caption)    → str | None
"""

import json
import re
from pathlib import Path

import PIL
from docling_core.types.doc import CoordOrigin


# ---- Caption-number Extraction -----------------------------------------------
# Single regex per asset type; case-insensitive and handles abbreviated forms.

_FIG_NUMBER_RE = re.compile(
    r'\b(?:Figure|Fig\.?)\s*([A-Za-z]?\d+)', re.IGNORECASE
)
_TBL_NUMBER_RE = re.compile(
    r'\b(?:Table|Tab\.?)\s*(\d+)', re.IGNORECASE
)
# Subfigure label at the START of a caption: "(a)", "(ii)", "(b1)", …
_SUBFIG_RE = re.compile(r'^\s*\([a-zA-Z\d]+\)')


def extract_figure_number(caption: str | None) -> str | None:
    """Return the first figure number found in *caption*, or None."""
    if not caption:
        return None
    m = _FIG_NUMBER_RE.search(caption)
    return m.group(1) if m else None


def extract_table_number(caption: str | None) -> str | None:
    """Return the first table number found in *caption*, or None."""
    if not caption:
        return None
    m = _TBL_NUMBER_RE.search(caption)
    return m.group(1) if m else None


# ---- Bounding-box Geometry Helpers -------------------------------------------

def _bbox_gaps(b1, b2):
    """Return (x_gap, y_gap, min_dim) between two docling bboxes (PDF points)."""
    x_gap = max(0.0, max(b1.l, b2.l) - min(b1.r, b2.r))
    if b1.coord_origin == CoordOrigin.BOTTOMLEFT:
        y_gap = max(0.0, max(b1.b, b2.b) - min(b1.t, b2.t))
        h1, w1 = b1.t - b1.b, b1.r - b1.l
        h2, w2 = b2.t - b2.b, b2.r - b2.l
    else:
        y_gap = max(0.0, max(b1.t, b2.t) - min(b1.b, b2.b))
        h1, w1 = b1.b - b1.t, b1.r - b1.l
        h2, w2 = b2.b - b2.t, b2.r - b2.l
    return x_gap, y_gap, min(h1, w1, h2, w2)


def _union_bbox(bboxes):
    """Return the union BoundingBox of a list of bboxes (same coord_origin)."""
    from docling_core.types.doc import BoundingBox
    origin = bboxes[0].coord_origin
    l, r = min(b.l for b in bboxes), max(b.r for b in bboxes)
    if origin == CoordOrigin.BOTTOMLEFT:
        return BoundingBox(l=l, t=max(b.t for b in bboxes),
                           r=r, b=min(b.b for b in bboxes), coord_origin=origin)
    return BoundingBox(l=l, t=min(b.t for b in bboxes),
                       r=r, b=max(b.b for b in bboxes), coord_origin=origin)


def _crop_page_for_group(group_pics, raw_result,
                          caption_margin_frac=0.02, side_margin_frac=0.03):
    """Crop the page image to the union bbox of a compound figure group.

    The caption is stored separately in the metadata, so we intentionally do
    NOT include it in the image crop.  A tiny bottom margin (2% of group height)
    is kept as a visual buffer to avoid accidentally clipping the last pixel row
    of the figure content itself.
    Returns a PIL Image or None.
    """
    page_no = group_pics[0]['page_no']
    page = raw_result.document.pages.get(page_no)
    if page is None or page.image is None or page.image.pil_image is None:
        return None
    bboxes = [p['bbox'] for p in group_pics if p['bbox'] is not None]
    if not bboxes:
        return None

    union = _union_bbox(bboxes)
    img = page.image.pil_image
    iw, ih = img.size
    pw, ph = page.size.width, page.size.height
    sx, sy = iw / pw, ih / ph

    if union.coord_origin == CoordOrigin.BOTTOMLEFT:
        x1, y1 = union.l * sx, (ph - union.t) * sy
        x2, y2 = union.r * sx, (ph - union.b) * sy
    else:
        x1, y1, x2, y2 = union.l * sx, union.t * sy, union.r * sx, union.b * sy

    gw, gh = x2 - x1, y2 - y1
    x1 = max(0, x1 - gw * side_margin_frac)
    y1 = max(0, y1 - gh * side_margin_frac)
    x2 = min(iw, x2 + gw * side_margin_frac)
    y2 = min(ih, y2 + gh * caption_margin_frac)
    return img.crop((int(x1), int(y1), int(x2), int(y2)))


# ---- Figure Grouping (Compound-figure Detection) -----------------------------

def _build_text_caps_by_page(raw_result):
    """Scan non-picture text elements for 'Figure N:' captions.

    Returns {page_no: [(figure_number, text, cx, cy), ...]} covering every
    'Figure N' mention in the document text, whether or not docling linked it
    to a PictureItem.  Used for:
      1. Cross-column guard: compare nearest captions of two figures to decide
         if they are truly separate (different numbers) or subfigures (same).
      2. Fallback caption recovery: assign unlinked captions to figure groups.
    """
    from docling_core.types.doc import PictureItem, TableItem
    result = {}
    try:
        for el, _ in raw_result.document.iterate_items():
            if isinstance(el, (PictureItem, TableItem)):
                continue
            text = (getattr(el, 'text', None) or '').strip()
            fn = extract_figure_number(text) if text else None
            if not fn:
                continue
            prov = getattr(el, 'prov', None)
            if not prov:
                continue
            p0 = prov[0] if isinstance(prov, (list, tuple)) else prov
            pg = getattr(p0, 'page_no', None)
            if not pg:
                continue
            b = getattr(p0, 'bbox', None)
            cx = ((b.l + b.r) / 2) if b else 0.0
            cy = ((b.t + b.b) / 2) if b else 0.0
            result.setdefault(pg, []).append((fn, text, cx, cy))
    except Exception as e:
        print(f"   ⚠️  Caption text scan failed: {e}")
    return result


def _build_table_caps_by_page(raw_result):
    """Scan non-picture text elements for 'Table N:' captions.

    Returns {page_no: [(table_number, text, cx, cy), ...]} covering text
    elements that START with a proper "Table N" caption prefix.  Body sentences
    that merely cite a table (e.g. "results in Table 2 show...") are excluded
    via the match-at-start requirement.

    Used to recover the correct table caption when Docling mis-links a nearby
    Figure caption to a TableItem.
    """
    from docling_core.types.doc import PictureItem, TableItem
    result = {}
    try:
        for el, _ in raw_result.document.iterate_items():
            if isinstance(el, (PictureItem, TableItem)):
                continue
            text = (getattr(el, 'text', None) or '').strip()
            if not text:
                continue
            # Only accept text that STARTS with "Table N" (proper caption prefix).
            # This filters out body sentences like "as shown in Table 2".
            if not _TBL_NUMBER_RE.match(text):
                continue
            tn = extract_table_number(text)
            if not tn:
                continue
            prov = getattr(el, 'prov', None)
            if not prov:
                continue
            p0 = prov[0] if isinstance(prov, (list, tuple)) else prov
            pg = getattr(p0, 'page_no', None)
            if not pg:
                continue
            b = getattr(p0, 'bbox', None)
            cx = ((b.l + b.r) / 2) if b else 0.0
            cy = ((b.t + b.b) / 2) if b else 0.0
            result.setdefault(pg, []).append((tn, text, cx, cy))
    except Exception as e:
        print(f"   ⚠️  Table caption text scan failed: {e}")
    return result


def _nearest_figure_number(cx, cy, page_no, text_caps_by_page, max_dist_pts=250):
    """Return the figure number of the nearest text caption within max_dist_pts."""
    best_d, best_fn = float('inf'), None
    for fn, _text, ccx, ccy in text_caps_by_page.get(page_no, []):
        d = ((cx - ccx) ** 2 + (cy - ccy) ** 2) ** 0.5
        if d < best_d:
            best_d, best_fn = d, fn
    return best_fn if best_d <= max_dist_pts else None


def _group_figures(picture_elements, raw_result, gap_threshold_frac=0.5):
    """Cluster PictureItems into compound-figure groups.

    Two rules drive the clustering (union-find):

    Rule 1 — Explicit number match
        Two figures with the SAME 'Figure N' number in their linked captions
        on the same page are always merged into one compound figure.

    Rule 2 — Spatial proximity (orphan subfigures)
        A figure WITHOUT a linked 'Figure N' number (orphan) is merged with
        a nearby figure when the bounding-box gap ≤ gap_threshold_frac × the
        smallest dimension of either figure.
        Orphans with a single numbered anchor use a more lenient threshold
        (0.75 ×) to handle layouts where subfigures are further apart.

    Cross-column guard
        When two column-wide figures (each > 30 % of page width) sit in
        opposite halves of the page, the guard compares the nearest text-
        element caption for each:
          • Different figure numbers  → block merge (truly separate figures).
          • Same / no nearby caption  → allow through, let Rule 2 decide
                                        (may be subfigures spanning both cols).

        This distinguishes PartCATSeg Fig 2 (left) / Fig 3 (right) — separate
        figures, each with its own nearby text caption — from Transformer
        Fig 2 — two subfigures sharing a single 'Figure 2:' caption below.

    Table-image guard
        Docling sometimes classifies visual tables (e.g. icon/image grids) as
        PictureItems instead of TableItems.  Two checks identify these:
          1. The element's linked caption starts with "Table N." → explicit.
          2. Orphan element (no "Figure N" caption): nearest "Table N." text is
             closer than nearest "Figure N." text → spatial.
        Elements flagged as table images are never merged into figure groups.

    Returns: list of groups, each group being a list of pic-dicts:
        {element, caption, figure_number, is_table_img, is_subfig, page_no, bbox}
    """
    # Pre-compute text captions once for the whole document.
    text_caps = _build_text_caps_by_page(raw_result)
    table_caps = _build_table_caps_by_page(raw_result)

    def _is_table_image(cap, page_no, bbox):
        """Return True if this PictureItem is a visual table, not a figure."""
        cap = cap or ''
        # Explicit "Table N." caption linked by Docling → definitely a table.
        if extract_table_number(cap) is not None:
            return True
        # Orphan (no Figure-N caption): nearest "Table N." text closer than
        # nearest "Figure N." text → treat as table image.
        if extract_figure_number(cap) is None and bbox is not None and page_no:
            tbl_entries = table_caps.get(page_no, [])
            if tbl_entries:
                cx = (bbox.l + bbox.r) / 2
                cy = (bbox.t + bbox.b) / 2
                d_tbl = min(
                    ((cx - tcx) ** 2 + (cy - tcy) ** 2) ** 0.5
                    for _, _, tcx, tcy in tbl_entries
                )
                d_fig = min(
                    (((cx - fcx) ** 2 + (cy - fcy) ** 2) ** 0.5
                     for _, _, fcx, fcy in text_caps.get(page_no, [])),
                    default=float('inf')
                )
                if d_tbl < d_fig:
                    return True
        return False

    pics = []
    for elem in picture_elements:
        cap = elem.caption_text(raw_result.document)
        prov = elem.prov[0] if elem.prov else None
        page_no = prov.page_no if prov else None
        bbox = prov.bbox if prov else None
        pics.append({
            'element': elem,
            'caption': cap,
            'figure_number': extract_figure_number(cap),
            'is_table_img': _is_table_image(cap, page_no, bbox),
            'is_subfig': bool(_SUBFIG_RE.match(cap)) if cap else False,
            'page_no': page_no,
            'bbox': bbox,
        })

    n = len(pics)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    page_width_cache = {}

    def page_width(page_no):
        if page_no not in page_width_cache:
            pg = raw_result.document.pages.get(page_no)
            page_width_cache[page_no] = pg.size.width if (pg and pg.size) else None
        return page_width_cache[page_no]

    for i in range(n):
        for j in range(i + 1, n):
            pi, pj = pics[i], pics[j]
            if pi['page_no'] != pj['page_no'] or pi['page_no'] is None:
                continue

            # Table-image guard: never merge a visual-table PictureItem into a
            # figure group — it belongs to table extraction, not figure grouping.
            if pi['is_table_img'] or pj['is_table_img']:
                continue

            # Rule 1: same figure number → always merge
            if pi['figure_number'] and pj['figure_number'] and \
               pi['figure_number'] == pj['figure_number']:
                union(i, j)
                continue

            if pi['bbox'] is None or pj['bbox'] is None:
                continue

            x_gap, y_gap, min_dim = _bbox_gaps(pi['bbox'], pj['bbox'])

            # -- Caption-based merge decision ---------------------------------
            # We use the nearest unlinked text captions as the primary signal
            # to decide whether two figures are (a) truly separate or (b) parts
            # of the same compound figure.
            pw = page_width(pi['page_no'])
            ci = (pi['bbox'].l + pi['bbox'].r) / 2
            cj = (pj['bbox'].l + pj['bbox'].r) / 2
            byi = (pi['bbox'].t + pi['bbox'].b) / 2
            byj = (pj['bbox'].t + pj['bbox'].b) / 2
            max_d = (pw * 0.35) if pw else 250
            fi = _nearest_figure_number(ci, byi, pi['page_no'], text_caps, max_d)
            fj = _nearest_figure_number(cj, byj, pj['page_no'], text_caps, max_d)
            # Nearest text captions point to DIFFERENT numbers → separate figures
            if fi and fj and fi != fj:
                continue

            # Rule 2: at least one must be an orphan (no figure number) OR
            # their nearest text captions agree (same number or no caption),
            # indicating they share one compound caption — even if docling
            # incorrectly linked a different number to one of them (e.g. Figure 3
            # mis-linked to a right-hand subfig of Figure 2).
            #
            # We only skip merging when:
            #   • BOTH have figure numbers that are different from each other, AND
            #   • Their nearest text captions also point to different numbers
            #     (already handled above).
            # In all other cases (orphan, is_subfig, or matching text caps),
            # fall through to the proximity threshold.
            both_numbered_differently = (
                pi['figure_number'] and pj['figure_number'] and
                pi['figure_number'] != pj['figure_number']
            )
            if both_numbered_differently:
                # Extra check: if their nearest text captions agree on one number,
                # docling may have mis-labelled one subfig — allow merge.
                # e.g. both fi and fj point to "2", but docling linked "3" to pj.
                if not (fi == fj and fi is not None):
                    continue  # truly separate figures

            # -- Proximity / cross-column threshold ---------------------------
            # At this point the caption check already confirmed that the two
            # figures share the same figure number (or one is un-numbered), so
            # they are *candidates* for being subfigures of the same compound.
            #
            # For the anchor + orphan case (one has "Figure N:", one doesn't),
            # the diagrams can be far apart because each subfig may be narrow
            # and the combined caption sits below *both*.  Basing the threshold
            # on min_dim (the smaller figure's smallest edge) is too strict when
            # the subfigures have a large gap between them (e.g. Transformer
            # Fig 2: gap=105 pt, min_dim=67 pt → old threshold 50 pt fails).
            #
            # We instead use a page-fraction threshold:
            #   • Same nearest-caption (fi==fj) : merge if gap < 35% of page width
            #                                      (handles compound figs within one
            #                                       column or spanning two columns)
            #   • fi/fj absent or unknown        : fall back to min_dim × frac
            one_has_number = bool(pi['figure_number']) != bool(pj['figure_number'])

            if pw and pw > 0 and one_has_number and (fi == fj) and fi is not None:
                # Shared text caption → use page-relative threshold.
                # Also guard against truly separate figures that share a caption
                # only by coincidence: the x_gap must be < 35% of page width
                # AND the y_gap must be small (same row of figures).
                if x_gap <= pw * 0.35 and y_gap <= pw * 0.35:
                    union(i, j)
                continue  # don't fall through to min_dim threshold

            # Cross-column guard: column-wide figures in opposite halves
            # (both > 30% of page width) → only block if gap is very large.
            if pw and pw > 0:
                wi = (pi['bbox'].r - pi['bbox'].l) / pw
                wj = (pj['bbox'].r - pj['bbox'].l) / pw
                if ((ci < pw / 2) != (cj < pw / 2)) and wi > 0.30 and wj > 0.30:
                    if x_gap > pw * 0.30:  # > 30% of page width apart → separate
                        continue

            # Standard proximity threshold (min_dim based)
            frac = 0.75 if one_has_number else gap_threshold_frac
            if x_gap <= min_dim * frac and y_gap <= min_dim * frac:
                union(i, j)

    groups: dict[int, list] = {}
    first_idx: dict[int, int] = {}  # root → earliest element index in document order
    for i, pic in enumerate(pics):
        root = find(i)
        groups.setdefault(root, []).append(pic)
        if root not in first_idx:
            first_idx[root] = i
        else:
            first_idx[root] = min(first_idx[root], i)
    # Sort groups by the document order of their earliest element so that
    # compound figures containing early subfigures don't jump ahead of
    # logically earlier figures (e.g. Figure 3 subfigure at index a < Figure 2
    # at index d caused Figure 3's group to be returned before Figure 2).
    return [v for _, v in sorted(groups.items(), key=lambda kv: first_idx[kv[0]])]


# ---- Asset Metadata and Filters ----------------------------------------------

def _asset_metadata(img: PIL.Image.Image, path: str,
                    caption: str, caption_number, counter: int,
                    is_compound: bool, subfigure_count: int) -> dict:
    """Build the standard metadata dict for a figure or table asset."""
    return {
        'caption': caption,
        'figure_path': path,
        'width': img.width,
        'height': img.height,
        'figure_size': img.width * img.height,
        'figure_aspect': img.width / max(img.height, 1),
        'original_index': counter,
        'caption_number': caption_number,
        'is_compound': is_compound,
        'subfigure_count': subfigure_count,
    }


_SUPPL_ID_RE = re.compile(
    r'^[A-Za-z][\.\d]|^(Supp|Sup|App|Appendix)\w*\d', re.IGNORECASE
)
# Caption-level supplementary detection.
# Rules:
#   • Only match FULL PHRASES that unambiguously describe supplementary material.
#   • "appendix" alone is EXCLUDED — legitimate main figures frequently say
#     "...as shown in Appendix A.1" as a cross-reference, which is not
#     supplementary content.
#   • "supp\w*" is EXCLUDED — it matches common words like "support", "suppose".
#   • "SM" / "SI" (2-char) are EXCLUDED — too short, too many false positives
#     (e.g. "3DGM", "SM model").
#   • Kept: full phrases that appear only in captions of genuinely supplementary
#     figures: "supplementary figure", "supplementary material", "extended data".
_SUPPL_CAP_RE = re.compile(
    r'\b(supplementary\s+(?:figure|material|section|table|video|content)'
    r'|additional\s+material'
    r'|extended\s+data)',
    re.IGNORECASE,
)


# ---- Step 1-3 [Entry]: Figure and Table Extraction ---------------------------
def extract_figures_and_tables(args, raw_result, step_dir=None, boundary_seq=None):
    """Extract figures and tables from a Docling conversion result.

    Parameters
    ----------
    boundary_seq : int | None
        iterate_items() sequence index of the first post-conclusion element
        (References, Appendix, …).  Elements at or after this index are
        excluded, allowing finer-grained filtering than page-level cutoffs.
    """
    # ---- Output paths -------------------------------------------------------
    if step_dir:
        output_dir = Path(step_dir) / args.poster_name
        json_dir = Path(step_dir)
    else:
        base = f'<{args.model_name_t}_{args.model_name_v}>_images_and_tables'
        output_dir = Path(base) / args.poster_name
        json_dir = Path(base)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    name = args.poster_name

    # ---- Build element→sequence-index map (for boundary filtering) ----------
    elem_seq = {}
    if boundary_seq is not None:
        for idx, (el, _) in enumerate(raw_result.document.iterate_items()):
            elem_seq[id(el)] = idx

    def _past_boundary(element):
        if boundary_seq is None:
            return False
        seq = elem_seq.get(id(element))
        return seq is not None and seq >= boundary_seq

    def _group_past_boundary(group):
        seqs = [elem_seq[id(p['element'])] for p in group if id(p['element']) in elem_seq]
        return seqs and min(seqs) >= boundary_seq

    # ---- Save page images ---------------------------------------------------
    for _pno, page in raw_result.document.pages.items():
        pno = page.page_no
        with (output_dir / f"{name}-{pno}.png").open("wb") as fp:
            page.image.pil_image.save(fp, format="PNG")

    # ---- Tables -------------------------------------------------------------
    tables, tables_no_caption = {}, {}
    used_tbl_ids: set[str] = set()
    tbl_counter = 1
    print("📊 Processing tables:")

    # Pre-build table-caption index for fallback recovery.
    # When Docling mis-links a Figure caption to a TableItem (e.g. Figure 5's
    # caption text is attached to Table 2 because they share the same page),
    # we use this index to find the nearest proper "Table N:" caption instead.
    _table_caps_by_page = _build_table_caps_by_page(raw_result)
    _claimed_tbl_caps: set[tuple] = set()  # (page_no, tbl_num) already claimed

    for tbl in raw_result.document.tables:
        if _past_boundary(tbl):
            print(f"   ⏭️ Table skipped (past boundary)")
            continue
        caption = tbl.caption_text(raw_result.document)

        # Validate caption type: reject Figure captions mis-linked to a TableItem.
        # Docling sometimes assigns a nearby Figure N caption to a TableItem when
        # the figure and table share the same page.
        if caption and _FIG_NUMBER_RE.match(caption):
            print(f"   ⚠️  Table {tbl_counter}: Docling linked a Figure caption — trying recovery")
            print(f"         Wrong: '{caption[:70]}'")
            prov = tbl.prov[0] if tbl.prov else None
            pg   = prov.page_no if prov else None
            bbox = prov.bbox    if prov else None
            recovered_cap, recovered_tn = None, None
            if pg and bbox and pg in _table_caps_by_page:
                gcx = (bbox.l + bbox.r) / 2
                gcy = (bbox.t + bbox.b) / 2
                best_d = float('inf')
                for tn, cap, cx, cy in _table_caps_by_page[pg]:
                    if (pg, tn) in _claimed_tbl_caps:
                        continue
                    d = ((gcx - cx) ** 2 + (gcy - cy) ** 2) ** 0.5
                    if d < best_d:
                        best_d, recovered_tn, recovered_cap = d, tn, cap
            if recovered_cap:
                _claimed_tbl_caps.add((pg, recovered_tn))
                print(f"         Recovered: '{recovered_cap[:70]}'")
                caption = recovered_cap
            else:
                print(f"         Recovery failed — treating as no-caption table")
                caption = ''

        img_path = str(output_dir / f"{name}-table-{tbl_counter}.png")
        try:
            tbl.get_image(raw_result.document).save(img_path, "PNG")
            img = PIL.Image.open(img_path)
        except Exception as e:
            print(f"   ⚠️ Table {tbl_counter}: {e}")
            tbl_counter += 1
            continue

        if caption:
            tbl_num = extract_table_number(caption)
            tbl_id = tbl_num if (tbl_num and tbl_num not in used_tbl_ids) else str(tbl_counter)
            used_tbl_ids.add(tbl_id)
            tables[tbl_id] = {
                'caption': caption,
                'table_path': img_path,
                'width': img.width, 'height': img.height,
                'figure_size': img.width * img.height,
                'figure_aspect': img.width / max(img.height, 1),
                'original_index': tbl_counter,
                'caption_number': tbl_num,
            }
            print(f"   📊 Table {tbl_id}: {caption[:60]}...")
        else:
            tables_no_caption[f"no_caption_{tbl_counter}"] = {
                'caption': '', 'table_path': img_path,
                'width': img.width, 'height': img.height,
                'figure_size': img.width * img.height,
                'figure_aspect': img.width / max(img.height, 1),
                'original_index': tbl_counter, 'caption_number': None,
            }
        tbl_counter += 1

    # ---- Figures ------------------------------------------------------------
    figures, figures_no_caption = {}, {}
    used_fig_ids: set[str] = set()
    fig_counter = 1
    print("\n🖼️ Processing figures:")

    groups = _group_figures(raw_result.document.pictures, raw_result)
    n_compound = sum(1 for g in groups if len(g) > 1)
    print(f"   📐 {len(raw_result.document.pictures)} elements → {len(groups)} groups "
          f"({n_compound} compound)")

    # Build the set of figure numbers already linked via PictureItem captions.
    # Used in fallback caption recovery to avoid re-claiming a linked caption.
    linked_fig_nums = {p['figure_number'] for g in groups for p in g if p.get('figure_number')}

    # Pre-compute text captions for fallback recovery (reuse the same scan already
    # done inside _group_figures – but we call it once more here for simplicity;
    # the scan is fast and the result is a small dict).
    text_caps = _build_text_caps_by_page(raw_result)
    claimed_caps: set[tuple] = set()   # (page_no, fig_number) already assigned

    for group in groups:
        if boundary_seq is not None and _group_past_boundary(group):
            print(f"   ⏭️ Figure group skipped (past boundary)")
            continue

        # Best caption: prefer the one with an explicit 'Figure N' number
        best_cap, best_fn = '', None
        for p in group:
            cap, fn = p['caption'] or '', p['figure_number']
            if fn and (best_fn is None or len(cap) > len(best_cap)):
                best_cap, best_fn = cap, fn
        if not best_fn:
            best_cap = max((p['caption'] or '' for p in group), key=len)
            best_fn = extract_figure_number(best_cap)

        # Fallback: search nearby text elements for a 'Figure N:' caption that
        # Docling failed to link to this PictureItem.
        #
        # Root cause of CVF-logo false positive (RichHF):
        #   • CVF logo has NO Docling caption → best_cap = "" → fallback runs
        #   • text_caps[page1] contains ALL text elements that merely MENTION
        #     "Figure N" anywhere, including paper body sentences like
        #     "Text-to-image models [..] are becoming... as shown in Figure 1."
        #   • The body sentence is the NEAREST text on page 1 → logo gets
        #     best_fn="1", displacing the real Figure 1.
        #
        # Fix (two gates):
        #
        # Gate A — skip fallback when best_cap is non-empty body text:
        #   If Docling linked any non-empty text to this PictureItem AND that text
        #   doesn't start with "Figure N", it is body text, not a caption.
        #   Trust the Docling assignment; the image is not a real figure.
        #
        # Gate B — in fallback, only accept text that STARTS WITH "Figure N":
        #   text_caps stores any text mentioning "Figure N" (needed for cross-column
        #   guard in _group_figures).  For fallback caption assignment we only want
        #   proper captions (e.g. "Figure 1. An illustration of..."), not body
        #   sentences that happen to cite a figure number.
        _cap_is_fig_caption = bool(_FIG_NUMBER_RE.match(best_cap)) if best_cap else False
        # Gate A
        _run_fallback = (not best_cap) or _cap_is_fig_caption

        if not best_fn and _run_fallback:
            pg = next((p['page_no'] for p in group if p.get('page_no')), None)
            bboxes = [p['bbox'] for p in group if p.get('bbox')]
            if pg and bboxes and pg in text_caps:
                gcx = sum((b.l + b.r) / 2 for b in bboxes) / len(bboxes)
                gcy = sum((b.t + b.b) / 2 for b in bboxes) / len(bboxes)
                best_d = float('inf')
                for fn, cap, cx, cy in text_caps[pg]:
                    if (pg, fn) in claimed_caps or fn in linked_fig_nums:
                        continue
                    # Gate B: only accept text that IS a proper figure caption
                    # (starts with "Figure N" / "Fig. N").  Body sentences that
                    # merely cite a figure are rejected here.
                    if not _FIG_NUMBER_RE.match(cap):
                        continue
                    d = ((gcx - cx) ** 2 + (gcy - cy) ** 2) ** 0.5
                    if d < best_d:
                        best_d, best_fn, best_cap = d, fn, cap
                if best_fn:
                    claimed_caps.add((pg, best_fn))
                    print(f"   📎 Figure {best_fn}: caption recovered from text (page {pg})")
        elif not best_fn and not _run_fallback:
            print(f"   ⏭️ Figure (page {next((p['page_no'] for p in group if p.get('page_no')), '?')}): "
                  f"skipping fallback — body text linked by Docling: '{best_cap[:60]}'")

        # Build image
        is_compound = len(group) > 1
        if is_compound:
            fig_img = _crop_page_for_group(group, raw_result)
            if fig_img is None:
                fb = next((p for p in group if p['caption']), group[0])
                fig_img = fb['element'].get_image(raw_result.document)
        else:
            fig_img = group[0]['element'].get_image(raw_result.document)

        if fig_img is None:
            print(f"   ⚠️ Figure group {fig_counter}: get_image() returned None, skipping")
            fig_counter += 1
            continue

        img_path = (str(output_dir / f"{name}-figure-{fig_counter}.png") if step_dir else
                    f'<{args.model_name_t}_{args.model_name_v}>_images_and_tables'
                    f'/{name}/{name}-figure-{fig_counter}.png')
        label = f"COMPOUND({len(group)})" if is_compound else "single"

        try:
            fig_img.save(img_path, "PNG")
            loaded = PIL.Image.open(img_path)

            # Skip supplementary / appendix figures
            is_suppl = (
                (best_fn and bool(_SUPPL_ID_RE.match(best_fn))) or
                (best_cap and bool(_SUPPL_CAP_RE.search(best_cap)))
            )
            if is_suppl:
                figures_no_caption[f"suppl_{fig_counter}"] = _asset_metadata(
                    loaded, img_path, best_cap or '', best_fn,
                    fig_counter, is_compound, len(group))
                print(f"   ⏭️ Figure {best_fn} [{label}] skipped (supplementary)")
                fig_counter += 1
                continue

            # Skip very small images — these are typically conference logos, icons,
            # or decorative elements (e.g. CVF/IEEE logos on the first page) that
            # happen to be detected as PictureItems.  A legitimate content figure
            # is always substantially larger than a small logo.
            # Threshold: less than 8000 px² (≈ 90×90) is too small to be meaningful.
            _px_area = loaded.width * loaded.height
            if _px_area < 8000:
                figures_no_caption[f"tiny_{fig_counter}"] = _asset_metadata(
                    loaded, img_path, best_cap or '', best_fn,
                    fig_counter, is_compound, len(group))
                print(f"   ⏭️ Figure {best_fn or fig_counter} [{label}] skipped "
                      f"(too small: {loaded.width}×{loaded.height}={_px_area}px²)")
                fig_counter += 1
                continue

            meta = _asset_metadata(loaded, img_path, best_cap or '', best_fn,
                                   fig_counter, is_compound, len(group))

            if best_fn:
                # Only include figures that have an explicit "Figure N" caption number.
                # Images with some text caption but no figure number (logos, icons,
                # decorative elements like conference logos) are excluded from the poster.
                fig_id = best_fn if best_fn not in used_fig_ids else f"unk_{fig_counter}"
                used_fig_ids.add(fig_id)
                if fig_id in figures:
                    # Collision: compound group wins over singleton
                    if meta['subfigure_count'] > figures[fig_id].get('subfigure_count', 1):
                        figures[fig_id] = meta
                        print(f"   ⚠️ Figure {fig_id}: replaced with larger compound")
                    else:
                        print(f"   ⚠️ Figure {fig_id}: kept existing, skipped smaller group")
                else:
                    figures[fig_id] = meta
                    print(f"   🖼️ Figure {fig_id} [{label}]: {best_cap[:60]}...")
            else:
                # No "Figure N" number found — caption may be incidental text (logos,
                # header images, etc.). Move to no_caption bucket, excluded from poster.
                nc_id = f"no_caption_{fig_counter}"
                figures_no_caption[nc_id] = meta
                if best_cap:
                    print(f"   ⏭️ Figure {fig_counter} [{label}] → no_caption (has text but no Figure N: {best_cap[:50]}...)")

        except Exception as e:
            print(f"   ⚠️ Figure group {fig_counter}: {e}")

        fig_counter += 1

    # ---- Save markdown / HTML previews --------------------------------------
    from docling_core.types.doc import ImageRefMode
    try:
        raw_result.document.save_as_markdown(
            output_dir / f"{name}-with-figures.md", image_mode=ImageRefMode.EMBEDDED)
    except Exception:
        pass
    try:
        raw_result.document.save_as_markdown(
            output_dir / f"{name}-with-image-refs.md", image_mode=ImageRefMode.REFERENCED)
    except Exception:
        pass
    try:
        raw_result.document.save_as_html(
            output_dir / f"{name}-with-image-refs.html", image_mode=ImageRefMode.REFERENCED)
    except Exception:
        pass

    # ---- Save JSON ----------------------------------------------------------
    figs_path  = json_dir / f'{name}_images.json'
    tbls_path  = json_dir / f'{name}_tables.json'
    figs_nc    = json_dir / f'{name}_images_no_caption.json'
    tbls_nc    = json_dir / f'{name}_tables_no_caption.json'
    json.dump(figures,           open(figs_path,  'w'), indent=4)
    json.dump(tables,            open(tbls_path,  'w'), indent=4)
    json.dump(figures_no_caption, open(figs_nc,   'w'), indent=4)
    json.dump(tables_no_caption,  open(tbls_nc,   'w'), indent=4)

    print(f"\n📊 Extraction complete: {len(figures)} figures, {len(tables)} tables "
          f"(+{len(figures_no_caption)} figs / {len(tables_no_caption)} tables without captions)")
    return figures, tables, str(figs_path), str(tbls_path), str(output_dir)
