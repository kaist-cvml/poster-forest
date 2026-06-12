#!/bin/bash
# GPT-4o: all papers, landscape (48×36)
# Requires OPENAI_API_KEY in .env
# Usage: bash scripts/run_gpt4o_landscape.sh

set -e
cd "$(dirname "$0")/.."  # project root

PAPERS=(
  "papers/2017_NIPS/Attention_Is_All_You_Need.pdf"
  "papers/2026_ACL/PosterForest.pdf"
  # "papers/2023_CVPR/LaserMix.pdf"
  # "papers/2023_NeurIPS/CVPT.pdf"
  # "papers/2024_ACL/FIM-SE.pdf"
  # "papers/2024_CVPR/RichHF.pdf"
  # "papers/2024_CVPR/YOLO-World.pdf"
  # "papers/2024_ECCV/FiT3D.pdf"
  # "papers/2024_ICML/COREP.pdf"
  # "papers/2024_NeurIPS/3DGM.pdf"
  # "papers/2024_NeurIPS/PartCLIPSeg.pdf"
  # "papers/2024_NeurIPS/PIIP.pdf"
  # "papers/2024_NeurIPS/VAR.pdf"
  # "papers/2025_CVPR/PartCATSeg.pdf"
  # "papers/2025_ICLR/EOM.pdf"
  # "papers/2026_ICIP/Scribble.pdf"
  # "papers/2026_ICLR/Paper2Code.pdf"
)

echo "======================================================"
echo "  LANDSCAPE (48×36 in) — ${#PAPERS[@]} papers  [GPT-4o]"
echo "  $(date)"
echo "======================================================"

FAIL=0 DONE=0
for PAPER in "${PAPERS[@]}"; do
  NAME=$(basename "$PAPER" .pdf)
  DONE=$((DONE + 1))
  echo "  [$DONE/${#PAPERS[@]}] $NAME"
  set +e
  python -m PosterForest.main \
    --paper_path="$PAPER" \
    --model_name_t="4o" \
    --model_name_v="4o" \
    --poster_width_inches=48 \
    --poster_height_inches=36
  EXIT=$?
  set -e
  [ $EXIT -eq 0 ] && echo "  ✅ $NAME" || { echo "  ❌ $NAME (exit $EXIT)"; FAIL=$((FAIL + 1)); }
done

echo ""
echo "======================================================"
echo "  DONE — $(date)  |  ✅ $((${#PAPERS[@]} - FAIL)) / ${#PAPERS[@]}"
echo "======================================================"
[ $FAIL -gt 0 ] && exit 1 || exit 0
