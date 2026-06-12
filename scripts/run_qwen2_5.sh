#!/bin/bash
# Qwen2.5: all papers, portrait (36×48) then landscape (48×36)
# Requires vLLM servers to be running: bash scripts/start_vllm_qwen2_5.sh
# Usage: bash scripts/run_qwen2_5.sh

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

# Check vLLM servers
echo "Checking vLLM servers..."
curl -sf http://localhost:8005/health > /dev/null || { echo "ERROR: LLM server (port 8005) not ready"; exit 1; }
curl -sf http://localhost:8010/health > /dev/null || { echo "ERROR: VLM server (port 8010) not ready"; exit 1; }
echo "  LLM (8005) ✓   VLM (8010) ✓"

run_papers() {
  local WIDTH=$1 HEIGHT=$2 LABEL=$3
  local FAIL=0 DONE=0
  echo ""
  echo "======================================================"
  echo "  $LABEL — ${#PAPERS[@]} papers  [Qwen2.5]"
  echo "  $(date)"
  echo "======================================================"
  for PAPER in "${PAPERS[@]}"; do
    NAME=$(basename "$PAPER" .pdf)
    DONE=$((DONE + 1))
    echo "  [$DONE/${#PAPERS[@]}] $NAME"
    set +e
    python -m PosterForest.main \
      --paper_path="$PAPER" \
      --model_name_t="vllm_qwen2_5" \
      --model_name_v="vllm_qwen2_5_vl" \
      --poster_width_inches=$WIDTH \
      --poster_height_inches=$HEIGHT
    EXIT=$?
    set -e
    [ $EXIT -eq 0 ] && echo "  ✅ $NAME" || { echo "  ❌ $NAME (exit $EXIT)"; FAIL=$((FAIL + 1)); }
  done
  echo ""
  echo "  $LABEL done — ✅ $((${#PAPERS[@]} - FAIL)) / ${#PAPERS[@]}"
  return $FAIL
}

run_papers 36 48 "PORTRAIT (36×48 in)"
FAIL_P=$?

run_papers 48 36 "LANDSCAPE (48×36 in)"
FAIL_L=$?

echo ""
echo "======================================================"
echo "  ALL DONE — $(date)"
echo "  Portrait : ✅ $((${#PAPERS[@]} - FAIL_P)) / ${#PAPERS[@]}"
echo "  Landscape: ✅ $((${#PAPERS[@]} - FAIL_L)) / ${#PAPERS[@]}"
echo "======================================================"
[ $((FAIL_P + FAIL_L)) -gt 0 ] && exit 1 || exit 0
