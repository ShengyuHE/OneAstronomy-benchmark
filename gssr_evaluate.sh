#!/bin/bash
set -euo pipefail

source /users/she/bin/packages/miniconda3/etc/profile.d/conda.sh
conda activate AION

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <NUM>" >&2
  exit 1
fi

NUM="$1"

DIR='/capstor/store/cscs/pasc/c39/swiss-ai/test/reports'
LOG_DIR="${DIR}/${NUM}"

GSSR_ANALYZE=/users/she/bin/packages/GPU-Saturation-Scorer/gssr-analyze
GSSR_INPUT=${LOG_DIR}/gssr_report/alps-daint_${NUM}
PDF_OUTPUT=${LOG_DIR}/gssr-report-${NUM}.pdf
GSSR_UV_ACTIVE=1 python "$GSSR_ANALYZE" "$GSSR_INPUT" -o "$PDF_OUTPUT"
echo "Saved GSSR report to: $PDF_OUTPUT"