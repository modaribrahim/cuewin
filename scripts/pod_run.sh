#!/usr/bin/env bash
# Revision-2 GPU run. Usage on the pod, from the repo root:
#   bash scripts/pod_run.sh
# Models are downloaded from the HuggingFace Hub on first use.
set -euo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1
python -m pip install -q "transformers==5.5.0" "tokenizers==0.22.1" "pandas==2.2.2" "numpy==1.26.4"
python -m pip install -q --no-deps -e .
python -c "import torch; assert torch.cuda.is_available(), 'no GPU'; print(torch.cuda.get_device_name(0))"
export PYTHONUNBUFFERED=1
python -m cuewin.pipeline.revision2 2>&1 | grep --line-buffered -v "UNEXPECTED\|LOAD REPORT\|Loading weights" | tee -a data/rev2_log.txt
tar czf rev2_results.tar.gz data/rev2_*
echo "done: download rev2_results.tar.gz"
