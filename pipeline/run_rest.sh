#!/usr/bin/env bash
# asli: train as soon as training fakes exist; test-only generators (XTTS, Odia/Assamese MMS) run in parallel.
set -uo pipefail
cd "$(dirname "$0")"
ROOT=$(cd .. && pwd); . "$ROOT/../.venv-ml/bin/activate"
log() { echo "[$(date +%H:%M:%S)] $*"; }
log "waiting for training-split MMS (pa, ur)"
while pgrep -f "gen_mms.py" >/dev/null; do sleep 20; done
log "preprocess fakes: train, dev"
WORKERS=4 python prepare.py fake train,dev || { log "prepare fake FAILED"; exit 1; }
log "train"
python train.py --epochs 8 || { log "train FAILED"; exit 1; }
log "waiting for test-only generators (XTTS, Odia/Assamese)"
while pgrep -f "gen_xtts.py|gen_mms_test.py" >/dev/null; do sleep 20; done
log "preprocess fakes: test"
WORKERS=4 python prepare.py fake test || { log "prepare fake test FAILED"; exit 1; }
log "evaluate + export"
python evaluate.py asli && python export.py || { log "eval/export FAILED"; exit 1; }
for m in MelodyMachine/Deepfake-audio-detection-V2 mo-thecreator/Deepfake-audio-detection Gustking/wav2vec2-large-xlsr-deepfake-audio-classification; do
  log "baseline $m"; python evaluate.py "hf:$m" --max 300 || log "baseline $m failed"
done
python site_data.py && python readme_results.py
log "ALL_DONE"
