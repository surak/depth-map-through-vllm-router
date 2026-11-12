#!/bin/bash
#SBATCH --job-name=visual-geometry
#SBATCH --partition=gpus
#SBATCH --nodelist=haicluster1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=UNLIMITED
#SBATCH --requeue
#SBATCH --output=/p/haicluster/llama/blablador-fleet/logs/visual-geometry/%j.slurm.out
#SBATCH --error=/p/haicluster/llama/blablador-fleet/logs/visual-geometry/%j.slurm.err
# VGGT-1B mapping behind the OpenAI API, registered with synapse as alias-visual-geometry.
# The repository copy is the source; the fleet runs the copy in blablador-fleet/launchers.
set -euo pipefail
ROOT=/p/haicluster/llama/FastChat
F=/p/haicluster/llama/blablador-fleet
APP=/home/strube1/Devel/depth-map-through-vllm-router
PORT=8030
cd "$APP"
VLLM_API_KEY="$(cat "$ROOT/synapse-external/backend-key-$PORT")"
export VLLM_API_KEY
export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS:-0}
export OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
ENGINE= CLIENT=
cleanup() { [[ -z "$CLIENT" ]] || kill -TERM "$CLIENT" 2>/dev/null || true; [[ -z "$ENGINE" ]] || kill -TERM -- "-$ENGINE" 2>/dev/null || true; }
trap cleanup EXIT
trap 'exit 0' TERM INT
setsid "$APP/.venv/bin/python" -m mapper.server --model "$ROOT/models/VGGT-1B" \
 --host 0.0.0.0 --port $PORT --served-model-name alias-visual-geometry vggt-1b \
 --data-dir "$ROOT/visual-geometry-sessions" \
 >"$F/logs/visual-geometry/$SLURM_JOB_ID.engine.log" 2>&1 &
ENGINE=$!
SYNAPSE_EXTERNAL_CLIENT="$F/bin/synapse-client-metadata-20261028" /usr/bin/python3 "$ROOT/synapse-external/supervisor.py" "$SLURM_JOB_ID" $PORT 0 >"$F/logs/visual-geometry/$SLURM_JOB_ID.synapse.log" 2>&1 &
CLIENT=$!
wait -n "$ENGINE" "$CLIENT"
