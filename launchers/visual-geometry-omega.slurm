#!/bin/bash
#SBATCH --job-name=visual-geometry-omega
#SBATCH --partition=gpus
#SBATCH --nodelist=haicluster2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=UNLIMITED
#SBATCH --requeue
#SBATCH --output=/p/haicluster/llama/blablador-fleet/logs/visual-geometry-omega/%j.slurm.out
#SBATCH --error=/p/haicluster/llama/blablador-fleet/logs/visual-geometry-omega/%j.slurm.err
# VGGT-Omega-1B mapping behind the OpenAI API, registered with synapse as alias-visual-geometry-omega.
# Runs next to visual-geometry.slurm (VGGT-1B, port 8030) from its own venv, .venv-omega (torch>=2.6).
# The repository copy is the source; the fleet runs the copy in blablador-fleet/launchers,
# and the code is checked out in FastChat/depth-map-through-vllm-router.
set -euo pipefail
ROOT=/p/haicluster/llama/FastChat
F=/p/haicluster/llama/blablador-fleet
APP=/p/haicluster/llama/FastChat/depth-map-through-vllm-router
PORT=8031
cd "$APP"
VLLM_API_KEY="$(cat "$ROOT/synapse-external/backend-key-$PORT")"
export VLLM_API_KEY
export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS:-0}
export OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
ENGINE= CLIENT=
cleanup() { [[ -z "$CLIENT" ]] || kill -TERM "$CLIENT" 2>/dev/null || true; [[ -z "$ENGINE" ]] || kill -TERM -- "-$ENGINE" 2>/dev/null || true; }
trap cleanup EXIT
trap 'exit 0' TERM INT
setsid "$APP/.venv-omega/bin/python" -m mapper.server --model "$ROOT/models/VGGT-Omega/vggt_omega_1b_512.pt" \
 --host 0.0.0.0 --port $PORT --served-model-name alias-visual-geometry-omega vggt-omega-1b \
 --data-dir "$ROOT/visual-geometry-omega-sessions" \
 >"$F/logs/visual-geometry-omega/$SLURM_JOB_ID.engine.log" 2>&1 &
ENGINE=$!
SYNAPSE_EXTERNAL_CLIENT="$F/bin/synapse-client-metadata-20261028" /usr/bin/python3 "$ROOT/synapse-external/supervisor.py" "$SLURM_JOB_ID" $PORT 0 >"$F/logs/visual-geometry-omega/$SLURM_JOB_ID.synapse.log" 2>&1 &
CLIENT=$!
wait -n "$ENGINE" "$CLIENT"
