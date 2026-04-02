export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu --extra fa4

source .venv/bin/activate

if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

python -m nanochat.report reset

python -m nanochat.dataset -n 8

python -m nanochat.dataset -n 170 &

DATASET_DOWNLOAD_PID=$!

python -m scripts.tok_train
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval

echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

python -m scripts.base_train --depth=20 --target-param-data-ratio=8 --device-batch-size=4 --run=d20 --resume-from-step 89

python -m scripts.base_eval --device-batch-size=4

