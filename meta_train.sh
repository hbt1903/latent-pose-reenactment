DATASET_ROOT="/content/latent-pose-reenactment"
IDENTITY_NAME="id1"
MAX_BATCH_SIZE=8             # pick the largest possible, start with 8 and decrease until it fits in VRAM
CHECKPOINT_PATH="/content/latent-pose-reenactment/model/latent-pose-release.pth"
OUTPUT_PATH="/content/latent-pose-reenactment/outputs/"       # a directory for outputs, will be created
RUN_NAME="tony_hawk_take_1"  # give your run a name if you want

# Important. See the note below
TARGET_NUM_ITERATIONS=10

# Don't change these
NUM_IMAGES=`ls -1 "$DATASET_ROOT/images-cropped/$IDENTITY_NAME" | wc -l`
BATCH_SIZE=$((NUM_IMAGES<MAX_BATCH_SIZE ? NUM_IMAGES : MAX_BATCH_SIZE))
ITERATIONS_IN_EPOCH=$(( NUM_IMAGES / BATCH_SIZE ))

cd latent-pose-reenactment
mkdir -p $OUTPUT_PATH
python3 train.py \
    --config finetuning-base                 \
    --checkpoint_path "$CHECKPOINT_PATH"     \
    --data_root "$DATASET_ROOT"              \
    --train_split_path "$IDENTITY_NAME"      \
    --batch_size $BATCH_SIZE                 \
    --num_epochs $(( (TARGET_NUM_ITERATIONS + ITERATIONS_IN_EPOCH - 1) / ITERATIONS_IN_EPOCH )) \
    --experiments_dir "$OUTPUT_PATH"         \
    --experiment_name "$RUN_NAME"
