cd ..
# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"
DATASET_DIR="datasets"

# Set up the working directories.
PQR_FOLDER="uwis"
EXP_FOLDER="exp/train_on_trainval_set"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${PQR_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PQR_FOLDER}/${EXP_FOLDER}/train"


EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${PQR_FOLDER}/${EXP_FOLDER}/export"


mkdir -p "${EXPORT_DIR}"

NUM_ITERATIONS=17548
CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph.pb"



python "${WORK_DIR}"/export_model.py \
  --logtostderr \
  --checkpoint_path="${CKPT_PATH}" \
  --export_path="${EXPORT_PATH}" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --num_classes=3 \
  --crop_size=513 \
  --crop_size=513 \
  --inference_scales=1.0
