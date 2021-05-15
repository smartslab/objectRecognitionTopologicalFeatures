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
DATASET="${WORK_DIR}/${DATASET_DIR}/${PQR_FOLDER}/tfrecords_livingroom"

mkdir -p "${WORK_DIR}/${DATASET_DIR}/${PQR_FOLDER}/exp"
mkdir -p "${TRAIN_LOGDIR}"


#

NUM_ITERATIONS=20000
python "${WORK_DIR}"/train.py \
  --logtostderr \
  --train_split="train" \
  --dataset="uwis_livingroom" \
  --model_variant="xception_65" \
  --num_clone=2\
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --train_crop_size=513,513 \
  --optimizer=momentum \
  --top_k_percent_pixels=0.01 \
  --hard_example_mining_step=2500 \
  --train_batch_size=4 \
  --log_steps=10 \
  --save_interval_secs=600 \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --tf_initial_checkpoint="${INIT_FOLDER}/deeplabv3_pascal_train_aug/model.ckpt" \
  --initialize_last_layer=false \
  --fine_tune_batch_norm=false \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${DATASET}"
