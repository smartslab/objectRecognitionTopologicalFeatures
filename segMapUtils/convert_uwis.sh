CURRENT_DIR=$(pwd)
WORK_DIR="./uwis"
PQR_ROOT="${WORK_DIR}/data"
SEG_FOLDER="${PQR_ROOT}/foreground_livingroom"
SEMANTIC_SEG_FOLDER="${PQR_ROOT}/foregroundraw_livingroom"
# Build TFRecords of the dataset.
OUTPUT_DIR="${WORK_DIR}/tfrecords_livingroom"
mkdir -p "${OUTPUT_DIR}"
IMAGE_FOLDER="${PQR_ROOT}/JPEGImages_livingroom"
LIST_FOLDER="${PQR_ROOT}/ImageSets_livingroom"
echo "Converting scene dataset..."
python ./build_ara_scene_data.py \
  --image_folder="${IMAGE_FOLDER}" \
  --semantic_segmentation_folder="${SEMANTIC_SEG_FOLDER}" \
  --list_folder="${LIST_FOLDER}" \
  --image_format="jpg" \
  --output_dir="${OUTPUT_DIR}"
