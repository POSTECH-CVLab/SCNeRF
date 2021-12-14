WORKSPACE_PATH=$1
IMAGE_PATH=$WORKSPACE_PATH/images
DATABASE_PATH=$WORKSPACE_PATH/database.db
OUTPUT_PATH=$WORKSPACE_PATH/sparse
COLMAP=colmap

DISPLAY=:0 $COLMAP feature_extractor \
    --image_path $IMAGE_PATH \
    --database_path $DATABASE_PATH \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model PINHOLE

DISPLAY=:0 $COLMAP exhaustive_matcher \
    --database_path $DATABASE_PATH \
    --SiftMatching.guided_matching 1

mkdir $OUTPUT_PATH

DISPLAY=:0 $COLMAP mapper --database_path $DATABASE_PATH \
    --image_path $IMAGE_PATH \
    --output_path $OUTPUT_PATH \
    --Mapper.multiple_models 0

python3 -m colmap_utils.read_sparse_model \
    --working_dir $WORKSPACE_PATH \
    --reorganize

python3 -m colmap_utils.post_colmap \
    --working_dir $WORKSPACE_PATH