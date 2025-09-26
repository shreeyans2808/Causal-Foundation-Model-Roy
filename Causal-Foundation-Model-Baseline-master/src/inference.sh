# main script for inference

######### inference params
CUDA=${CUDA:-0}

######### data params

# NOTE: name of YAML file and run save folder
# see ./config for more options
TAG=${TAG:-"aggregator_tf_gies"}
#TAG="baseline"  # baseline never requires training
CONFIG=${CONFIG:-"config/${TAG}.yaml"}

# override the dataset CSV and output locations at runtime
DATA_FILE=${DATA_FILE:-"data/intervention_8160.csv"}
SAVE_PATH=${SAVE_PATH:-"outputs/${TAG}"}
RESULTS_FILE=${RESULTS_FILE:-"predictions.npy"}

PATH_GIES="checkpoints/gies_synthetic/model_best_epoch=535_auprc=0.849.ckpt"
PATH_FCI="checkpoints/fci_synthetic/model_best_epoch=373_auprc=0.842.ckpt"
PATH_SERGIO="checkpoints/fci_sergio/model_best_epoch=341_auprc=0.646.ckpt"

case "$TAG" in
    "aggregator_tf_fci")
        DEFAULT_CKPT=$PATH_FCI
        ;;
    "aggregator_tf_fci_sergio")
        DEFAULT_CKPT=$PATH_SERGIO
        ;;
    "aggregator_tf_gies")
        DEFAULT_CKPT=$PATH_GIES
        ;;
    *)
        DEFAULT_CKPT=$PATH_GIES
        ;;
esac

# set the appropriate --checkpoint_path variable that MATCHES with $TAG
CHECKPOINT_PATH=${CHECKPOINT_PATH:-$DEFAULT_CKPT}

mkdir -p "$SAVE_PATH"

python src/inference.py \
    --config_file $CONFIG \
    --run_name $TAG \
    --gpu $CUDA \
    --data_file $DATA_FILE \
    --save_path $SAVE_PATH \
    --results_file $RESULTS_FILE \
    --checkpoint_path $CHECKPOINT_PATH

