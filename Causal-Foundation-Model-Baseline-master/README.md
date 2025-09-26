# SEA Foundation Model - My Version Usage Guide

## Overview
This guide covers the complete workflow: running the SEA (Sample/Estimate/Aggregate) Foundation Model on your data and evaluating the results.

## Prerequisites

### 1. Environment Setup
```bash
# Create conda environment
conda env create -f environment.yml
conda activate sea-foundation

# Install additional evaluation dependencies
pip install gcastle==1.0.4rc1
```

### 2. Required Files
- **Model checkpoints**: `checkpoints/` directory with trained models
- **Configuration files**: `config/` directory with YAML configs
- **Evaluation script**: `evaluate_predictions.py`
- **Source code**: `src/` directory

## Part 1: Running the Model on Your Data

### Data Preparation
1. **Format your data**:
   - CSV format with numerical values
   - Rows = samples, Columns = variables
   - No missing values (impute if necessary)
   - Example: `data/hepar2.csv`

2. **Place data file**:
   ```
   data/
   ├── your_dataset.csv
   └── your_ground_truth.npy (optional, for evaluation)
   ```

### Model Execution

**Run inference**:
```
python src/inference.py --config_file config/aggregator_tf_gies.yaml --run_name hepar2_gies --data_file data/hepar2.csv --save_path outputs/hepar2_test --results_file predictions.npy --checkpoint_path checkpoints/gies_synthetic/model_best_epoch=535_auprc=0.849.ckpt --gpu -1
```


### Expected Output
After successful execution, you'll find:
```
outputs/your_results/
├── predictions.npy          # Main output (N(N-1)/2 predictions)
├── adjacency_matrix.npy     # Reconstructed NxN matrix
├── metrics.json            # Basic statistics
└── config.yaml            # Configuration used
```

## Part 2: Evaluating Results

### Understanding predictions.npy
- **Format**: 1D array of shape `[N(N-1)/2]` where N = number of variables
- **Content**: Edge probability predictions for upper triangular matrix
- **Range**: [0, 1] representing causal relationship strength
- **Reconstruction**: Use `evaluate_predictions.py` to convert back to NxN matrix

### Basic Evaluation (No Ground Truth)
```bash
python evaluate_predictions.py --predictions outputs/your_results/predictions.npy
```

**Output**:
- Summary statistics (min, max, mean, median, std)
- Threshold analysis with edge counts
- Sparsity recommendations
- Visualization-ready matrices

### Full Evaluation (With Ground Truth)
```bash
python evaluate_predictions.py \
    --predictions outputs/your_results/predictions.npy \
    --ground_truth data/your_ground_truth.npy \
    --output evaluation_results.json
```

**Output**:
- All basic evaluation metrics
- **AUROC**: Area under ROC curve
- **AUPRC**: Area under Precision-Recall curve  
- **gcastle metrics**: Comprehensive causal discovery evaluation
- **Threshold optimization**: Best F1, precision, recall scores
- **Performance comparison**: Against random and trivial baselines

### Advanced Evaluation Options
```bash
# Custom thresholds
python evaluate_predictions.py \
    --predictions outputs/your_results/predictions.npy \
    --ground_truth data/your_ground_truth.npy \
    --thresholds 0.01,0.05,0.1,0.2,0.5 \
    --output detailed_evaluation.json

# Batch processing multiple results
python evaluate_predictions.py \
    --predictions outputs/*/predictions.npy \
    --batch_mode \
    --output batch_results/
```

## Complete Workflow Example

### Example: Running on hepar2 dataset
```bash
# 1. Prepare data (already provided)
ls data/hepar2.csv data/hepar2adj.npy

# 2. Run the model
cd src
python inference.py --data ../data/hepar2.csv --output ../outputs/hepar2_test/

# 3. Evaluate results
cd ..
python evaluate_predictions.py \
    --predictions outputs/hepar2_test/predictions.npy \
    --ground_truth data/hepar2adj.npy \
    --output hepar2_evaluation.json

# 4. View results
cat hepar2_evaluation.json
```

## Understanding Output Metrics

### Basic Metrics
- **AUROC**: Higher is better (0.5 = random, 1.0 = perfect)
- **AUPRC**: Higher is better, accounts for class imbalance
- **Precision**: True edges / Predicted edges
- **Recall**: True edges found / Total true edges
- **F1**: Harmonic mean of precision and recall

### gcastle Metrics (when available)
- **SHD**: Structural Hamming Distance (lower is better)
- **SID**: Structural Intervention Distance
- **FDR**: False Discovery Rate
- **TPR**: True Positive Rate
- **FPR**: False Positive Rate

### Threshold Selection Guidelines
- **Conservative (high precision)**: Use threshold ≥ 0.1
- **Balanced**: Use threshold that maximizes F1 score
- **Liberal (high recall)**: Use threshold ≤ 0.05
- **Sparse networks**: Use higher thresholds (0.2+)
- **Dense networks**: Use lower thresholds (0.01-0.05)

## Troubleshooting

### Common Issues
1. **"CUDA not available"**: Model automatically uses CPU, this is normal
2. **Memory errors**: Reduce batch size in config file
3. **Dimension mismatches**: Check data format and preprocessing
4. **Poor performance**: Try different thresholds or model checkpoints

### Performance Tips
- **Large datasets**: Use CPU with multiple cores
- **Multiple runs**: Use different random seeds for robustness
- **Cross-validation**: Split data and evaluate on held-out sets

### File Formats
- **CSV**: Standard comma-separated values
- **NPY**: NumPy binary format for arrays
- **JSON**: Human-readable results and configurations

## Output Interpretation

### Good Results Indicators
- AUROC > 0.7
- AUPRC > 0.1 (depending on sparsity)
- F1 score > 0.1
- Clear threshold-performance trade-offs

### Warning Signs
- AUROC ≈ 0.5 (random performance)
- All predictions near 0 or 1
- No clear optimal threshold
- Extremely sparse or dense predictions

## Next Steps
1. **Visualization**: Use adjacency matrices for network plots
2. **Domain validation**: Check if discovered edges make sense
3. **Intervention planning**: Use high-confidence edges for experiments
4. **Model comparison**: Try different checkpoints or algorithms

For technical details about the predictions format, see `email_explanation.md`.



------------------------------------------------------------------  ORIGINAL -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Sample, estimate, aggregate: A recipe for causal discovery foundation models

Official implementation of SEA (under review).

## Overview

Inspired by foundation models, we propose a causal
discovery framework where a deep learning model is pretrained to resolve
predictions from classical discovery algorithms run over smaller subsets of
variables. This method is enabled by the observations that the outputs from
classical algorithms are fast to compute for small problems, informative of
(marginal) data structure, and their structure outputs as objects remain
comparable across datasets.

If you find our work interesting, please check out our paper to learn more:
[Sample, estimate, aggregate: A recipe for causal discovery foundation
models](http://arxiv.org/abs/2402.01929).

```
@article{wu2024sea,
  title={Sample, estimate, aggregate: A recipe for causal discovery
  foundation models},
  author={Wu, Menghua and Bao, Yujia and Barzilay, Regina and Jaakkola, Tommi},
  journal={arXiv 2402.01929},
  year={2024}
}
```

## Installation

```
conda create -y --name sea pip python=3.10
conda activate sea

pip install tqdm pyyaml numpy==1.23 pandas matplotlib seaborn
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install pytorch-lightning==1.9.0 torchmetrics==1.3.0 causal-learn==0.1.3.7 pulp==2.8.0 wandb
```

SEA was tested using Python 3.10 with PyTorch 1.13-cu116.
We ran training across two A6000 GPUs and inference on one V100 GPU.

## Quickstart

To run inference using our pretrained models, please modify the data and model
paths in `src/inference.sh`, specify the appropriate config file, and run:
```
./src/inference.sh
```
When benchmarking runtimes, it is assumed that `batch_size=1`.
If you do not need runtimes, you may increase `batch_size` for faster
completion.

`DATA_FILE` can now point directly to a raw observational CSV (one row per
sample, one column per variable). The script will convert it into the manifest
format expected by SEA, writing the intermediate `.npy` tensors under
`${SAVE_PATH}/converted_csv/`. Existing manifest CSVs continue to work without
modification.

To train your own SEA, please modify the data and model paths in
`src/train.sh`, specify the appropriate config file, change the wandb
project to your own, and run:
```
./src/train.sh
```
We recommend at least 10-20 data workers per GPU and a batch size of at least
16.

## Models

We provide pretrained weights for 3 versions of SEA under `checkpoints`:
- GIES on synthetic data with (primarily) additive noise
- FCI on synthetic data with (primarily) additive noise
- FCI on SERGIO-simulated data

Please note that, while these models are expected to work well across a
variety of synthetic datasets, there are cases like e.g. sigmoid with
multiplicative noise which are hard, given our limited training data.

## Datasets

You may download our datasets [here](https://zenodo.org/records/10611036).
Sample data split files are specified in `data`.
- Synthetic testing datasets (Erdos-Renyi and scale-free)
- SERGIO testing datasets

Our datasets follow the [DCDI](https://github.com/slachapelle/dcdi) data format.
- Each file has a suffix of `dataset_id`, which distinguishes between datasets
  generated under the same setting.
- `DAG[id].npy` is a NumPy array containing the `N*N` ground truth graph.
- `data[id].npy` is a NumPy array containing `M*N` observational data.
- `data_interv[id].npy` is a NumPy array containing `M*N` interventional data.
- `regimes[id].csv` is a length `M` text file in which every line `i` specifies
  the regime index of sample `i`.
- `interventions[id].csv` is a length `M` text file in which every line `i` specifies
  the nodes intervened in sample `i`, delimited by `,`.

## Results

Our outputs are stored as `.npy` files containing the Python dictionaries used
during evaluation (we rely on `numpy.save(..., allow_pickle=True)`). You can
load them via `numpy.load(path, allow_pickle=True).item()`. Example parsing code
is provided in `examples/SEA-results.ipynb`.

We have uploaded the predictions of all traditional baselines and our models
to the [Zenodo archive](https://zenodo.org/records/10611036) as well.

