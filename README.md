# Assignment 2 - SASRec on MovieLens 1M

This repository contains our implementation of SASRec (Self-Attentive Sequential Recommendation) for Assignment 2 in Recommender Systems.

The project trains and evaluates a sequential recommendation model on the MovieLens 1M dataset using PyTorch and reports the required ranking metrics:

- NDCG@10
- NDCG@20
- Recall@10
- Recall@20

--------------------------------------------------
Project Structure
--------------------------------------------------

.
├── MainGBCE.py
├── ModelGBCE.py
├── UtilsGCrossEntropy.py
├── ExperimentsGBCE.py
├── HelperGBCE.py
├── ratings.dat
├── results/
├── logs/
└── README.md

Main files

- MainGBCE.py
  Main training script for a single SASRec run using BCE loss and early stopping.

- ModelGBCE.py
  SASRec model implementation, including:
  - item embeddings
  - positional embeddings
  - self-attention blocks
  - causal masking
  - feedforward layers
  - layer normalization
  - prediction function

- UtilsGCrossEntropy.py
  Data preprocessing, leave-one-out split, batch sampling, and evaluation functions.

- ExperimentsGBCE.py
  Script for running the required ablation experiments:
  - number of self-attention blocks
  - hidden size
  - number of attention heads
  - maximum sequence length

- HelperGBCE.py
  Utility functions for plotting learning curves.

--------------------------------------------------
Dataset
--------------------------------------------------

This project uses the MovieLens 1M dataset.

Expected input format:

userId::movieId::rating::timestamp

Preprocessing rules

- Ratings >= 4 are treated as positive interactions
- Ratings < 4 are ignored
- Each user's interactions are sorted chronologically
- Users with fewer than 5 interactions are removed
- Leave-one-out split:
  - training: all but the last two interactions
  - validation: second-to-last interaction
  - test: last interaction

Place the dataset file as:

ratings.dat

in the project root folder.

--------------------------------------------------
Requirements
--------------------------------------------------

Python packages
- Python 3.x
- PyTorch
- NumPy
- Pandas
- Matplotlib

Example lab environment

On the lab machine, we used the module system:

module purge
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load matplotlib/3.7.2-gfbf-2023a

--------------------------------------------------
How to Run
--------------------------------------------------

1. Single training run

Run one complete training session with early stopping:

python MainGBCE.py

This will:
- preprocess the data
- train SASRec
- evaluate on validation data each epoch
- save the best checkpoint
- evaluate on the test set at the end

--------------------------------------------------
2. Single experiment run with plots
--------------------------------------------------

python ExperimentsGBCE.py --mode single

This generates:
- a single run result JSON
- train loss curve
- validation NDCG@10 curve

Saved inside:

results/

--------------------------------------------------
3. Run the required ablation experiments
--------------------------------------------------

python ExperimentsGBCE.py --mode all_required --n_repetitions 1 --verbose

This runs the required comparisons for:

- num_blocks
- hidden_units
- num_heads
- maxlen

Example with more repetitions

python ExperimentsGBCE.py --mode all_required --n_repetitions 3 --verbose

--------------------------------------------------
4. Run individual ablations
--------------------------------------------------

Number of transformer blocks
python ExperimentsGBCE.py --mode ablation_blocks --n_repetitions 1 --verbose

Hidden size
python ExperimentsGBCE.py --mode ablation_hidden --n_repetitions 1 --verbose

Number of attention heads
python ExperimentsGBCE.py --mode ablation_heads --n_repetitions 1 --verbose

Maximum sequence length
python ExperimentsGBCE.py --mode ablation_maxlen --n_repetitions 1 --verbose

--------------------------------------------------
Running in Background
--------------------------------------------------

For longer experiments, you can run them in the background:

mkdir -p logs
nohup python ExperimentsGBCE.py --mode all_required --n_repetitions 3 --verbose > logs/all_required_bce_r3.log 2>&1 &

Check progress with:

tail -f logs/all_required_bce_r3.log

Check if the process is still running:

ps -fu $USER | grep ExperimentsGBCE.py

Check GPU usage:

nvidia-smi

--------------------------------------------------
Output Files
--------------------------------------------------

The scripts save outputs to the results/ folder, including:

- model checkpoints (.pth)
- single-run JSON result files
- ablation summary CSV files
- ablation summary JSON files
- learning curve plots (.png)

Examples:

results/single_bce_result.json
results/single_bce_train_loss.png
results/single_bce_valid_ndcg10.png
results/ablation_num_blocks_summary.csv
results/ablation_num_blocks_curve.png

--------------------------------------------------
Notes
--------------------------------------------------

- The final working version uses BCEWithLogitsLoss with negative sampling.
- Evaluation is based on ranking the true next item against sampled negative items.
- The implementation uses early stopping based on validation NDCG@10.
- GPU is recommended for experiments, especially when running multiple configurations.

--------------------------------------------------
Authors
--------------------------------------------------

- Jesus Elenes Uriarte
- Muhamad Iqbal Arsa
- Santiago Moreno Mercado