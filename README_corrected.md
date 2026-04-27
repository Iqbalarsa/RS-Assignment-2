Assignment 2 - SASRec on MovieLens 1M

This repository contains our implementation of SASRec (Self-Attentive Sequential Recommendation) for Assignment 2 in Recommender Systems.

The project trains and evaluates a sequential recommendation model on the MovieLens 1M dataset using PyTorch and reports the required ranking metrics:

- Recall@10
- Recall@20
- NDCG@10
- NDCG@20

The final evaluation is performed using full-ranking evaluation over all items.


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


Main files:

- MainGBCE.py
  Main training script for a single SASRec run using BCE loss, negative sampling, validation evaluation, and early stopping.

- ModelGBCE.py
  SASRec model implementation, including:
  - item embeddings
  - positional embeddings
  - masked self-attention blocks
  - causal attention masking
  - feedforward layers
  - dropout
  - layer normalization
  - prediction function for ranking candidate items

- UtilsGCrossEntropy.py
  Data preprocessing, leave-one-out split, batch sampling, negative sampling, and full-ranking evaluation functions.

- ExperimentsGBCE.py
  Script for running:
  - a single baseline run
  - all required ablation experiments
  - individual ablations
  - the final best configuration with multiple random seeds
  - learning curve plots with error bands

- HelperGBCE.py
  Utility functions for plotting learning curves.


--------------------------------------------------
Dataset
--------------------------------------------------

This project uses the MovieLens 1M dataset.

Expected input format:

userId::movieId::rating::timestamp

Preprocessing rules:

- Ratings >= 4 are treated as positive interactions.
- Ratings < 4 are ignored.
- Each user's interactions are sorted chronologically by timestamp.
- Users with fewer than 5 interactions are removed.
- User IDs and item IDs are reindexed from 1.
- ID 0 is reserved for padding.
- A leave-one-out split is used:
  - training: all interactions except the last two
  - validation: second-to-last interaction
  - test: last interaction

Place the dataset file as:

ratings.dat

in the project root folder.


--------------------------------------------------
Requirements
--------------------------------------------------

Python packages:

- Python 3.x
- PyTorch
- NumPy
- Pandas
- Matplotlib

Example lab environment:

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
- build chronological user sequences
- split the dataset into train, validation, and test sets
- train the SASRec model using BCEWithLogitsLoss and negative sampling
- evaluate on the validation set after each epoch
- save the best checkpoint according to validation NDCG@10
- evaluate the best model on the test set


--------------------------------------------------
2. Single experiment run with plots
--------------------------------------------------

python ExperimentsGBCE.py --mode single

This generates:
- a single run result JSON file
- a training loss curve
- a validation NDCG@10 curve

Outputs are saved inside:

results/


--------------------------------------------------
3. Run all required ablation experiments
--------------------------------------------------

python ExperimentsGBCE.py --mode all_required --n_repetitions 3 --verbose

This runs the required comparisons for:

- number of Transformer blocks
- hidden size
- number of attention heads
- maximum sequence length

The ablation experiments compare the following values:

- num_blocks: 1, 2, 3
- hidden_units: 32, 50, 100
- num_heads: 1, 2, 5
- maxlen: 50, 100, 200

The resulting CSV summaries and plots are saved in:

results/


--------------------------------------------------
4. Run individual ablations
--------------------------------------------------

Number of Transformer blocks:

python ExperimentsGBCE.py --mode ablation_blocks --n_repetitions 3 --verbose

Hidden size:

python ExperimentsGBCE.py --mode ablation_hidden --n_repetitions 3 --verbose

Number of attention heads:

python ExperimentsGBCE.py --mode ablation_heads --n_repetitions 3 --verbose

Maximum sequence length:

python ExperimentsGBCE.py --mode ablation_maxlen --n_repetitions 3 --verbose


--------------------------------------------------
5. Run the final best configuration
--------------------------------------------------

The selected final configuration after the ablation study is:

- maxlen = 200
- hidden_units = 100
- num_blocks = 1
- num_heads = 1
- batch_size = 128
- learning_rate = 0.001
- dropout_rate = 0.2
- max_epochs = 50
- early_stopping_patience = 5

Run the final configuration with three random seeds using:

python ExperimentsGBCE.py --mode final_best --n_repetitions 3 --verbose

This generates:
- final_best_config_summary.json
- final_best_config_details.csv
- final_best_config_valid_ndcg10_std.png
- final_best_config_valid_ndcg10_stderr.png
- final_best_config_train_loss_std.png
- final_best_config_train_loss_stderr.png


--------------------------------------------------
Evaluation
--------------------------------------------------

The model is evaluated with:

- Recall@10
- Recall@20
- NDCG@10
- NDCG@20

Validation evaluation:

- The prefix consists of the training sequence.
- The target is the validation item.
- Previously seen training items are masked.
- The model ranks the validation item against all items.

Test evaluation:

- The prefix consists of the training sequence plus the validation item.
- The target is the test item.
- Previously seen training and validation items are masked.
- The model ranks the test item against all items.

Therefore, the reported validation and test metrics are based on full-ranking evaluation over all items, not sampled evaluation.


--------------------------------------------------
Training Objective
--------------------------------------------------

The model is trained using next-item prediction.

For each user sequence, the training sampler constructs:

- input sequence
- positive next-item targets
- negative sampled items

The loss function is Binary Cross-Entropy with logits:

- positive items are labeled as 1
- negative sampled items are labeled as 0
- padding positions are ignored when computing the loss

The optimizer is Adam with learning rate 0.001.


--------------------------------------------------
Running in Background
--------------------------------------------------

For longer experiments, the scripts can be run in the background.

Example: all required ablations

mkdir -p logs
nohup bash -lc '
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load matplotlib/3.7.2-gfbf-2023a
cd /path/to/project
python ExperimentsGBCE.py --mode all_required --n_repetitions 3 --verbose
' > logs/all_required_bce_r3.log 2>&1 &

Example: final best configuration

mkdir -p logs
nohup bash -lc '
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load matplotlib/3.7.2-gfbf-2023a
cd /path/to/project
python ExperimentsGBCE.py --mode final_best --n_repetitions 3 --verbose
' > logs/final_best_config_r3.log 2>&1 &

Check progress with:

tail -f logs/all_required_bce_r3.log

or:

tail -f logs/final_best_config_r3.log

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
- final best configuration summaries and curves

Examples:

results/single_bce_result.json
results/single_bce_train_loss.png
results/single_bce_valid_ndcg10.png
results/ablation_num_blocks_summary.csv
results/ablation_hidden_units_summary.csv
results/ablation_num_heads_summary.csv
results/ablation_maxlen_summary.csv
results/ablation_hidden_units_curve.png
results/ablation_num_blocks_curve.png
results/ablation_num_heads_curve.png
results/ablation_maxlen_curve.png
results/final_best_config_summary.json
results/final_best_config_details.csv
results/final_best_config_valid_ndcg10_std.png
results/final_best_config_train_loss_std.png


--------------------------------------------------
Final Reported Results
--------------------------------------------------

The final selected configuration was evaluated over three random seeds.

Final configuration:

- maxlen = 200
- hidden_units = 100
- num_blocks = 1
- num_heads = 1

Final test performance over three runs:

- Recall@10 = 0.194 ± 0.003
- NDCG@10 = 0.102 ± 0.001
- Recall@20 = 0.289 ± 0.004
- NDCG@20 = 0.126 ± 0.001


--------------------------------------------------
Notes
--------------------------------------------------

- The model is implemented manually in PyTorch.
- No ready-made SASRec package is used.
- Training uses BCEWithLogitsLoss with negative sampling.
- Evaluation uses full ranking over all items.
- Early stopping is based on validation NDCG@10.
- GPU is recommended for running the experiments, especially for full-ranking evaluation and multiple ablation runs.


--------------------------------------------------
Authors
--------------------------------------------------

- Jesus Elenes Uriarte
- Muhamad Iqbal Arsa
- Santiago Moreno Mercado
