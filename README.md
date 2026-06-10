# DSTEC: Decoupled Spatio-Temporal Expert Collaboration

This repository provides the core implementation of **DSTEC**, a traffic flow forecasting framework designed for robust prediction under anomalous or incomplete traffic observations. The implementation includes temporal pattern restoration, LLM-guided spatial graph expansion, and stage-wise spatio-temporal expert collaboration.

## Overview

DSTEC contains three main components:

1. **Pattern Restoration Temporal Expert**
   A temporal pattern bank is constructed from historical traffic windows. During training, the pattern bank is used to support residual-driven temporal relation learning and temporal expert modeling.

2. **Complex-Topology Knowledge-Guided Spatial Expert**
   Candidate latent spatial relations are generated from structural cues and verified by an LLM-based relation evaluator. Verified relations are added to the original traffic topology to build an expanded spatial graph.

3. **Spatio-Temporal Expert Collaboration**
   Temporal and spatial experts are trained with a stage-wise collaboration mechanism. A historical-best checkpoint is used as a reference model, and perturbation stability is used to select the teacher expert for selective guidance.

## Repository Structure

```text
DSTEC/
├── train_ours.py                         # Training script for DSTEC
├── model/
│   ├── ours.py                           # Main DSTEC model
│   └── pattern_bank.py                   # Pattern bank imputation module
├── scripts/
│   ├── build_pattern_bank_allinone.py    # Build temporal pattern banks
│   └── build_space_graph_allinone.py     # Build LLM-guided spatial graph
├── lib/
│   ├── utils.py                          # Data loading and utility functions
│   └── metrics.py                        # Evaluation metrics
└── README.md
```

## Environment

The code is implemented with PyTorch. The main dependencies include:

```text
python >= 3.8
torch
numpy
scipy
scikit-learn
networkx
tqdm
tensorboardX
requests
```

A detailed environment file will be provided in the full release.

## Data Preparation

The current code follows the commonly used ASTGCN-style traffic data format. Raw traffic datasets and large processed files are not included in this repository due to file size and redistribution constraints.

Expected data files include:

```text
data/
└── PEMS04/
    ├── PEMS04.npz
    ├── PEMS04_r1_d0_w0_astcgn.npz
    ├── PEMS04_As_ours.npy
    └── PEMS04_pattern_bank_hybrid_U256_P256_r1d0w0.npz
```

The complete preprocessing scripts and processed data instructions will be provided in the full release.

## Build Temporal Pattern Bank

Example command:

```bash
python scripts/build_pattern_bank_allinone.py \
  --config configurations/PEMS04_astgcn.conf \
  --variants U256_P256 U192_P64
```

This script constructs uniform, prototype, and hybrid temporal pattern banks from the prepared training windows.

## Build LLM-Guided Spatial Graph

For debugging or checking the pipeline without calling an external LLM API, use:

```bash
python scripts/build_space_graph_allinone.py \
  --config configurations/PEMS04_astgcn.conf \
  --llm stub
```

For real LLM-guided relation verification, set the corresponding API key and use:

```bash
python scripts/build_space_graph_allinone.py \
  --config configurations/PEMS04_astgcn.conf \
  --llm deepseek \
  --space_graph_policy force
```

or:

```bash
python scripts/build_space_graph_allinone.py \
  --config configurations/PEMS04_astgcn.conf \
  --llm openrouter \
  --space_graph_policy force
```

The script first generates candidate latent spatial relations, then verifies them using a rubric-guided LLM relation evaluator. Relations with high plausibility and low risk are added to the original topology.

## Train DSTEC

Example command:

```bash
python train_ours.py \
  --config configurations/PEMS04_astgcn.conf \
  --as_path data/PEMS04/PEMS04_As_ours.npy \
  --bank_path data/PEMS04/PEMS04_pattern_bank_hybrid_U256_P256_r1d0w0.npz
```


## Notes

This repository currently provides the core implementation of the proposed method, including the model architecture, temporal pattern bank construction, LLM-guided spatial graph expansion, and stage-wise expert collaboration training logic.

The current version is intended to help reviewers inspect the main implementation details. Full cleaned reproduction scripts, complete configuration files, and detailed data preprocessing instructions will be released after acceptance.
