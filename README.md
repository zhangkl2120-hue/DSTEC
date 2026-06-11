# DSTEC: Decoupled Spatio-Temporal Expert Collaboration

This repository provides the core implementation of **DSTEC**, a decoupled spatio-temporal expert collaboration framework for robust traffic flow forecasting under anomalous or incomplete traffic observations.

DSTEC explicitly separates temporal pattern modeling and spatial topology modeling into two expert branches, and coordinates them through a stability-guided expert collaboration mechanism. The implementation includes detrended temporal pattern cache construction, LLM-guided topology reasoning, expanded spatial graph construction, and stage-wise expert mutual learning.

## Overview

DSTEC consists of three main components:

1. **Temporal Expert with Detrended Pattern Cache**

   The temporal expert first removes smooth traffic trends from historical traffic segments and constructs a detrended pattern cache based on residual evolution patterns. Given an anomalous input window, it retrieves similar historical residual patterns according to the observed context and uses them to supplement missing observations. The restored sequence is further used to construct a residual-aware temporal relation graph for temporal prediction.

2. **Spatial Expert with LLM-Guided Topology Reasoning**

   The spatial expert constructs candidate latent topological links from structural cues, such as multi-hop neighborhoods, centrality statistics, common neighbors, and local structural support. A rubric-guided LLM relation evaluator is then used to reason over the plausibility and risk of each candidate relation. Verified relations are added to the original traffic topology to build an expanded spatial graph.

3. **Spatio-Temporal Expert Mutual Learning**

   The temporal and spatial experts are trained with a stage-wise collaboration mechanism. At each stage, a historical-best checkpoint is used as the reference model, and perturbation-based stability is used to select the more stable expert as the teacher. The selected expert provides guidance to the other expert, enabling mutual learning and reducing the disturbance caused by unreliable spatio-temporal interaction.

## Repository Structure

```text
DSTEC/
├── train_ours.py                         # Training script for DSTEC
├── model/
│   ├── ours.py                           # Main DSTEC model
│   └── pattern_bank.py                   # Detrended pattern cache / restoration module
├── scripts/
│   ├── build_pattern_bank_allinone.py    # Build detrended temporal pattern cache
│   └── build_space_graph_allinone.py     # Build LLM-guided expanded spatial graph
├── configurations/
│   └── PEMS04_astgcn.conf                # Example configuration file
├── lib/
│   ├── utils.py                          # Data loading and utility functions
│   └── metrics.py                        # Evaluation metrics
├── requirements.txt                      # Python dependencies
└── README.md
````

## Dataset Sources

The traffic datasets used in this project are based on the California Performance Measurement System (PeMS). PeMS traffic data are collected from highway loop detectors deployed across California freeway systems.

Users may refer to the following public sources for raw or processed PeMS-style datasets:

* Caltrans PeMS official source: https://pems.dot.ca.gov/
* ASTGCN-style PeMSD4 and PeMSD8 datasets: https://github.com/guoshnBJTU/ASTGCN-2019-pytorch

Due to dataset size and redistribution constraints, this repository does not include raw datasets or large processed files. Users should download the corresponding datasets and organize them into the ASTGCN-style data format.

An example data structure is:

```text
data/
└── PEMS04/
    ├── PEMS04.npz
    ├── PEMS04_r1_d0_w0_astcgn.npz
    ├── PEMS04_As_ours.npy
    └── PEMS04_pattern_bank_hybrid_U256_P256_r1d0w0.npz
```

Here, `PEMS04_As_ours.npy` denotes the expanded spatial adjacency matrix constructed by the LLM-guided topology reasoning module, and `PEMS04_pattern_bank_hybrid_U256_P256_r1d0w0.npz` denotes the temporal pattern cache used by the temporal expert.

## Environment

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

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

## Data Preparation

The current implementation follows the commonly used ASTGCN-style traffic data format. Before training DSTEC, users should prepare:

1. the processed traffic sequence file;
2. the original road network adjacency matrix;
3. the detrended temporal pattern cache;
4. the LLM-guided expanded spatial graph.

For PeMS04, the expected files are:

```text
data/
└── PEMS04/
    ├── PEMS04.npz
    ├── PEMS04_r1_d0_w0_astcgn.npz
    ├── PEMS04_As_ours.npy
    └── PEMS04_pattern_bank_hybrid_U256_P256_r1d0w0.npz
```

## Build Detrended Temporal Pattern Cache

Example command:

```bash
python scripts/build_pattern_bank_allinone.py \
  --config configurations/PEMS04_astgcn.conf \
  --variants U256_P256 U192_P64
```

This script constructs temporal pattern caches from the prepared training windows. Historical traffic segments are detrended before being stored, so that the cached patterns focus more on local residual dynamics rather than smooth global trends.

## Build LLM-Guided Spatial Graph

For debugging or checking the pipeline without calling an external LLM API, use:

```bash
python scripts/build_space_graph_allinone.py \
  --config configurations/PEMS04_astgcn.conf \
  --llm stub
```

For real LLM-guided relation verification, set the corresponding API key and use one of the following commands:

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

The script first generates candidate latent spatial relations from structural evidence and then verifies them using a rubric-guided LLM relation evaluator. Candidate relations with high plausibility and low risk are added to the original topology. The resulting expanded spatial graph is saved and reused during model training and inference.

The LLM is only used in the offline spatial graph construction stage and does not participate in iterative model training or online prediction.

## Train DSTEC

Example command:

```bash
python train_ours.py \
  --config configurations/PEMS04_astgcn.conf \
  --as_path data/PEMS04/PEMS04_As_ours.npy \
  --bank_path data/PEMS04/PEMS04_pattern_bank_hybrid_U256_P256_r1d0w0.npz
```

During training, DSTEC uses the detrended pattern cache for temporal restoration and the expanded spatial graph for topology-aware spatial propagation. The temporal and spatial experts are coordinated through stage-wise stability-guided expert collaboration.

## Notes

This repository currently provides the core implementation of DSTEC, including:

* temporal expert modeling with detrended pattern cache;
* residual-aware temporal relation learning;
* LLM-guided topology reasoning and spatial graph expansion;
* stage-wise stability-based expert selection;
* stability-guided spatio-temporal expert collaboration.

Raw datasets and large processed files are not included due to file size and redistribution constraints. Users should download the corresponding PeMS datasets from public sources and follow the expected data structure described above.

