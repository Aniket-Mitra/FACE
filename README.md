# FACE: Counterfactual Pairwise Evaluation of Physiognomic Social Bias in Vision-Language Models

This repository contains the code, dataset structure, and experimental results for **FACE**, a benchmark for systematically evaluating appearance-conditioned social bias in Vision-Language Models (VLMs).

FACE uses controlled synthetic portraits and counterfactual pairwise comparisons to examine how demographic and appearance-related attributes are associated with VLM judgments across prosocial and accusatory social contexts.

## Paper

**Looks Can Mislead: Auditing Appearance-Based Bias in Vision-Language Models**

This repository accompanies the above work and contains the resources used for the FACE benchmark and its associated experiments.

## Repository Structure

```text
FACE/
├── dataset/    # Base portraits, counterfactual variants, and image-pair definitions
├── code/       # Dataset generation, pairing, VLM inference, and analysis scripts
└── results/    # Experimental outputs and derived evaluation metrics
```

### `dataset/`

Contains the synthetic base portraits, controlled counterfactual appearance variants, and JSON files defining the image pairs used during evaluation.

A separate README inside the `dataset/` folder describes the dataset organization and pairing structure in more detail.

### `code/`

Contains the scripts used for:

- Base portrait generation
- Counterfactual appearance generation
- Controlled image-pair construction
- VLM inference
- Response parsing and bias analysis
- Abstention-enabled evaluation
- Controlled multi-attribute experiments

Each Python file contains a short description of its purpose, expected input, and output. The `code/` README provides an overview of how the scripts are organized.

### `results/`

Contains experimental outputs and derived evaluation metrics from the FACE experiments.

## FACE Dataset

The full FACE dataset is available at:

**[Download the FACE Dataset](https://www.jioaicloud.com/l/?u=dr2XIL6SoUw14fFNjjq8rJ_N5QMzSPdwrIVo7VKvcd8=VaU)**

## Evaluation Overview

FACE evaluates VLM behavior through controlled pairwise portrait comparisons. The benchmark examines associations involving demographic and appearance-related attributes under both **prosocial** and **accusatory** social judgments.

The evaluation includes:

- Selection Frequency
- Log-Odds Ratio (LOR)
- Parsing Success
- Position Bias
- Swap Consistency
- Abstention Behavior
- Multi-Attribute Context Sensitivity

The primary analysis uses controlled pairwise comparisons to characterize appearance-conditioned associations in model outputs. Additional experiments examine whether models abstain from unsupported judgments when provided with a `Cannot Determine` option and whether observed associations change under multi-attribute visual contexts.

## Responsible Use and Interpretation

FACE is designed for auditing and research on social bias in VLMs. The dataset consists of synthetically generated portraits and controlled appearance modifications.

The demographic and appearance labels represent experimental generation conditions. Associations measured by FACE describe model behavior under the benchmark protocol and should **not** be interpreted as properties of the depicted individuals or as evidence that physical appearance predicts real-world behavior, personality, morality, trustworthiness, or intent.

## Citation

Citation information will be added upon publication.
