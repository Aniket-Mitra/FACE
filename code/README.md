# FACE Code

This folder contains the code used for dataset construction, image pairing, VLM inference, and analysis in the **FACE benchmark**.

Each Python file contains a short description at the beginning explaining its purpose, input, output, and role in the experimental pipeline. Please refer to the individual code files for implementation details.

## Main Code

The main scripts cover the primary FACE pipeline, including:

- Base portrait generation
- Counterfactual appearance generation
- Controlled image-pair construction
- VLM inference
- Response parsing and analysis

The analysis includes selection frequency, log-odds ratio (LOR), and model-reliability measurements such as parsing success and position bias.

## `abstention/`

The `abstention/` folder contains the code for the **abstention-enabled evaluation**.

These scripts largely follow the same inference and analysis procedures as the corresponding main scripts. The primary difference is the addition of a **`Cannot Determine`** response option alongside **A** and **B**.

This experiment evaluates whether models abstain from making an appearance-based social judgment when they are not forced to choose between the two presented individuals.

## `controlled_experiment/`

The `controlled_experiment/` folder contains the code for the **controlled multi-attribute experiment**.

The scripts follow the same overall experimental procedure but use **model-specific appearance-attribute configurations**. Positive and negative attribute settings are selected for each evaluated model based on the primary analysis and are used to construct the controlled multi-attribute conditions.

The generation procedure and experimental structure are described in detail inside `generate_llama4scout.py`. The other model-specific scripts follow the same overall procedure using their corresponding model configurations.

## Code Navigation

For detailed information about a particular script, please open the corresponding Python file. Each code file contains an introductory description explaining what the script does and how it fits within the FACE experimental pipeline.