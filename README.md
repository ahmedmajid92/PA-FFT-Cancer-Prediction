# Pathway-Aware Feature-Tokenizing Transformer (PA-FFT) for Cancer Prediction

This project implements a novel deep learning architecture for cancer prediction using multi-omics data (Gene Expression) and biological pathway knowledge. The model leverages a **Pathway-Aware Feature Tokenizer** to group genes into biological units before processing them with a Transformer.

## Project Structure

- `data/`
  - `raw/`: Contains original datasets (Mendeley, GSE45827, TCGA, MSigDB).
  - `processed/`: Contains generated artifacts (vocabularies, masks, tensors).
- `src/`: Source code modules.
- `main_phase1.py`: Execution script for Phase 1.

## Phase 1: Data Ingestion & Intersection

**Goal**: Create a unified gene vocabulary across diverse datasets ensuring biological relevance.

### Implemented Logic

1.  **Data Loading**:
    - **Mendeley**: Benchmark dataset (Gene Symbols).
    - **TCGA PANCAN**: Large-scale pre-training dataset (Gene Symbols).
    - **GSE45827**: Challenge dataset (Affymetrix Probes).
    - **MSigDB**: Knowledge base (KEGG & REACTOME pathways).

2.  **Probe-to-Gene Mapping**:
    - Since GSE45827 uses **Affymetrix Probe IDs** (e.g., `1007_s_at`) and TCGA uses **Gene Symbols** (e.g., `DDR1`), direct intersection is impossible (0 matches).
    - **Solution**: We use the `mygene` library to query the human genome database and map ~30k probes to ~15k unique gene symbols.

3.  **Intersection Strategy**:
    - **Primary Goal**: Maximize the feature space while ensuring compatibility between the Pre-training (TCGA) and Fine-tuning (GSE) datasets.
    - **Result**:
      - Intersection (All 3 datasets): ~918 genes (Too small).
      - **Intersection (GSE ∩ TCGA): ~12,166 genes** (Selected).
      - This vocabulary covers ~35 genes per pathway on average.

### Artifacts Generated

- `data/processed/gene_vocab.json`: The fixed vocabulary of 12,166 common genes.
- `data/processed/pathway_mask.pt`: A binary tensor `[Num_Pathways, Num_Genes]` indicating which genes belong to which pathway.
- `data/processed/pathway_names.json`: The list of 2,683 pathway names.

## Usage

1.  **Install Environment**:
    ```powershell
    conda env create -f environment.yml
    conda activate pa_fft_cancer
    ```
2.  **Run Phase 1**:
    ```powershell
    python main_phase1.py
    ```
    _Note: The first run requires internet access to perform the probe mapping via MyGene.info._
