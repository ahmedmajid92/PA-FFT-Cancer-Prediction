import os
import json
import torch
import numpy as np
from src.preprocessor import GeneDataProcessor

def main():
    # Ensure reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    print("=== Starting Phase 1: Data Ingestion & Intersection ===")
    
    processor = GeneDataProcessor()

    # 1. Load Data
    # Note: Ensure data files are present in data/raw/... as per README/Instructions
    mendeley_genes = processor.load_mendeley()
    gse_genes = processor.load_gse()
    tcga_genes = processor.load_tcga()
    
    pathways = processor.load_pathways()

    if not gse_genes or not tcga_genes:
        print("\nCRITICAL WARNING: One or more key datasets (GSE, TCGA) failed to load.")
        print("Please check if the data files are placed correctly in 'data/raw/'.")
        print("Terminating Phase 1.")
        return

    # 2. Compute Intersection
    vocab = processor.intersect_genes(mendeley_genes, gse_genes, tcga_genes)
    print(f"\nFinal Vocabulary Size: {len(vocab)} genes")

    # 3. Create Pathway Mask
    mask, pathway_names = processor.create_pathway_mask(vocab, pathways)

    # 4. Save Artifacts
    output_dir = processor.processed_dir
    os.makedirs(output_dir, exist_ok=True)

    # Save Vocab
    vocab_path = os.path.join(output_dir, "gene_vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"Saved gene vocabulary to {vocab_path}")

    # Save Pathway Mask
    mask_path = os.path.join(output_dir, "pathway_mask.pt")
    torch.save(mask, mask_path)
    print(f"Saved pathway mask to {mask_path}")

    # Save Pathway Names (for reference/reproducibility)
    pnames_path = os.path.join(output_dir, "pathway_names.json")
    with open(pnames_path, "w") as f:
        json.dump(pathway_names, f, indent=2)
    print(f"Saved pathway names to {pnames_path}")

    print("\n=== Phase 1 Completed Successfully ===")
    print(f"Summary:")
    print(f"- Genes (Vocab): {len(vocab)}")
    print(f"- Pathways: {len(pathway_names)}")
    print(f"- Mask Shape: {mask.shape}")

if __name__ == "__main__":
    main()
