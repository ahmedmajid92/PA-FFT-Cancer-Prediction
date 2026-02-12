import os
import pandas as pd
import numpy as np
import scipy.io
import torch
import gzip

class GeneDataProcessor:
    def __init__(self, data_dir="data"):
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        
        # Paths to specific datasets
        self.mendeley_path = os.path.join(self.raw_dir, "mendeley_data", "data.mat")
        self.gse_path = os.path.join(self.raw_dir, "gse45827", "GSE45827_series_matrix.txt.gz")
        self.tcga_path = os.path.join(self.raw_dir, "tcga_pancan", "EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv")
        self.pathway_path = os.path.join(self.raw_dir, "msigdb", "c2.all.v2026.1.Hs.symbols.gmt")

    def load_mendeley(self):
        """
        Load Mendeley .mat file. 
        Returns: 
            genes (list): List of gene symbols.
            data: The actual data matrix (not strictly needed for intersection, but good to check).
        """
        print(f"Loading Mendeley data from {self.mendeley_path}...")
        try:
            mat = scipy.io.loadmat(self.mendeley_path)
            # Inspect keys to find gene list
            genes = []
            if 'geneIds' in mat:
                # Based on inspection: [array(['AARS'], dtype='<U4'), ...]
                raw_genes = mat['geneIds'].flatten()
                for g in raw_genes:
                    if isinstance(g, np.ndarray) and g.size > 0:
                        genes.append(str(g[0]).strip())
                    else:
                        genes.append(str(g).strip())
            elif 'genes' in mat:
                 genes = [str(g[0]).strip() for g in mat['genes'].flatten()]
            else:
                # Fallback: look for likely candidates
                print("Warning: Could not find explicit 'genes' or 'geneIds' key. Checking other keys...")
                for k in mat.keys():
                    if k.startswith('__'): continue
                    val = mat[k]
                    if val.dtype.kind in {'U', 'S'} and val.ndim <= 2: 
                         genes = [str(g).strip() for g in val.flatten()]
                         print(f"Found potential gene list in key: {k}")
                         break
                if not genes:
                     raise ValueError("Could not locate gene symbols in .mat file")

            print(f"Mendeley: Loaded {len(genes)} genes.")
            return set(genes)
        except Exception as e:
            print(f"Error loading Mendeley: {e}")
            import traceback
            traceback.print_exc()
            return set()

    def load_gse(self):
        """
        Load GSE45827 series matrix.
        Returns:
            genes (list): List of gene symbols.
        """
        print(f"Loading GSE data from {self.gse_path}...")
        try:
            # Series Matrix usually has '!' comment lines. The data starts after them.
            # It is tab separated. Rows are usually keys (genes), cols are samples.
            # ID_REF is usually the first column.
            
            df = pd.read_csv(self.gse_path, compression='gzip', sep='\t', comment='!', index_col=0)
            
            # The index is usually 'ID_REF'. In series matrix, this maps to probes, 
            # but processed matrix often has gene symbols. 
            # If it's probes, we would need a platform mapping. 
            # The prompt implies strictly "common gene intersection", suggesting we expect gene symbols here.
            # If the index is not gene symbols, we might need to map it. 
            # Start with assumption that index contains gene symbols or we can treat them as such for intersection.
            
            genes = df.index.astype(str).tolist()
            # Clean genes
            genes = [g.replace('"', '').strip() for g in genes]
            
            # Check for probe IDs
            sample_genes = genes[:10]
            is_probe = any(g.endswith('_at') or (g[0].isdigit() and '_' in g) for g in sample_genes if g)
            
            if is_probe:
                print("Detected probe IDs in GSE data. Mapping to Gene Symbols using mygene...")
                import mygene
                mg = mygene.MyGeneInfo()

                # Query mygene with manual batching
                all_results = []
                batch_size = 1000
                total_probes = len(genes)
                print(f"Querying {total_probes} probes in batches of {batch_size}...")

                for i in range(0, total_probes, batch_size):
                    batch = genes[i:i+batch_size]
                    try:
                        res = mg.querymany(batch, scopes='reporter', fields='symbol', species='human', verbose=False)
                        all_results.extend(res)
                        print(f"Processed batch {i//batch_size + 1}/{(total_probes + batch_size - 1)//batch_size}")
                    except Exception as e:
                        print(f"Error querying batch {i}: {e}. Retrying once...")
                        import time
                        time.sleep(2)
                        try:
                            res = mg.querymany(batch, scopes='reporter', fields='symbol', species='human', verbose=False)
                            all_results.extend(res)
                        except Exception as e2:
                             print(f"Failed again: {e2}. Skipping batch.")

                mapped_genes = set()
                mapped_count = 0

                for res in all_results:
                    if 'symbol' in res:
                        mapped_genes.add(res['symbol'])
                        mapped_count += 1

                print(f"GSE: Mapped {mapped_count} probes to {len(mapped_genes)} unique gene symbols.")
                return mapped_genes

            print(f"GSE: Loaded {len(genes)} genes/probes (No mapping needed).")
            return set(genes)
        except Exception as e:
            print(f"Error loading GSE: {e}")
            import traceback
            traceback.print_exc()
            return set()

    def load_tcga(self):
        """
        Load TCGA PANCAN data.
        Index format: GeneSymbol|EntrezID (e.g., TP53|7157).
        """
        print(f"Loading TCGA data from {self.tcga_path}...")
        try:
            # Read only the index first to save memory if needed, but we need intersection.
            # The file is large. use chunking if we were processing data, 
            # but for just getting index `usecols=[0]` is efficient.
            
            df_idx = pd.read_csv(self.tcga_path, sep='\t', usecols=[0], index_col=0)
            
            raw_index = df_idx.index.astype(str)
            gene_symbols = []
            
            for item in raw_index:
                if '|' in item:
                    # Split and keep symbol (part 0)
                    symbol = item.split('|')[0]
                    # Handle cases like "?|1234" -> usually unmapped
                    if symbol == '?': 
                         continue
                    gene_symbols.append(symbol)
                else:
                    gene_symbols.append(item)
            
            # Remove duplicates
            unique_genes = set(gene_symbols)
            print(f"TCGA: Loaded {len(unique_genes)} unique genes (from {len(raw_index)} rows).")
            return unique_genes
        except Exception as e:
            print(f"Error loading TCGA: {e}")
            return set()

    def load_pathways(self):
        """
        Parse .gmt file. Filter for KEGG_ or REACTOME_.
        Returns:
            pathways (dict): {PathwayName: [GeneList]}
        """
        print(f"Loading pathways from {self.pathway_path}...")
        pathways = {}
        try:
            with open(self.pathway_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 3: continue
                    name = parts[0]
                    # url = parts[1] # skip
                    genes = parts[2:]
                    
                    if "KEGG_" in name or "REACTOME_" in name:
                        pathways[name] = genes
            
            print(f"Pathways: Loaded {len(pathways)} KEGG/REACTOME pathways.")
            return pathways
        except Exception as e:
            print(f"Error loading pathways: {e}")
            return {}

    def intersect_genes(self, mendeley_genes, gse_genes, tcga_genes):
        """
        Find intersection. Prioritize GSE & TCGA.
        """
        print("Computing intersections...")
        common_all = mendeley_genes & gse_genes & tcga_genes
        common_gse_tcga = gse_genes & tcga_genes
        
        print(f"Intersection (All 3): {len(common_all)} genes")
        print(f"Intersection (GSE & TCGA): {len(common_gse_tcga)} genes")
        
        # Logic: If all 3 < 5000, use GSE & TCGA
        if len(common_all) < 5000:
            print("Intersection of all 3 is too small (<5000). Using GSE & TCGA intersection.")
            vocab = sorted(list(common_gse_tcga))
        else:
            print("Using intersection of all 3 datasets.")
            vocab = sorted(list(common_all))
            
        return vocab

    def create_pathway_mask(self, vocab, pathways):
        """
        Create binary mask (Num_Pathways, Num_Genes).
        Rows: Pathways
        Cols: Genes in Vocab
        """
        print("Creating pathway mask...")
        vocab_map = {gene: i for i, gene in enumerate(vocab)}
        num_genes = len(vocab)
        pathway_names = sorted(list(pathways.keys()))
        num_pathways = len(pathways)
        
        mask = torch.zeros((num_pathways, num_genes), dtype=torch.float32)
        
        coverage_counts = []
        
        for i, pname in enumerate(pathway_names):
            p_genes = pathways[pname]
            count = 0
            for g in p_genes:
                if g in vocab_map:
                    idx = vocab_map[g]
                    mask[i, idx] = 1.0
                    count += 1
            coverage_counts.append(count)
            
        print(f"Mask created. Shape: {mask.shape}")
        print(f"Avg genes per pathway in vocab: {np.mean(coverage_counts):.2f}")
        
        return mask, pathway_names

