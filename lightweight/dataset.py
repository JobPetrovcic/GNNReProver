import os
import torch
import argparse
from torch import Tensor
from loguru import logger
from tqdm import tqdm
from typing import Optional, Dict, Any, List

from torch import LongTensor

# New imports to replace GNNDataModule dependencies
from common import Corpus
from retrieval.datamodule import RetrievalDataset
# We need the model to generate embeddings if they don't exist
from retrieval.model import PremiseRetriever 


class LightweightGraphDataset:
    """
    Lightweight graph dataset for GNN training on formal theorem proving tasks.
    
    Dataset dimensions:
        n_premises: Number of mathematical premises (theorems, definitions, lemmas)
        n_contexts: Number of tactic instances (individual proof steps)
        n_files: Number of source files in the corpus
        embedding_dim: Dimension of node embeddings
    Attributes:
        premise_embeddings (Tensor): ReProver embeddings for all premises. Shape: (n_premises, embedding_dim)
        premise_edge_index (LongTensor): Directed adjacency list for the premise graph. Shape: (2, n_premise_edges)
        premise_edge_attr (LongTensor): Edge types for the premise graph. Shape: (n_premise_edges,)
        context_embeddings (Tensor): ReProver embeddings for all tactic instances. Shape: (n_contexts, embedding_dim)
        context_edge_index (LongTensor): Directed edges from premises to tactic instances (bipartite graph). Shape: (2, n_context_edges)
        context_edge_attr (LongTensor): Edge types for the premise-to-instance edges. Shape: (n_context_edges,)
        context_premise_labels (LongTensor): Ground-truth links from instances to correct answer premises. Shape: (2, n_labels)
        train_mask, val_mask, test_mask (Tensor): Boolean masks for data splits, applied to instances. Shape: (n_contexts,)
        premise_pos (Tensor): Start and end positions for each premise [start_line, start_col, end_line, end_col]. Shape: (n_premises, 4)
        context_theorem_pos (Tensor): Theorem position for each tactic instance [line, column]. Shape: (n_contexts, 2)
        premise_to_file_idx_map (LongTensor): Maps premise index to its file index. Shape: (n_premises,)
        context_to_file_idx_map (LongTensor): Maps tactic instance index to its file index. Shape: (n_contexts,)
        file_dependency_matrix (Tensor): Dense adjacency matrix of file imports. 1.0 indicates dependency. Shape: (n_files, n_files)
        file_idx_to_path_map (List[str]): Maps file index to its string path. Length: n_files
        premise_idx_to_name_map (List[str]): List of premise names indexed by premise ID. Length: n_premises
        edge_types_map (Dict[str, int]): Maps edge type names to integer IDs (e.g., 'signature_lctx', 'signature_goal')
        
    Notes:
        - All edge indices follow PyTorch Geometric format: source nodes in row 0, target nodes in row 1
        - Premise and context nodes are separately indexed starting from 0
        - Labels tensor format: [context_indices, premise_indices] for positive premise retrieval
        - File paths are sorted alphabetically for consistent indexing
    """

    def __init__(
        self,
        premise_embeddings: Tensor,
        premise_edge_index: LongTensor,
        premise_edge_attr: LongTensor,
        context_embeddings: Tensor,
        context_edge_index: LongTensor,
        context_edge_attr: LongTensor,
        context_premise_labels: LongTensor,
        train_mask: Tensor, val_mask: Tensor, test_mask: Tensor,
        premise_to_file_idx_map: LongTensor,
        context_to_file_idx_map: LongTensor,
        file_dependency_matrix: Tensor,
        file_idx_to_path_map: List[str],
        premise_idx_to_name_map: List[str],
        edge_types_map: Dict[str, int],
        premise_pos: Tensor,
        context_theorem_pos: Tensor,
    ):
        self.premise_embeddings = premise_embeddings
        self.premise_edge_index = premise_edge_index
        self.premise_edge_attr = premise_edge_attr
        self.context_embeddings = context_embeddings
        self.context_edge_index = context_edge_index
        self.context_edge_attr = context_edge_attr
        self.context_premise_labels = context_premise_labels
        self.train_mask, self.val_mask, self.test_mask = train_mask, val_mask, test_mask
        self.premise_to_file_idx_map = premise_to_file_idx_map
        self.context_to_file_idx_map = context_to_file_idx_map
        self.file_dependency_matrix = file_dependency_matrix
        self.file_idx_to_path_map = file_idx_to_path_map
        self.premise_idx_to_name_map = premise_idx_to_name_map
        self.edge_types_map = edge_types_map
        self.premise_pos = premise_pos
        self.context_theorem_pos = context_theorem_pos

    def to(self, device: torch.device) -> "LightweightGraphDataset":
        """Moves all tensor attributes to the specified device."""
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, Tensor):
                setattr(self, attr_name, attr_value.to(device))
        return self

    @classmethod
    def load_or_create(
        cls,
        save_dir: str,
        data_path: Optional[str] = None,
        corpus_path: Optional[str] = None,
        retriever_ckpt_path: Optional[str] = None,
        gnn_config: Optional[Dict[str, Any]] = None,
    ) -> "LightweightGraphDataset":
        """
        Loads or creates the dataset where each "context" node corresponds to a
        tactic instance, not a unique proof state.
        """
        instance_save_dir = save_dir + "_instances"
        os.makedirs(instance_save_dir, exist_ok=True)
        filenames = {
            "premise_embeddings.pt", "premise_edge_index.pt", "premise_edge_attr.pt",
            "context_embeddings.pt", "context_edge_index.pt", "context_edge_attr.pt",
            "context_premise_labels.pt", "train_mask.pt", "val_mask.pt", "test_mask.pt",
            "premise_to_file_idx_map.pt", "context_to_file_idx_map.pt",
            "file_dependency_matrix.pt", "file_idx_to_path_map.pt",
            "premise_idx_to_name_map.pt", "edge_types_map.pt", "premise_pos.pt",
            "context_theorem_pos.pt",
        }

        if all(os.path.exists(os.path.join(instance_save_dir, fn)) for fn in filenames):
            logger.info(f"Loading instance-based graph dataset from {instance_save_dir}...")
            data_dict = { k.replace('.pt', ''): torch.load(os.path.join(instance_save_dir, k)) for k in filenames }
            return cls(**data_dict)
        else:
            logger.info("Cached instance-based dataset not found. Creating from source...")
            if not all([data_path, corpus_path, retriever_ckpt_path, gnn_config]):
                raise ValueError("All source data paths and config must be provided for first-time creation.")
            data_dict = cls._create_from_source(data_path, corpus_path, retriever_ckpt_path, gnn_config, instance_save_dir)
            return cls(**data_dict)

    @staticmethod
    def _create_from_source(
        data_path: str, corpus_path: str, retriever_ckpt_path: str, gnn_config: Dict[str, Any], save_dir: str
    ) -> Dict[str, Any]:
        logger.info("Processing source data to create dataset...")
        
        # Initialize Corpus
        corpus = Corpus(corpus_path, gnn_config)
        
        # Initialize Dataset Loader (combining all splits to get all instances)
        logger.info("Loading all instances from train/val/test...")
        full_ds = RetrievalDataset(
            [os.path.join(data_path, f"{split}.json") for split in ("train", "val", "test")],
            corpus,
            0, 0, 0, None, # No negatives, no tokenizer needed for raw loading
            is_train=False,
            graph_dependencies_config=gnn_config
        )
        all_examples = full_ds.data
        num_instances = len(all_examples)
        logger.info(f"Data processing complete. Found {num_instances} total tactic instances.")

        # --- Embedding Generation Logic ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        retriever_name = retriever_ckpt_path.replace("/", "_")
        
        premise_embeddings_path = os.path.join(
            os.path.dirname(corpus_path), f"premise_embeddings_{retriever_name}.pt"
        )
        
        node_features = None
        retriever_loaded = False
        retriever = None
        
        def load_retriever_if_needed():
            nonlocal retriever, retriever_loaded
            if not retriever_loaded:
                logger.info("Loading PremiseRetriever model...")
                retriever = PremiseRetriever.load_hf(retriever_ckpt_path, 2048, device)
                retriever_loaded = True
            return retriever

        if os.path.exists(premise_embeddings_path):
            logger.info(f"Loading cached premise embeddings from {premise_embeddings_path}")
            node_features = torch.load(premise_embeddings_path)
        else:
            logger.info("Cached premise embeddings not found. Generating them now...")
            retriever = load_retriever_if_needed()
            retriever.load_corpus(corpus)
            retriever.reindex_corpus(batch_size=4)
            node_features = retriever.corpus_embeddings.cpu()
            torch.save(node_features, premise_embeddings_path)

        # 2. Context Embeddings
        context_embeddings_path = os.path.join(data_path, f"context_embeddings_{retriever_name}.pt")
        context_embeddings_dict = {}

        if os.path.exists(context_embeddings_path):
            logger.info(f"Loading cached context embeddings from {context_embeddings_path}")
            context_embeddings_dict = torch.load(context_embeddings_path)
        else:
            logger.info("Cached context embeddings not found. Generating them now...")
            retriever = load_retriever_if_needed()
            
            unique_contexts = {ex["context"].serialize(): None for ex in all_examples}
            context_list = list(unique_contexts.keys())
            logger.info(f"Found {len(context_list)} unique contexts to embed.")
            
            batch_size = 4
            
            with torch.no_grad():
                for i in tqdm(range(0, len(context_list), batch_size), desc="Preembedding contexts"):
                    batch_contexts = context_list[i : i + batch_size]
                    tokenized = retriever.tokenizer(
                        batch_contexts,
                        padding="longest",
                        max_length=retriever.max_seq_len,
                        truncation=True,
                        return_tensors="pt",
                    )
                    input_ids = tokenized.input_ids.to(device)
                    attention_mask = tokenized.attention_mask.to(device)
                    
                    embeddings = retriever._encode(input_ids, attention_mask).cpu()

                    for j, context_str in enumerate(batch_contexts):
                        context_embeddings_dict[context_str] = embeddings[j]
            
            torch.save(context_embeddings_dict, context_embeddings_path)

        # --- Construct Dataset Tensors ---

        # Premise and File Info
        premise_embeddings = node_features.clone()
        premise_edge_index = corpus.premise_dep_graph.edge_index.clone()
        premise_edge_attr = corpus.premise_dep_graph.edge_attr.clone()
        premise_idx_to_name_map = [p.full_name for p in corpus.all_premises]
        premise_pos_data = [[p.start.line_nb, p.start.column_nb, p.end.line_nb, p.end.column_nb] for p in corpus.all_premises]
        premise_pos = torch.tensor(premise_pos_data, dtype=torch.long)
        edge_types_map = corpus.edge_types_map

        # Build Dense File Dependency Matrix
        file_paths = sorted(list(corpus.transitive_dep_graph.nodes()))
        path_to_file_idx = {path: i for i, path in enumerate(file_paths)}
        premise_to_file_idx_map = torch.tensor([path_to_file_idx[p.path] for p in corpus.all_premises], dtype=torch.long)
        
        num_files = len(file_paths)
        # Initialize dense matrix
        file_dependency_matrix = torch.zeros((num_files, num_files), dtype=torch.float)
        
        if corpus.transitive_dep_graph.edges():
            for u, v in corpus.transitive_dep_graph.edges():
                u_idx = path_to_file_idx.get(u)
                v_idx = path_to_file_idx.get(v)
                if u_idx is not None and v_idx is not None:
                    file_dependency_matrix[u_idx, v_idx] = 1.0

        # Tactic Instance Info
        logger.info("Building tensors based on all tactic instances...")
        embedding_dim = node_features.shape[1]
        context_emb_dtype = next(iter(context_embeddings_dict.values())).dtype
        
        context_embeddings = torch.zeros((num_instances, embedding_dim), dtype=context_emb_dtype)
        context_to_file_idx_map = torch.full((num_instances,), -1, dtype=torch.long)
        context_theorem_pos = torch.full((num_instances, 2), -1, dtype=torch.long)

        # For edges and labels
        src_edges, dst_edges, attr_edges = [], [], []
        label_ctx_indices, label_p_indices = [], []
        lctx_id = edge_types_map.get('signature_lctx')
        goal_id = edge_types_map.get('signature_goal')

        for i, ex in enumerate(tqdm(all_examples, desc="Building instance tensors")):
            # 1. Get the pre-computed embedding for this instance's context string
            context_str = ex["context"].serialize()
            context_embeddings[i] = context_embeddings_dict[context_str]

            # 2. Map this instance to its file index and theorem position
            context_to_file_idx_map[i] = path_to_file_idx.get(ex["context"].path, -1)
            context_theorem_pos[i] = torch.tensor((ex["context"].theorem_pos.line_nb, ex["context"].theorem_pos.column_nb), dtype=torch.long)

            # 3. Build context edges (p -> instance)
            for p_name in ex["lctx_premises"]:
                p_idx = corpus.name2idx.get(p_name)
                if p_idx is not None and lctx_id is not None:
                    src_edges.append(p_idx); dst_edges.append(i); attr_edges.append(lctx_id)
            for p_name in ex["goal_premises"]:
                p_idx = corpus.name2idx.get(p_name)
                if p_idx is not None and goal_id is not None:
                    src_edges.append(p_idx); dst_edges.append(i); attr_edges.append(goal_id)

            # 4. Build ground-truth labels (instance -> p)
            for pos_premise in ex["all_pos_premises"]:
                p_idx = corpus.name2idx.get(pos_premise.full_name)
                if p_idx is not None:
                    label_ctx_indices.append(i); label_p_indices.append(p_idx)

        context_edge_index = torch.tensor([src_edges, dst_edges], dtype=torch.long)
        context_edge_attr = torch.tensor(attr_edges, dtype=torch.long)
        context_premise_labels = torch.tensor([label_ctx_indices, label_p_indices], dtype=torch.long)

        # --- Masks ---
        logger.info("Building train/val/test masks for instances...")
        train_mask, val_mask, test_mask = (torch.zeros(num_instances, dtype=torch.bool) for _ in range(3))
        instance_key_to_idx = {
            (ex["file_path"], ex["full_name"], tuple(ex["start"]), ex["tactic_idx"]): i
            for i, ex in enumerate(all_examples)
        }

        for split, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
            split_json_path = os.path.join(data_path, f"{split}.json")
            if os.path.exists(split_json_path):
                split_ds = RetrievalDataset([split_json_path], corpus, 0, 0, 0, None, False, gnn_config)
                for ex in split_ds.data:
                    key = (ex["file_path"], ex["full_name"], tuple(ex["start"]), ex["tactic_idx"])
                    instance_idx = instance_key_to_idx.get(key)
                    if instance_idx is not None:
                        mask[instance_idx] = True

        # --- Save Data ---
        logger.info(f"Saving processed instance-based data to {save_dir}...")
        data_to_save = {
            "premise_embeddings": premise_embeddings, "premise_edge_index": premise_edge_index, "premise_edge_attr": premise_edge_attr,
            "context_embeddings": context_embeddings, "context_edge_index": context_edge_index, "context_edge_attr": context_edge_attr,
            "context_premise_labels": context_premise_labels, "train_mask": train_mask, "val_mask": val_mask, "test_mask": test_mask,
            "premise_to_file_idx_map": premise_to_file_idx_map, "context_to_file_idx_map": context_to_file_idx_map,
            "file_dependency_matrix": file_dependency_matrix, "file_idx_to_path_map": file_paths,
            "premise_idx_to_name_map": premise_idx_to_name_map, "edge_types_map": edge_types_map,
            "premise_pos": premise_pos, "context_theorem_pos": context_theorem_pos,
        }
        for name, data in data_to_save.items():
            torch.save(data, os.path.join(save_dir, f"{name}.pt"))
        return data_to_save


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or load the LightweightGraphDataset.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save/load the cached lightweight dataset.")
    parser.add_argument("--data_path", type=str, default="data/leandojo_benchmark_4/random", help="Path to the original dataset (e.g., .../random). Required for first run.")
    parser.add_argument("--corpus_path", type=str, default="data/leandojo_benchmark_4/corpus.jsonl", help="Path to the corpus.jsonl file. Required for first run.")
    parser.add_argument("--retriever_ckpt_path", type=str, default="kaiyuy/leandojo-lean4-retriever-byt5-small", help="Path to the retriever model. Required for first run.")
    args = parser.parse_args()

    gnn_config = {
        'mode': 'custom', 'use_proof_dependencies': True,
        'signature_and_state': {'verbosity': 'clickable', 'distinguish_lctx_goal': True}
    }

    dataset = LightweightGraphDataset.load_or_create(
        save_dir=args.save_dir, data_path=args.data_path, corpus_path=args.corpus_path,
        retriever_ckpt_path=args.retriever_ckpt_path, gnn_config=gnn_config,
    )

    logger.info("Dataset loaded successfully. Statistics:")
    logger.info(f"  - Premise Embeddings Shape: {dataset.premise_embeddings.shape}")
    logger.info(f"  - Premise Edge Index Shape: {dataset.premise_edge_index.shape}, Attr Shape: {dataset.premise_edge_attr.shape}")
    logger.info(f"  - Context Embeddings Shape: {dataset.context_embeddings.shape}")
    logger.info(f"  - Context Edge Index Shape: {dataset.context_edge_index.shape}, Attr Shape: {dataset.context_edge_attr.shape}")
    logger.info(f"  - Context->Premise Labels Shape: {dataset.context_premise_labels.shape}")
    logger.info(f"  - Num Train Contexts: {dataset.train_mask.sum().item()}")
    logger.info(f"  - Num Val Contexts:   {dataset.val_mask.sum().item()}")
    logger.info(f"  - Num Test Contexts:  {dataset.test_mask.sum().item()}")
    logger.info(f"  - Edge Types Map: {dataset.edge_types_map}")
    logger.info(f"  - Premise Positions Shape: {dataset.premise_pos.shape}")
    logger.info(f"  - Context Theorem Positions Shape: {dataset.context_theorem_pos.shape}")
    logger.info(f"  - File Graph Info:")
    logger.info(f"    - Num Files: {len(dataset.file_idx_to_path_map)}")
    logger.info(f"    - Premise-to-File Map Shape: {dataset.premise_to_file_idx_map.shape}")
    logger.info(f"    - Context-to-File Map Shape: {dataset.context_to_file_idx_map.shape}")
    logger.info(f"    - File Dependency Matrix Shape: {dataset.file_dependency_matrix.shape}")

if __name__ == "__main__":
    main()