from concurrent.futures import Future, ProcessPoolExecutor, wait, FIRST_COMPLETED
import logging
import time
import warnings
import wandb
import torch
from torch import Tensor
from torch_geometric.nn import RGCNConv, RGATConv, GATConv, GCNConv # type: ignore
from typing import Callable, Dict, Any, List, Literal, Tuple, Optional, Union
import copy
from tqdm import tqdm
import yaml
from lightweight.dataset import LightweightGraphDataset
import optuna
import torch.multiprocessing as mp

from torch.optim.adamw import AdamW

def to_undirected(edge_index: Tensor, edge_attr: Tensor) -> Tuple[Tensor, Tensor]:
    undirected_edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    undirected_edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    return undirected_edge_index, undirected_edge_attr

class L2Norm(torch.nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__() # type: ignore

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.normalize(x, p=2, dim=-1)

def calculate_mad(x: Tensor) -> float:
    if x.size(0) > 1000:
        perm = torch.randperm(x.size(0), device=x.device)[:1000]
        x = x[perm]
    
    x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
    sim_matrix = torch.mm(x_norm, x_norm.t())
    dist_matrix = 1 - sim_matrix
    n = x.size(0)
    sum_dist = torch.sum(dist_matrix)
    mad = sum_dist / (n * (n - 1))
    return mad.item()

def calculate_dirichlet_energy(x: Tensor, edge_index: Tensor) -> float:
    x = torch.nn.functional.normalize(x, p=2, dim=-1) # Fix: make it scale-invariant
    src, dst = edge_index
    x_src = x[src]
    x_dst = x[dst]
    diff = x_src - x_dst
    sq_diff = torch.sum(diff ** 2, dim=1)
    energy = torch.mean(sq_diff)
    return energy.item()

class GNN(torch.nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(GNN, self).__init__() # type: ignore
        self.config = config
        
        input_size = config['input_size']
        hidden_size = config['hidden_size']

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        layer_type = config['layer_type']
        norm_type = config.get('normalization', 'none')

        for i in range(config['n_layers']):
            in_channels = input_size if i == 0 else hidden_size
            
            if layer_type == 'RGCN':
                conv = RGCNConv(in_channels, hidden_size, num_relations=config['n_relations'])
            elif layer_type == 'RGAT':
                assert hidden_size % config['heads'] == 0
                conv = RGATConv(in_channels, hidden_size // config['heads'], num_relations=config['n_relations'], heads=config['heads'])
            elif layer_type == 'GAT':
                assert hidden_size % config['heads'] == 0
                conv = GATConv(in_channels, hidden_size // config['heads'], heads=config['heads'])
            elif layer_type == 'GCN':
                conv = GCNConv(in_channels, hidden_size)
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
            self.convs.append(conv)

            if norm_type in ['batchnorm', 'batch']:
                norm = torch.nn.BatchNorm1d(hidden_size)
            elif norm_type in ['layernorm', 'layer']:
                norm = torch.nn.LayerNorm(hidden_size)
            elif norm_type in ['l2']:
                norm = L2Norm()
            else:
                norm = torch.nn.Identity()
            self.norms.append(norm)

        if self.config['residual'] and input_size != hidden_size:
            self.residual_projection = torch.nn.Linear(input_size, hidden_size)
        else:
            self.residual_projection = torch.nn.Identity()

        if config['n_layers'] == 0 and input_size != hidden_size:
            self.input_projection = torch.nn.Linear(input_size, hidden_size)
        else:
            self.input_projection = torch.nn.Identity()

    def forward(self, x: Tensor, edge_index: Tensor, edge_type: Tensor) -> Tensor:
        if self.config['n_layers'] == 0:
            return self.input_projection(x)

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_res = x
            if isinstance(conv, (RGCNConv, RGATConv)):
                x = conv(x, edge_index, edge_type)
            else:
                x = conv(x, edge_index)

            if self.config['activation'] == 'relu':
                x = torch.relu(x)
            elif self.config['activation'] == 'gelu':
                x = torch.nn.functional.gelu(x)

            if self.config.get("debug", False) and self.training:
                dead_neurons = (x == 0).float().mean().item()
                prefix = self.config.get("name", "gnn")
                
                mad = calculate_mad(x)
                dirichlet = calculate_dirichlet_energy(x, edge_index)
                
                wandb.log({
                    f"{prefix}_layer_{i}_dead_neurons": dead_neurons,
                    f"{prefix}_layer_{i}_mad": mad,
                    f"{prefix}_layer_{i}_dirichlet": dirichlet
                })

            if self.config['dropout'] > 0.0:
                x = torch.nn.functional.dropout(x, p=self.config['dropout'], training=self.training)
            
            if self.config['residual']:
                if i == 0:
                    x_res = self.residual_projection(x_res)
                x = x + x_res

            x = norm(x)
                
        return x
    
class Scorer(torch.nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(Scorer, self).__init__() # type: ignore
        self.config = config
        self.heads: int = config.get('heads', 1)

        preprocess_config = config['preprocess']
        if preprocess_config["type"] == 'linear':
            self.preprocess = torch.nn.Linear(config['embedding_dim'], config['embedding_dim'])
        elif preprocess_config["type"] == 'nn':
            depth = preprocess_config['depth']
            layers: list[torch.nn.Module] = []
            hidden_size = config['embedding_dim']
            for _ in range(depth):
                layers.append(torch.nn.Linear(hidden_size, hidden_size))
                layers.append(torch.nn.ReLU())
                hidden_size = config['embedding_dim']
            self.preprocess = torch.nn.Sequential(*layers)
        elif preprocess_config["type"] == 'cosine':
            self.preprocess: Callable[[Tensor], Tensor] = lambda x: torch.nn.functional.normalize(x, p=2, dim=-1)
        elif preprocess_config["type"] == 'none':
            self.preprocess = lambda x: x
        else:
            raise ValueError(f"Unknown preprocess type: {preprocess_config['type']}")
        
        aggregation_type = config['aggregation']
        if aggregation_type == 'logsumexp':
            self.aggregate : Callable[[Tensor], Tensor] = lambda x: torch.logsumexp(x, dim=1)
        elif aggregation_type == 'mean':
            if self.heads == 1:
                self.aggregate : Callable[[Tensor], Tensor] = lambda x: x.squeeze_(1)
            else:
                self.aggregate : Callable[[Tensor], Tensor] = lambda x: torch.mean(x, dim=1)
        elif aggregation_type == 'max':
            self.aggregate : Callable[[Tensor], Tensor] = lambda x: torch.max(x, dim=1).values
        elif aggregation_type == 'gated':
            self.gate = torch.nn.Linear(config['embedding_dim'], 1)
            self.aggregate : Callable[[Tensor], Tensor] = lambda x: torch.sum(torch.sigmoid(self.gate(x)) * x, dim=1)
        else:
            raise ValueError(f"Unknown aggregation type: {aggregation_type}")
        
        self.temperature = torch.nn.Parameter(torch.tensor(1.0)) if config["learn_temperature"] else 1.0


    def forward(self, context_embeddings: Tensor, premise_embeddings: Tensor) -> Tensor:
        b, e = context_embeddings.shape
        p, e_prime = premise_embeddings.shape
        assert e == e_prime, f"Embedding dimensions must match: {e} != {e_prime}"

        context_embeddings = self.preprocess(context_embeddings)
        premise_embeddings = self.preprocess(premise_embeddings)

        context_embeddings = context_embeddings.view(b, self.heads, e // self.heads)
        premise_embeddings = premise_embeddings.view(p, self.heads, e // self.heads)
        scores = torch.einsum('bhd, phd -> bhp', context_embeddings, premise_embeddings)
        scores = self.aggregate(scores).squeeze(1)
        scores *= self.temperature

        assert scores.shape == (b, p), f"Scores shape must be (batch_size, n_premises): {scores.shape}"
        return scores


class LossFunction:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.weigh_by_class: bool = config["weigh_by_class"]
        self.loss_function: Literal["bce", "mse", "info_nce"] = config["loss_function"]

        if self.loss_function == "info_nce":
            assert self.weigh_by_class == False, "Class weighting not supported for InfoNCE loss"
            self.temperature = config.get("temperature", 1.0)

    def compute(self, logits: Tensor, targets: Tensor) -> Tensor:
        weights = torch.ones_like(logits, dtype=torch.float)
        if self.weigh_by_class:
            n_positives = targets.sum().item()
            n_negatives = targets.numel() - n_positives
            total = targets.numel()
            
            if n_positives > 0:
                weights[targets] = total / (2.0 * n_positives)
            if n_negatives > 0:
                weights[~targets] = total / (2.0 * n_negatives)

        targets_float = targets.float()
        if self.loss_function == "bce":
            unweighted_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets_float, reduction='none')
        elif self.loss_function == "mse":
            unweighted_loss = torch.nn.functional.mse_loss(torch.sigmoid(logits), targets_float, reduction='none')
        elif self.loss_function == "info_nce":
            logits = logits / self.temperature
            log_probs = torch.nn.functional.log_softmax(logits, dim=1)
            positive_log_probs = log_probs[targets]
            loss = -positive_log_probs
            return loss.mean()
        else:
            raise ValueError(f"Unknown loss function: {self.loss_function}")
        
        weighted_loss = (unweighted_loss * weights).mean()
        return weighted_loss

class NegativeSampler(torch.nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(NegativeSampler, self).__init__() # type: ignore
        self.config = config
        self.sampler: Literal["random", "hard_top_k", "hard_sampling", "all"] = config["sampler"]
        self.scorer = Scorer(config["scorer"])


    def forward(self, context_embeddings: Tensor, premise_embeddings: Tensor, retrieved_labels : Tensor, get_metrics: bool) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        n_contexts = context_embeddings.shape[0]
        n_premises = premise_embeddings.shape[0]

        label_premise_indices = retrieved_labels[1]
        if self.sampler == "random":
            n_negatives = self.config["n_negatives_per_context"] * n_contexts
            negative_premise_indices = torch.randint(0, n_premises, (n_negatives,), device=premise_embeddings.device)
            premise_indices = torch.cat([label_premise_indices, negative_premise_indices], dim=0)
        elif self.sampler == "all":
            #assert self.config.get("n_negatives_per_context") is None, "n_negatives_per_context should not be set when using 'all' sampler"
            premise_indices = torch.arange(n_premises, device=premise_embeddings.device)
        elif self.sampler == "hard_top_k" or self.sampler == "hard_sampling":
            with torch.no_grad():
                all_scores = self.scorer(context_embeddings, premise_embeddings) 
            k = self.config["n_negatives_per_context"]
            if self.sampler == "hard_top_k":
                _, hard_negative_indices = torch.topk(all_scores, k=k, dim=1)
                assert hard_negative_indices.shape == (n_contexts, k)
            elif self.sampler == "hard_sampling":
                probs = torch.softmax(all_scores, dim=1) 
                hard_negative_indices = torch.multinomial(probs, num_samples=k, replacement=False)
            else:
                raise RuntimeError("Unreachable")
            hard_negative_indices = hard_negative_indices.flatten()
            premise_indices = torch.cat([label_premise_indices, hard_negative_indices], dim=0)
        else:
            raise ValueError(f"Unknown sampler: {self.sampler}")

        premise_indices: torch.Tensor = torch.unique(premise_indices) # type: ignore
        premise_indices, _ = torch.sort(premise_indices) # type: ignore
        selected_premise_embeddings = premise_embeddings[premise_indices]
        
        scores = self.scorer(context_embeddings, selected_premise_embeddings)

        if get_metrics:
            metrics: Dict[str, Tensor] = {}
            if self.sampler in ["all", "hard_sampling", "hard_top_k"]:
                with torch.no_grad():
                    if "all_scores" not in locals():
                        all_scores = self.scorer(context_embeddings, premise_embeddings)
                    metrics = calculate_metrics_batch(all_scores, retrieved_labels) # type: ignore
        else:
            metrics = {}

        targets = torch.zeros_like(scores, dtype=torch.bool)
        context_indices_correct = retrieved_labels[0]
        premise_indices_correct = retrieved_labels[1]
        
        assert torch.all(torch.isin(premise_indices_correct, premise_indices)), "Some correct premise indices are not in the selected premise indices" # type: ignore
        premise_positions = torch.searchsorted(premise_indices, premise_indices_correct) # type: ignore

        targets[context_indices_correct, premise_positions] = True

        return scores, targets, metrics

def calculate_metrics_batch(all_premise_scores_batch: Tensor, batch_labels : Tensor) -> Dict[str, Tensor]:
    targets = torch.zeros_like(all_premise_scores_batch, dtype=torch.bool)
    context_indices = batch_labels[0]
    premise_indices_correct = batch_labels[1]
    targets[context_indices, premise_indices_correct] = True

    context_premise_count = targets.sum(dim=1) 

    valid_contexts_mask = context_premise_count > 0
    if not valid_contexts_mask.any():
        warnings.warn("No valid contexts with at least one correct premise found for metrics calculation.")
        return {"R@1": torch.tensor(0.0), "R@10": torch.tensor(0.0), "MRR": torch.tensor(0.0)}

    metrics: Dict[str, Any] = {}
    for k in [1, 10]:
        valid_scores = all_premise_scores_batch[valid_contexts_mask]
        valid_targets = targets[valid_contexts_mask]
        valid_context_premise_count = context_premise_count[valid_contexts_mask]

        topk_indices = torch.topk(valid_scores, k=k, dim=1).indices 
        assert topk_indices.shape == (valid_scores.shape[0], k)
        
        hits_at_k = valid_targets.gather(1, topk_indices).sum(dim=-1)
        assert hits_at_k.shape == (valid_scores.shape[0],)
        single_R_at_k = hits_at_k / valid_context_premise_count 
        metrics[f"R@{k}"] = single_R_at_k
    
    ranks = torch.argsort(torch.argsort(all_premise_scores_batch, dim=1, descending=True), dim=1) + 1 
    
    ranks_of_correct_only = torch.where(targets, ranks.float(), torch.inf)
    
    min_ranks, _ = torch.min(ranks_of_correct_only, dim=1) 
    
    reciprocal_ranks = 1.0 / min_ranks
    
    single_MRR = reciprocal_ranks[valid_contexts_mask]
    metrics["MRR"] = single_MRR

    return metrics

def edge_dropout(
    edge_index: Tensor, edge_type: Tensor, p: float, training: bool = True
) -> Tuple[Tensor, Tensor]:
    if p == 0.0 or not training:
        return edge_index, edge_type
    
    keep_prob = 1 - p
    mask = torch.rand(edge_index.size(1), device=edge_index.device) < keep_prob
    
    return edge_index[:, mask], edge_type[mask]

def batchify_dataset(dataset: LightweightGraphDataset, split_indices: Tensor, batch_size: int, shuffle: bool = False):
    n_contexts = dataset.context_embeddings.shape[0]

    if shuffle:
        perm = torch.randperm(split_indices.size(0), device=split_indices.device)
        split_indices = split_indices[perm]

    premise_to_context_edge_mask = torch.isin(dataset.context_edge_index[1], split_indices)
    premise_to_context_edge_index = dataset.context_edge_index[:, premise_to_context_edge_mask]
    premise_to_context_edge_type = dataset.context_edge_attr[premise_to_context_edge_mask]

    assert (dataset.context_to_file_idx_map == -1).any() == False, "Some contexts have invalid file index (-1)"

    for start in range(0, len(split_indices), batch_size):
        end = min(start + batch_size, len(split_indices))
        batch_global_indices = split_indices[start:end]

        batch_global_to_local_map = torch.full((n_contexts,), -1, dtype=torch.long, device=split_indices.device)
        batch_global_to_local_map[batch_global_indices] = torch.arange(len(batch_global_indices), device=split_indices.device)
        batch_context_embeddings = dataset.context_embeddings[batch_global_indices]
        
        batch_context_file_indices = dataset.context_to_file_idx_map[batch_global_indices].clone()
        batch_context_theorem_pos = dataset.context_theorem_pos[batch_global_indices].clone()
        
        batch_edge_mask = torch.isin(premise_to_context_edge_index[1], batch_global_indices)
        batch_premise_to_context_edge_index_global = premise_to_context_edge_index[:, batch_edge_mask].clone()
        batch_premise_to_context_edge_type = premise_to_context_edge_type[batch_edge_mask].clone()
        batch_premise_to_context_edge_index_global[1] = batch_global_to_local_map[batch_premise_to_context_edge_index_global[1]]

        retrieved_labels_mask = torch.isin(dataset.context_premise_labels[0], batch_global_indices)
        retrieved_labels_global = dataset.context_premise_labels[:, retrieved_labels_mask]
        retrieved_labels = retrieved_labels_global.clone()
        retrieved_labels[0] = batch_global_to_local_map[retrieved_labels[0]]

        data : Dict[str, Any] = {
            "context_embeddings": batch_context_embeddings.float(),
            "context_to_file_idx_map": batch_context_file_indices,
            "context_theorem_pos": batch_context_theorem_pos,
            "premise_embeddings": dataset.premise_embeddings.float(),

            "premise_edge_index": dataset.premise_edge_index,
            "premise_edge_type": dataset.premise_edge_attr,

            "premise_to_context_edge_index": batch_premise_to_context_edge_index_global,
            "premise_to_context_edge_type": batch_premise_to_context_edge_type,

            "retrieved_labels" : retrieved_labels
        }
        yield data

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__() # type: ignore
    
    def evaluate(self, dataset: LightweightGraphDataset, split: str, batch_size: int, mask_accessible: bool) -> Dict[str, float]:
        self.eval()
        with torch.no_grad():
            mask = getattr(dataset, f"{split}_mask", None)
            if mask is None: raise ValueError(f"Invalid split: {split}")
            split_indices = mask.nonzero(as_tuple=False).view(-1)

            eval_generator = batchify_dataset(dataset, split_indices, batch_size)

            metrics_data : list[Dict[str, Tensor]] = []
            pbar = tqdm(eval_generator, desc=f"Evaluating on {split} split")
            for batch_data in pbar:
                metrics_data.append(self.evaluate_batch(batch_data, dataset, mask_accessible))

        all_metrics_lists: Dict[str, list[Tensor]] = {}
        for m in metrics_data:
            for key, value in m.items():
                if key not in all_metrics_lists:
                    all_metrics_lists[key] = []
                all_metrics_lists[key].append(value)

        all_metrics_tensor = {key: torch.cat(value, dim=0) for key, value in all_metrics_lists.items()}
        
        metrics : Dict[str, float] = {}
        for key, value in all_metrics_tensor.items():
            metrics[key] = value.mean().item()

        return metrics

    def get_scores(self, batch_data: Dict[str, Any]) -> Tensor:
        raise NotImplementedError

    def evaluate_batch(self, batch_data: Dict[str, Any], dataset: LightweightGraphDataset, mask_accessible: bool) -> Dict[str, Tensor]:
        self.eval()
        with torch.no_grad():
            scores = self.get_scores(batch_data)
            if mask_accessible:
                batch_context_file_indices = batch_data["context_to_file_idx_map"]
                batch_context_theorem_pos = batch_data["context_theorem_pos"]
                premise_to_file_idx_map = dataset.premise_to_file_idx_map
                premise_end_pos = dataset.premise_pos[:, 2:]

                batch_same_file_mask = batch_context_file_indices.unsqueeze(1) == premise_to_file_idx_map.unsqueeze(0)

                batch_before_pos_mask = (premise_end_pos.unsqueeze(0)[:, :, 0] < batch_context_theorem_pos.unsqueeze(1)[:, :, 0]) | \
                                        ((premise_end_pos.unsqueeze(0)[:, :, 0] == batch_context_theorem_pos.unsqueeze(1)[:, :, 0]) & \
                                        (premise_end_pos.unsqueeze(0)[:, :, 1] <= batch_context_theorem_pos.unsqueeze(1)[:, :, 1]))
                
                in_file_accessible_mask = batch_same_file_mask & batch_before_pos_mask

                file_dependency_adj = dataset.file_dependency_matrix
                
                imported_mask = file_dependency_adj.bool()[batch_context_file_indices][:, premise_to_file_idx_map] # TODO for future, save as bool

                accessible_mask = in_file_accessible_mask | imported_mask
                scores.masked_fill_(~accessible_mask, -torch.inf)
            
            retrieved_labels = batch_data["retrieved_labels"]
            metrics = calculate_metrics_batch(scores, retrieved_labels)
            return metrics

class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.model = model
        self.decay = decay
        self.shadow: Dict[str, Tensor] = {}
        self.backup: Dict[str, Tensor] = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

    def to(self, device: torch.device):
        for name in self.shadow:
            self.shadow[name] = self.shadow[name].to(device)
        for name in self.backup:
            self.backup[name] = self.backup[name].to(device)

class GNNRetrievalModel(Model):
    def __init__(self, config : Dict[str, Any]):
        super(GNNRetrievalModel, self).__init__() # type: ignore
        self.config = config
        self.data_config = config["data_config"]
        self.use_random_embeddings: bool = config["use_random_embeddings"]
        self.embedding_size: int = self.data_config["embedding_dim"]
        
        premise_gnn_config = config["gnn_premises"].copy()
        premise_gnn_config["debug"] = config.get("debug", False)
        premise_gnn_config["name"] = "premise_gnn"
        
        if self.use_random_embeddings:
            hidden_size = premise_gnn_config["hidden_size"]
            premise_gnn_config['input_size'] = hidden_size
            
            self.random_initial_premise_embeddings = torch.nn.Parameter(torch.randn(self.data_config["n_premises"], hidden_size))
            self.random_initial_premise_embeddings_for_context = torch.nn.Parameter(torch.randn(self.data_config["n_premises"], hidden_size))
        else:
            assert 0
            premise_gnn_config['input_size'] = self.embedding_size

        self.premise_gnn = GNN(premise_gnn_config)

        if "gnn_contexts" in config:
            context_gnn_config = config["gnn_contexts"].copy()
            context_gnn_config["debug"] = config.get("debug", False)
            context_gnn_config["name"] = "context_gnn"
            
            if self.use_random_embeddings:
                self.random_initial_premise_embeddings = torch.nn.Parameter(torch.randn(self.data_config["n_premises"], hidden_size)) # type: ignore
                context_gnn_config['input_size'] = hidden_size # type: ignore
            else:
                assert 0
                context_gnn_config['input_size'] = self.embedding_size

            self.context_gnn = GNN(context_gnn_config)
        else:
            self.context_gnn = self.premise_gnn
        
        self.loss_function = LossFunction(config["loss"])
        self.negative_sampler = NegativeSampler(config["negative_sampler"])
        self.scorer = self.negative_sampler.scorer

        optimizer_config = config["optimizer"]
        self.optimizer = AdamW(self.parameters(), lr=optimizer_config["lr"], weight_decay=optimizer_config.get("weight_decay", 0.0))

        if config["training"].get("ema_decay"):
            self.ema = EMA(self, config["training"]["ema_decay"])

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if hasattr(self, 'ema'):
            try:
                param = next(self.parameters())
                self.ema.to(param.device)
            except StopIteration:
                pass
        return self

    @staticmethod
    def load(path: str, device: torch.device = torch.device('cpu')) -> 'GNNRetrievalModel':
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        model = GNNRetrievalModel(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'ema_state_dict' in checkpoint and hasattr(model, 'ema'):
            model.ema.shadow = checkpoint['ema_state_dict']
        model.to(device)
        return model

    def apply_ema(self):
        if hasattr(self, 'ema'):
            self.ema.apply_shadow()

    def restore_ema(self):
        if hasattr(self, 'ema'):
            self.ema.restore()

    def evaluate(self, dataset: LightweightGraphDataset, split: str, batch_size: int, mask_accessible: bool) -> Dict[str, float]:
        metrics = super().evaluate(dataset, split, batch_size, mask_accessible)
        if hasattr(self, 'ema'):
            self.ema.apply_shadow()
            ema_metrics = super().evaluate(dataset, split, batch_size, mask_accessible)
            self.ema.restore()
            for k, v in ema_metrics.items():
                metrics[f"EMA_{k}"] = v
        return metrics

    def forward(self, batch_data: Dict[str, Any]) -> Tuple[Tensor, Tensor]:
        initial_LM_premise_embeddings = batch_data["premise_embeddings"]
        initial_LM_premise_embeddings = torch.zeros_like(initial_LM_premise_embeddings) # Safeguard against accidental usage
        initial_batch_context_embeddings = batch_data["context_embeddings"]
        initial_batch_context_embeddings = torch.zeros_like(initial_batch_context_embeddings) # Safeguard against accidental usage
        device = initial_LM_premise_embeddings.device
        assert device == initial_batch_context_embeddings.device

        n_premises = initial_LM_premise_embeddings.shape[0]
        assert n_premises == self.data_config["n_premises"]
        n_contexts = initial_batch_context_embeddings.shape[0]
        hidden_size = self.config["gnn_premises"]["hidden_size"]
        assert initial_LM_premise_embeddings.shape[1] == initial_batch_context_embeddings.shape[1] == self.embedding_size

        premise_edge_index = batch_data["premise_edge_index"]
        premise_edge_type : torch.Tensor = batch_data["premise_edge_type"]
        if self.config["undirected_graphs"]:
            premise_edge_index, premise_edge_type = to_undirected(premise_edge_index, premise_edge_type)

        premise_to_context_edge_index : torch.Tensor = batch_data["premise_to_context_edge_index"].clone()
        assert premise_to_context_edge_index[1].max() <= n_contexts
        premise_to_context_edge_index[1] += n_premises 
        premise_to_context_edge_type : torch.Tensor = batch_data["premise_to_context_edge_type"]

        if self.config["undirected_graphs"]:
            premise_to_context_edge_index, premise_to_context_edge_type = to_undirected(premise_to_context_edge_index, premise_to_context_edge_type)

        all_edge_index = torch.cat([premise_edge_index, premise_to_context_edge_index], dim=1)
        all_edge_type = torch.cat([premise_edge_type, premise_to_context_edge_type], dim=0)
        
        edge_dropout_p = self.config['gnn_premises'].get('edge_dropout', 0.0)
        edge_dropout_p_context = self.config['gnn_contexts'].get('edge_dropout', 0.0)

        # TODO: plot stuff
        premise_ei_train, premise_et_train = edge_dropout(
            premise_edge_index, premise_edge_type, p=edge_dropout_p, training=self.training
        )
        all_ei_train, all_et_train = edge_dropout(
            all_edge_index, all_edge_type, p=edge_dropout_p_context, training=self.training
        )
        if self.use_random_embeddings:
            initial_premise_features = self.random_initial_premise_embeddings
            dummy_context_embeddings = torch.zeros(n_contexts, hidden_size, device=device)
            initial_all_features = torch.cat([self.random_initial_premise_embeddings_for_context, dummy_context_embeddings], dim=0)
        else:
            assert 0
            initial_premise_features = initial_LM_premise_embeddings
            initial_all_features = torch.cat([initial_LM_premise_embeddings, initial_batch_context_embeddings], dim=0)

        premise_embeddings = self.premise_gnn(initial_premise_features, premise_ei_train, premise_et_train)
        all_embeddings_for_context = self.context_gnn(initial_all_features, all_ei_train, all_et_train)

        context_embeddings = all_embeddings_for_context[n_premises:]

        assert context_embeddings.shape[0] == n_contexts
        assert premise_embeddings.shape[0] == n_premises

        return context_embeddings, premise_embeddings
    
    def predict_batch(self, batch_data: Dict[str, Any]):
        context_embeddings, premise_embeddings = self.forward(batch_data)
        return self.scorer.forward(context_embeddings, premise_embeddings)

    def compute_loss(self, batch_data: Dict[str, Any], get_metrics: bool):
        context_embeddings, premise_embeddings = self.forward(batch_data)
        retrieved_labels = batch_data["retrieved_labels"]
        logits, targets, metrics = self.negative_sampler.forward(context_embeddings, premise_embeddings, retrieved_labels, get_metrics)
        loss = self.loss_function.compute(logits, targets)
        return loss, metrics

    def train_epoch(
            self, 
            dataset: LightweightGraphDataset, 
            config: Dict[str, Any], 
            get_metrics: bool,
            #n_batch_overfitting_test : Optional[int] = None
        ) -> Dict[str, Any]:
        training_config = config["training"]
        gradient_accumulation_steps = training_config.get("gradient_accumulation_steps", 1)
        
        train_split_indices = dataset.train_mask.nonzero(as_tuple=False).view(-1)
        train_generator = batchify_dataset(dataset, split_indices=train_split_indices, batch_size=training_config["batch_size"], shuffle=training_config.get("shuffle", False))
        
        num_batches = (len(train_split_indices) + training_config["batch_size"] - 1) // training_config["batch_size"]
        pbar = tqdm(enumerate(train_generator), total=num_batches, desc="Training")

        all_batch_metrics: Dict[str, List[Any]] = {}
        for i, batch_data in pbar:
            self.train()
            should_update = ((i + 1) % gradient_accumulation_steps == 0) or (i + 1 == num_batches)
            
            loss, batch_metrics_tensors = self.compute_loss(batch_data, get_metrics=get_metrics)
            memory = torch.cuda.memory_allocated(device=loss.device) / (1024 ** 3)
            loss.backward() # type: ignore

            if config.get("debug", False):
                parameters = [p for p in self.parameters() if p.grad is not None]
                norms : list[float] = [p.grad.detach().data.norm(2).item() for p in parameters] # type: ignore
                if norms:
                    total_norm = sum(n**2 for n in norms) ** 0.5
                    mean_norm = sum(norms) / len(norms)
                    std_norm = (sum((n - mean_norm) ** 2 for n in norms) / len(norms)) ** 0.5
                    
                    wandb.log({
                        "grad_norm_total": total_norm,
                        "grad_norm_mean": mean_norm,
                        "grad_norm_std": std_norm,
                        "grad_norm_hist": wandb.Histogram(norms)
                    })

            if should_update:
                self.optimizer.step() # type: ignore
                self.optimizer.zero_grad()
                if hasattr(self, 'ema'):
                    self.ema.update()
            
            batch_metrics :Dict[str, float] = {}
            if get_metrics:
                for k, v in batch_metrics_tensors.items():
                    batch_metrics[k] = v.mean().item() 
            batch_metrics["loss"] = loss.item()
            batch_metrics["memory"] = memory
            
            for key, value in batch_metrics.items():
                if key not in all_batch_metrics:
                    all_batch_metrics[key] = []
                all_batch_metrics[key].append(value)
            
            mean_metrics = {key: sum(values) / len(values) for key, values in all_batch_metrics.items()}
            pbar.set_postfix({key: f"{value:.4f}" for key, value in mean_metrics.items()}) # type: ignore
        
        return {
            "train_metrics": {key: sum(values) / len(values) for key, values in all_batch_metrics.items()}
        }

    def get_scores(self, batch_data: Dict[str, Any]) -> Tensor:
        context_embeddings, premise_embeddings = self.forward(batch_data)
        scores = self.scorer(context_embeddings, premise_embeddings)
        return scores

class LMRetrievalModel(Model):
    def __init__(self):
        super().__init__()

    def get_scores(self, batch_data: Dict[str, Any]) -> Tensor:
        lm_context = batch_data["context_embeddings"]
        lm_prem = batch_data["premise_embeddings"]
        return torch.matmul(lm_context, lm_prem.t())

class EnsembleGNNRetrievalModel(Model):
    def __init__(self, models: List[GNNRetrievalModel]):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.config = models[0].config
        self.loss_function = models[0].loss_function

    def apply_ema(self):
        for m in self.models:
            if hasattr(m, 'apply_ema'):
                m.apply_ema()
            elif hasattr(m, 'ema'):
                 m.ema.apply_shadow()

    def restore_ema(self):
        for m in self.models:
            if hasattr(m, 'restore_ema'):
                m.restore_ema()
            elif hasattr(m, 'ema'):
                 m.ema.restore()

    def evaluate(self, dataset: LightweightGraphDataset, split: str, batch_size: int, mask_accessible: bool) -> Dict[str, float]:
        metrics = super().evaluate(dataset, split, batch_size, mask_accessible)
        
        has_ema = any(hasattr(m, 'ema') or hasattr(m, 'apply_ema') for m in self.models)
        if has_ema:
            self.apply_ema()
            ema_metrics = super().evaluate(dataset, split, batch_size, mask_accessible)
            self.restore_ema()
            for k, v in ema_metrics.items():
                metrics[f"EMA_{k}"] = v
        return metrics

    def get_scores(self, batch_data: Dict[str, Any]) -> Tensor:
        all_scores = []
        for model in self.models:
            all_scores.append(model.get_scores(batch_data))
        return torch.stack(all_scores).mean(dim=0)

class WeightedSumRetrieval(Model):
    def __init__(self, gnn_model_or_path: Union[str, Model], device: torch.device):
        super().__init__()
        if isinstance(gnn_model_or_path, str):
            self.gnn_model = GNNRetrievalModel.load(gnn_model_or_path, device)
        else:
            self.gnn_model = gnn_model_or_path
            if isinstance(self.gnn_model, torch.nn.Module):
                self.gnn_model.to(device)
        
        self.gnn_model.eval()
        
        # Disable debug logging in submodules to prevent wandb errors
        if isinstance(self.gnn_model, EnsembleGNNRetrievalModel):
             for sub_model in self.gnn_model.models:
                 if hasattr(sub_model, 'premise_gnn'):
                     sub_model.premise_gnn.config['debug'] = False
                 if hasattr(sub_model, 'context_gnn'):
                     sub_model.context_gnn.config['debug'] = False
        else:
            if hasattr(self.gnn_model, 'premise_gnn'):
                 self.gnn_model.premise_gnn.config['debug'] = False
            if hasattr(self.gnn_model, 'context_gnn'):
                 self.gnn_model.context_gnn.config['debug'] = False

        # Freeze GNN parameters so we only train the ensemble weights
        for param in self.gnn_model.parameters():
            param.requires_grad = False
            
        # Learnable weight (start at 0.5)
        self.alpha = torch.nn.Parameter(torch.tensor(0.5))
        
        # Learnable temperature scaling to adjust "confidence" distributions
        self.gnn_temp = torch.nn.Parameter(torch.tensor(1.0))
        self.lm_temp = torch.nn.Parameter(torch.tensor(1.0))
        
        # Optimizer for these 3 parameters
        self.optimizer = AdamW([self.alpha, self.gnn_temp, self.lm_temp], lr=0.01)

    def evaluate(self, dataset: LightweightGraphDataset, split: str, batch_size: int, mask_accessible: bool) -> Dict[str, float]:
        metrics = super().evaluate(dataset, split, batch_size, mask_accessible)
        
        if hasattr(self.gnn_model, 'apply_ema'):
            self.gnn_model.apply_ema()
            ema_metrics = super().evaluate(dataset, split, batch_size, mask_accessible)
            self.gnn_model.restore_ema()
            for k, v in ema_metrics.items():
                metrics[f"EMA_{k}"] = v
        elif hasattr(self.gnn_model, 'ema'):
            self.gnn_model.ema.apply_shadow()
            ema_metrics = super().evaluate(dataset, split, batch_size, mask_accessible)
            self.gnn_model.ema.restore()
            for k, v in ema_metrics.items():
                metrics[f"EMA_{k}"] = v
        
        return metrics

    def normalize(self, x: Tensor) -> Tensor:
        # Standardize per-batch (Mean 0, Std 1)
        # This brings the "loud" GNN scores and "quiet" LM scores to the same range
        return (x - x.mean()) / (x.std() + 1e-8)

    def get_scores(self, batch_data: Dict[str, Any]) -> Tensor:
        with torch.no_grad():
            gnn_raw = self.gnn_model.get_scores(batch_data)
            
            lm_context = batch_data["context_embeddings"]
            lm_prem = batch_data["premise_embeddings"]
            lm_raw = torch.matmul(lm_context, lm_prem.t())
        
        # 1. Normalize both to the same scale (Z-score)
        gnn_norm = self.normalize(gnn_raw)
        lm_norm = self.normalize(lm_raw)

        # 2. Apply temperature scaling & mixing
        w = torch.sigmoid(self.alpha)
        
        combined_scores = (w * (gnn_norm / self.gnn_temp)) + \
                          ((1 - w) * (lm_norm / self.lm_temp))
        
        return combined_scores

    def train_epoch(self, dataset: LightweightGraphDataset, config: Dict[str, Any], get_metrics: bool) -> Dict[str, Any]:
        # NOTE: We typically train ensemble weights on the Validation set (or a held-out split)
        # because the GNN is likely overconfident on the Training set.
        mask = dataset.val_mask
        split_indices = mask.nonzero(as_tuple=False).view(-1)
        
        generator = batchify_dataset(dataset, split_indices, config["training"]["batch_size"], shuffle=True)
        
        total_loss = 0.0
        steps = 0
        
        self.train() 
        
        for batch_data in generator:
            self.optimizer.zero_grad()
            
            # This calls get_scores, which uses the learnable alpha/temps
            scores = self.get_scores(batch_data)
            
            # Reconstruct targets for the loss function
            retrieved_labels = batch_data["retrieved_labels"]
            targets = torch.zeros_like(scores, dtype=torch.bool)
            targets[retrieved_labels[0], retrieved_labels[1]] = True
            
            # Calculate loss (using the same loss config as the GNN)
            loss = self.gnn_model.loss_function.compute(scores, targets)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            
        return {
            "loss": total_loss / steps if steps > 0 else 0.0, 
            "alpha": torch.sigmoid(self.alpha).item(), # Log the actual weight [0,1]
            "gnn_temp": self.gnn_temp.item(), 
            "lm_temp": self.lm_temp.item()
        }

class MLPCombinationRetrieval(Model):
    def __init__(self, gnn_model_path: str, device: torch.device, hidden_dim: int = 128):
        super().__init__()
        self.gnn_model = GNNRetrievalModel.load(gnn_model_path, device)
        self.gnn_model.eval()
        for param in self.gnn_model.parameters():
            param.requires_grad = False
            
        gnn_hidden = self.gnn_model.config["gnn_premises"]["hidden_size"]
        lm_dim = self.gnn_model.data_config["embedding_dim"]
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(gnn_hidden + lm_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.optimizer = AdamW(self.mlp.parameters(), lr=0.001)
        self.loss_function = self.gnn_model.loss_function

    def get_embeddings(self, batch_data: Dict[str, Any]) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            gnn_ctx, gnn_prem = self.gnn_model.forward(batch_data)
        
        lm_ctx = batch_data["context_embeddings"]
        lm_prem = batch_data["premise_embeddings"]
        
        combined_ctx = torch.cat([gnn_ctx, lm_ctx], dim=1)
        combined_prem = torch.cat([gnn_prem, lm_prem], dim=1)
        
        new_ctx = self.mlp(combined_ctx)
        new_prem = self.mlp(combined_prem)
        
        return new_ctx, new_prem

    def get_scores(self, batch_data: Dict[str, Any]) -> Tensor:
        ctx, prem = self.get_embeddings(batch_data)
        return torch.matmul(ctx, prem.t())

    def train_epoch(self, dataset: LightweightGraphDataset, config: Dict[str, Any], get_metrics: bool) -> Dict[str, Any]:
        mask = dataset.train_mask
        split_indices = mask.nonzero(as_tuple=False).view(-1)
        generator = batchify_dataset(dataset, split_indices, config["training"]["batch_size"], shuffle=True)
        
        total_loss = 0.0
        steps = 0
        for batch_data in generator:
            self.optimizer.zero_grad()
            
            scores = self.get_scores(batch_data)
            
            retrieved_labels = batch_data["retrieved_labels"]
            targets = torch.zeros_like(scores, dtype=torch.bool)
            targets[retrieved_labels[0], retrieved_labels[1]] = True
            
            loss = self.loss_function.compute(scores, targets)
            loss.backward() # type: ignore
            self.optimizer.step()
            total_loss += loss.item()
            steps += 1
            
        return {"loss": total_loss / steps}

def get_data_config(dataset: LightweightGraphDataset) -> Dict[str, Any]:
    data_config = {
        "n_premises": dataset.premise_embeddings.shape[0],
        "n_contexts": dataset.context_embeddings.shape[0],
        "embedding_dim": dataset.premise_embeddings.shape[1],
    }
    return data_config

SAVE_DIR = "verbose"
def load_dataset(save_dir: str=SAVE_DIR) -> LightweightGraphDataset:
    dataset = LightweightGraphDataset.load_or_create(save_dir=save_dir)
    return dataset

def run_experiment(
        dataset : LightweightGraphDataset, 
        config: Dict[str, Any], 
        gpu_id: int, 
        save_path: Optional[str], 
        early_stopping_metric:str = "R@10", 
        #n_batch_overfitting_test: Optional[int] = None, 
        time_limit: Optional[int] = None,
        seed: Optional[int] = None) -> Dict[str, Any]:
    start_time = time.time()
    torch.cuda.set_device(gpu_id)
    device = "cuda"
    if seed is None:
        seed = 42 + gpu_id
    torch.manual_seed(seed) # type: ignore

    if config.get("debug", False):
        wandb.init(project="gnn_reprover_debug", config=config)

    model = GNNRetrievalModel(config=config).to(device)

    best_val_metrics = None
    epoch = config["training"]["epochs"]
    assert epoch > 0, "Number of epochs must be positive"

    best_val_metrics = None
    test_metrics_at_best_val = None
    best_train_metrics = None

    def eval_and_compare():
        nonlocal best_val_metrics
        val_metrics = model.evaluate(dataset, 'val', batch_size=config['evaluation']['batch_size'], mask_accessible=True)
        print(f"Val: {val_metrics}")
        if (best_val_metrics is None) or val_metrics["R@10"] > best_val_metrics["R@10"]:
            best_val_metrics = val_metrics
            nonlocal test_metrics_at_best_val
            test_metrics_at_best_val = model.evaluate(dataset, 'test', batch_size=config['evaluation']['batch_size'], mask_accessible=True)
            print(f"Test at best Val: {test_metrics_at_best_val}")
            # Save the model checkpoint
            if save_path is not None:
                save_dict = {
                    'model_state_dict': model.state_dict(),
                    'config': config
                }
                if hasattr(model, 'ema'):
                    save_dict['ema_state_dict'] = model.ema.shadow
                
                torch.save(save_dict, save_path)
                print(f"Saved best model to {save_path}")

        nonlocal best_train_metrics
        train_metrics = model.evaluate(dataset, 'train', batch_size=config['evaluation']['batch_size'], mask_accessible=False)
        best_train_metrics = train_metrics
        print(f"Train: {train_metrics}")

    for epoch in range(1, epoch + 1):
        model.train_epoch(
            dataset, 
            config=config, 
            get_metrics=False
        )
        
        if epoch % config["evaluation"]["eval_every_n_epochs"] == 0:
            eval_and_compare()

        elapsed_time = time.time() - start_time
        if time_limit is not None and elapsed_time > time_limit:
            print(f"Time limit of {time_limit} seconds reached. Stopping training.")
            break

    eval_and_compare()
    assert test_metrics_at_best_val is not None
    assert best_train_metrics is not None
    assert best_val_metrics is not None
    return {
        "val_metrics": best_val_metrics,
        "test_metrics": test_metrics_at_best_val,
        "train_metrics": best_train_metrics,
    }

def objective(trial: optuna.Trial, base_config: Dict[str, Any], base_dataset: LightweightGraphDataset) -> float:
    gpu_id = trial.user_attrs["gpu_id"]
    
    config = copy.deepcopy(base_config)
    
    config["gnn_premises"]["n_layers"] = trial.suggest_int("gnn_premises_n_layers", 0, 3)
    config["gnn_premises"]["normalization"] = trial.suggest_categorical("gnn_premises_normalization", ["none", "batchnorm", "layernorm", "l2"])
    config["gnn_premises"]["dropout"] = trial.suggest_float("gnn_premises_dropout", 0.0, 0.5)
    config["gnn_premises"]["edge_dropout"] = trial.suggest_float("gnn_premises_edge_dropout", 0.0, 0.5)
    config["gnn_premises"]["hidden_size"] = trial.suggest_categorical("gnn_s_hidden_size", [64, 128, 256, 512])

    config["gnn_contexts"]["n_layers"] = trial.suggest_int("gnn_contexts_n_layers", 0, 3)
    config["gnn_contexts"]["normalization"] = trial.suggest_categorical("gnn_contexts_normalization", ["none", "batchnorm", "layernorm", "l2"])
    config["gnn_contexts"]["dropout"] = trial.suggest_float("gnn_contexts_dropout", 0.0, 0.5)
    config["gnn_contexts"]["edge_dropout"] = trial.suggest_float("gnn_contexts_edge_dropout", 0.0, 0.5)
    config["gnn_contexts"]["hidden_size"] = trial.suggest_categorical("gnn_s_hidden_size", [64, 128, 256, 512])

    config["negative_sampler"]["scorer"]["learn_temperature"] = trial.suggest_categorical("learn_temperature", [True, False])

    config["training"]["epochs"] = trial.suggest_int("training_epochs", 1, 100)
    
    
    config["optimizer"]["lr"] = trial.suggest_float("lr", 0.000001, 0.01, log=True)
    config["optimizer"]["weight_decay"] = trial.suggest_float("weight_decay", 0.000001, 0.01, log=True)
    
    config["training"]["batch_size"] = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])

    dataset = copy.deepcopy(base_dataset).to(device=torch.device(f"cuda:{gpu_id}"))
    
    try:
        metrics = run_experiment(dataset, config, gpu_id, None, time_limit=3600 * 3)
        val_metrics = metrics["val_metrics"]
        train_metrics = metrics["train_metrics"]
        test_metrics = metrics["test_metrics"]

        trial.set_user_attr("R@10 (Train)", train_metrics["R@10"])
        trial.set_user_attr("R@1 (Train)", train_metrics["R@1"])
        trial.set_user_attr("MRR (Train)", train_metrics["MRR"])

        trial.set_user_attr("R@10 (Val)", val_metrics["R@10"])
        trial.set_user_attr("R@1 (Val)", val_metrics["R@1"])
        trial.set_user_attr("MRR (Val)", val_metrics["MRR"])

        trial.set_user_attr("R@10 (Test)", test_metrics["R@10"])
        trial.set_user_attr("R@1 (Test)", test_metrics["R@1"])
        trial.set_user_attr("MRR (Test)", test_metrics["MRR"])

        return val_metrics["R@10"]
    except Exception as e:
        raise e

def optune(base_config: Dict[str, Any], base_dataset: LightweightGraphDataset, gpu_ids: List[int], storage: str, study_name: str):
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Optuna study: {study_name}")
    logger.info(f"Using GPUs: {gpu_ids}")

    data_config = get_data_config(base_dataset)
    base_config["data_config"] = data_config
    
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True
    )

    n_experiments_per_gpu = 1
    max_workers = n_experiments_per_gpu * len(gpu_ids)
    
    # Set multiprocessing start method to 'spawn' to avoid CUDA initialization errors
    mp_context = mp.get_context('spawn')
    executor = ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context, max_tasks_per_child=1)

    futures: Dict[Future[Any], Tuple[optuna.Trial, int]] = {}
    
    # Initial submission
    for gpu_id in gpu_ids:
        for _ in range(n_experiments_per_gpu):
            trial = study.ask()
            trial.set_user_attr("gpu_id", gpu_id)
            future = executor.submit(objective, trial, base_config, base_dataset) 
            futures[future] = (trial, gpu_id)
        
    try:
        # CHANGED: Use a while loop with wait() instead of as_completed()
        # as_completed() only sees futures that exist when the loop starts.
        # This approach handles dynamically added futures.
        while futures:
            done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)

            for future in done:
                trial, gpu_id = futures.pop(future)
                try:
                    result = future.result()
                    study.tell(trial, result)
                except Exception as e:
                    logger.error(f"Trial {trial.number} failed: {e}", exc_info=True)
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)

                # Submit a new trial to the FREED gpu_id
                new_trial = study.ask()
                new_trial.set_user_attr("gpu_id", gpu_id)
                new_future = executor.submit(objective, new_trial, base_config, base_dataset)
                futures[new_future] = (new_trial, gpu_id)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Terminating all running trials...")
        executor.shutdown(wait=False)
        logger.info("Optimization terminated.")
        logger.info("Optimization interrupted by user.")

def run_baseline(dataset: LightweightGraphDataset, config: Dict[str, Any], gpu_id: int):
    device = torch.device(f"cuda:{gpu_id}")
    print("Evaluating LM Baseline...")
    lm_model = LMRetrievalModel().to(device)
    print("LM Results:")
    #print("Train:", lm_model.evaluate(dataset, 'train', config['evaluation']['batch_size'], mask_accessible=False))
    print("Val:", lm_model.evaluate(dataset, 'val', config['evaluation']['batch_size'], mask_accessible=True))
    print("Test:", lm_model.evaluate(dataset, 'test', config['evaluation']['batch_size'], mask_accessible=True))

def run_combined_model(dataset: LightweightGraphDataset, config: Dict[str, Any], gpu_id: int, gnn_path: str) -> None:
    device = torch.device(f"cuda:{gpu_id}")
    
    # Weighted Sum Ensemble
    print("\nTraining Weighted Sum Ensemble...")
    weighted_model = WeightedSumRetrieval(gnn_path, device).to(device)
    for epoch in range(20):
        metrics = weighted_model.train_epoch(dataset, config, get_metrics=False)
        print(f"Epoch {epoch}: {metrics}")

    print("Weighted Sum Results:")
    #print("Train:", weighted_model.evaluate(dataset, 'train', config['evaluation']['batch_size'], mask_accessible=False))
    print("Val:", weighted_model.evaluate(dataset, 'val', config['evaluation']['batch_size'], mask_accessible=True))
    print("Test:", weighted_model.evaluate(dataset, 'test', config['evaluation']['batch_size'], mask_accessible=True))

    ## MLP Ensemble
    #print("\nTraining MLP Ensemble...")
    #mlp_model = MLPCombinationRetrieval(gnn_path, device).to(device)
    #for epoch in range(10):
    #    metrics = mlp_model.train_epoch(dataset, config, get_metrics=False)
    #    print(f"Epoch {epoch}: {metrics}")
#
    #print("MLP Results:")
    #print("Train:", mlp_model.evaluate(dataset, 'train', config['evaluation']['batch_size'], mask_accessible=False))
    #print("Val:", mlp_model.evaluate(dataset, 'val', config['evaluation']['batch_size'], mask_accessible=True))
    #print("Test:", mlp_model.evaluate(dataset, 'test', config['evaluation']['batch_size'], mask_accessible=True))

def run_diagnose(dataset: LightweightGraphDataset, config: Dict[str, Any], gpu_id: int, gnn_path: str):
    device = torch.device(f"cuda:{gpu_id}")
    gnn_model = GNNRetrievalModel.load(gnn_path, device)
    gnn_model.eval()
    
    gnn_means, gnn_stds = [], []
    lm_means, lm_stds = [], []
    intersection_sum, union_sum = 0, 0
    oracle_hits, total_samples = 0, 0
    
    gnn_ranks_when_lm_correct = []
    lm_ranks_when_gnn_correct = []
    
    mask = dataset.val_mask
    split_indices = mask.nonzero(as_tuple=False).view(-1)
    generator = batchify_dataset(dataset, split_indices, config["evaluation"]["batch_size"])
    
    for batch_data in tqdm(generator, desc="Diagnosing"):
        with torch.no_grad():
            gnn_scores = gnn_model.get_scores(batch_data)
            lm_ctx = batch_data["context_embeddings"]
            lm_prem = batch_data["premise_embeddings"]
            lm_scores = torch.matmul(lm_ctx, lm_prem.t())

        batch_context_file_indices = batch_data["context_to_file_idx_map"]
        batch_context_theorem_pos = batch_data["context_theorem_pos"]
        premise_to_file_idx_map = dataset.premise_to_file_idx_map
        premise_end_pos = dataset.premise_pos[:, 2:]

        batch_same_file_mask = batch_context_file_indices.unsqueeze(1) == premise_to_file_idx_map.unsqueeze(0)
        batch_before_pos_mask = (premise_end_pos.unsqueeze(0)[:, :, 0] < batch_context_theorem_pos.unsqueeze(1)[:, :, 0]) | \
                                ((premise_end_pos.unsqueeze(0)[:, :, 0] == batch_context_theorem_pos.unsqueeze(1)[:, :, 0]) & \
                                (premise_end_pos.unsqueeze(0)[:, :, 1] <= batch_context_theorem_pos.unsqueeze(1)[:, :, 1]))
        
        in_file_accessible_mask = batch_same_file_mask & batch_before_pos_mask
        file_dependency_adj = dataset.file_dependency_matrix
        imported_mask = file_dependency_adj.bool()[batch_context_file_indices][:, premise_to_file_idx_map]
        accessible_mask = in_file_accessible_mask | imported_mask
        
        gnn_scores.masked_fill_(~accessible_mask, -torch.inf)
        lm_scores.masked_fill_(~accessible_mask, -torch.inf)

        valid_mask = accessible_mask
        gnn_means.append(gnn_scores[valid_mask].mean().item())
        gnn_stds.append(gnn_scores[valid_mask].std().item())
        lm_means.append(lm_scores[valid_mask].mean().item())
        lm_stds.append(lm_scores[valid_mask].std().item())

        k = 10
        _, gnn_topk = torch.topk(gnn_scores, k=k, dim=1)
        _, lm_topk = torch.topk(lm_scores, k=k, dim=1)
        
        retrieved_labels = batch_data["retrieved_labels"]
        
        for i in range(gnn_scores.shape[0]):
            s_gnn = set(gnn_topk[i].tolist())
            s_lm = set(lm_topk[i].tolist())
            
            intersection_sum += len(s_gnn & s_lm)
            union_sum += len(s_gnn | s_lm)
            
            targets = retrieved_labels[1][retrieved_labels[0] == i]
            if len(targets) == 0: continue
            
            total_samples += 1
            
            gnn_hit = any(t.item() in s_gnn for t in targets)
            lm_hit = any(t.item() in s_lm for t in targets)
            
            if gnn_hit or lm_hit:
                oracle_hits += 1

            target = targets[0].item()
            
            r_gnn = (gnn_scores[i] > gnn_scores[i, target]).sum().item() + 1
            r_lm = (lm_scores[i] > lm_scores[i, target]).sum().item() + 1
            
            if r_lm <= 10:
                gnn_ranks_when_lm_correct.append(r_gnn)
            
            if r_gnn <= 10:
                lm_ranks_when_gnn_correct.append(r_lm)

    print("-" * 30)
    print("DIAGNOSTICS REPORT")
    print("-" * 30)
    print(f"Scale (GNN): Mean={sum(gnn_means)/len(gnn_means):.4f}, Std={sum(gnn_stds)/len(gnn_stds):.4f}")
    print(f"Scale (LM) : Mean={sum(lm_means)/len(lm_means):.4f}, Std={sum(lm_stds)/len(lm_stds):.4f}")
    print("-" * 30)
    print(f"Jaccard Similarity @ 10: {intersection_sum / union_sum:.4f}")
    print(f"Oracle R@10            : {oracle_hits / total_samples:.4f}")
    print("-" * 30)
    print(f"Avg GNN Rank when LM is Correct (R@10): {sum(gnn_ranks_when_lm_correct)/len(gnn_ranks_when_lm_correct) if gnn_ranks_when_lm_correct else 0:.1f}")
    print(f"Avg LM Rank when GNN is Correct (R@10): {sum(lm_ranks_when_gnn_correct)/len(lm_ranks_when_gnn_correct) if lm_ranks_when_gnn_correct else 0:.1f}")
    print("-" * 30)

def train_single_model(config: Dict[str, Any], dataset: LightweightGraphDataset, gpu_id: int, save_path: str, seed: int) -> str:
    dataset_gpu = copy.deepcopy(dataset).to(torch.device(f"cuda:{gpu_id}"))
    run_experiment(dataset_gpu, config, gpu_id, save_path, seed=seed)
    return save_path

def run_ensemble(dataset: LightweightGraphDataset, config: Dict[str, Any], gpu_ids: List[int], n_models: int, train: bool) -> None:
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Ensemble training with {n_models} models on GPUs: {gpu_ids}")
    
    # Create a directory for ensemble models
    ensemble_dir = "weights/ensemble"
    import os
    os.makedirs(ensemble_dir, exist_ok=True)
    
    if train:
        mp_context = mp.get_context('spawn')
        executor = ProcessPoolExecutor(max_workers=len(gpu_ids), mp_context=mp_context, max_tasks_per_child=1)
        
        futures: Dict[Future[Any], Tuple[int, int]] = {} # future -> (model_idx, gpu_id)
        
        model_paths = []
        
        # Initial submission
        model_idx = 0
        for gpu_id in gpu_ids:
            if model_idx < n_models:
                save_path = os.path.join(ensemble_dir, f"model_{model_idx}.pt")
                seed = 42 + model_idx # Ensure different seeds
                future = executor.submit(train_single_model, config, dataset, gpu_id, save_path, seed)
                futures[future] = (model_idx, gpu_id)
                model_idx += 1
                
        while futures:
            done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                m_idx, gpu_id = futures.pop(future)
                try:
                    path = future.result()
                    model_paths.append(path)
                    print(f"Model {m_idx} trained and saved to {path}")
                except Exception as e:
                    print(f"Model {m_idx} failed: {e}")
                
                if model_idx < n_models:
                    save_path = os.path.join(ensemble_dir, f"model_{model_idx}.pt")
                    seed = 42 + model_idx
                    new_future = executor.submit(train_single_model, config, dataset, gpu_id, save_path, seed)
                    futures[new_future] = (model_idx, gpu_id)
                    model_idx += 1

        print("All models trained. Loading ensemble...")
    else:
        model_paths = [os.path.join(ensemble_dir, f"model_{i}.pt") for i in range(n_models)]
    device = torch.device(f"cuda:{gpu_ids[0]}") # Use first GPU for ensemble evaluation
    
    models = []
    for path in model_paths:
        models.append(GNNRetrievalModel.load(path, device))
        
    ensemble_model = EnsembleGNNRetrievalModel(models)
    ensemble_model.to(device)
    
    dataset.to(device)

    print("Evaluating Ensemble...")
    #print("Train:", ensemble_model.evaluate(dataset, 'train', config['evaluation']['batch_size'], mask_accessible=False))
    print("Val:", ensemble_model.evaluate(dataset, 'val', config['evaluation']['batch_size'], mask_accessible=True))
    print("Test:", ensemble_model.evaluate(dataset, 'test', config['evaluation']['batch_size'], mask_accessible=True))
    
    print("\nTraining Weighted Sum Ensemble with Ensemble GNN...")
    weighted_model = WeightedSumRetrieval(ensemble_model, device).to(device)
    
    # Train weighted sum
    for epoch in range(20):
        metrics = weighted_model.train_epoch(dataset, config, get_metrics=False)
        print(f"Epoch {epoch}: {metrics}")

    print("Weighted Sum Ensemble Results:")
    print("Val:", weighted_model.evaluate(dataset, 'val', config['evaluation']['batch_size'], mask_accessible=True))
    print("Test:", weighted_model.evaluate(dataset, 'test', config['evaluation']['batch_size'], mask_accessible=True))

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    dataset = load_dataset()
    #config = load_config("configs/config.yaml")
    #config["data_config"] = get_data_config(dataset)
    #study_name = "optuna_after_overfitting_study"
    #optune(config, dataset, gpu_ids=[1, 2, 3], storage=f"sqlite:///{study_name}.db", study_name=study_name)

    path = "weights/pure_gnn_retrieval_model.pt"
    config = load_config("configs/config_optuna_val.yaml")
    config["data_config"] = get_data_config(dataset)
    gpu_id = 3
    dataset.to(device=torch.device(f"cuda:{gpu_id}"))

    #run_baseline(dataset, config, gpu_id)

    #run_experiment(dataset, config, gpu_id=gpu_id, save_path=path)

    #run_combined_model(dataset, config, gpu_id, path)
    
    #run_diagnose(dataset, config, gpu_id, path)

    gpu_ids = list(range(torch.cuda.device_count()))
    if not gpu_ids:
        gpu_ids = [0]
    
    # Ensure dataset is on CPU before spawning processes to avoid CUDA context issues
    dataset.to(torch.device("cpu"))
    
    run_ensemble(dataset, config, gpu_ids=gpu_ids, n_models=12, train=False)