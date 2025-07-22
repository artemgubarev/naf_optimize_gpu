import numpy as np
from base import AttentionForest
from forests import FORESTS, ForestKind, ForestType, TaskType
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Tuple, Union, Callable
from dataclasses import InitVar, dataclass, field
import logging
import joblib
import os

import time
from numba import njit
from tqdm import tqdm, trange
import torch
from naf_nn import NAFNetwork
from sklearn.utils.validation import check_random_state

from torch.utils.data import TensorDataset, DataLoader

from IPython.core.debugger import set_trace

import logging
logging.getLogger('numba').setLevel(logging.WARNING)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.DEBUG, format="%(message)s", force=True)
import scipy.sparse


@njit
def _prepare_leaf_sparse(xs, leaf_ids):
    """
    Args:
        xs: Input data of shape (n_samples, n_features).
        leaf_ids: Leaf id for each sample and tree, of shape (n_samples, n_trees)
    Returns:
        Array of shape (n_samples, n_trees, n_leaves).
    """
    # leaf_ids shape: (n_samples, n_trees)
    max_leaf_id = leaf_ids.max()
    n_leaves = max_leaf_id + 1
    n_trees = leaf_ids.shape[1]
    n_samples = xs.shape[0]
    result = np.zeros((n_samples, n_trees, n_leaves), dtype=np.uint8)
    for i in range(n_samples):
        for j in range(n_trees):
            result[i, j, leaf_ids[i, j]] = 1
    return result


@dataclass
class NAFParams:
    """Parameters of Neural Attention Forest."""

    kind: Union[ForestKind, str]
    task: TaskType
    loss: Union[str, Callable] = "mse"
    eps: Optional[int] = None
    mode: str = "end_to_end"
    n_epochs: int = 100
    lr: float = 1.0e-3
    lam: float = 0.0
    hidden_size: int = 16
    n_layers: int = 1
    target_loss_weight: float = 1.0
    forest: dict = field(default_factory=lambda: {})
    use_weights_random_init: bool = True
    weights_init_type: str = "default"
    random_state: Optional[int] = None
    gpu: bool = False
    gpu_device: int = 0

    def __post_init__(self):
        if not isinstance(self.kind, ForestKind):
            self.kind = ForestKind.from_name(self.kind)


class NeuralAttentionForest(AttentionForest):
    def __init__(self, params: NAFParams, run_agent=None):
        self.params = params
        self.forest = None
        self.run_agent = run_agent
        self._after_init()

    def _make_nn(self, n_features):
        self.nn = NAFNetwork(n_features, self.params.hidden_size, self.params.n_layers)
        if self.params.use_weights_random_init:
            MAX_INT = np.iinfo(np.int32).max
            rng = check_random_state(self.params.random_state)
            seed = rng.randint(MAX_INT)
            torch.manual_seed(seed)

            def _init_weights(m):
                if isinstance(m, torch.nn.Linear):
                    # torch.nn.init.uniform_(m.weight)
                    if self.params.weights_init_type == "xavier":
                        torch.nn.init.xavier_normal_(m.weight)
                        m.bias.data.fill_(0.0)
                    elif self.params.weights_init_type == "uniform":
                        torch.nn.init.uniform_(m.weight)
                        m.bias.data.fill_(0.0)
                    elif self.params.weights_init_type == "general_rule_uniform":
                        n = m.in_features
                        y = 1.0 / np.sqrt(n)
                        m.weight.data.uniform_(-y, y)
                        m.bias.data.fill_(0.0)
                    elif self.params.weights_init_type == "general_rule_normal":
                        y = m.in_features
                        m.weight.data.normal_(0.0, 1.0 / np.sqrt(y))
                        m.bias.data.fill_(0.0)
                    elif self.params.weights_init_type == "default":
                        m.reset_parameters()
                    else:
                        raise ValueError(f"Wrong {self.params.weights_init_type=}")

            self.nn.apply(_init_weights)

    def _base_fit(self, X, y) -> "NeuralAttentionForest":
        forest_cls = FORESTS[ForestType(self.params.kind, self.params.task)]
        self.forest = forest_cls(**self.params.forest)
        self.forest.random_state = self.params.random_state
        #logging.debug("Start fitting Random forest")
        #start_time = time.time()
        self.forest.fit(X, y)
        #end_time = time.time()
        #logging.info("Random forest fit time: %f", end_time - start_time)
        # store training X and y
        self.training_xs = X.copy()
        self.training_y = self._preprocess_target(y.copy())
        # store leaf id for each point in X
        #start_time = time.time()
        self.training_leaf_ids = self.forest.apply(self.training_xs)
        #end_time = time.time()
        #logging.info("Random forest apply time: %f", end_time - start_time)
        # make a tree-leaf-points correspondence
        #logging.debug("Generating leaves data")
        #start_time = time.time()
        self.leaf_sparse = _prepare_leaf_sparse(
            self.training_xs, self.training_leaf_ids
        )
        #end_time = time.time()
        #logging.info("Leaf generation time: %f", end_time - start_time)
        # self.tree_weights = np.ones(self.forest.n_estimators)
        #logging.debug("Initializing the neural network")
        self.n_trees = self.forest.n_estimators
        self._make_nn(n_features=X.shape[1])
        return self

    def fit(self, x, y):
        self._base_fit(x, y)

    def optimize_weights(self, X, y_orig, batch_size=2048, background_batch_size=1024) -> "NeuralAttentionForest":
        assert self.forest is not None, "Need to fit before weights optimization"

        if self.params.gpu and torch.cuda.is_available():
            torch.cuda.set_device(self.params.gpu_device)
            self.device = torch.device(f"cuda:{self.params.gpu_device}")
        else:
            self.device = torch.device("cpu")
        
        if self.params.mode == "end_to_end":
            self._optimize_weights_end_to_end(X, y_orig, batch_size, background_batch_size)
        elif self.params.mode == "two_step":
            self._optimize_weights_two_step(X, y_orig)
        else:
            raise ValueError(f"Wrong mode: {self.params.mode!r}")

    def _make_loss(self):
        if callable(self.params.loss):
            return self.params.loss
        elif self.params.loss == "mse":
            return torch.nn.MSELoss()
        elif self.params.loss == "cross_entropy":
            return torch.nn.CrossEntropyLoss()
        elif self.params.task == TaskType.CLASSIFICATION:
            return torch.nn.CrossEntropyLoss()
        elif self.params.task == TaskType.REGRESSION:
            return torch.nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss setting: {self.params.loss!r}")
        

    def _optimize_weights_end_to_end(self, X, y_orig, batch_size=2048, background_batch_size=1024) -> 'NeuralAttentionForest':
        import gc
        from tqdm import trange
        import matplotlib.pyplot as plt
        assert self.forest is not None, "Need to fit before weights optimization"
        
        # device, precision
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_fp16 = getattr(self.params, 'use_fp16', False)
        dtype = torch.float16 if use_fp16 else torch.float32
        self.nn = self.nn.to(device).to(dtype)
        
        X_tensor = torch.tensor(X, dtype=dtype)
        
        if self.params.task == TaskType.CLASSIFICATION:
            y_tensor = torch.tensor(y_orig, dtype=torch.long)
        else:
            y_tensor = torch.tensor(y_orig[:, np.newaxis] if y_orig.ndim == 1 else y_orig, dtype=torch.float32)
            
        indices = torch.arange(X.shape[0])
        
        # dataloader
        dataset = TensorDataset(X_tensor, y_tensor, indices)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # background data
        dense_y = self.training_y.toarray() if scipy.sparse.issparse(self.training_y) else self.training_y
        background_X_cpu = torch.tensor(self.training_xs, dtype=dtype)
        background_y_cpu = torch.tensor(dense_y, dtype=dtype)
        if background_y_cpu.ndim == 1:
            background_y_cpu = background_y_cpu.unsqueeze(1)
        
        # loss function
        optim = torch.optim.AdamW(self.nn.parameters(), lr=self.params.lr)
        loss_fn = self._make_loss()
        n_epochs = self.params.n_epochs

        if self.params.task == TaskType.CLASSIFICATION:
            y_tensor = y_tensor.long()

        losses_per_epoch = []
        background_batch_size = background_batch_size

        # learn
        if self.params.lam == 0.0:
            for epoch in trange(n_epochs, desc="Training (no reconstruction)"):
                epoch_losses = []
                
                for batch_x, batch_y, batch_idx in loader:
                    batch_x = batch_x.to(device).to(dtype)
                    batch_y = batch_y.to(device)

                    idx = torch.randint(0, background_X_cpu.size(0), (background_batch_size,))
                    background_X = background_X_cpu[idx].to(device)
                    background_y = background_y_cpu[idx].to(device)

                    X_batch_np = X[batch_idx.cpu().numpy()]
                    seg_batch = self._get_leaf_data_segments_gpu(
                        X_batch_np,
                        exclude_input=True,
                        sample_batch_size=X_batch_np.shape[0],
                        background_batch_size=background_batch_size,
                        device=device
                    )
                    seg_batch = torch.tensor(seg_batch, dtype=torch.bool, device=device)
                    neighbors_hot_batch = seg_batch[:, idx, :]

                    predictions = self.nn(batch_x, background_X, background_y, neighbors_hot_batch)
                
                    optim.zero_grad()
                    loss = loss_fn(predictions, batch_y)
                    loss.backward()
                    optim.step()
                    epoch_losses.append(loss.item())

                    # clean memory TODO: выбрать способ по лучше чистить память
                    torch.cuda.empty_cache()
                    gc.collect()

                losses_per_epoch.append(np.mean(epoch_losses))
        # else:
        #     tlw = self.params.target_loss_weight
        #     lam = self.params.lam
        #     for epoch in trange(n_epochs, desc="Training (with reconstruction)"):
        #         epoch_losses = []
        #         for batch_x, batch_y, batch_idx in loader:
        #             batch_x = batch_x.to(device).to(dtype)
        #             batch_y = batch_y.to(device)

        #             idx = torch.randint(0, background_X_cpu.size(0), (background_batch_size,))
        #             background_X = background_X_cpu[idx].to(device)
        #             background_y = background_y_cpu[idx].to(device)
                    
        #             neighbors_hot_batch = neighbors_hot[batch_idx][:, idx, :].to(device)

        #             predictions, xs_reconstruction, *_ = self.nn(
        #                 batch_x, background_X, background_y,  neighbors_hot_batch, need_attention_weights=True
        #             )
        #             optim.zero_grad()
        #             loss = tlw * loss_fn(predictions, batch_y) + lam * loss_fn(xs_reconstruction, batch_x)
        #             loss.backward()
        #             optim.step()
        #             epoch_losses.append(loss.item())

        #             # clean memory
        #             torch.cuda.empty_cache()
        #             gc.collect()

        #         losses_per_epoch.append(np.mean(epoch_losses))

        # loss visualizatiopn
        plt.figure(figsize=(8, 5))
        plt.plot(losses_per_epoch, label='Loss per epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        return self
    
    
    # def _optimize_weights_end_to_end(self, X, y_orig, batch_size=2048, background_batch_size=1024) -> 'NeuralAttentionForest':
    #     import gc
    #     from tqdm import trange
    #     import matplotlib.pyplot as plt

    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     use_fp16 = getattr(self.params, 'use_fp16', False)
    #     dtype = torch.float16 if use_fp16 else torch.float32

    #     self.nn = self.nn.to(device).to(dtype)
    #     self.nn.train()
    #     scaler = GradScaler(enabled=use_fp16)

    #     X_tensor = torch.tensor(X, dtype=dtype, device=device)
    #     if self.params.task == TaskType.CLASSIFICATION:
    #         y_tensor = torch.tensor(y_orig, dtype=torch.long, device=device)
    #     else:
    #         y_tensor = torch.tensor(
    #             y_orig[:, None] if y_orig.ndim == 1 else y_orig,
    #             dtype=torch.float32,
    #             device=device
    #         )
    #     indices = torch.arange(X.shape[0], device=device)
    #     loader = DataLoader(
    #         TensorDataset(X_tensor, y_tensor, indices),
    #         batch_size=batch_size,
    #         shuffle=True,
    #         pin_memory=True,
    #         num_workers=4
    #     )

    #     dense_y = self.training_y.toarray() if scipy.sparse.issparse(self.training_y) else self.training_y
    #     background_X_gpu = torch.tensor(self.training_xs, dtype=dtype, device=device)
    #     background_y_gpu = torch.tensor(dense_y, dtype=dtype, device=device)
    #     if background_y_gpu.ndim == 1:
    #         background_y_gpu = background_y_gpu.unsqueeze(1)

    #     optim = torch.optim.AdamW(self.nn.parameters(), lr=self.params.lr)
    #     loss_fn = self._make_loss()
    #     n_epochs = self.params.n_epochs
    #     losses_per_epoch = []

    #     for epoch in trange(n_epochs, desc="Train E2E"):
    #         epoch_losses = []
    #         for batch_x, batch_y, batch_idx in loader:
                
    #             bg_idx = torch.randint(0, background_X_gpu.size(0), (background_batch_size,), device=device)
    #             bg_X = background_X_gpu[bg_idx]
    #             bg_y = background_y_gpu[bg_idx]

    #             leaf_ids_batch = self.forest.apply(X[batch_idx.cpu().numpy()])
    #             seg_batch = self._compute_seg_for_batch(
    #                 leaf_ids_batch, 
    #                 exclude_input=True, 
    #                 background_batch_size=background_batch_size
    #             )
    #             neighbors_hot_batch = seg_batch
                
    #             optim.zero_grad()
    #             with autocast(enabled=use_fp16):
    #                 preds = self.nn(batch_x, bg_X, bg_y, neighbors_hot_batch)
    #                 loss = loss_fn(preds, batch_y)
    #             scaler.scale(loss).backward()
    #             scaler.step(optim)
    #             scaler.update()

    #             epoch_losses.append(loss.item())

    #         losses_per_epoch.append(sum(epoch_losses) / len(epoch_losses))

    #         gc.collect()
            
    #     plt.figure(figsize=(8, 5))
    #     plt.plot(losses_per_epoch, label='Loss per epoch')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.title('Training Loss Curve')
    #     plt.grid(True)
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()
    #     return self
    
    def _compute_seg_for_batch(
        self,
        leaf_ids_batch: np.ndarray,
        batch_idx: np.ndarray,
        exclude_input: bool,
        background_batch_size: int
    ) -> torch.Tensor:
        
        device = self.leaf_sparse_gpu.device
        leaf_ids = torch.from_numpy(leaf_ids_batch).long().to(device)  # (B, n_trees)
        n_trees = leaf_ids.shape[1]
        ls = self.leaf_sparse_gpu                                     # (n_background, n_trees, n_leaves)

    # Собираем сегменты по деревьям
        segs = []
        for t in range(n_trees):
            leaf_idx_t = leaf_ids[:, t]                               # (B,)
        # ls[:, t, :] — (n_background, n_leaves)
        # выберем по dim=1 листья у всех background для каждого из B
            seg_t = ls[:, t, :].index_select(1, leaf_idx_t)           # (n_background, B)
            segs.append(seg_t.T)                                      # (B, n_background)

        seg = torch.stack(segs, dim=2).bool()                         # (B, n_background, n_trees)

        if exclude_input:
        # для каждого i обнуляем self-input, если batch_idx[i] < n_background
            for i, global_i in enumerate(batch_idx):
                if global_i < seg.shape[1]:
                    seg[i, global_i, :] = False

    # Случайно выбираем background_batch_size колонок
        bg_idx = torch.randperm(seg.shape[1], device=device)[:background_batch_size]
        return seg[:, bg_idx, :]

    def _optimize_weights_two_step(self, X, y_orig) -> "NeuralAttentionForest":
        assert self.forest is not None, "Need to fit before weights optimization"
        neighbors_hot = self._get_leaf_data_segments(X, exclude_input=True)
        X_tensor = torch.tensor(X, dtype=torch.double)
        background_X = torch.tensor(self.training_xs, dtype=torch.double)
        background_y = torch.tensor(self.training_y, dtype=torch.double)
        if len(background_y.shape) == 1:
            background_y = background_y.unsqueeze(1)
            y_orig = y_orig[:, np.newaxis]
        y_true = torch.tensor(y_orig, dtype=torch.double)
        neighbors_hot = torch.tensor(neighbors_hot, dtype=torch.bool)

        # first step
        first_nn = self.nn.leaf_network
        optim = torch.optim.AdamW(first_nn.parameters(), lr=self.params.lr)
        loss_fn = self._make_loss()
        n_epochs = self.params.n_epochs
        n_trees = neighbors_hot.shape[2]
        y_true_per_tree = y_true[:, None].repeat(1, n_trees, 1)
        n_out = y_true_per_tree.shape[-1]
        for epoch in range(n_epochs // 2):
            _first_leaf_xs, first_leaf_y, _first_alphas = first_nn(
                X_tensor,
                background_X,
                background_y,
                neighbors_hot,
            )
            # first_leaf_y shape: (n_samples, n_trees, n_out)
            optim.zero_grad()
            loss = loss_fn(
                first_leaf_y.view(-1, n_out), y_true_per_tree.view(-1, n_out)
            )
            loss.backward()
            optim.step()

        self.nn.tree_network.second_encoder.weight.data[:] = (
            first_nn.first_encoder.weight.data
        )
        self.nn.tree_network.second_encoder.bias.data[:] = (
            first_nn.first_encoder.bias.data
        )
        # second step
        optim = torch.optim.AdamW(self.nn.tree_network.parameters(), lr=self.params.lr)
        loss_fn = torch.nn.MSELoss()
        for epoch in range(n_epochs // 2):
            predictions = self.nn(
                X_tensor,
                background_X,
                background_y,
                neighbors_hot,
            )
            optim.zero_grad()
            loss = loss_fn(predictions, y_true)
            loss.backward()
            optim.step()

        return self

    def optimize_weights_unlabeled(self, X) -> "NeuralAttentionForest":
        assert self.forest is not None, "Need to fit before weights optimization"
        if self.params.mode == "end_to_end":
            self._optimize_weights_unlabeled_end_to_end(X)
        else:
            raise ValueError(f"Wrong mode: {self.params.mode!r}")

    def _optimize_weights_unlabeled_end_to_end(self, X) -> "NeuralAttentionForest":
        assert self.forest is not None, "Need to fit before weights optimization"
        neighbors_hot = self._get_leaf_data_segments(X, exclude_input=False)
        X_tensor = torch.tensor(X, dtype=torch.double)
        background_X = torch.tensor(self.training_xs, dtype=torch.double)
        background_y = torch.tensor(self.training_y, dtype=torch.double)
        if len(background_y.shape) == 1:
            background_y = background_y.unsqueeze(1)
        neighbors_hot = torch.tensor(neighbors_hot, dtype=torch.bool)

        optim = torch.optim.AdamW(self.nn.parameters(), lr=self.params.lr)
        loss_fn = self._make_loss()
        n_epochs = self.params.n_epochs

        for epoch in range(n_epochs):
            # second_y, second_xs, first_alphas, second_betas
            predictions, xs_reconstruction, _alphas, _betas = self.nn(
                X_tensor,
                background_X,
                background_y,
                neighbors_hot,
                need_attention_weights=True,
            )
            optim.zero_grad()
            loss = loss_fn(xs_reconstruction, X_tensor)
            loss.backward()
            optim.step()
        return self
    
    def _get_leaf_data_segments_gpu(self,
                                X: np.ndarray,
                                exclude_input: bool = False,
                                sample_batch_size: int = 512,
                                background_batch_size: int = 512,
                                device: str = 'cuda') -> np.ndarray:
        
        leaf_ids = self.forest.apply(X)  # shape (n_samples, n_trees)
        n_samples, n_trees = leaf_ids.shape
        n_background, _, n_leaves = self.leaf_sparse.shape
        
        result = np.zeros((n_samples, n_background, n_trees), dtype=np.uint8)
        
        for si in range(0, n_samples, sample_batch_size):
            sb = min(sample_batch_size, n_samples - si)
            leaf_ids_batch = torch.from_numpy(leaf_ids[si:si + sb])  # (sb, n_trees)

            for bi in range(0, n_background, background_batch_size):
                try:
                    bb = min(background_batch_size, n_background - bi)
                    ls_cpu = self.leaf_sparse[bi:bi + bb]  # (bb, n_trees, n_leaves)
                    ls = torch.from_numpy(ls_cpu).to(device)                # (bb, n_trees, n_leaves)
                    ids = leaf_ids_batch.to(device)                        # (sb, n_trees)
                    ls = ls.unsqueeze(0).expand(sb, bb, n_trees, n_leaves)
                    idx = ids.unsqueeze(1).unsqueeze(-1).expand(sb, bb, n_trees, 1)
                    seg = torch.gather(ls, dim=3, index=idx).squeeze(3)

                    if exclude_input:
                        for local_i in range(sb):
                            global_i = si + local_i
                            if bi <= global_i < bi + bb:
                                seg[local_i, global_i - bi, :] = 0

                    result[si:si + sb, bi:bi + bb, :] = seg.to('cpu').numpy()
                    del ls, ids, idx, seg
                except RuntimeError as e:
                    err_str = str(e)
                    mem_summary = torch.cuda.memory_summary(device=device, abbreviated=True)
                    print(f"Crash at si={si}, bi={bi}\n")
                    print(err_str + "\n\n")
                    print("=== Memory summary ===\n")
                    print(mem_summary)
                    raise
                
        return result
    

        

    # def predict(self, X, need_attention_weights=False) -> np.ndarray:
    #     assert self.forest is not None, "Need to fit before predict"
        
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    #     neighbors_hot = self._get_leaf_data_segments_gpu(X, exclude_input=False)
    #     X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    #     background_X = torch.tensor(self.training_xs, dtype=torch.float32).to(device)
        
    #     if scipy.sparse.issparse(self.training_y):
    #         background_y_np = self.training_y.toarray()
    #     else:
    #         background_y_np = self.training_y
    #     background_y = torch.tensor(background_y_np, dtype=torch.float32).to(device)
        
    #     if len(background_y.shape) == 1:
    #         background_y = background_y.unsqueeze(1)
    #     neighbors_hot = torch.tensor(neighbors_hot, dtype=torch.bool).to(device)
        
    #     self.nn = self.nn.to(device)
        
    #     with torch.no_grad():
    #         output = self.nn(
    #             X_tensor,
    #             background_X,
    #             background_y,
    #             neighbors_hot,
    #             need_attention_weights=need_attention_weights,
    #         )
    #         if isinstance(output, tuple):
    #             output = tuple([
    #                 out.detach().cpu().numpy()
    #                 for out in output
    #             ])
    #             predictions, X_reconstruction, alphas, betas = output
    #         else:
    #             predictions = output.detach().cpu().numpy()

    #     if self.params.kind.need_add_init():
    #         predictions += self.forest.init_.predict(X)[:, np.newaxis]
    #     if not need_attention_weights:
    #         return predictions
    #     else:
    #         return predictions, X_reconstruction, alphas, betas
    
    def predict(self, X, need_attention_weights=False, batch_size=256) -> np.ndarray:
        assert self.forest is not None, "Need to fit before predict"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        neighbors_hot_full = self._get_leaf_data_segments_gpu(X, exclude_input=False)
        neighbors_hot_full = torch.tensor(neighbors_hot_full, dtype=torch.bool)

        background_X = torch.tensor(self.training_xs, dtype=torch.float32).to(device)

        if scipy.sparse.issparse(self.training_y):
            background_y_np = self.training_y.toarray()
        else:
            background_y_np = self.training_y
        background_y = torch.tensor(background_y_np, dtype=torch.float32).to(device)
        if background_y.ndim == 1:
            background_y = background_y.unsqueeze(1)

        self.nn = self.nn.to(device)

        predictions_all = []
        if need_attention_weights:
            x_recon_all, alphas_all, betas_all = [], [], []

        for i in tqdm(range(0, len(X), batch_size), 'predict'):
            xb = X[i:i + batch_size]
            nb_full = neighbors_hot_full[i:i + batch_size]  

            xb_tensor = torch.tensor(xb, dtype=torch.float32).to(device)
            bg_idx = torch.randperm(nb_full.shape[1])[:256]
            background_X_batch = background_X[bg_idx]
            background_y_batch = background_y[bg_idx]
            nb_tensor = nb_full[:, bg_idx, :].to(device)

            with torch.no_grad():
                output = self.nn(
                    xb_tensor,
                    background_X_batch,
                    background_y_batch,
                    nb_tensor,
                    need_attention_weights=need_attention_weights,
                )
                if isinstance(output, tuple):
                    preds, x_recon, alphas, betas = output
                    predictions_all.append(preds.cpu())
                    x_recon_all.append(x_recon.cpu())
                    alphas_all.append(alphas.cpu())
                    betas_all.append(betas.cpu())
                else:
                    predictions_all.append(output.cpu())

        predictions = torch.cat(predictions_all, dim=0).numpy()

        if self.params.kind.need_add_init():
            predictions += self.forest.init_.predict(X)[:, np.newaxis]

        if not need_attention_weights:
            return predictions
        else:
            return (
                predictions,
                torch.cat(x_recon_all, dim=0).numpy(),
                torch.cat(alphas_all, dim=0).numpy(),
                torch.cat(betas_all, dim=0).numpy(),
            )

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save(self.nn.state_dict(), os.path.join(path, "nn_weights.pth"))
        joblib.dump(self.forest, os.path.join(path, "forest.pkl"))
        joblib.dump(self.params, os.path.join(path, "params.pkl"))
        np.save(os.path.join(path, "training_xs.npy"), self.training_xs)
        y = self.training_y.toarray() if scipy.sparse.issparse(self.training_y) else self.training_y
        np.save(os.path.join(path, "training_y.npy"), y)
        np.save(os.path.join(path, "leaf_sparse.npy"), self.leaf_sparse)
        
    def load(self, n_features, path: str):
        self.forest = joblib.load(os.path.join(path, "forest.pkl"))
        self.params = joblib.load(os.path.join(path, "params.pkl"))
        self.training_xs = np.load(os.path.join(path, "training_xs.npy"))
        self.leaf_sparse = np.load(os.path.join(path, "leaf_sparse.npy"))

        raw_y = np.load(os.path.join(path, "training_y.npy"), allow_pickle=True)

        if raw_y.dtype == object and raw_y.ndim == 0:
            elem = raw_y.item()
            if scipy.sparse.isspmatrix(elem):
                raw_y = elem
                
        if scipy.sparse.isspmatrix(raw_y):
            raw_y = raw_y.toarray()

        self.training_y = np.asarray(raw_y, dtype=np.float32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._make_nn(n_features)
        self.nn.load_state_dict(torch.load(
            os.path.join(path, "nn_weights.pth"),
            map_location=device
        ))
        self.nn = self.nn.to(device).float()
        self.nn.eval()