


import gc
import torch
import logging

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange
from base import AttentionForest
from forests import FORESTS, ForestKind, ForestType, TaskType
from typing import Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from torch.utils.data import TensorDataset, DataLoader
from naf_nn import NAFNetwork
from sklearn.utils.validation import check_random_state
from torch.utils.data import TensorDataset, DataLoader
from IPython.core.debugger import set_trace
from numba import njit, prange

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
    
    print("min          =", leaf_ids.min())
    print("max          =", leaf_ids.max())
    print("std          =", round(leaf_ids.std(), 2))
    print("median       =", np.median(leaf_ids))
    print("unique count =", np.unique(leaf_ids).size)
    
    n_leaves = max_leaf_id + 1
    n_trees = leaf_ids.shape[1]
    n_samples = xs.shape[0]
    result = np.zeros((n_samples, n_trees, n_leaves), dtype=np.uint8)
    for i in range(n_samples):
        for j in range(n_trees):
            result[i, j, leaf_ids[i, j]] = 1
    return result

@njit(parallel=True)
def _get_leaf_data_segments_numba(leaf_sparse, leaf_ids, exclude_input):
    n_samples, n_trees = leaf_ids.shape
    n_background = leaf_sparse.shape[0]
    result = np.zeros((n_samples, n_background, n_trees), dtype=np.uint8)

    for i in prange(n_samples):
        for j in range(n_trees):
            lid = leaf_ids[i, j]
            for b in range(n_background):
                result[i, b, j] = leaf_sparse[b, j, lid]

        if exclude_input:
            for j in range(n_trees):
                result[i, i, j] = 0

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
        self.forest.fit(X, y)
        self.training_xs = X.copy()
        self.training_y = self._preprocess_target(y.copy())
        self.training_leaf_ids = self.forest.apply(self.training_xs)
        self.leaf_sparse = _prepare_leaf_sparse(
            self.training_xs, self.training_leaf_ids
        )
        self.n_trees = self.forest.n_estimators
        self._make_nn(n_features=X.shape[1])
        return self

    def fit(self, x, y):
        self._base_fit(x, y)

    def optimize_weights(self, X, y_orig, batch_size=2048, background_batch_size=1024, n_parts=96) -> "NeuralAttentionForest":
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
        
    # ---- with part
    # def _optimize_weights_end_to_end(self, X, y_orig, batch_size=512, background_batch_size=256, n_parts=96) -> 'NeuralAttentionForest':
    
    #     assert self.forest is not None, "Need to fit before weights optimization"
    
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     use_fp16 = getattr(self.params, 'use_fp16', False)
    #     dtype = torch.float16 if use_fp16 else torch.float32
    #     self.nn = self.nn.to(device).to(dtype)

    #     dense_y = self.training_y.toarray() if scipy.sparse.issparse(self.training_y) else self.training_y
    #     background_X_cpu = torch.tensor(self.training_xs, dtype=dtype)
    #     background_y_cpu = torch.tensor(dense_y, dtype=dtype)
    #     if background_y_cpu.ndim == 1:
    #         background_y_cpu = background_y_cpu.unsqueeze(1)

    #     optim = torch.optim.AdamW(self.nn.parameters(), lr=self.params.lr)
    #     loss_fn = self._make_loss()
    #     n_epochs = self.params.n_epochs

    #     all_indices = np.arange(X.shape[0])
    #     parts = np.array_split(all_indices, n_parts)

    #     losses_per_epoch = []

    #     for part_id, part_idx in enumerate(parts, 1):

    #         X_part = X[part_idx]
    #         y_part = y_orig[part_idx]
    #         X_tensor = torch.tensor(X_part, dtype=dtype)
    #         if self.params.task == TaskType.CLASSIFICATION:
    #             y_tensor = torch.tensor(y_part, dtype=torch.long)
    #         else:
    #             y_tensor = torch.tensor(
    #                 y_part[:, np.newaxis] if y_part.ndim == 1 else y_part,
    #                 dtype=torch.float32
    #             )

    #         part_indices = torch.arange(X_part.shape[0])
    #         dataset = TensorDataset(X_tensor, y_tensor, part_indices)
    #         loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    #         seg_np = self._get_leaf_data_segments_gpu(
    #             X_part,
    #             exclude_input=True,
    #             sample_batch_size=256,
    #             background_batch_size=128,
    #             device=device
    #         )
            
    #         #seg_np = self._get_leaf_data_segments(X_part, exclude_input=True)
            
    #         full_seg = torch.tensor(seg_np, dtype=torch.bool, device=device)

    #         for epoch in trange(n_epochs, desc=f"Part {part_id}/{n_parts}"):
    #             epoch_losses = []
    #             for batch_x, batch_y, batch_idx in loader:
    #                 batch_x = batch_x.to(device).to(dtype)
    #                 batch_y = batch_y.to(device)

    #                 # random context
    #                 idx = torch.randint(0, background_X_cpu.size(0), (background_batch_size,))
    #                 background_X = background_X_cpu[idx].to(device)
    #                 background_y = background_y_cpu[idx].to(device)
    #                 neighbors_hot = full_seg[batch_idx][:, idx].to(device)

    #                 preds = self.nn(batch_x, background_X, background_y, neighbors_hot)
    #                 optim.zero_grad()
    #                 loss = loss_fn(preds, batch_y)
    #                 loss.backward()
    #                 optim.step()

    #                 epoch_losses.append(loss.item())

    #             losses_per_epoch.append(np.mean(epoch_losses))

    #         print(f'mean loss = {np.mean(losses_per_epoch):.5f}')
    #         del full_seg, seg_np, dataset, loader, X_tensor, y_tensor
    #         torch.cuda.empty_cache()
    #         gc.collect()
            
    #     plt.plot(losses_per_epoch)
    #     plt.show()   

    #     return self
    
    # ---------- seg for batch
    def _optimize_weights_end_to_end(self, X, y_orig, batch_size=512, background_batch_size=256) -> 'NeuralAttentionForest':
        assert self.forest is not None, "Need to fit before weights optimization"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_fp16 = getattr(self.params, 'use_fp16', False)
        dtype = torch.float16 if use_fp16 else torch.float32
        self.nn = self.nn.to(device).to(dtype)

        dense_y = self.training_y.toarray() if scipy.sparse.issparse(self.training_y) else self.training_y
        background_X_cpu = torch.tensor(self.training_xs, dtype=dtype)
        background_y_cpu = torch.tensor(dense_y, dtype=dtype)
        if background_y_cpu.ndim == 1:
            background_y_cpu = background_y_cpu.unsqueeze(1)

        optim = torch.optim.AdamW(self.nn.parameters(), lr=self.params.lr)
        loss_fn = self._make_loss()
        n_epochs = self.params.n_epochs

        X_tensor = torch.tensor(X, dtype=dtype)
        if self.params.task == TaskType.CLASSIFICATION:
            y_tensor = torch.tensor(y_orig, dtype=torch.long)
        else:
            y_tensor = torch.tensor(
                y_orig[:, np.newaxis] if y_orig.ndim == 1 else y_orig,
                dtype=torch.float32
            )
        all_indices = torch.arange(X.shape[0])
        dataset = TensorDataset(X_tensor, y_tensor, all_indices)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        losses_per_epoch = []

        for epoch in trange(n_epochs, desc="Epochs"):
            epoch_losses = []
            for batch_x, batch_y, batch_idx in loader:
                batch_x = batch_x.to(device).to(dtype)
                batch_y = batch_y.to(device)

                bg_idx = torch.randint(0, background_X_cpu.size(0), (background_batch_size,), dtype=torch.long)
                background_X = background_X_cpu[bg_idx].to(device)
                background_y = background_y_cpu[bg_idx].to(device)

                leaf_ids_batch = self.forest.apply(batch_x.cpu().numpy())  # (B, T)
                neighbors_hot = self._get_leaf_data_segments_from_leaf_ids_gpu(
                    leaf_ids_batch,
                    bg_idx.cpu().numpy(),
                    device=device,
                    exclude_input=True)

                preds = self.nn(batch_x, background_X, background_y, neighbors_hot)
                optim.zero_grad()
                loss = loss_fn(preds, batch_y)
                loss.backward()
                optim.step()

                epoch_losses.append(loss.item())

            mean_loss = np.mean(epoch_losses)
            losses_per_epoch.append(mean_loss)
            print(f"Epoch {epoch+1}/{n_epochs} — mean loss: {mean_loss:.5f}")

            torch.cuda.empty_cache()
            gc.collect()

        plt.plot(losses_per_epoch)
        plt.show()

        return self
    
    # ----------- without parts full seg
    # def _optimize_weights_end_to_end(self, X, y_orig, batch_size=512, background_batch_size=256) -> 'NeuralAttentionForest':
    #     assert self.forest is not None, "Need to fit before weights optimization"

    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     use_fp16 = getattr(self.params, 'use_fp16', False)
    #     dtype = torch.float16 if use_fp16 else torch.float32
    #     self.nn = self.nn.to(device).to(dtype)

    #     dense_y = self.training_y.toarray() if scipy.sparse.issparse(self.training_y) else self.training_y
    #     background_X_cpu = torch.tensor(self.training_xs, dtype=dtype)
    #     background_y_cpu = torch.tensor(dense_y, dtype=dtype)
    #     if background_y_cpu.ndim == 1:
    #         background_y_cpu = background_y_cpu.unsqueeze(1)

    #     optim = torch.optim.AdamW(self.nn.parameters(), lr=self.params.lr)
    #     loss_fn = self._make_loss()
    #     n_epochs = self.params.n_epochs

    #     X_tensor = torch.tensor(X, dtype=dtype)
    #     if self.params.task == TaskType.CLASSIFICATION:
    #         y_tensor = torch.tensor(y_orig, dtype=torch.long)
    #     else:
    #         y_tensor = torch.tensor(
    #             y_orig[:, np.newaxis] if y_orig.ndim == 1 else y_orig,
    #             dtype=torch.float32
    #         )
    #     all_indices = torch.arange(X.shape[0])
    #     dataset = TensorDataset(X_tensor, y_tensor, all_indices)
    #     loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    #     seg_np = self._get_leaf_data_segments(X, exclude_input=True)  # shape: (n_samples, n_background, n_trees)
    #     full_seg = torch.tensor(seg_np, dtype=torch.bool, device=device)

    #     losses_per_epoch = []

    #     for epoch in trange(n_epochs, desc="Epochs"):
    #         epoch_losses = []
    #         for batch_x, batch_y, batch_idx in loader:
    #             batch_x = batch_x.to(device).to(dtype)
    #             batch_y = batch_y.to(device)

    #             idx = torch.randint(0, background_X_cpu.size(0), (background_batch_size,))
    #             background_X = background_X_cpu[idx].to(device)
    #             background_y = background_y_cpu[idx].to(device)
    #             neighbors_hot = full_seg[batch_idx][:, idx]

    #             preds = self.nn(batch_x, background_X, background_y, neighbors_hot)
    #             optim.zero_grad()
    #             loss = loss_fn(preds, batch_y)
    #             loss.backward()
    #             optim.step()

    #             epoch_losses.append(loss.item())

    #         mean_loss = np.mean(epoch_losses)
    #         losses_per_epoch.append(mean_loss)
    #         print(f"Epoch {epoch+1}/{n_epochs} — mean loss: {mean_loss:.5f}")

    #         torch.cuda.empty_cache()
    #         gc.collect()

    #     plt.plot(losses_per_epoch)
    #     plt.xlabel("Epoch")
    #     plt.ylabel("Mean Loss")
    #     plt.title("Training Loss")
    #     plt.show()

    #     return self
    
    # def _optimize_weights_end_to_end(self, X, y_orig,
    #                              batch_size=512,
    #                              background_batch_size=256,
    #                              n_parts=96) -> 'NeuralAttentionForest':
    #     assert self.forest is not None, "Need to fit before weights optimization"

    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     use_fp16 = getattr(self.params, 'use_fp16', False)
    #     dtype = torch.float16 if use_fp16 else torch.float32
    #     self.nn = self.nn.to(device).to(dtype)

    #     dense_y = (self.training_y.toarray()
    #            if scipy.sparse.issparse(self.training_y)
    #            else self.training_y)
    #     background_X_cpu = torch.tensor(self.training_xs, dtype=dtype)
    #     background_y_cpu = torch.tensor(dense_y, dtype=dtype)
    #     if background_y_cpu.ndim == 1:
    #         background_y_cpu = background_y_cpu.unsqueeze(1)

    #     optim = torch.optim.AdamW(self.nn.parameters(), lr=self.params.lr)
    #     loss_fn = self._make_loss()
    #     n_epochs = self.params.n_epochs

    #     all_indices = np.arange(X.shape[0])
    #     parts = np.array_split(all_indices, n_parts)

    #     losses_per_epoch = []

    #     for epoch in trange(n_epochs, desc="Epochs"):
    #         epoch_loss_parts = []

    #         for part_id, part_idx in enumerate(parts, 1):
    #             X_part = X[part_idx]
    #             y_part = y_orig[part_idx]
    #             X_tensor = torch.tensor(X_part, dtype=dtype)
    #             if self.params.task == TaskType.CLASSIFICATION:
    #                 y_tensor = torch.tensor(y_part, dtype=torch.long)
    #             else:
    #                 y_tensor = torch.tensor(
    #                     y_part[:, None] if y_part.ndim == 1 else y_part,
    #                     dtype=torch.float32
    #                 )
    #             idx_tensor = torch.arange(X_part.shape[0])
    #             dataset = TensorDataset(X_tensor, y_tensor, idx_tensor)
    #             loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    #             seg_np = self._get_leaf_data_segments_gpu(
    #                 X_part,
    #                 exclude_input=True,
    #                 sample_batch_size=256,
    #                 background_batch_size=128,
    #                 device=device
    #             )
    #             full_seg = torch.tensor(seg_np, dtype=torch.bool, device=device)

    #             part_losses = []
    #             for batch_x, batch_y, batch_idx in loader:
    #                 batch_x = batch_x.to(device).to(dtype)
    #                 batch_y = batch_y.to(device)

    #                 idx = torch.randint(0, background_X_cpu.size(0),
    #                                 (background_batch_size,))
    #                 background_X = background_X_cpu[idx].to(device)
    #                 background_y = background_y_cpu[idx].to(device)
    #                 neighbors_hot = full_seg[batch_idx][:, idx]

    #                 preds = self.nn(batch_x, background_X,
    #                             background_y, neighbors_hot)
    #                 optim.zero_grad()
    #                 loss = loss_fn(preds, batch_y)
    #                 loss.backward()
    #                 optim.step()

    #                 part_losses.append(loss.item())

    #             mean_part_loss = float(np.mean(part_losses))
    #             epoch_loss_parts.append(mean_part_loss)

    #             del full_seg, seg_np, dataset, loader, X_tensor, y_tensor
    #             torch.cuda.empty_cache()
    #             gc.collect()

    #         epoch_mean_loss = float(np.mean(epoch_loss_parts))
    #         losses_per_epoch.append(epoch_mean_loss)
    #         print(f"Epoch {epoch+1}/{n_epochs} — mean loss: {epoch_mean_loss:.5f}")

    #     plt.plot(losses_per_epoch)
    #     plt.xlabel("Epoch")
    #     plt.ylabel("Mean Loss")
    #     plt.show()

    #     return self
   
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
    
    def _get_leaf_data_segments(self, X, exclude_input=False) -> Tuple[np.ndarray, np.ndarray]:
        # """
        # Args:
        #     X: Input points.
        #     exclude_input: Exclude leaf points that are exactly the same as input point.
        #                    It is useful to unbias training when fitting and optimizing
        #                    on the same data set.
        # """
        # leaf_ids = self.forest.apply(X)
        # # shape of leaf_ids: (n_samples, n_trees)
        # result = np.zeros((X.shape[0], self.leaf_sparse.shape[0], self.leaf_sparse.shape[1]), dtype=np.uint8)
        # # shape of `self.leaf_sparse`: (n_background_samples, n_trees, n_leaves)
        # for i in range(leaf_ids.shape[0]):
        #     for j in range(leaf_ids.shape[1]):
        #         result[i, :, j] = self.leaf_sparse[:, j, leaf_ids[i, j]]
        #     if exclude_input:
        #         result[i, i, :] = 0
        # # result shape: (n_samples, n_background_samples, n_trees)
        # return result
        leaf_ids = self.forest.apply(X)  # (n_samples, n_trees)
        return _get_leaf_data_segments_numba(self.leaf_sparse, leaf_ids, exclude_input)
    
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
        
        for si in tqdm(range(0, n_samples, sample_batch_size), 'leafs'):
            sb = min(sample_batch_size, n_samples - si)
            leaf_ids_batch = torch.from_numpy(leaf_ids[si:si + sb])  # (sb, n_trees)

            for bi in range(0, n_background, background_batch_size):
                try:
                    bb = min(background_batch_size, n_background - bi)
                    ls_cpu = self.leaf_sparse[bi:bi + bb]
                    ls = torch.from_numpy(ls_cpu).to(device)
                    ids = leaf_ids_batch.to(device)
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
    
    def _get_leaf_data_segments_from_leaf_ids_gpu(self,
                                              leaf_ids_batch: np.ndarray,
                                              bg_idx: np.ndarray,
                                              device: str = 'cuda',
                                              exclude_input: bool = True) -> torch.Tensor:
        B, T = leaf_ids_batch.shape
        M = bg_idx.shape[0]
        _, _, L = self.leaf_sparse.shape
        ls = torch.from_numpy(self.leaf_sparse[bg_idx]).to(device)
        ls = ls.unsqueeze(0).expand(B, M, T, L)
        ids = torch.from_numpy(leaf_ids_batch).to(device)  # (B, T)
        ids = ids.unsqueeze(1).unsqueeze(-1).expand(B, M, T, 1)
        seg = torch.gather(ls, dim=3, index=ids).squeeze(3)  # (B, M, T)
        if exclude_input:
            for i in range(B):
                if i < M:
                    seg[i, i, :] = 0
        return seg.bool()
    
    # def predict_batch(self, X, batch_size=512, background_batch_size=256, n_parts=64, need_attention_weights=False):
        
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     use_fp16 = getattr(self.params, 'use_fp16', False)
    #     dtype = torch.float16 if use_fp16 else torch.float32

    #     self.nn = self.nn.to(device).to(dtype)
    #     self.nn.eval()

    #     dense_y = self.training_y.toarray() if scipy.sparse.issparse(self.training_y) else self.training_y
    #     background_X_cpu = torch.tensor(self.training_xs, dtype=dtype)
    #     background_y_cpu = torch.tensor(dense_y, dtype=dtype)
    #     if background_y_cpu.ndim == 1:
    #         background_y_cpu = background_y_cpu.unsqueeze(1)

    #     all_indices = np.arange(X.shape[0])
    #     parts = np.array_split(all_indices, n_parts)

    #     predictions_all = []
    #     if need_attention_weights:
    #         x_recon_all, alphas_all, betas_all = [], [], []

    #     with torch.no_grad():
    #         for part_id, part_idx in enumerate(parts, 1):
    #             X_part = X[part_idx]
    #             X_tensor = torch.tensor(X_part, dtype=dtype)
    #             part_indices = torch.arange(X_part.shape[0])
    #             dataset = TensorDataset(X_tensor, part_indices)
    #             loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    #             # neighbors_hot for full part
    #             seg_np = self._get_leaf_data_segments_gpu(
    #                 X_part,
    #                 exclude_input=False,
    #                 sample_batch_size=256,
    #                 background_batch_size=128,
    #                 device=device
    #             )
    #             full_seg = torch.tensor(seg_np, dtype=torch.bool, device=device)

    #             for batch_x, batch_idx in tqdm(loader, desc="Predict batches"):
    #                 batch_x = batch_x.to(device).to(dtype)

    #                 # random context
    #                 idx = torch.randint(0, background_X_cpu.size(0), (background_batch_size,))
    #                 background_X = background_X_cpu[idx].to(device)
    #                 background_y = background_y_cpu[idx].to(device)
    #                 neighbors_hot = full_seg[batch_idx][:, idx].to(device)

    #                 output = self.nn(
    #                     batch_x,
    #                     background_X,
    #                     background_y,
    #                     neighbors_hot,
    #                     need_attention_weights=need_attention_weights,
    #                 )

    #                 if isinstance(output, tuple):
    #                     preds, x_recon, alphas, betas = output
    #                     predictions_all.append(preds.cpu())
    #                     x_recon_all.append(x_recon.cpu())
    #                     alphas_all.append(alphas.cpu())
    #                     betas_all.append(betas.cpu())
    #                 else:
    #                     predictions_all.append(output.cpu())

    #             del full_seg, seg_np, dataset, loader, X_tensor
    #             torch.cuda.empty_cache()
    #             gc.collect()

    #     predictions = torch.cat(predictions_all, dim=0).numpy()

    #     if self.params.kind.need_add_init():
    #         predictions += self.forest.init_.predict(X)[:, np.newaxis]

    #     if not need_attention_weights:
    #         return predictions
    #     else:
    #         return (
    #             predictions,
    #             torch.cat(x_recon_all, dim=0).numpy(),
    #             torch.cat(alphas_all, dim=0).numpy(),
    #             torch.cat(betas_all, dim=0).numpy(),
    #         )
            
    def predict_batch(self, X, batch_size=512, background_batch_size=256, need_attention_weights=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_fp16 = getattr(self.params, 'use_fp16', False)
        dtype = torch.float16 if use_fp16 else torch.float32

        self.nn = self.nn.to(device).to(dtype)
        self.nn.eval()

        dense_y = self.training_y.toarray() if scipy.sparse.issparse(self.training_y) else self.training_y
        background_X_cpu = torch.tensor(self.training_xs, dtype=dtype)
        background_y_cpu = torch.tensor(dense_y, dtype=dtype)
        if background_y_cpu.ndim == 1:
            background_y_cpu = background_y_cpu.unsqueeze(1)

        X_tensor = torch.tensor(X, dtype=dtype)
        all_indices = torch.arange(X.shape[0])
        dataset = TensorDataset(X_tensor, all_indices)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        predictions_all = []
        if need_attention_weights:
            x_recon_all, alphas_all, betas_all = [], [], []

        with torch.no_grad():
            for batch_x, batch_idx in tqdm(loader, desc="Predicting"):
                batch_x = batch_x.to(device).to(dtype)

                bg_idx = torch.randint(0, background_X_cpu.size(0), (background_batch_size,), dtype=torch.long)
                background_X = background_X_cpu[bg_idx].to(device)
                background_y = background_y_cpu[bg_idx].to(device)

                leaf_ids_batch = self.forest.apply(batch_x.cpu().numpy())
                neighbors_hot = self._get_leaf_data_segments_from_leaf_ids_gpu(
                    leaf_ids_batch, bg_idx.cpu().numpy(), device=device, exclude_input=False
                )

                output = self.nn(
                    batch_x,
                    background_X,
                    background_y,
                    neighbors_hot,
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

                torch.cuda.empty_cache()
                gc.collect()

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
