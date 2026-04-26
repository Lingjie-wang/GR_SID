import functools
import logging
from typing import Optional, Tuple

import torch
import torch.distributed as dist

from src.components.distance_functions import DistanceFunction
from src.components.clustering_initializers import ClusteringInitializer
from src.components.loss_functions import WeightedSquaredError
from src.components.quantization_strategies import QuantizationStrategy
from src.models.modules.clustering.base_clustering_module import BaseClusteringModule


class VectorQuantization(BaseClusteringModule):
    def __init__(
        self,
        n_clusters: int,
        n_features: int,
        distance_function: DistanceFunction,
        initializer: ClusteringInitializer,
        quantization_strategy: QuantizationStrategy,
        loss_function: torch.nn.Module = WeightedSquaredError(),
        optimizer: torch.optim.Optimizer = functools.partial(
            torch.optim.SGD,
            lr=0.5,
        ),
        init_buffer_size: int = 1000,
        restart_unused_codes: bool = False,
        restart_noise_scale: float = 0.01,
        restart_ema_decay: float = 0.99,
        restart_usage_threshold: float = 1.0,
    ):
        """
        Initialize the VectorQuantization module.

        Args:
            n_clusters: Number of clusters.
            n_features: Number of features in the input data.
            distance_function: Distance function to use for computing distances between points.
            loss_function: Loss function to use for training.
            optimizer: Optimizer to use for training.
            init_method: Initialization method ("random" or "k-means++").
            init_buffer_size: Number of points to buffer for initialization.
            restart_unused_codes: Whether to restart unused codes during training.
            restart_noise_scale: Scale of the noise to add when restarting unused codes.
            restart_ema_decay: EMA decay used for code usage statistics.
            restart_usage_threshold: Codes with EMA usage below this value are restarted.
        """

        super().__init__(
            n_clusters=n_clusters,
            n_features=n_features,
            distance_function=distance_function,
            loss_function=loss_function,
            optimizer=optimizer,
            initializer=initializer,
            init_buffer_size=init_buffer_size,
        )

        self.quantization_strategy = quantization_strategy
        self.restart_unused_codes = restart_unused_codes
        self.restart_noise_scale = restart_noise_scale
        self.restart_ema_decay = restart_ema_decay
        self.restart_usage_threshold = restart_usage_threshold
        self.last_restart_count = 0
        self.register_buffer("cluster_size_ema", torch.zeros(self.n_clusters))

    @torch.no_grad()
    def _tile_with_noise(self, vectors: torch.Tensor, target_n: int) -> torch.Tensor:
        """Tile vectors and add small noise so we can sample enough restart candidates."""
        n_vectors, embed_dim = vectors.shape
        n_repeats = (target_n + n_vectors - 1) // n_vectors
        std = vectors.new_ones(embed_dim) * self.restart_noise_scale / (embed_dim**0.5)
        vectors = vectors.repeat(n_repeats, 1)
        vectors = vectors + torch.rand_like(vectors) * std
        return vectors

    @torch.no_grad()
    def _update_cluster_size_ema(self, assignments: torch.Tensor) -> None:
        """
        Update EMA usage counts for each code in the current batch.
        
        Args:
            assignments: Tensor of shape (batch_size,) containing the cluster assignment for each input point.
        """
        if assignments.numel() == 0:
            return

        flat_assignments = assignments.reshape(-1).to(device=self.centroids.device)
        cluster_size = torch.bincount(
            flat_assignments,
            minlength=self.n_clusters,
        ).to(device=self.centroids.device, dtype=self.cluster_size_ema.dtype) #* 计算每个聚类中心被分配的输入点数量

        #* 分布式场景下做全局求和
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(cluster_size, op=dist.ReduceOp.SUM)

        #* 更新 cluster_size_ema，使用指数移动平均来平滑每个聚类中心的使用频率。这样可以更稳定地判断哪些聚类中心是未使用的，从而决定是否需要重启它们。
        self.cluster_size_ema.mul_(self.restart_ema_decay).add_(
            cluster_size,
            alpha=1.0 - self.restart_ema_decay,
        )

    @torch.no_grad()
    def _restart_unused_codes_if_needed(
        self, batch: torch.Tensor, assignments: torch.Tensor
    ) -> int:
        '''
        Restart unused codes with random vectors from the current batch if needed.

        Args:
            batch: Tensor of shape (batch_size, n_features) containing the input data points.
            assignments: Tensor of shape (batch_size,) containing the cluster assignment for each input point.
        '''

        #* 只有在训练阶段并且设置了 restart_unused_codes 时才会执行重启未使用的代码
        if not self.training or not self.restart_unused_codes:
            return 0

        vectors = batch.reshape(-1, self.n_features).detach()
        if vectors.shape[0] == 0:
            return 0
        
        #? 如果输入的向量数量少于聚类中心的数量，我们需要通过重复输入向量并添加噪声来生成足够的候选重启向量。这是为了确保我们有足够的候选向量来替换未使用的聚类中心。
        if vectors.shape[0] < self.n_clusters:
            vectors = self._tile_with_noise(vectors, self.n_clusters)

        n_vectors = vectors.shape[0]
        random_vectors = vectors[torch.randperm(n_vectors, device=vectors.device)][
            : self.n_clusters
        ]
        random_vectors = random_vectors.to(
            device=self.centroids.device, dtype=self.centroids.dtype
        )

        if dist.is_available() and dist.is_initialized():
            dist.broadcast(random_vectors, src=0)

        self._update_cluster_size_ema(assignments)
        
        #* 按照 EMA 判死码
        usage = (
            self.cluster_size_ema.view(-1, 1) >= self.restart_usage_threshold
        ).to(dtype=random_vectors.dtype)
        unused_mask = ~(usage.view(-1).bool())
        restart_count = int(unused_mask.sum().item()) #* 未使用的码字数量

        if restart_count > 0:
            # Keep used codes as-is and only replace dead codes with sampled vectors.
            self.centroids.data.mul_(usage).add_(random_vectors * (1.0 - usage))
            # After restart, reset EMA usage of revived codes so they are treated as active.
            self.cluster_size_ema.mul_(usage.view(-1)).add_(
                (1.0 - usage).view(-1) * self.restart_usage_threshold
            )

            trainer = getattr(self, "_trainer", None)
            if trainer is not None and getattr(getattr(trainer, "model", None), "verbose", False):
                logging.info(
                    "Device %s: Restarted %s unused codes in VectorQuantization",
                    self.device,
                    restart_count,
                )

        return restart_count

    def forward(
        self, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass of the K-Means model on the input batch.

        This function computes the cluster assignments for each input point, the number
        of points in each cluster, and the sum of points in each cluster.

        Args:
            batch: Data points of shape (batch_size, n_features)

        Returns:
            assignments: Cluster assignments of shape (batch_size,)
            embeddings: Embeddings of shape (batch_size, n_features).
                These embeddings will be used for computing the quantization loss.
            reconstruction_loss_embeddings: Embeddings of shape (batch_size, n_features)
                computed in a way that enables gradient backpropagation through the input
                embeddings. If the quantization strategy does not support this, this will
                be None.
        """
        codebook = self.get_centroids()
        (
            ids,
            embeddings,
            reconstruction_loss_embeddings,
        ) = self.quantization_strategy.quantize(
            codebook=codebook,
            batch=batch,
        )
        return ids, embeddings, reconstruction_loss_embeddings

    def model_step(
        self,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Perform a forward pass of the K-Means model on the batch and compute the loss.
        ? 一定要是 K-Means Model 吗？不能是其他的残差量化方法吗？
        ? 我的理解是：这里的注释是其他地方复制过来的，不一定是 K-Means model
        This function may be called by another LightningModule, such as a residual
        K-means module, that is using this MiniBatchKMeans module as a submodule.

        Calling this function along will not update the centroids, and will not
        increment self.global_step.（执行的全局步数， global_step 的上界应该是一个超参数） 
        If a parent module is using this module as a
        submodule, the parent will be responsible for updating those parameters.
        Otherwise, these will be updated by Lightning after it calls
        training_step.

        Args:
            batch: Data points of shape (batch_size, n_features)

        Returns:
            assignments: Cluster assignments of shape (batch_size,)
            global_loss_embeddings: Embeddings of shape (batch_size, n_features)
            loss: Loss value. Tensor of shape (1,)
        """
        if batch.device != self.device:
            batch = batch.to(self.device)

        #* 初始化中心点
        # Initialize centroids using the chosen method
        # Buffer initial batches for better initialization
        if self.is_initial_step:
            self.is_initial_step = False
            self.is_initialized = True
        if not self.is_initialized:
            return self.initialization_step(batch)

        assignments, embeddings, reconstruction_loss_embeddings = self.forward(batch)
        self.last_restart_count = self._restart_unused_codes_if_needed(batch, assignments)
        if self.last_restart_count > 0:
            assignments, embeddings, reconstruction_loss_embeddings = self.forward(batch)

        loss = self.loss_function(batch, embeddings)  # quantization loss
        return (
            assignments,
            reconstruction_loss_embeddings
            if reconstruction_loss_embeddings is not None
            else embeddings,
            loss,
        )
