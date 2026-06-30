import torch
from typing import Tuple, List, Optional, Dict, Any
from abc import ABC, abstractmethod
import logging

# --- 缓存基类和统一接口定义 ---

class BaseClusterReducer(ABC):
    """
    支持集群 ID 缓存的神经网络参数降维方法的抽象基类。
    为每个唯一的 cluster_ids 维护一个独立的投影矩阵。
    """
    def __init__(self, target_dim: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        self.target_dim = target_dim
        # 存储投影矩阵的缓存: {cluster_ids: V_matrix}
        self.projection_matrices_cache: Dict[Tuple, torch.Tensor] = {} 
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.float32
        self.logger = logging.getLogger("ClusterReducer")

    @abstractmethod
    def _fit_logic(self, X: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        核心 fit 逻辑，计算并返回投影基 V (n_features, target_dim)。
        子类必须实现此方法。
        """
        pass

    def fit(self, cluster_ids: Tuple, X: torch.Tensor, **kwargs: Any):
        """
        学习降维基，并将其与 cluster_ids 关联存储。
        """
        # 调用核心逻辑计算投影基
        projection_matrix = self._fit_logic(X, **kwargs)
        
        # 存储到缓存
        self.projection_matrices_cache[cluster_ids] = projection_matrix
        # print(f"Fitted and cached V for cluster {cluster_ids}. Shape: {projection_matrix.shape}")

    def transform(self, cluster_ids: Tuple, X: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        使用指定 cluster_ids 对应的投影基进行投影。
        """
        if cluster_ids not in self.projection_matrices_cache:
            # raise KeyError(f"Projection matrix for cluster {cluster_ids} not found. Call fit() first.")
            self.logger.info(f"Projection initialize for cluster {cluster_ids}.")
            self.fit(cluster_ids, X, **kwargs)  # 自动 fit
            
        V = self.projection_matrices_cache[cluster_ids]
        
        # 确保输入数据和投影基在同一设备和类型
        X = X.to(device=self.device, dtype=self.dtype)
        
        # 投影: X @ V
        return X @ V

    def fit_transform(self, cluster_ids: Tuple, X: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """ 组合 fit 和 transform。 """
        self.fit(cluster_ids, X, **kwargs)
        return self.transform(cluster_ids, X)


# --- 1. 标准 PCA (支持集群缓存) ---

class ClusterStandardPCA(BaseClusterReducer):
    """
    标准 PCA，为每个集群 ID 独立计算和缓存投影基。
    """
    @torch.no_grad()
    def _fit_logic(self, X: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """ 
        计算标准 PCA 的投影基 V。 
        X: (n_samples, n_features)
        """
        X = X.to(device=self.device, dtype=self.dtype)
        n_samples, n_features = X.shape

        # SVD 分解: X = U Sigma V^T
        _, _, Vt = torch.linalg.svd(X, full_matrices=False)
        
        # 提取 V 的前 K 列 (即 Vt.T 的前 K 列)
        V = Vt.T 
        
        k = min(self.target_dim, V.shape[1])
        # if k < self.target_dim:
        #     print(f"Warning: Target dimension {self.target_dim} exceeds max possible rank {V.shape[1]}. Using K={k}.")
            
        return V[:, :k]
    


# --- 2. 随机 PCA (基于您的原始逻辑修正，支持集群缓存) ---

class ClusterRandomizedPCA(BaseClusterReducer):
    """
    随机化 PCA (RPCA)，为每个集群 ID 独立计算和缓存投影基。
    """
    def __init__(self, target_dim: int, oversample: int = 5, n_iter: int = 2, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__(target_dim, device, dtype)
        self.oversample = oversample
        self.n_iter = n_iter
        self.n_random = self.target_dim + self.oversample
        # self.logger = kwargs.get('logger', None) # 可以在实际应用中加入日志

    @torch.no_grad()
    def _fit_logic(self, X: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """ 
        计算 RPCA 的投影基 V。
        X: (n_samples, n_features)
        """
        X = X.to(device=self.device, dtype=self.dtype)
        n_samples, n_features = X.shape
        
        # if n_samples < self.n_random:
        #     print(f"Warning: n_samples ({n_samples}) < n_random ({self.n_random}). Max effective rank is n_samples.")
        
        # 1. 随机投影矩阵 R: (n_features, n_random)
        rand_mat = torch.randn(n_features, self.n_random, device=self.device, dtype=self.dtype)
        
        # 2. 初始投影 Y = X @ R: (n_samples, n_random)
        Y = X @ rand_mat
        
        # 3. 幂迭代 (交替投影，修正您的原始逻辑)
        for _ in range(self.n_iter):
            # Y = X.T @ Y -> (n_features, n_random)
            Y = X.T @ Y 
            # Y = X @ Y -> (n_samples, n_random)
            Y = X @ Y
            # QR 分解 (正交化)
            Y, _ = torch.linalg.qr(Y, mode='reduced')
            
        # 4. 最终 QR 分解: Q 是 X 投影空间的近似基 Q
        Q, _ = torch.linalg.qr(Y, mode='reduced')
        
        # 5. 投影到小矩阵 B = Q.T @ X
        B = Q.T @ X
        
        # 6. SVD 分解 B = U Sigma V^T
        _, _, Vt = torch.linalg.svd(B, full_matrices=False)
        
        # 7. 提取前 K 个主成分方向 V
        V = Vt.T
        
        k = min(self.target_dim, V.shape[1])
        return V[:, :k]


# --- 3. 直接截取 (支持集群缓存) ---

class ClusterFeatureTruncation(BaseClusterReducer):
    """
    特征直接截取，为每个集群 ID 独立构造相同的投影基 (恒等截取)。
    """
    @torch.no_grad()
    def _fit_logic(self, X: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """ 
        构造一个稀疏的截取矩阵作为投影基。 
        """
        X = X.to(device=self.device, dtype=self.dtype)
        n_features = X.shape[1]
        
        # 投影基 V: (n_features, target_dim)
        V = torch.zeros(n_features, self.target_dim, device=self.device, dtype=self.dtype)
        
        # 截取前 K 个特征
        k = min(self.target_dim, n_features)
        V[:k, :k] = torch.eye(k, device=self.device, dtype=self.dtype)
        
        return V

class TorchPCA:
    def __init__(self, target_dim, device=None):
        self.target_dim = target_dim
        self.device = device
        self.mean = None
        self.components = None  # shape: (k, d)

    def fit(self, X):
        """
        X: Tensor, shape (n_samples, n_features)
        """
        if self.device is None:
            self.device = X.device

        X = X.to(self.device)

        # compute mean
        self.mean = X.mean(dim=0, keepdim=True)
        X_centered = X - self.mean

        # SVD decomposition
        # X_centered = U * S * Vh
        U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)

        # principal directions: take top-k rows of Vh
        self.components = Vh[:self.target_dim]  # (k, d)

    def transform(self, X):
        """
        X: (n_samples, n_features)
        return reduced X: (n_samples, k)
        """
        X = X.to(self.device)

        X_centered = X - self.mean
        X_reduced = X_centered @ self.components.T
        return X_reduced.cpu()
    
    def fit_transform(self, X):
        """
        Fit PCA on X and return reduced representation
        """
        self.fit(X)
        return self.transform(X)


class IncrementalPCA:
    """
    Incremental / Online PCA for PyTorch
    Can update projection with new batches without storing all past samples.
    """
    def __init__(self, target_dim, device=None, smoothing=0.05):
        """
        target_dim: int, number of principal components to keep
        device: torch.device
        smoothing: float, optional exponential smoothing factor for components update
        """
        self.target_dim = target_dim
        self.device = device
        self.mean = None         # (1, d)
        self.components = None   # (k, d)
        self.n_samples = 0       # total number of seen samples
        self.smoothing = smoothing

    def fit(self, X):
        """
        X: Tensor, shape (n_samples_batch, n_features)
        Incrementally update mean and top-k components
        """
        if self.device is None:
            self.device = X.device
        X = X.to(self.device)
        batch_size, n_features = X.shape

        # compute batch mean and centered X
        batch_mean = X.mean(dim=0, keepdim=True)
        X_centered = X - batch_mean

        # update global mean
        if self.mean is None:
            self.mean = batch_mean
        else:
            total_n = self.n_samples + batch_size
            self.mean = (self.mean * self.n_samples + batch_mean * batch_size) / total_n

        # center with updated mean
        X_centered = X - self.mean

        # compute SVD of centered batch
        U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
        batch_components = Vh[:self.target_dim]  # (k, d)

        # update components with smoothing
        if self.components is None:
            self.components = batch_components
        else:
            dot_product = torch.sum(self.components * batch_components, dim=1, keepdim=True)
            batch_components = batch_components * torch.sign(dot_product)
            self.components = (1 - self.smoothing) * self.components + self.smoothing * batch_components
            # re-orthogonalize to maintain orthonormality
            q, _ = torch.linalg.qr(self.components.T)
            self.components = q.T

        self.n_samples += batch_size

    def transform(self, X):
        """
        Project X onto the top-k components
        X: Tensor, shape (n_samples, n_features)
        Returns: Tensor, shape (n_samples, target_dim)
        """
        X = X.to(self.device)
        X_centered = X - self.mean
        X_reduced = X_centered @ self.components.T
        return X_reduced.cpu()

    def fit_transform(self, X):
        """
        Convenience method for initial batch
        """
        self.fit(X)
        return self.transform(X)

class RandomProjection:
    def __init__(self, target_dim, device=None):
        self.input_dim = None
        self.output_dim = target_dim
        self.R = None

    def fit(self, X):
        if self.R is None:
            self.input_dim = X.shape[1]
            # Gaussian random matrix
            self.R = torch.randn(
                self.output_dim, self.input_dim
            ) / (self.output_dim ** 0.5)

    def transform(self, X):
        return (X @ self.R.T).cpu()

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def set_feature_reducer(method: str) -> BaseClusterReducer:
    """
    根据指定方法创建对应的集群降维器实例。
    """
    method = method.upper()
    if method == 'IPCA':
        return IncrementalPCA
    elif method == 'PCA':
        return TorchPCA
    elif method == 'RP':
        return RandomProjection
    else:
        raise ValueError(f"Unknown feature reduction method: {method}")


# --- 示例使用 ---

if __name__ == '__main__':
    # 模拟数据和集群设置
    N_A, N_B = 1000, 800  # 集群 A 和 B 的样本数
    P = 5000            # 特征数
    K = 32              # 目标降维维度
    
    # 模拟两个不同的集群数据
    X_A = torch.randn(N_A, P, dtype=torch.float64) 
    X_B = torch.randn(N_B, P, dtype=torch.float64)
    
    # 集群 ID
    CLUSTER_A = (1, 2, 3)
    CLUSTER_B = (4, 5, 6)
    
    print(f"Feature Dimension P: {P}, Target Dimension K: {K}\n")
    
    # --- 实例化并使用 RPCA ---
    rpca_reducer = ClusterRandomizedPCA(target_dim=K, oversample=10, n_iter=3, dtype=torch.float64)
    
    print("--- Fitting Cluster A (RPCA) ---")
    rpca_reduced_A = rpca_reducer.fit_transform(CLUSTER_A, X_A)
    print(f"RPCA Reduced A shape: {rpca_reduced_A.shape}")

    print("\n--- Fitting Cluster B (RPCA) ---")
    rpca_reducer.fit(CLUSTER_B, X_B)
    rpca_reduced_B = rpca_reducer.transform(CLUSTER_B, X_B)
    print(f"RPCA Reduced B shape: {rpca_reduced_B.shape}")
    
    # 检查缓存中是否有两个不同的矩阵
    V_A = rpca_reducer.projection_matrices_cache[CLUSTER_A]
    V_B = rpca_reducer.projection_matrices_cache[CLUSTER_B]
    print(f"\nRPCA Cache Size: {len(rpca_reducer.projection_matrices_cache)}")
    print(f"V_A == V_B? {torch.equal(V_A, V_B)}") # 应为 False (随机性或数据不同)
    print("-" * 30)
    
    # --- 实例化并使用标准 PCA ---
    pca_reducer = ClusterStandardPCA(target_dim=K, dtype=torch.float64)
    
    print("--- Fitting Cluster A (Standard PCA) ---")
    pca_reduced_A = pca_reducer.fit_transform(CLUSTER_A, X_A)
    print(f"PCA Reduced A shape: {pca_reduced_A.shape}")

    # 尝试转换，但不先 fit (会抛出 KeyError)
    try:
        pca_reducer.transform(CLUSTER_B, X_B)
    except KeyError as e:
        print(f"\nExpected Error: {e}")
        
    print("\n--- Fitting Cluster B (Standard PCA) ---")
    pca_reducer.fit(CLUSTER_B, X_B)
    pca_reduced_B = pca_reducer.transform(CLUSTER_B, X_B)
    print(f"PCA Reduced B shape: {pca_reduced_B.shape}")