
import numpy as np
import pandas as pd
import torch
import warnings

from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from datapoisontoolkit.detectors import base_defense


class DifferentiableLOF(base_defense.DifferentiableDefense):
    def __init__(self, n_neighbors=20, temperature=1.0):
        self.k = n_neighbors
        self.temperature = temperature

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Soft LOF Implementation
        # 1. Compute pairwise distances
        pairwise_dist = torch.cdist(x, x)
        
        # 2. Soft k-distance using logsumexp (smooth max approximation)
        # Using negative distances because logsumexp is soft-max
        # We want the (k+1)-th smallest distance
        sorted_dists, _ = torch.sort(pairwise_dist, dim=1)
        k_dist_hard = sorted_dists[:, self.k]
        
        # Soft approximation around the hard k-dist
        scaled_dists = -(pairwise_dist - k_dist_hard.unsqueeze(1))**2 / self.temperature
        soft_k_dist = -self.temperature * torch.logsumexp(scaled_dists, dim=1) + k_dist_hard
        
        return soft_k_dist

class LOFDefense(base_defense.Defense):
    def __init__(self, n_neighbors=20):
        self.n_neighbors = n_neighbors

    def detect(self, series: pd.Series) -> pd.Series:
        model = LocalOutlierFactor(n_neighbors=self.n_neighbors, contamination='auto')
        model.fit_predict(series.values.reshape(-1, 1))
        return pd.Series(-model.negative_outlier_factor_, index=series.index)

    def get_differentiable_proxy(self) -> base_defense.DifferentiableDefense:
        return DifferentiableLOF(self.n_neighbors)

# ### 5.2. PCA-based Defense
class DifferentiablePCA(base_defense.DifferentiableDefense):
    def __init__(self, n_components=1):
        self.n_components = n_components

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # A differentiable version of PCA: SVD (Singular Value Decomposition)
        # Center the data
        x_centered = x - x.mean(dim=0, keepdim=True)
        U, S, V = torch.svd(x_centered)
        
        # Reconstruct using top-k components
        # x_reconstructed = U[:, :k] @ S[:k] @ V[:, :k].T + mean
        # Simpler: Project to subspace and back
        components = V[:, :self.n_components]
        projection = x_centered @ components
        reconstruction = projection @ components.T + x.mean(dim=0, keepdim=True)
        
        # The reconstruction error is the anomaly score
        return torch.sum((x - reconstruction)**2, dim=1)

class PCADefense(base_defense.Defense):
    def __init__(self, n_components=1):
        self.n_components = n_components

    def detect(self, series: pd.Series) -> pd.Series:
        data = series.values.reshape(-1, 1)
        pca = PCA(n_components=self.n_components)
        projected = pca.fit_transform(data)
        reconstructed = pca.inverse_transform(projected)
        error = np.sum((data - reconstructed)**2, axis=1)
        return pd.Series(error, index=series.index)

    def get_differentiable_proxy(self) -> base_defense.DifferentiableDefense:
        return DifferentiablePCA(self.n_components)

# ### 5.3. Kernel Density Estimation (KDE) Defense
class DifferentiableKDE(base_defense.DifferentiableDefense):
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # A differentiable version of KDE using a Gaussian Kernel
        pairwise_dist = torch.cdist(x, x)
        # Anomaly score = negative log-likelihood
        # -log( sum( exp( -dist^2 / (2h^2) ) ) )
        log_prob = torch.logsumexp(-0.5 * (pairwise_dist / self.bandwidth)**2, dim=1)
        return -log_prob

class KDEDefense(base_defense.Defense):
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth

    def detect(self, series: pd.Series) -> pd.Series:
        data = series.values.reshape(-1, 1)
        kde = KernelDensity(bandwidth=self.bandwidth)
        kde.fit(data)
        log_density = kde.score_samples(data)
        return pd.Series(-log_density, index=series.index)

    def get_differentiable_proxy(self) -> base_defense.DifferentiableDefense:
        return DifferentiableKDE()

# ### 5.4 Isolation Forest Defense (Soft Implementation)
class DifferentiableIsolationForest(base_defense.DifferentiableDefense):
    def __init__(self, n_trees=100, max_depth=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        # Random projection vectors (hyperplanes)
        # We initialize them lazily in __call__ or need to store them
        self.w = None
        self.b = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_features = x.shape
        
        # Initialize random splits if not done
        if self.w is None or self.w.device != x.device:
            # w: [n_trees * max_depth, n_features]
            self.w = torch.randn(self.n_trees * self.max_depth, n_features, device=x.device)
            # b: [n_trees * max_depth] - biases/thresholds
            # Assuming standardized data, thresholds around 0
            self.b = torch.randn(self.n_trees * self.max_depth, device=x.device)

        # Compute "split probability" or "path length proxy"
        # Projections: p = x @ w.T + b
        # sigmoid(p) gives probability of going right vs left
        # Ideally we want to measure how "isolated" a point is.
        # Points that fall into small partitions (far from others) are anomalies.
        
        # Simplified differentiable proxy:
        # Measures how far a point is from the "center" of random partitions
        projections = x @ self.w.T + self.b # [batch, n_projections]
        
        # If point is far from boundaries (projections are large), it's "easy to isolate" or "extreme"
        # We can sum the absolute values or squared values of these random projections
        # This acts like a random basis expansion similar to isolation forest intuition
        return torch.mean(torch.sigmoid(torch.abs(projections)), dim=1)

class IsolationForestDefense(base_defense.Defense):
    def __init__(self, contamination='auto'):
        self.contamination = contamination

    def detect(self, series: pd.Series) -> pd.Series:
        if series.nunique() < 2:
            return pd.Series(0, index=series.index)
            
        model = IsolationForest(contamination=self.contamination, random_state=42)
        preds = model.fit_predict(series.values.reshape(-1, 1))
        # 1: outlier, 1: normal
        scores = pd.Series(-preds, index=series.index)
        return scores

    def get_differentiable_proxy(self) -> base_defense.DifferentiableDefense:
        return DifferentiableIsolationForest()

# ### 5.5 Median/Quantile Filter (Soft Implementation)
class DifferentiableMedian(base_defense.DifferentiableDefense):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Soft Median using sigmoid approximation for rank
        # rank(x_i) = sum_j sigmoid(beta * (x_i - x_j))
        beta = 2.0 
        x_flat = x.squeeze()
        if x_flat.ndim > 1: x_flat = x_flat.mean(dim=1) # Handle multi-dim
        
        diffs = x_flat.unsqueeze(0) - x_flat.unsqueeze(1) # [N, N]
        ranks = torch.sigmoid(beta * diffs).sum(dim=1) / x_flat.shape[0] # [0, 1]
        
        # Median is where rank is close to 0.5
        # We want to minimize distance to median
        # Vanishing gradient fix: Add a linear term for extreme values
        # score = abs(rank - 0.5) + alpha * abs(x - median_val_approx)
        # Simplified: Just ensure rank gradient doesn't die
        loss = torch.abs(ranks - 0.5)
        # Regularization: pull extreme values towards mean to re-enter solvable range
        loss += 0.01 * torch.abs(x_flat - x_flat.mean()) 
        return loss

class MedianFilterDefense(base_defense.Defense):
    def __init__(self, window=7):
        self.window = window

    def detect(self, series: pd.Series) -> pd.Series:
        median = series.rolling(window=self.window, center=True, min_periods=1).median()
        error = np.abs(series - median)
        return pd.Series(error, index=series.index)

    def get_differentiable_proxy(self) -> base_defense.DifferentiableDefense:
        return DifferentiableMedian()

class DifferentiableQuantile(base_defense.DifferentiableDefense):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Soft Quantile Implementation
        # Similar to Median, but we penalize being in tails
        beta = 2.0
        x_flat = x.squeeze()
        if x_flat.ndim > 1: x_flat = x_flat.mean(dim=1)
        
        # Calculate distance matrix for ranking approximation
        diffs = x_flat.unsqueeze(0) - x_flat.unsqueeze(1)
        ranks = torch.sigmoid(beta * diffs).sum(dim=1) / x_flat.shape[0]
        
        # Penalize if rank is < 0.05 or > 0.95 (approx)
        # Fix vanishing gradient: use softplus for smoother penalties out of bounds
        lower_penalty = torch.nn.functional.softplus(50 * (0.05 - ranks))
        upper_penalty = torch.nn.functional.softplus(50 * (ranks - 0.95))
        
        return lower_penalty + upper_penalty

class QuantileDefense(base_defense.Defense):
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def detect(self, series: pd.Series) -> pd.Series:
        lq = series.quantile(self.lower)
        uq = series.quantile(self.upper)
        mask = (series < lq) | (series > uq)
        return pd.Series(mask.astype(int), index=series.index)

    def get_differentiable_proxy(self) -> base_defense.DifferentiableDefense:
        return DifferentiableQuantile()

# ### 5.6 DBSCAN (Soft Implementation)
class DifferentiableDBSCAN(base_defense.DifferentiableDefense):
    def __init__(self, eps=0.5, beta=1.0):
        self.eps = eps
        self.beta = beta # Smoother gradient (was 5.0)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Soft neighbor count
        pairwise_dist = torch.cdist(x, x)
        
        # Soft threshold: sigmoid(beta * (eps - dist))
        soft_neighbors = torch.sigmoid(self.beta * (self.eps - pairwise_dist)).sum(dim=1)
        
        # Anomaly score is inverse of density. 
        # Fix vanishing gradient: If density is very low (isolated),
        # added a term that explicitly minimizes distance to nearest neighbors.
        # This "pulls" the point towards the cluster even if sigmoid saturated.
        
        # Calculate distance to nearest 5 neighbors (approx)
        k_nearest_dist, _ = torch.topk(pairwise_dist, k=min(10, x.shape[0]), dim=1, largest=False)
        dist_penalty = k_nearest_dist.mean(dim=1)
        
        return -torch.log(soft_neighbors + 1e-6) + 0.5 * dist_penalty

class DBSCANDefense(base_defense.Defense):
    def detect(self, series: pd.Series) -> pd.Series:
        X = series.values.reshape(-1, 1)
        model = DBSCAN(eps=0.5, min_samples=10)
        labels = model.fit_predict(X)
        # Label -1: outlier
        scores = (labels == -1).astype(int)
        return pd.Series(scores, index=series.index)
        
    def get_differentiable_proxy(self) -> base_defense.DifferentiableDefense:
        return DifferentiableDBSCAN()

def get_defense(name: str) -> base_defense.Defense:
    if name == "lof": return LOFDefense()
    if name == "pca": return PCADefense()
    if name == "kde": return KDEDefense()
    if name == 'forest' : return IsolationForestDefense()
    if name == "median" : return MedianFilterDefense()
    if name == 'quantile' : return QuantileDefense()
    if name == 'dbscan' : return DBSCANDefense()
    raise ValueError(f"Unknown defense: {name}")