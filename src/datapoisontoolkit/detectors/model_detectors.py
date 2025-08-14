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
	def __init__(self, n_neighbors=20):
		self.k = n_neighbors

	def __call__(self, x: torch.Tensor) -> torch.Tensor:
		pairwise_dist = torch.cdist(x, x)
		k_distances, _ = torch.kthvalue(pairwise_dist, self.k + 1, dim=1)
		# Basit, türevlenebilir proxy: k-mesafesi ne kadar büyükse, o kadar anomalidir.
		return k_distances

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
		# PCA'in türevlenebilir bir versiyonu: SVD (Singular Value Decomposition)
		U, S, V = torch.svd(x - x.mean())
		reconstruction = (U[:, :self.n_components] @ torch.diag(S[:self.n_components]) @ V[:, :self.n_components].T) + x.mean()
		# Yeniden yapılandırma hatası anomali skorudur
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
		# Gaussian Kernel kullanarak KDE'nin türevlenebilir bir versiyonu
		pairwise_dist = torch.cdist(x, x)
		# Anomali skoru = negatif log-olasılık
		log_density = torch.logsumexp(-0.5 * (pairwise_dist / self.bandwidth)**2, dim=1)
		return -log_density

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

# ### 5.4. Spectral Clustering Defense
#class SpectralDefense(Defense):
#	def detect(self, series: pd.Series) -> pd.Series:
#		data = StandardScaler().fit_transform(series.values.reshape(-1, 1))
#		# İki küme varsayıyoruz: normal ve anomali
#		model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=0)
#		labels = model.fit_predict(data)
#		# Daha küçük olan kümeyi anomali olarak etiketle
#		anomaly_cluster = 0 if np.sum(labels == 0) < np.sum(labels == 1) else 1
#		scores = np.zeros_like(labels)
#		scores[labels == anomaly_cluster] = 1
#		return pd.Series(scores, index=series.index)
#
#	def get_differentiable_proxy(self) -> base_defense.DifferentiableDefense:
#		# Spectral clustering'in türevlenebilir bir proxy'si karmaşıktır.
#		# Bu yüzden D-SPO'nun bu savunmayı hedeflemediğini varsayıyoruz.
#		# D-SPO'nun her savunmayı yenmek zorunda olmadığını, sadece birçoğunu
#		# yenebildiğini göstermek de bir bulgudur.
#		class NoOpProxy(base_defense.DifferentiableDefense):
#			def __call__(self, x: torch.Tensor) -> torch.Tensor:
#				return torch.zeros(x.shape[0], device=x.device)
#		return NoOpProxy()

class IsolationForestDefense(base_defense.Defense):
    def __init__(self, contamination='auto'):
        self.contamination = contamination

    def detect(self, series: pd.Series) -> pd.Series:
        # Düzeltme: Isolation Forest'ın çalışması için yeterli benzersiz değer olup olmadığını kontrol et
        if series.nunique() < 2:
            warnings.warn("IsolationForest için çok az sayıda benzersiz değer var. Boş anomali skorları döndürülüyor.", UserWarning)
            return pd.Series(0, index=series.index)
            
        model = IsolationForest(contamination=self.contamination, random_state=42)
        preds = model.fit_predict(series.values.reshape(-1, 1))
        # 1: outlier, 1: normal
        scores = pd.Series(-preds, index=series.index)
        return scores

class MedianFilterDefense(base_defense.Defense):
	def __init__(self, window=7):
		self.window = window

	def detect(self, series: pd.Series) -> pd.Series:
		median = series.rolling(window=self.window, center=True, min_periods=1).median()
		error = np.abs(series - median)
		return pd.Series(error, index=series.index)

	def get_differentiable_proxy(self):
		class NoOpProxy(base_defense.DifferentiableDefense):
			def __call__(self, x: torch.Tensor) -> torch.Tensor:
				return torch.zeros(x.shape[0], device=x.device)
		return NoOpProxy()

class QuantileDefense(base_defense.Defense):
	def __init__(self, lower=0.01, upper=0.99):
		self.lower = lower
		self.upper = upper

	def detect(self, series: pd.Series) -> pd.Series:
		lq = series.quantile(self.lower)
		uq = series.quantile(self.upper)
		mask = (series < lq) | (series > uq)
		return pd.Series(mask.astype(int), index=series.index)

	def get_differentiable_proxy(self):
		class NoOpProxy(base_defense.DifferentiableDefense):
			def __call__(self, x: torch.Tensor) -> torch.Tensor:
				return torch.zeros(x.shape[0], device=x.device)
		return NoOpProxy()

class DBSCANDefense(base_defense.Defense):
	def detect(self, series: pd.Series) -> pd.Series:
		X = series.values.reshape(-1, 1)
		model = DBSCAN(eps=0.5, min_samples=10)
		labels = model.fit_predict(X)
		# Label -1: outlier
		scores = (labels == -1).astype(int)
		return pd.Series(scores, index=series.index)
	def get_differentiable_proxy(self):
		class NoOpProxy(base_defense.DifferentiableDefense):
			def __call__(self, x: torch.Tensor) -> torch.Tensor:
				return torch.zeros(x.shape[0], device=x.device)
		return NoOpProxy()

def get_defense(name: str) -> base_defense.Defense:
	if name == "lof": return LOFDefense()
	if name == "pca": return PCADefense()
	if name == "kde": return KDEDefense()
	#if name == "spectral": return SpectralDefense()
	if name == 'forest' : return IsolationForestDefense()
	if name == "median" : return MedianFilterDefense()
	if name == 'quantile' : return QuantileDefense()
	if name == 'dbscan' : return DBSCANDefense()
	raise ValueError(f"Unknown defense: {name}")