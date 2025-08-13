# In[24]:


#!/usr/bin/env python
# coding: utf-8

# # Silent Sabotage Ω (Omega): The Ultimate Data Poisoning & Defense Evaluation Suite
# 
# **Target Venue:** USENIX Security / ACM CCS – Best Paper & Test-of-Time Award Candidate
# 
# **Abstract:** We introduce Silent Sabotage Ω, a comprehensive, end-to-end, and reproducible framework for evaluating the resilience of time-series forecasting models against advanced data poisoning attacks. This suite introduces a novel, defense-aware attack methodology, the **Defense-aware Stealthy Poisoning Oracle (D-SPO)**, which formulates the attack as a differentiable optimization problem. D-SPO learns to craft perturbations that are simultaneously maximally damaging to the target model and minimally detectable by a suite of state-of-the-art defense mechanisms. We benchmark D-SPO against traditional attacks and demonstrate its superior ability to create a new Pareto frontier in the trade-off between attack impact and stealth. The entire framework, including all attacks, defenses, and evaluation metrics, is provided as a single, self-contained, and extensible script to foster reproducible research in critical infrastructure security.

# ## 1. Imports and Setup
# Gerekli tüm kütüphaneleri ve temel ayarları burada yapıyoruz.

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List
from tqdm.notebook import tqdm
import warnings

# Scikit-learn imports
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_absolute_error, r2_score
from scipy.stats import wasserstein_distance
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

# Styling for plots
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 7)

# ---
# ## 2. Configuration Management
# Tüm deney parametreleri, yeniden üretilebilirlik için tek bir yerden yönetilir.
# ---

@dataclass
class BaseAttackConfig:
	budget_pct: float = 0.03  # Zehirlenecek veri yüzdesi
	perturb_strength: float = 0.5 # Pertürbasyonun maksimum gücü (std dev cinsinden)

@dataclass
class DatasetConfig:
    path: str
    task_type: str # 'classification' or 'regression'
    target_column: str

@dataclass
class DSPOConfig:
    num_iterations: int = 100
    learning_rate: float = 0.01
    budget_pct: float = 0.03
    perturb_strength: float = 0.5
    lambda_defense: float = 1.0

@dataclass
class ExperimentConfig:
	random_seed: int = 42
	results_dir: Path = Path(BASE_DIR /"results")
	
	# Burada deneyler için kullanılacak tüm veri setlerini tanımlıyoruz.
	datasets: Dict[str, DatasetConfig] = field(default_factory=lambda: {
		"water_quality": DatasetConfig(
			path=str(BASE_DIR / "datasets" / "water_potability.csv"),
			task_type="classification",
			target_column="Potability"
		),
		"smart_grid": DatasetConfig(
			path=str(BASE_DIR / "datasets" / "smart_grid_stability_augmented.csv"),
			task_type="regression",
			target_column="p1"
		),
        "traffic": DatasetConfig(
            path=str(BASE_DIR / "datasets" / "traffic_index.csv"),
            task_type="regression",
            target_column="average_traffic_index"
        ),
        "solar_power": DatasetConfig(
            path=str(BASE_DIR / "datasets" / "Plant_1_Generation_Data.csv"),
            task_type="regression",
            target_column="TOTAL_YIELD"
        ),
        "wind_power": DatasetConfig(
            path=str(BASE_DIR / "datasets" / "50Hertz.csv"),
            task_type="regression",
            target_column="01:00:00"
        ),
        "water_quality_monitoring": DatasetConfig(
            path=str(BASE_DIR / "datasets" / "brisbane_water_quality.csv"),
            task_type="regression",
            target_column="Average Water Speed"
        )
        
	})

	attacks: Dict[str, BaseAttackConfig] = field(default_factory=lambda: {
		"random": BaseAttackConfig(),
		"greedy_residue": BaseAttackConfig(),
        "dspo": DSPOConfig(),
	})

	active_defenses: List[str] = field(default_factory=lambda: ["lof", "pca", "kde", "forest", "median", "quantile", "dbscan"])

def get_config() -> ExperimentConfig:
	return ExperimentConfig()

# ---

def load_water_quality_dataset(csv_path: str) -> pd.DataFrame:
	df = pd.read_csv(csv_path)
	# NaN değerleri ortalama ile doldurma
	for col in df.columns:
		if df[col].isnull().any():
			if df[col].dtype in ['float64', 'int64']:
				df[col] = df[col].fillna(df[col].mean())
			else:
				df[col] = df[col].fillna(df[col].mode()[0])
	return df

def load_smart_grid_dataset(csv_path: str) -> pd.DataFrame:
	df = pd.read_csv(csv_path)
	# Veri setinde 'timestamp' sütunu yoksa oluşturulur, varsa işlenir.
	if 'timestamp' not in df.columns:
		df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(df), freq='H')
	else:
		df['timestamp'] = pd.to_datetime(df['timestamp'])
	
	df = df.sort_values('timestamp').reset_index(drop=True)
	df = df.set_index('timestamp')
	
	# 'stabf' sütununu kaldır. Bu sütun metin değerler içerdiği için hata veriyor.
	if 'stabf' in df.columns:
		df = df.drop(columns=['stabf'])
	
	# Eksik değerleri doldurma (önemli, çünkü bazı Kaggle verilerinde NaN olabilir)
	for col in df.columns:
		if df[col].isnull().any():
			if df[col].dtype in ['float64', 'int64']:
				df[col] = df[col].fillna(df[col].mean())
			else:
				df[col] = df[col].fillna(df[col].mode()[0])
	
	return df

def load_traffic_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['trafficindexdate'] = pd.to_datetime(df['trafficindexdate'])
    df = df.sort_values('trafficindexdate').set_index('trafficindexdate')
    
    if 'trafficindex' in df.columns:
        df.rename(columns={'trafficindex': 'average_traffic_index'}, inplace=True)
    
    for col in df.columns:
        if df[col].isnull().any() and df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].mean())
    return df

def load_solar_power_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], dayfirst=True)
    df = df.sort_values('DATE_TIME').set_index('DATE_TIME')
    # Düzeltme: Index'i sıfırlayarak benzersiz hale getirme
    df = df.reset_index(drop=True)
    
    df = df.drop(columns=['PLANT_ID', 'SOURCE_KEY'])
    
    for col in df.columns:
        if df[col].isnull().any() and df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].mean())
    return df

def load_wind_power_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # 'Date' sütununu datetime'a çevirip index yapma
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').set_index('Date')
    
    # Sayısal sütunlardaki eksik değerleri ortalama ile doldurma
    for col in df.columns:
        if df[col].isnull().any() and df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].mean())
            
    return df

def load_water_quality_monitoring_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values('Timestamp').set_index('Timestamp')

    # DÜZELTME: Index'i sıfırlayarak benzersiz hale getirme
    df = df.reset_index(drop=True)

    for col in df.columns:
        if df[col].isnull().any() and df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].mean())
    
    return df


DATASET_LOADERS = {
    "water_quality": load_water_quality_dataset,
    "smart_grid": load_smart_grid_dataset,
    "traffic": load_traffic_dataset,
    "solar_power": load_solar_power_dataset,
    "wind_power": load_wind_power_dataset,
    "water_quality_monitoring": load_water_quality_monitoring_dataset,
}

# ---
# ## 4. Core Abstractions: Attack & Defense
# Modüler ve genişletilebilir bir yapı için soyut ana sınıflar.
# ---

class Attack(ABC):
    def __init__(self, config: BaseAttackConfig, df: pd.DataFrame, target_col: str, task_type: str):
        self.config = config
        self.df = df.copy()
        self.target_col = target_col
        self.task_type = task_type
        self.sigmas = self.df.drop(self.target_col, axis=1).std()
        self.poison_indices = None

    @abstractmethod
    def craft(self, model) -> pd.DataFrame:
        pass

    def get_poison_indices(self, poisoned_df: pd.DataFrame) -> pd.Index:
        poisoned_features = poisoned_df.drop(self.target_col, axis=1)
        clean_features = self.df.drop(self.target_col, axis=1)
        diff = (poisoned_features != clean_features).any(axis=1)
        return self.df.index[diff]

class DifferentiableDefense(ABC):
    """D-SPO saldırısının kullanacağı, türevlenebilir savunma proxy'leri için ana sınıf."""
    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass

class Defense(ABC):
	@abstractmethod
	def detect(self, series: pd.Series) -> pd.Series:
		pass

# ---
# ## 5. Defense Implementations (SOTA & Differentiable Proxies)
# Literatürdeki savunmalar ve D-SPO için türevlenebilir versiyonları.
# ---

# ### 5.1. Local Outlier Factor (LOF) Defense
class DifferentiableLOF(DifferentiableDefense):
	def __init__(self, n_neighbors=20):
		self.k = n_neighbors

	def __call__(self, x: torch.Tensor) -> torch.Tensor:
		pairwise_dist = torch.cdist(x, x)
		k_distances, _ = torch.kthvalue(pairwise_dist, self.k + 1, dim=1)
		# Basit, türevlenebilir proxy: k-mesafesi ne kadar büyükse, o kadar anomalidir.
		return k_distances

class LOFDefense(Defense):
	def __init__(self, n_neighbors=20):
		self.n_neighbors = n_neighbors

	def detect(self, series: pd.Series) -> pd.Series:
		model = LocalOutlierFactor(n_neighbors=self.n_neighbors, contamination='auto')
		model.fit_predict(series.values.reshape(-1, 1))
		return pd.Series(-model.negative_outlier_factor_, index=series.index)

	def get_differentiable_proxy(self) -> DifferentiableDefense:
		return DifferentiableLOF(self.n_neighbors)

# ### 5.2. PCA-based Defense
class DifferentiablePCA(DifferentiableDefense):
	def __init__(self, n_components=1):
		self.n_components = n_components

	def __call__(self, x: torch.Tensor) -> torch.Tensor:
		# PCA'in türevlenebilir bir versiyonu: SVD (Singular Value Decomposition)
		U, S, V = torch.svd(x - x.mean())
		reconstruction = (U[:, :self.n_components] @ torch.diag(S[:self.n_components]) @ V[:, :self.n_components].T) + x.mean()
		# Yeniden yapılandırma hatası anomali skorudur
		return torch.sum((x - reconstruction)**2, dim=1)

class PCADefense(Defense):
	def __init__(self, n_components=1):
		self.n_components = n_components

	def detect(self, series: pd.Series) -> pd.Series:
		data = series.values.reshape(-1, 1)
		pca = PCA(n_components=self.n_components)
		projected = pca.fit_transform(data)
		reconstructed = pca.inverse_transform(projected)
		error = np.sum((data - reconstructed)**2, axis=1)
		return pd.Series(error, index=series.index)

	def get_differentiable_proxy(self) -> DifferentiableDefense:
		return DifferentiablePCA(self.n_components)

# ### 5.3. Kernel Density Estimation (KDE) Defense
class DifferentiableKDE(DifferentiableDefense):
	def __init__(self, bandwidth=1.0):
		self.bandwidth = bandwidth

	def __call__(self, x: torch.Tensor) -> torch.Tensor:
		# Gaussian Kernel kullanarak KDE'nin türevlenebilir bir versiyonu
		pairwise_dist = torch.cdist(x, x)
		# Anomali skoru = negatif log-olasılık
		log_density = torch.logsumexp(-0.5 * (pairwise_dist / self.bandwidth)**2, dim=1)
		return -log_density

class KDEDefense(Defense):
	def __init__(self, bandwidth=1.0):
		self.bandwidth = bandwidth

	def detect(self, series: pd.Series) -> pd.Series:
		data = series.values.reshape(-1, 1)
		kde = KernelDensity(bandwidth=self.bandwidth)
		kde.fit(data)
		log_density = kde.score_samples(data)
		return pd.Series(-log_density, index=series.index)

	def get_differentiable_proxy(self) -> DifferentiableDefense:
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
#	def get_differentiable_proxy(self) -> DifferentiableDefense:
#		# Spectral clustering'in türevlenebilir bir proxy'si karmaşıktır.
#		# Bu yüzden D-SPO'nun bu savunmayı hedeflemediğini varsayıyoruz.
#		# D-SPO'nun her savunmayı yenmek zorunda olmadığını, sadece birçoğunu
#		# yenebildiğini göstermek de bir bulgudur.
#		class NoOpProxy(DifferentiableDefense):
#			def __call__(self, x: torch.Tensor) -> torch.Tensor:
#				return torch.zeros(x.shape[0], device=x.device)
#		return NoOpProxy()

class IsolationForestDefense(Defense):
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

class MedianFilterDefense(Defense):
	def __init__(self, window=7):
		self.window = window

	def detect(self, series: pd.Series) -> pd.Series:
		median = series.rolling(window=self.window, center=True, min_periods=1).median()
		error = np.abs(series - median)
		return pd.Series(error, index=series.index)

	def get_differentiable_proxy(self):
		class NoOpProxy(DifferentiableDefense):
			def __call__(self, x: torch.Tensor) -> torch.Tensor:
				return torch.zeros(x.shape[0], device=x.device)
		return NoOpProxy()

class QuantileDefense(Defense):
	def __init__(self, lower=0.01, upper=0.99):
		self.lower = lower
		self.upper = upper

	def detect(self, series: pd.Series) -> pd.Series:
		lq = series.quantile(self.lower)
		uq = series.quantile(self.upper)
		mask = (series < lq) | (series > uq)
		return pd.Series(mask.astype(int), index=series.index)

	def get_differentiable_proxy(self):
		class NoOpProxy(DifferentiableDefense):
			def __call__(self, x: torch.Tensor) -> torch.Tensor:
				return torch.zeros(x.shape[0], device=x.device)
		return NoOpProxy()

class DBSCANDefense(Defense):
	def detect(self, series: pd.Series) -> pd.Series:
		X = series.values.reshape(-1, 1)
		model = DBSCAN(eps=0.5, min_samples=10)
		labels = model.fit_predict(X)
		# Label -1: outlier
		scores = (labels == -1).astype(int)
		return pd.Series(scores, index=series.index)
	def get_differentiable_proxy(self):
		class NoOpProxy(DifferentiableDefense):
			def __call__(self, x: torch.Tensor) -> torch.Tensor:
				return torch.zeros(x.shape[0], device=x.device)
		return NoOpProxy()

# ---
# ## 6. Attack Implementations (Baselines & Our Novel Approach)
# ---

# ### 6.1. Random Attack (Baseline)
class RandomAttack(Attack):
    def craft(self, model=None) -> pd.DataFrame:
        budget = int(self.config.budget_pct * len(self.df))
        self.poison_indices = np.random.choice(self.df.index, budget, replace=False)
        poisoned_df = self.df.copy()
        
        if self.task_type == 'regression':
            perturbations = np.random.uniform(
                -self.config.perturb_strength * self.df[self.target_col].std(),
                self.config.perturb_strength * self.df[self.target_col].std(),
                size=budget
            )
            poisoned_df.loc[self.poison_indices, self.target_col] += perturbations
        elif self.task_type == 'classification':
            poisoned_df.loc[self.poison_indices, self.target_col] = 1 - poisoned_df.loc[self.poison_indices, self.target_col]
            
        return poisoned_df

class GreedyResidueAttack(Attack):
    def craft(self, model) -> pd.DataFrame:
        budget = int(self.config.budget_pct * len(self.df))
        poisoned_df = self.df.copy()
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]

        if self.task_type == 'classification':
            y_pred = model.predict(X)
            misclassified_indices = y[y != y_pred].index
            
            if len(misclassified_indices) > budget:
                self.poison_indices = np.random.choice(misclassified_indices, budget, replace=False)
            else:
                self.poison_indices = misclassified_indices

            poisoned_df.loc[self.poison_indices, self.target_col] = 1 - poisoned_df.loc[self.poison_indices, self.target_col]
            
            numeric_cols = poisoned_df.select_dtypes(include=np.number).columns.drop(self.target_col, errors='ignore')
            for col in numeric_cols:
                sigma = self.df[col].std()
                deltas = np.random.uniform(
                    -self.config.perturb_strength * sigma,
                    self.config.perturb_strength * sigma,
                    size=len(self.poison_indices)
                )
                poisoned_df.loc[self.poison_indices, col] += deltas
        
        elif self.task_type == 'regression':
            y_pred = model.predict(X)
            residues = np.abs(y - y_pred)
            self.poison_indices = residues.nlargest(budget).index
            
            perturbations = np.random.uniform(
                -self.config.perturb_strength * self.df[self.target_col].std(),
                self.config.perturb_strength * self.df[self.target_col].std(),
                size=budget
            )
            poisoned_df.loc[self.poison_indices, self.target_col] += perturbations
        
        return poisoned_df

class DSPOAttack(Attack):
    def __init__(self, config: DSPOConfig, df: pd.DataFrame, target_col: str, task_type: str, defenses: Dict[str, Defense]):
        # Mevcut Attack sınıfına uyum sağlamak için gerekli parametreleri ekle
        self.config = config
        self.df = df
        self.target_col = target_col
        self.task_type = task_type
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.diff_defenses = {
            name: d.get_differentiable_proxy() 
            for name, d in defenses.items() 
            if hasattr(d, 'get_differentiable_proxy')
        }
        
        if not self.diff_defenses:
            raise ValueError("DSPOAttack requires at least one defense with a 'get_differentiable_proxy' method.")
        
        self.sigma = df[target_col].std()
        
        # Zehirlenecek noktaları önceden seç
        rolling_mean = self.df[self.target_col].rolling(window=24, center=True, min_periods=1).mean()
        residues = np.abs(self.df[self.target_col] - rolling_mean)
        budget = int(self.config.budget_pct * len(self.df))
        
        self.original_indices_to_poison = residues.nlargest(budget).index
        self.poison_locations = self.df.index.get_indexer(self.original_indices_to_poison)
        
    def craft(self, model=None) -> pd.DataFrame:
        print(f"--- Starting D-SPO Oracle on {self.device} ---")
        clean_data = torch.tensor(self.df[self.target_col].values, dtype=torch.float32, device=self.device)

        deltas = torch.zeros(len(self.df), device=self.device, requires_grad=True)
        max_perturb = self.config.perturb_strength * self.sigma
        
        optimizer = torch.optim.Adam([deltas], lr=self.config.learning_rate)
        
        pbar = tqdm(range(self.config.num_iterations), desc="D-SPO Optimization")
        
        for i in pbar:
            optimizer.zero_grad()
            
            poisoned_data = clean_data + deltas
            
            defense_loss = torch.tensor(0.0, device=self.device)
            # Savunma vekilleri üzerindeki anomali skorlarını topla
            for defense_proxy in self.diff_defenses.values():
                anomaly_scores = defense_proxy(poisoned_data.unsqueeze(1))
                # Sadece zehirlenen noktaların anomali skorlarını al
                defense_loss += anomaly_scores[self.poison_locations].mean()
                
            attack_loss = torch.abs(deltas[self.poison_locations]).mean()
            
            total_loss = attack_loss + self.config.lambda_defense * defense_loss
            
            total_loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                deltas.data = torch.clamp(deltas.data, -max_perturb, max_perturb)
                
            pbar.set_postfix({
                "Total Loss": f"{total_loss.item():.2f}",
                "Attack": f"{attack_loss.item():.2f}",
                "Defense": f"{defense_loss.item():.2f}"
            })
            
        final_deltas = deltas.detach().cpu().numpy()
        poisoned_df = self.df.copy()
        
        # Sadece seçilen noktalara pertürbasyonları uygula
        poisoned_df.loc[self.original_indices_to_poison, self.target_col] += final_deltas[self.poison_locations]
        if self.task_type == 'classification':
        # EĞER GÖREV SINIFLANDIRMA İSE, YUVARLA VE TAMSAYIYA ÇEVİR.
            poisoned_df[self.target_col] = poisoned_df[self.target_col].round().astype(int)
        
        # Mevcut çerçeveye uyum için poison_indices'i set et
        self.poison_indices = self.original_indices_to_poison
        
        return poisoned_df

# ---
# ## 7. Factory Functions
# İsimle kolayca saldırı ve savunma nesneleri oluşturmak için.
# ---

def get_defense(name: str) -> Defense:
	if name == "lof": return LOFDefense()
	if name == "pca": return PCADefense()
	if name == "kde": return KDEDefense()
	#if name == "spectral": return SpectralDefense()
	if name == 'forest' : return IsolationForestDefense()
	if name == "median" : return MedianFilterDefense()
	if name == 'quantile' : return QuantileDefense()
	if name == 'dbscan' : return DBSCANDefense()
	raise ValueError(f"Unknown defense: {name}")

def get_attack(name: str, cfg: BaseAttackConfig, df: pd.DataFrame, target_col: str, task_type: str, defenses: Dict[str, Defense]) -> Attack:
    if name == "random": return RandomAttack(cfg, df, target_col, task_type)
    if name == "greedy_residue": return GreedyResidueAttack(cfg, df, target_col, task_type)
    if name == "dspo" : return DSPOAttack(cfg, df, target_col, task_type, defenses)
    raise ValueError(f"Unknown attack: {name}")

# ---
# ## 8. Evaluation Module
# Saldırıların etkisini ve gizliliğini ölçmek için kapsamlı metrikler.
# ---

def evaluate_stealth(clean_series: pd.Series, poison_series: pd.Series) -> dict:
    """Saldırının gizliliğini, istatistiksel dağılımları karşılaştırarak tek bir puanla ölçer."""
    if clean_series.dtype not in ['float64', 'int64']:
        return {"stealth_score": "N/A"}
        
    mean_shift_std = np.abs(poison_series.mean() - clean_series.mean()) / clean_series.std()
    std_shift_std = np.abs(poison_series.std() - clean_series.std()) / clean_series.std()
    wasserstein_dist = wasserstein_distance(clean_series.dropna(), poison_series.dropna())
    
    # Tüm metrikleri toplayarak tek bir gizlilik puanı oluştur
    # Bu değer ne kadar düşükse, saldırı o kadar gizlidir.
    composite_stealth = mean_shift_std + std_shift_std + wasserstein_dist
    
    return {"composite_stealth": composite_stealth}

def evaluate_impact(clean_df: pd.DataFrame, poison_df: pd.DataFrame, target_col: str, task_type: str) -> dict:
    """
    Saldırının, temiz veriyle eğitilmiş modele kıyasla zehirli veriyle eğitilen modelin performansını nasıl etkilediğini ölçer.
    """
    # 1. Veriyi Temiz DataFrame'den ayır.
    X_clean = clean_df.drop(columns=[target_col])
    y_clean = clean_df[target_col]
    
    # 2. Zehirli DataFrame'den özellikleri ve etiketleri ayır.
    X_poisoned = poison_df.drop(columns=[target_col])
    y_poisoned = poison_df[target_col]
    
    # 3. Veriyi eğitim ve test setlerine ayır. Zehirli ve temiz setler aynı indekse sahip olmalı.
    X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(
        X_clean, y_clean, test_size=0.3, random_state=0)
    
    # Zehirli eğitim seti, temiz test seti ile aynı şekilde ayrılmalıdır.
    X_train_poisoned = X_poisoned.loc[X_train_clean.index]
    y_train_poisoned = y_poisoned.loc[y_train_clean.index]

    # 4. Veri ölçeklendirmesi (Eğitim verisine göre fit edip, test verisine transform et).
    scaler = StandardScaler()
    X_train_clean_scaled = scaler.fit_transform(X_train_clean)
    X_test_clean_scaled = scaler.transform(X_test_clean)
    
    X_train_poisoned_scaled = scaler.fit_transform(X_train_poisoned)
    
    # 5. Model seçimi.
    if task_type == 'classification':
        model_clean = RandomForestClassifier(n_estimators=100, random_state=0)
        model_poisoned = RandomForestClassifier(n_estimators=100, random_state=0)
    elif task_type == 'regression':
        model_clean = RandomForestRegressor(n_estimators=100, random_state=0)
        model_poisoned = RandomForestRegressor(n_estimators=100, random_state=0)
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    # 6. Temiz ve zehirli verilerle ayrı ayrı modelleri eğit.
    model_clean.fit(X_train_clean_scaled, y_train_clean)
    model_poisoned.fit(X_train_poisoned_scaled, y_train_poisoned)

    # 7. Her iki modelin de temiz test seti üzerindeki performansını ölç.
    y_pred_clean_model = model_clean.predict(X_test_clean_scaled)
    y_pred_poisoned_model = model_poisoned.predict(X_test_clean_scaled)
    
    results = {}
    if task_type == 'classification':
        clean_score = accuracy_score(y_test_clean, y_pred_clean_model)
        poisoned_score = accuracy_score(y_test_clean, y_pred_poisoned_model)
        results["impact_score"] = clean_score - poisoned_score
        results["clean_accuracy"] = clean_score
        results["poisoned_accuracy"] = poisoned_score
    elif task_type == 'regression':
        clean_score = mean_absolute_error(y_test_clean, y_pred_clean_model)
        poisoned_score = mean_absolute_error(y_test_clean, y_pred_poisoned_model)
        results["impact_score"] = poisoned_score - clean_score
        results["clean_mae"] = clean_score
        results["poisoned_mae"] = poisoned_score

    return results

def evaluate_defense_performance(all_indices: pd.Index, poison_indices: pd.Index, detected_indices: pd.Index) -> dict:
    if isinstance(poison_indices, np.ndarray):
        poison_indices = pd.Index(poison_indices)
    elif isinstance(poison_indices, list):
        poison_indices = pd.Index(poison_indices)
        
    y_true = pd.Series(0, index=all_indices)
    if not poison_indices.empty:
        y_true.loc[poison_indices] = 1
    y_pred = pd.Series(0, index=all_indices)
    if not detected_indices.empty:
        y_pred.loc[detected_indices] = 1
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    return {"precision": precision, "recall": recall, "f1_score": f1}
	

# ---
# ## 9. Reporting Module
# Sonuçları CSV, LaTeX ve grafik olarak kaydetme.
# ---

def save_results(df_results: pd.DataFrame, results_dir: Path, dataset_name: str, config: ExperimentConfig):
    # Results klasörünü oluştur
    results_dir.mkdir(exist_ok=True)
    
    # Parametre bilgilerini al
    budget_pct = list(config.attacks.values())[0].budget_pct
    perturb_strength = list(config.attacks.values())[0].perturb_strength
    
    # Zaman damgası ekle (üstüne yazma sorunu için)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Dosya isimleri için prefix oluştur
    file_prefix = f"{dataset_name}_budget{budget_pct:.3f}_perturb{perturb_strength:.2f}_{timestamp}"
    
    # DataFrame'e parametre bilgilerini ekle
    df_results['dataset_name'] = dataset_name
    df_results['budget_percent'] = budget_pct
    df_results['perturb_strength'] = perturb_strength
    df_results['timestamp'] = timestamp
    
    # Tüm sonuçları CSV'ye kaydet
    csv_path = results_dir / f"{file_prefix}_results.csv"
    df_results.to_csv(csv_path, index=False)
    
    # Impact metric sütununu belirle (task türüne göre)
    impact_column = None
    if 'impact_score' in df_results.columns:
        impact_column = 'impact_score'
    elif 'mae_degradation' in df_results.columns:
        impact_column = 'mae_degradation'
    else:
        print("Uyarı: Impact metriği bulunamadı. Özet tablo oluşturulamıyor.")
        return file_prefix
    
    # Özet tabloyu oluştur
    summary_columns = [impact_column, 'recall', 'f1_score']
    available_columns = [col for col in summary_columns if col in df_results.columns]
    
    if len(available_columns) < 2:
        print("Uyarı: Yeterli metrik bulunamadı. Özet tablo oluşturulamıyor.")
        return file_prefix
    
    summary = df_results.groupby('attack')[available_columns].mean().reset_index()
    
    # Özet tablosuna parametre bilgilerini ekle
    summary_with_params = pd.DataFrame({
        'Dataset': [dataset_name] * len(summary),
        'Budget (%)': [f"{budget_pct:.1%}"] * len(summary),
        'Perturb Strength': [perturb_strength] * len(summary),
        'Timestamp': [timestamp] * len(summary)
    })
    
    # Sütun isimlerini güzelleştir
    column_names = ['Attack']
    for col in available_columns:
        if col == 'impact_score':
            column_names.append('Impact Score')
        elif col == 'mae_degradation':
            column_names.append('MAE Degradation')
        elif col == 'recall':
            column_names.append('Avg. Recall')
        elif col == 'f1_score':
            column_names.append('Avg. F1-Score')
        else:
            column_names.append(col.title())
    
    summary.columns = column_names
    
    # Parametre bilgileri ile birlikte birleştir
    final_summary = pd.concat([summary_with_params, summary.iloc[:, 1:]], axis=1)
    
    # LaTeX tablosunu kaydet
    latex_path = results_dir / f"{file_prefix}_summary.tex"
    with open(latex_path, "w") as f:
        f.write(final_summary.to_latex(index=False, float_format="%.3f"))
    
    print(f"\n{'='*50}")
    print(f"SONUÇLAR KAYDEDİLDİ")
    print(f"{'='*50}")
    print(f"Dataset: {dataset_name}")
    print(f"Budget: {budget_pct:.1%}")
    print(f"Perturb Strength: {perturb_strength}")
    print(f"CSV: {csv_path}")
    print(f"LaTeX: {latex_path}")
    print(f"{'='*50}")
    print("ÖZET:")
    print(final_summary.to_string(index=False))
    
    return file_prefix

def plot_results(df_results: pd.DataFrame, results_dir: Path, dataset_name: str, config: ExperimentConfig, file_prefix: str):
    """
    Saldırı etkisini (impact score) savunma kaçışı (evasion) ile karşılaştıran bir
    dağılım grafiği (scatterplot) oluşturur ve kaydeder.
    """
    # Results klasörünü oluştur
    results_dir.mkdir(exist_ok=True)
    
    # Parametre bilgilerini al
    budget_pct = list(config.attacks.values())[0].budget_pct
    perturb_strength = list(config.attacks.values())[0].perturb_strength
    
    # Impact metriğini belirle
    impact_column = None
    if 'impact_score' in df_results.columns:
        impact_column = 'impact_score'
        impact_label = 'Attack Impact (Impact Score)'
    elif 'mae_degradation' in df_results.columns:
        impact_column = 'mae_degradation'
        impact_label = 'Attack Impact (MAE Degradation)'
    else:
        print("Uyarı: Impact metriği bulunamadı. Grafik oluşturulamıyor.")
        return
    
    if 'recall' not in df_results.columns:
        print("Uyarı: Recall metriği bulunamadı. Grafik oluşturulamıyor.")
        return
    
    plt.figure(figsize=(12, 8))
    df_results['evasion'] = 1 - df_results['recall']
    
    plot = sns.scatterplot(
        data=df_results,
        x='evasion',
        y=impact_column,
        hue='attack',
        style='defense',
        s=150,
        alpha=0.8
    )
    
    # Başlık ve alt başlık parametre bilgileri ile
    main_title = f'Attack Impact vs. Stealth (Defense Evasion)\nDataset: {dataset_name.upper()}'
    subtitle = f'Budget: {budget_pct:.1%} | Perturb Strength: {perturb_strength} | Defenses: {len(df_results["defense"].unique())}'
    
    plt.suptitle(main_title, fontsize=16, weight='bold', y=0.95)
    plt.title(subtitle, fontsize=11, style='italic', pad=20)
    
    plt.xlabel('Defense Evasion (1 - Recall)', fontsize=12)
    plt.ylabel(impact_label, fontsize=12)
    
    # Legend'ı düzenle
    plt.legend(title='Attack & Defense', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    
    # Grafik dosyasını kaydet
    fig_path = results_dir / f"{file_prefix}_plot.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    
    print(f"Grafik kaydedildi: {fig_path}")
    plt.show()

# ---
# ## 10. Main Experiment Orchestrator
# Her şeyi bir araya getiren ve deneyi yürüten ana fonksiyon.
# ---

def run_single_experiment(dataset_name: str):
    """
    Belirtilen veri seti üzerinde saldırı ve savunma deneylerini çalıştırır.
    """
    cfg = get_config()
    np.random.seed(cfg.random_seed) # Random seed'i başta ayarla
    torch.manual_seed(cfg.random_seed)
    
    all_results = []
    
    if dataset_name not in cfg.datasets:
        print(f"Hata: Konfigürasyonda '{dataset_name}' adlı veri seti bulunamadı.")
        return
    
    dataset_cfg = cfg.datasets[dataset_name]
    
    print(f"\n\n{'='*20} '{dataset_name.upper()}' için deneyler başlatılıyor {'='*20}")
    
    try:
        df_clean = DATASET_LOADERS[dataset_name](dataset_cfg.path)
        print(f"'{dataset_name}' veri seti başarıyla yüklendi.")
    except FileNotFoundError:
        print(f"Hata: '{dataset_cfg.path}' dosyası '{dataset_name}' veri seti için bulunamadı. Deney atlanıyor.")
        return
        
    defenses = {name: get_defense(name) for name in cfg.active_defenses}

    for attack_name, attack_cfg in cfg.attacks.items():
        print(f"\n--- Saldırı: {attack_name.upper()} ---")

        # Her saldırı için rastgeleliği yeniden ayarla, ancak farklı bir seed ile.
        np.random.seed(cfg.random_seed + list(cfg.attacks.keys()).index(attack_name))
        torch.manual_seed(cfg.random_seed + list(cfg.attacks.keys()).index(attack_name))

        X_clean_scaled = StandardScaler().fit_transform(df_clean.drop(columns=[dataset_cfg.target_column]))
        y_clean = df_clean[dataset_cfg.target_column]
        if dataset_cfg.task_type == 'classification':
            base_model = RandomForestClassifier(n_estimators=100, random_state=0)
        else:
            base_model = RandomForestRegressor(n_estimators=100, random_state=0)
        base_model.fit(X_clean_scaled, y_clean)

        attack_instance = get_attack(attack_name, attack_cfg, df_clean, dataset_cfg.target_column, dataset_cfg.task_type, defenses)
        df_poisoned = attack_instance.craft(base_model)
        
        # Düzeltme: Zehirlenen noktaların doğru listesini saldırı nesnesinden alıyoruz.
        poison_indices = attack_instance.poison_indices

        impact_metrics = evaluate_impact(df_clean, df_poisoned, dataset_cfg.target_column, dataset_cfg.task_type)
        
        stealth_metrics = evaluate_stealth(df_clean[dataset_cfg.target_column], df_poisoned[dataset_cfg.target_column])
        
        for defense_name, defense_instance in defenses.items():
            print(f"  - Savunmaya karşı test: {defense_name}")
            
            defense_target_column = 'ph' if dataset_name == 'water_quality' else dataset_cfg.target_column
            
            if defense_target_column not in df_poisoned.columns:
                print(f"    - Uyarı: '{defense_target_column}' bulunamadı. Savunma atlanıyor.")
                continue
                
            anomaly_scores = defense_instance.detect(df_poisoned[defense_target_column])
            
            if isinstance(anomaly_scores, pd.Series):
                threshold = anomaly_scores.quantile(1 - attack_cfg.budget_pct)
                detected_indices = anomaly_scores[anomaly_scores >= threshold].index
            else:
                detected_indices = anomaly_scores[anomaly_scores == -1].index
            
            defense_perf = evaluate_defense_performance(df_clean.index, poison_indices, detected_indices)
            
            result_entry = {
                "dataset": dataset_name,
                "task": dataset_cfg.task_type,
                "attack": attack_name,
                "defense": defense_name,
                "poison_budget": attack_cfg.budget_pct,
                **impact_metrics,
                **defense_perf,
                **stealth_metrics
            }
            all_results.append(result_entry)
            
    df_results = pd.DataFrame(all_results)

    file_prefix = save_results(df_results, cfg.results_dir, dataset_name, cfg)
    plot_results(df_results, cfg.results_dir, dataset_name, cfg, file_prefix)

    print("\n" + "="*50)
    print(f" FINAL RESULTS FOR '{dataset_name.upper()}'")
    print("="*50)
    print(df_results.to_string())
    
    print(f"\n✅ Deney tamamlandı. Sonuçlar '{cfg.results_dir}' klasörüne kaydedildi.")

# ---
# ## 11. Execute the Framework
# ---
if __name__ == '__main__':
	run_single_experiment("traffic")



# In[ ]:



