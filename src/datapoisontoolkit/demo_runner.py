import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from datapoisontoolkit.detectors import stat_detectors
from datapoisontoolkit.attacks import base_attack
from datapoisontoolkit.detectors import model_detectors

BASE_DIR = Path(__file__).resolve().parent

# Styling for plots
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 7)


@dataclass
class DatasetConfig:
    path: str
    task_type: str # 'classification' or 'regression'
    target_column: str

@dataclass
class ExperimentConfig:
	random_seed: int = 42
	results_dir: Path = Path(BASE_DIR.parent.parent /"results")
	
	# Burada deneyler için kullanılacak tüm veri setlerini tanımlıyoruz.
	datasets: Dict[str, DatasetConfig] = field(default_factory=lambda: {
		"water_quality": DatasetConfig(
			path=str(BASE_DIR.parent / "data" / "water_potability.csv"),
			task_type="classification",
			target_column="Potability"
		),
		"smart_grid": DatasetConfig(
			path=str(BASE_DIR.parent / "data" / "smart_grid_stability_augmented.csv"),
			task_type="regression",
			target_column="p1"
		),
        "traffic": DatasetConfig(
            path=str(BASE_DIR.parent / "data" / "traffic_index.csv"),
            task_type="regression",
            target_column="average_traffic_index"
        ),
        "solar_power": DatasetConfig(
            path=str(BASE_DIR.parent / "data" / "Plant_1_Generation_Data.csv"),
            task_type="regression",
            target_column="TOTAL_YIELD"
        ),
        "wind_power": DatasetConfig(
            path=str(BASE_DIR.parent / "data" / "50Hertz.csv"),
            task_type="regression",
            target_column="01:00:00"
        ),
        "water_quality_monitoring": DatasetConfig(
            path=str(BASE_DIR.parent / "data" / "brisbane_water_quality.csv"),
            task_type="regression",
            target_column="Average Water Speed"
        )
        
	})

	attacks: Dict[str, base_attack.BaseAttackConfig] = field(default_factory=lambda: {
		"random": base_attack.BaseAttackConfig(),
		"greedy_residue": base_attack.BaseAttackConfig(),
        "dspo": base_attack.DSPOConfig(),
	})

	active_defenses: List[str] = field(default_factory=lambda: ["lof", "pca", "kde", "forest", "median", "quantile", "dbscan"])



def get_config() -> ExperimentConfig:
	return ExperimentConfig()

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
    
    # Timestamp sütunu varsa işle, yoksa oluştur - SONRA KALDIR
    if 'timestamp' not in df.columns:
        df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(df), freq='H')
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 'stabf' sütununu kaldır (metin değerler içerdiği için)
    if 'stabf' in df.columns:
        df = df.drop(columns=['stabf'])
    
    # ÖNEMLİ: Timestamp sütununu tamamen kaldır (sklearn ile uyumsuz)
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])
    
    # Sayısal olmayan sütunları kaldır
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = df[numeric_cols]
    
    # Eksik değerleri doldur
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    
    print(f"Dataset loaded with {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types: {df.dtypes.to_dict()}")
    
    return df

def load_traffic_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load and preprocess traffic dataset for machine learning.
    Returns DataFrame with integer index to avoid datetime index issues.
    """
    df = pd.read_csv(csv_path)
    
    # Process datetime column but don't set it as index
    if 'trafficindexdate' in df.columns:
        df['trafficindexdate'] = pd.to_datetime(df['trafficindexdate'])
        # Sort by date
        df = df.sort_values('trafficindexdate').reset_index(drop=True)
        # Remove datetime column after sorting (causes issues with sklearn)
        df = df.drop(columns=['trafficindexdate'])
    
    # Rename target column if needed
    if 'trafficindex' in df.columns:
        df.rename(columns={'trafficindex': 'average_traffic_index'}, inplace=True)
    
    # Handle missing values only for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())
    
    # Remove any non-numeric columns that might cause issues
    df = df[numeric_cols]
    
    print(f"Traffic dataset loaded with {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types: {df.dtypes.to_dict()}")
    
    return df

def load_solar_power_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    
    # DateTime sütununu sıralama için kullan ama sonra kaldır (DSPO integer index gerektirir)
    if 'DATE_TIME' in df.columns:
        df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], dayfirst=True)
        df = df.sort_values('DATE_TIME')
        # DateTime sütununu kaldır çünkü DSPO sadece sayısal verilerle çalışır
        df = df.drop(columns=['DATE_TIME'])
    
    # Gereksiz sütunları kaldır
    columns_to_drop = ['PLANT_ID', 'SOURCE_KEY']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    # Sayısal sütunlardaki eksik değerleri ortalama ile doldur
    for col in df.columns:
        if df[col].isnull().any() and df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].mean())
    
    # Index'i sıfırla (DSPO integer RangeIndex gerektirir)
    df = df.reset_index(drop=True)
    
    return df

def load_wind_power_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    
    # 'Date' sütunu varsa datetime'a çevirip kaldır (DSPO integer index gerektirir)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        # Date sütununu kaldır çünkü DSPO sadece sayısal verilerle çalışır
        df = df.drop(columns=['Date'])
    
    # Sayısal sütunlardaki eksik değerleri ortalama ile doldurma
    for col in df.columns:
        if df[col].isnull().any() and df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].mean())
    
    # Index'i sıfırla (DSPO integer RangeIndex gerektirir)
    df = df.reset_index(drop=True)
            
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
        
    defenses = {name: model_detectors.get_defense(name) for name in cfg.active_defenses}

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

        attack_instance = base_attack.get_attack(attack_name, attack_cfg, df_clean, dataset_cfg.target_column, dataset_cfg.task_type, defenses)
        df_poisoned = attack_instance.craft(base_model)
        
        # Düzeltme: Zehirlenen noktaların doğru listesini saldırı nesnesinden alıyoruz.
        poison_indices = attack_instance.poison_indices

        impact_metrics = stat_detectors.evaluate_impact(df_clean, df_poisoned, dataset_cfg.target_column, dataset_cfg.task_type)
        
        stealth_metrics = stat_detectors.evaluate_stealth(df_clean[dataset_cfg.target_column], df_poisoned[dataset_cfg.target_column])
        
        for defense_name, defense_instance in defenses.items():
            print(f"  - Savunmaya karşı test: {defense_name}")
            
            defense_target_column = dataset_cfg.target_column
            
            if defense_target_column not in df_poisoned.columns:
                print(f"    - Uyarı: '{defense_target_column}' bulunamadı. Savunma atlanıyor.")
                continue
                
            anomaly_scores = defense_instance.detect(df_poisoned[defense_target_column])
            
            if isinstance(anomaly_scores, pd.Series):
                threshold = anomaly_scores.quantile(1 - attack_cfg.budget_pct)
                detected_indices = anomaly_scores[anomaly_scores >= threshold].index
            else:
                detected_indices = anomaly_scores[anomaly_scores == -1].index
            
            defense_perf = stat_detectors.evaluate_defense_performance(df_clean.index, poison_indices, detected_indices)
            
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
    
    print(f"\n[OK] Deney tamamlandi. Sonuclar '{cfg.results_dir}' klasorune kaydedildi.")

if __name__ == '__main__':
	run_single_experiment("solar_power")
