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
	
	# Here we define all datasets to be used for the experiments
	datasets: Dict[str, DatasetConfig] = field(default_factory=lambda: {
		"water_quality": DatasetConfig(
			path=str(BASE_DIR.parent / "data" / "water_potability.csv"),
			task_type="regression",
			target_column="Hardness"
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
	# Fill NaN values with the mean
	for col in df.columns:
		if df[col].isnull().any():
			if df[col].dtype in ['float64', 'int64']:
				df[col] = df[col].fillna(df[col].mean())
			else:
				df[col] = df[col].fillna(df[col].mode()[0])
	return df

def load_smart_grid_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    
    # If a Timestamp column exists, process it; otherwise, create one - REMOVE LATER
    if 'timestamp' not in df.columns:
        df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(df), freq='H')
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Drop the 'stabf' column (contains text values)
    if 'stabf' in df.columns:
        df = df.drop(columns=['stabf'])
    
    # IMPORTANT: Completely remove the Timestamp column (incompatible with sklearn)
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])
    
    # Remove non-numerical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = df[numeric_cols]
    
    # Fill missing values
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
    
    # Use the DateTime column for sorting, then drop it (DSPO requires integer index)
    if 'DATE_TIME' in df.columns:
        df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], dayfirst=True)
        df = df.sort_values('DATE_TIME')
        # Drop the DateTime column because DSPO works only with numerical data
        df = df.drop(columns=['DATE_TIME'])
    
    # Drop unnecessary columns
    columns_to_drop = ['PLANT_ID', 'SOURCE_KEY']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    # Fill missing values in numerical columns with the mean
    for col in df.columns:
        if df[col].isnull().any() and df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].mean())
    
    # Reset the index (DSPO requires an integer RangeIndex)
    df = df.reset_index(drop=True)
    
    return df

def load_wind_power_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    
    # If a 'Date' column exists, convert it to datetime and drop it (DSPO requires integer index)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        # Drop the Date column because DSPO works only with numerical data
        df = df.drop(columns=['Date'])
    
    
    # Fill missing values with the mean
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
            # Check for NaN values
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())
            
            # Check and clean infinite values
            if np.isinf(df[col]).any():
                print(f"Warning: Infinite values found in {col}, replacing with median")
                finite_values = df[col][np.isfinite(df[col])]
                if len(finite_values) > 0:
                    median_val = finite_values.median()
                    df[col] = df[col].replace([np.inf, -np.inf], median_val)
                else:
                    df[col] = df[col].replace([np.inf, -np.inf], 0)
            
            # Check for extremely large values (outlier clipping)
            q99 = df[col].quantile(0.99)
            q01 = df[col].quantile(0.01)
            if q99 > 1e6 or q01 < -1e6:
                print(f"Warning: Extreme values in {col}, clipping to reasonable range")
                df[col] = df[col].clip(lower=q01, upper=q99)
    
    # Reset the index (DSPO requires an integer RangeIndex)
    df = df.reset_index(drop=True)
    
    # Final check: Ensure all numerical values are finite
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if not np.isfinite(df[col]).all():
            print(f"Warning: Still non-finite values in {col} after cleaning")
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    return df

def load_water_quality_monitoring_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values('Timestamp').set_index('Timestamp')

    # Ensure uniqueness by resetting the index
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
    # Create the Results folder
    results_dir.mkdir(exist_ok=True)
    
    # Retrieve parameter information
    budget_pct = list(config.attacks.values())[0].budget_pct
    perturb_strength = list(config.attacks.values())[0].perturb_strength
    
    # Add a timestamp (to avoid overwrite issues)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a prefix for file names
    file_prefix = f"{dataset_name}_budget{budget_pct:.3f}_perturb{perturb_strength:.2f}_{timestamp}"
    
    # Add parameter information to the DataFrame
    df_results['dataset_name'] = dataset_name
    df_results['budget_percent'] = budget_pct
    df_results['perturb_strength'] = perturb_strength
    df_results['timestamp'] = timestamp
    
    # Save all results to CSV
    csv_path = results_dir / f"{file_prefix}_results.csv"
    df_results.to_csv(csv_path, index=False)
    
    # Determine the Impact metric column (depending on the task type)
    impact_column = None
    if 'impact_score' in df_results.columns:
        impact_column = 'impact_score'
    elif 'mae_degradation' in df_results.columns:
        impact_column = 'mae_degradation'
    else:
        print("Warning: Impact metric not found. Summary table cannot be created.")
        return file_prefix
    
    # Create a summary table
    summary_columns = [impact_column, 'recall', 'f1_score']
    available_columns = [col for col in summary_columns if col in df_results.columns]
    
    if len(available_columns) < 2:
        print("Warning: Not enough metrics available. Summary table cannot be created.")
        return file_prefix
    
    summary = df_results.groupby('attack')[available_columns].mean().reset_index()
    
    # Add parameter information to the summary table
    summary_with_params = pd.DataFrame({
        'Dataset': [dataset_name] * len(summary),
        'Budget (%)': [f"{budget_pct:.1%}"] * len(summary),
        'Perturb Strength': [perturb_strength] * len(summary),
        'Timestamp': [timestamp] * len(summary)
    })
    
    # Column names
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
    
    # Merge with parameter information
    final_summary = pd.concat([summary_with_params, summary.iloc[:, 1:]], axis=1)
    
    # Save the LaTeX table
    latex_path = results_dir / f"{file_prefix}_summary.tex"
    with open(latex_path, "w") as f:
        f.write(final_summary.to_latex(index=False, float_format="%.3f"))
    
    print(f"\n{'='*50}")
    print(f"RESULTS SAVED")
    print(f"{'='*50}")
    print(f"Dataset: {dataset_name}")
    print(f"Budget: {budget_pct:.1%}")
    print(f"Perturb Strength: {perturb_strength}")
    print(f"CSV: {csv_path}")
    print(f"LaTeX: {latex_path}")
    print(f"{'='*50}")
    print("SUMMARY:")
    print(final_summary.to_string(index=False))
    
    return file_prefix

def plot_results(df_results: pd.DataFrame, results_dir: Path, dataset_name: str, config: ExperimentConfig, file_prefix: str):
    """
    Creates and saves a scatter plot comparing attack impact (impact score) with defense evasion.
    """
    # Create the Results folder
    results_dir.mkdir(exist_ok=True)
    
    # Retrieve parameter information
    budget_pct = list(config.attacks.values())[0].budget_pct
    perturb_strength = list(config.attacks.values())[0].perturb_strength
    
    # Determine the Impact metric
    impact_column = None
    if 'impact_score' in df_results.columns:
        impact_column = 'impact_score'
        impact_label = 'Attack Impact (Impact Score)'
    elif 'mae_degradation' in df_results.columns:
        impact_column = 'mae_degradation'
        impact_label = 'Attack Impact (MAE Degradation)'
    else:
        print("Warning: Impact metric not found. Plot cannot be created.")
        return
    
    if 'recall' not in df_results.columns:
        print("Warning: Recall metric not found. Plot cannot be created.")
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
    
    # Set title and subtitle with parameter information
    main_title = f'Attack Impact vs. Stealth (Defense Evasion)\nDataset: {dataset_name.upper()}'
    subtitle = f'Budget: {budget_pct:.1%} | Perturb Strength: {perturb_strength} | Defenses: {len(df_results["defense"].unique())}'
    
    plt.suptitle(main_title, fontsize=16, weight='bold', y=0.95)
    plt.title(subtitle, fontsize=11, style='italic', pad=20)
    
    plt.xlabel('Defense Evasion (1 - Recall)', fontsize=12)
    plt.ylabel(impact_label, fontsize=12)
    
    # Adjust the legend
    plt.legend(title='Attack & Defense', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    
    # Save the plot file
    fig_path = results_dir / f"{file_prefix}_plot.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved: {fig_path}")
    plt.show()

def run_single_experiment(dataset_name: str):
    """
    Runs attack and defense experiments on the specified dataset.
    """
    cfg = get_config()
    np.random.seed(cfg.random_seed) # Set the random seed at the beginning
    torch.manual_seed(cfg.random_seed)
    
    all_results = []
    
    if dataset_name not in cfg.datasets:
        print(f"Error: Dataset named '{dataset_name}' not found in the configuration.")
        return
    
    dataset_cfg = cfg.datasets[dataset_name]
    
    print(f"\n\n{'='*20} Starting experiments for '{dataset_name.upper()}' {'='*20}")
    
    try:
        df_clean = DATASET_LOADERS[dataset_name](dataset_cfg.path)
        print(f"'{dataset_name}' loaded successfully.")
    except FileNotFoundError:
        print(f"Error: File '{dataset_cfg.path}' for dataset '{dataset_name}' not found. Skipping experiment.")
        return
        
    defenses = {name: model_detectors.get_defense(name) for name in cfg.active_defenses}

    for attack_name, attack_cfg in cfg.attacks.items():
        print(f"\n--- Attack: {attack_name.upper()} ---")

        # Reset randomness for each attack, but with a different seed
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
        
        # Fix: Obtain the correct list of poisoned points from the attack object
        poison_indices = attack_instance.poison_indices

        impact_metrics = stat_detectors.evaluate_impact(df_clean, df_poisoned, dataset_cfg.target_column, dataset_cfg.task_type)
        
        stealth_metrics = stat_detectors.evaluate_stealth(df_clean[dataset_cfg.target_column], df_poisoned[dataset_cfg.target_column])
        
        for defense_name, defense_instance in defenses.items():
            print(f"  - Testing against defense: {defense_name}")
            
            defense_target_column = dataset_cfg.target_column
            
            if defense_target_column not in df_poisoned.columns:
                print(f"    - Warning: '{defense_target_column}' not found. Skipping defense.")
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
    
    print(f"\n[OK] Experiment completed. Results saved in '{cfg.results_dir}' folder.")

def main():
    """
    Main entry point for the Data Poisoning Toolkit console application.
    Supports command line arguments for dataset selection and configuration.
    """
    import argparse
    import sys
    
    # Print toolkit information
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                  🛡️ Data Poisoning Toolkit                      ║
    ║            A Comprehensive Framework for ML Security Research     ║
    ║                                                                  ║
    ║  Authors: M.A. Yagiz, M.A. Dayi, Dr. P. Goktas                 ║
    ║  BlackHat Europe 2025 | IEEE Conference Submission              ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    parser = argparse.ArgumentParser(
        description="Data Poisoning Toolkit - ML Security Research Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  datapoisontoolkit                           # Run with default water_quality dataset
  datapoisontoolkit --dataset smart_grid     # Run with smart_grid dataset
  datapoisontoolkit --list-datasets          # Show available datasets
  datapoisontoolkit --dataset water_quality --attack dspo  # Run specific attack

For academic citation and more information, visit:
https://github.com/[username]/data-poison-toolkit
        """
    )
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="water_quality",
        help="Dataset to use for experiments (default: water_quality)"
    )
    
    parser.add_argument(
        "--attack",
        type=str,
        help="Specific attack to run (random, greedy_residue, dspo)"
    )
    
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List all available datasets and exit"
    )
    
    parser.add_argument(
        "--list-attacks",
        action="store_true",
        help="List all available attacks and exit"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="Data Poisoning Toolkit v1.0.0"
    )
    
    args = parser.parse_args()
    
    # Get configuration
    cfg = get_config()
    
    # Handle list operations
    if args.list_datasets:
        print("\n📊 Available Datasets:")
        print("=" * 50)
        for name, config in cfg.datasets.items():
            print(f"  • {name:<25} | {config.task_type:<12} | {config.target_column}")
        print("\n💡 Use --dataset <name> to select a specific dataset")
        sys.exit(0)
    
    if args.list_attacks:
        print("\n⚔️ Available Attacks:")
        print("=" * 40)
        for name in cfg.attacks.keys():
            print(f"  • {name}")
        print("\n💡 Use --attack <name> to select a specific attack")
        sys.exit(0)
    
    # Validate dataset
    if args.dataset not in cfg.datasets:
        print(f"❌ Error: Dataset '{args.dataset}' not found.")
        print(f"Available datasets: {', '.join(cfg.datasets.keys())}")
        print("Use --list-datasets to see all options.")
        sys.exit(1)
    
    # Filter attacks if specified
    if args.attack:
        if args.attack not in cfg.attacks:
            print(f"❌ Error: Attack '{args.attack}' not found.")
            print(f"Available attacks: {', '.join(cfg.attacks.keys())}")
            print("Use --list-attacks to see all options.")
            sys.exit(1)
        
        # Keep only the specified attack
        cfg.attacks = {args.attack: cfg.attacks[args.attack]}
        print(f"🎯 Running only '{args.attack}' attack")
    
    print(f"\n🚀 Starting experiment with dataset: {args.dataset}")
    print(f"📋 Active defenses: {', '.join(cfg.active_defenses)}")
    print(f"⚔️ Attacks to test: {', '.join(cfg.attacks.keys())}")
    print("\n" + "="*60)
    
    try:
        run_single_experiment(args.dataset)
        print("\n✅ Experiment completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠️ Experiment interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Experiment failed with error: {e}")
        print("💡 Check dataset paths and configuration.")
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main()
    else:
        # Default behavior when run without arguments
        run_single_experiment("water_quality")
