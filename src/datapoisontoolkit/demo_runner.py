import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import argparse
import warnings

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple , Any, Optional
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datapoisontoolkit.detectors import stat_detectors
from datapoisontoolkit.attacks import base_attack
from datapoisontoolkit.detectors import model_detectors
from datapoisontoolkit.attacks.base_attack import RandomAttack
from datapoisontoolkit.attacks.greedy_residue_mock import GreedyResidueAttack

BASE_DIR = Path(__file__).resolve().parent

# Styling for plots
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 7)

@dataclass
class ExperimentConfig:
	random_seed: int = 42
	results_dir: Path = Path(BASE_DIR.parent.parent /"results")

	attacks: Dict[str, base_attack.BaseAttackConfig] = field(default_factory=lambda: {
		"random": base_attack.BaseAttackConfig(),
		"greedy_residue": base_attack.BaseAttackConfig(),
        "dspo": base_attack.DSPOConfig(),
        "old_dspo": base_attack.DSPOConfig(),
	})

	active_defenses: List[str] = field(default_factory=lambda: ["lof", "pca", "kde", "forest", "median", "quantile", "dbscan"])



def get_config() -> ExperimentConfig:
	return ExperimentConfig()

@dataclass
class InteractiveDatasetConfig:
    """Dataset configuration for interactive mode"""
    path: str
    task_type: str
    target_column: str
    dataset_name: str

@dataclass
class AutoDatasetConfig:
    """Automatically configured dataset configuration"""
    path: str
    task_type: str  # 'classification' or 'regression' 
    target_column: str
    feature_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    datetime_columns: List[str] = field(default_factory=list)
    columns_to_drop: List[str] = field(default_factory=list)
    original_shape: Tuple[int, int] = (0, 0)
    preprocessing_info: Dict[str, Any] = field(default_factory=dict)

class FlexibleDatasetLoader:
    """
    Flexible dataset loader that can automatically prepare any CSV dataset 
    for the DSPO attack system.
    """
    
    def __init__(self):
        self.encoders = {}
        self.scalers = {}
        self.imputers = {}
        
    def analyze_dataset(self, csv_path: str, target_column: Optional[str] = None) -> AutoDatasetConfig:
        """
        Automatically analyze a dataset and determine its characteristics
        """
        print(f"Analyzing dataset: {csv_path}")
        df = pd.read_csv(csv_path)
        original_shape = df.shape
        print(f"Original dataset shape: {original_shape}")
        
        # Detect column types
        datetime_columns = self._detect_datetime_columns(df)
        categorical_columns = self._detect_categorical_columns(df)
        numeric_columns = self._detect_numeric_columns(df)
        
        print(f"Detected column types:")
        print(f"  - Datetime columns: {datetime_columns}")
        print(f"  - Categorical columns: {categorical_columns}")
        print(f"  - Numeric columns: {numeric_columns}")
        
        # Auto-detect target column if not specified
        if target_column is None:
            target_column = self._auto_detect_target(df, numeric_columns)
            print(f"Auto-detected target column: {target_column}")
        
        # Determine task type
        task_type = self._determine_task_type(df, target_column)
        print(f"Detected task type: {task_type}")
        
        # Determine columns to drop
        columns_to_drop = self._identify_columns_to_drop(df, datetime_columns)
        
        # Feature columns (everything except target and dropped columns)
        feature_columns = [col for col in df.columns 
                          if col != target_column and col not in columns_to_drop]
        
        config = AutoDatasetConfig(
            path=csv_path,
            task_type=task_type,
            target_column=target_column,
            feature_columns=feature_columns,
            categorical_columns=[col for col in categorical_columns if col in feature_columns],
            datetime_columns=datetime_columns,
            columns_to_drop=columns_to_drop,
            original_shape=original_shape,
            preprocessing_info={}
        )
        
        return config
    
    def load_and_preprocess_dataset(self, config: AutoDatasetConfig) -> pd.DataFrame:
        """
        Load and preprocess dataset according to the configuration
        """
        print(f"Loading and preprocessing dataset: {config.path}")
        df = pd.read_csv(config.path)
        
        # Step 1: Handle datetime columns (sort by them, then drop)
        df = self._handle_datetime_columns(df, config.datetime_columns)
        
        # Step 2: Drop unnecessary columns
        df = self._drop_columns(df, config.columns_to_drop)
        
        # Step 3: Handle missing values
        df = self._handle_missing_values(df, config)
        
        # Step 4: Encode categorical variables
        df = self._encode_categorical_variables(df, config.categorical_columns)
        
        # Step 5: Handle infinite and extreme values
        df = self._handle_infinite_values(df)
        
        # Step 6: Ensure target column is properly formatted
        df = self._format_target_column(df, config.target_column, config.task_type)
        
        # Step 8: ADDITIONAL SAFETY CHECK - Remove any remaining non-numeric columns
        df = self._ensure_all_numeric(df, config.target_column)
    
        # Step 9: Validate dataset for DSPO requirements
        self._validate_dataset_for_dspo(df, config)
        
        print(f"Preprocessing completed. Final shape: {df.shape}")
        print(f"Final columns: {list(df.columns)}")
        print(f"Target column '{config.target_column}' stats:")
        if config.task_type == 'regression':
            print(f"  Mean: {df[config.target_column].mean():.4f}")
            print(f"  Std: {df[config.target_column].std():.4f}")
            print(f"  Min: {df[config.target_column].min():.4f}")
            print(f"  Max: {df[config.target_column].max():.4f}")
        else:
            print(f"  Classes: {sorted(df[config.target_column].unique())}")
            print(f"  Class distribution: {df[config.target_column].value_counts().to_dict()}")
        
        return df
    
    def _ensure_all_numeric(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Final safety check to ensure all columns are numeric.
        Drop any remaining non-numeric columns (except if they're the target).
        """
        print("Performing final numeric validation...")
    
        # Get non-numeric columns
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
        if non_numeric_cols:
            print(f"Found remaining non-numeric columns: {non_numeric_cols}")
        
            # Don't drop target column even if it's non-numeric (we'll handle it separately)
            cols_to_drop = [col for col in non_numeric_cols if col != target_column]
        
            if cols_to_drop:
                print(f"Dropping non-numeric feature columns: {cols_to_drop}")
                df = df.drop(columns=cols_to_drop)
        
            # Handle target column if it's non-numeric
            if target_column in non_numeric_cols:
                print(f"Converting non-numeric target column '{target_column}' to numeric...")
                try:
                    # Try to convert to numeric
                    df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
                
                    # If that fails, use label encoding
                    if df[target_column].isnull().all():
                        print(f"Numeric conversion failed, using label encoding for target column...")
                        original_values = pd.read_csv(self.config.path if hasattr(self, 'config') else 'temp')[target_column]
                        encoder = LabelEncoder()
                        df[target_column] = encoder.fit_transform(original_values.astype(str))
                        self.encoders[f"{target_column}_target"] = encoder
                    
                except Exception as e:
                    print(f"Error converting target column: {e}")
                    # Last resort: label encoding
                    encoder = LabelEncoder()
                    df[target_column] = encoder.fit_transform(df[target_column].astype(str))
                    self.encoders[f"{target_column}_target"] = encoder
    
        print(f"All columns are now numeric. Final columns: {list(df.columns)}")
        return df

    def _detect_datetime_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect datetime columns in a dataframe by name patterns, dtype, or sample parsing."""
        datetime_columns = []

        # Comprehensive datetime keyword list
        datetime_patterns = [
            'date', 'time', 'timestamp', 'created', 'updated', 'modified',
            'datetime', 'datum', 'tarih', 'zaman', 'day', 'month', 'year',
            'jour', 'mois', 'année', 'jahr', 'monat', 'tag', 'fecha', 'tiempo'
        ]

        for col in df.columns:
            col_lower = col.lower()

            # 1) Search for datetime patterns in column name
            matched_pattern = next((pattern for pattern in datetime_patterns if pattern in col_lower), None)
            if matched_pattern:
                datetime_columns.append(col)
                print(f"[Pattern Match] Detected datetime column '{col}' by pattern '{matched_pattern}'")
                continue

            # 2) Check if DataFrame column dtype is already datetime64
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_columns.append(col)
                print(f"[Dtype Match] Detected datetime column '{col}' by dtype")
                continue

            # 3) For object columns, try parsing as datetime
            if df[col].dtype == 'object':
                sample = df[col].dropna().head(50)  # Look at first 50 non-null samples
                if len(sample) > 3:
                    try:
                        parsed_dates = pd.to_datetime(sample, errors='coerce', infer_datetime_format=True)
                        # If more than 70% can be parsed as dates, consider it datetime
                        success_rate = parsed_dates.notna().sum() / len(sample)
                        if success_rate > 0.7:
                            datetime_columns.append(col)
                            print(f"[Parsing] Detected datetime column '{col}' with {success_rate:.1%} success rate")
                    except Exception as e:
                        # Silently continue - not all parsing failures are worth reporting
                        pass

        return list(set(datetime_columns))  # Remove duplicates
    
    def _detect_categorical_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect categorical columns"""
        categorical_columns = []
        
        for col in df.columns:
            # String/object columns
            if df[col].dtype == 'object':
                categorical_columns.append(col)
            # Low cardinality numeric columns
            elif df[col].dtype in ['int64', 'int32'] and df[col].nunique() < 20:
                categorical_columns.append(col)
            # Boolean columns
            elif df[col].dtype == 'bool':
                categorical_columns.append(col)
        
        return categorical_columns
    
    def _detect_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect numeric columns"""
        return df.select_dtypes(include=[np.number]).columns.tolist()
    
    def _auto_detect_target(self, df: pd.DataFrame, numeric_columns: List[str]) -> str:
        """Auto-detect the most likely target column"""
        
        # Common target column names
        target_patterns = ['target', 'label', 'class', 'outcome', 'result', 'prediction',
                          'price', 'amount', 'value', 'score', 'rating', 'yield', 'output']
        
        for col in df.columns:
            for pattern in target_patterns:
                if pattern in col.lower():
                    return col
        
        # If no pattern match, choose the last numeric column
        if numeric_columns:
            return numeric_columns[-1]
        
        # Fallback: last column
        return df.columns[-1]
    
    def _determine_task_type(self, df: pd.DataFrame, target_column: str) -> str:
        """Determine if task is classification or regression"""
        
        target_series = df[target_column]
        
        # Check if target is categorical/string
        if target_series.dtype == 'object' or target_series.dtype == 'bool':
            return 'classification'
        
        # Check number of unique values
        unique_values = target_series.nunique()
        total_values = len(target_series)
        
        # If unique values ratio is low, likely classification
        if unique_values / total_values < 0.05 and unique_values < 50:
            return 'classification'
        
        # Check if all values are integers and there are few unique values
        if target_series.dtype in ['int64', 'int32'] and unique_values < 20:
            return 'classification'
        
        return 'regression'
    
    def _identify_columns_to_drop(self, df: pd.DataFrame, datetime_columns: List[str]) -> List[str]:
        """Identify columns that should be dropped"""
        columns_to_drop = []
    
        # Don't add datetime columns here since they'll be handled separately
        # This prevents double-dropping issues
    
        # Drop ID-like columns
        for col in df.columns:
            if any(pattern in col.lower() for pattern in ['id', 'key', 'index', 'identifier']):
                # But keep if it's not unique (might be a categorical feature)
                if df[col].nunique() == len(df):
                    columns_to_drop.append(col)
    
        # Drop constant columns
        for col in df.columns:
            if df[col].nunique() <= 1:
                columns_to_drop.append(col)
    
        # Drop columns with too many missing values (>80%)
        for col in df.columns:
            if df[col].isnull().sum() / len(df) > 0.8:
                columns_to_drop.append(col)
    
        return list(set(columns_to_drop))  # Remove duplicates
    
    def _handle_datetime_columns(self, df: pd.DataFrame, datetime_columns: List[str]) -> pd.DataFrame:
        """Handle datetime columns - use for sorting then drop"""
    
        if not datetime_columns:
            print("No datetime columns detected.")
            return df
    
        print(f"Processing {len(datetime_columns)} datetime columns: {datetime_columns}")
    
        for col in datetime_columns:
            if col in df.columns:
                try:
                    # Convert to datetime
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                
                    # Sort by this datetime column if conversion was successful
                    if not df[col].isnull().all():
                        df = df.sort_values(col)
                        print(f"Sorted dataset by datetime column: {col}")
                    else:
                        print(f"Warning: Could not convert datetime column {col} - all values became NaT")
                    
                except Exception as e:
                    print(f"Warning: Could not process datetime column {col}: {e}")
    
        # Actually drop the datetime columns after sorting
        datetime_cols_to_drop = [col for col in datetime_columns if col in df.columns]
        if datetime_cols_to_drop:
            df = df.drop(columns=datetime_cols_to_drop)
            print(f"Dropped datetime columns: {datetime_cols_to_drop}")
        
            # Double check that they were actually dropped
            remaining_datetime_cols = [col for col in datetime_cols_to_drop if col in df.columns]
            if remaining_datetime_cols:
                print(f"ERROR: Failed to drop datetime columns: {remaining_datetime_cols}")
                # Force drop them
                df = df.drop(columns=remaining_datetime_cols, errors='ignore')
                print(f"Force-dropped remaining datetime columns: {remaining_datetime_cols}")
    
        return df
    
    def _drop_columns(self, df: pd.DataFrame, columns_to_drop: List[str]) -> pd.DataFrame:
        """Drop specified columns"""
        columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            print(f"Dropped columns: {columns_to_drop}")
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame, config: AutoDatasetConfig) -> pd.DataFrame:
        """Handle missing values intelligently"""
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = [col for col in config.categorical_columns if col in df.columns]
        
        # Handle numeric columns
        for col in numeric_columns:
            if df[col].isnull().any():
                # Use median for numeric columns (more robust than mean)
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
                print(f"Filled missing values in {col} with median: {median_value}")
        
        # Handle categorical columns
        for col in categorical_columns:
            if df[col].isnull().any():
                # Use mode for categorical columns
                mode_value = df[col].mode()
                if len(mode_value) > 0:
                    df[col] = df[col].fillna(mode_value[0])
                    print(f"Filled missing values in {col} with mode: {mode_value[0]}")
                else:
                    # If no mode, use 'Unknown'
                    df[col] = df[col].fillna('Unknown')
                    print(f"Filled missing values in {col} with 'Unknown'")
        
        return df
    
    def _encode_categorical_variables(self, df: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
        """Encode categorical variables"""
        
        for col in categorical_columns:
            if col in df.columns and df[col].dtype == 'object':
                # Use label encoding for simplicity
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col].astype(str))
                self.encoders[col] = encoder
                print(f"Label encoded column: {col}")
        
        return df
    
    def _handle_infinite_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle infinite and extreme values"""
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            # Replace infinite values with NaN first
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            
            # If there are NaN values (from inf replacement), fill with median
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"Replaced infinite values in {col} with median: {median_val}")
            
            # Clip extreme outliers (beyond 99.9th percentile)
            if df[col].std() > 0:  # Only if there's variation
                q001 = df[col].quantile(0.001)
                q999 = df[col].quantile(0.999)
                
                outliers_low = (df[col] < q001).sum()
                outliers_high = (df[col] > q999).sum()
                
                if outliers_low > 0 or outliers_high > 0:
                    df[col] = df[col].clip(lower=q001, upper=q999)
                    print(f"Clipped {outliers_low + outliers_high} extreme values in {col}")
        
        return df
    
    def _format_target_column(self, df: pd.DataFrame, target_column: str, task_type: str) -> pd.DataFrame:
        """Format target column appropriately"""
        
        if task_type == 'classification':
            # Ensure target is integer
            if df[target_column].dtype == 'object':
                encoder = LabelEncoder()
                df[target_column] = encoder.fit_transform(df[target_column])
                self.encoders[f"{target_column}_target"] = encoder
            else:
                df[target_column] = df[target_column].astype(int)
            print(f"Formatted classification target column: {target_column}")
        else:
            # Ensure target is float
            df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
            if df[target_column].isnull().any():
                median_val = df[target_column].median()
                df[target_column] = df[target_column].fillna(median_val)
            print(f"Formatted regression target column: {target_column}")
        
        return df
    
    def _validate_dataset_for_dspo(self, df: pd.DataFrame, config: AutoDatasetConfig):
        """Validate that dataset is ready for DSPO attack"""

        print("Validating dataset for DSPO compatibility...")
    
        # Check if target column exists
        if config.target_column not in df.columns:
            raise ValueError(f"Target column '{config.target_column}' not found in dataset")

        # Check for non-numeric columns with detailed information
        non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            print(f"ERROR: Non-numeric columns detected: {non_numeric}")
            print(f"All columns in dataset: {list(df.columns)}")
            print(f"Column dtypes:")
            for col in df.columns:
                print(f"  {col}: {df[col].dtype} (sample values: {df[col].dropna().head(3).tolist()})")
        
            # Try to provide helpful information
            if config.target_column in non_numeric:
                print(f"ERROR: Target column '{config.target_column}' is non-numeric!")
                print(f"Target column dtype: {df[config.target_column].dtype}")
                print(f"Target column unique values: {df[config.target_column].unique()[:10]}")
        
            raise ValueError(f"Non-numeric columns found: {non_numeric}. All columns must be numeric for DSPO.")

        # Check for missing values
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            print(f"Missing values found in columns: {missing_cols}")
            for col in missing_cols:
                missing_count = df[col].isnull().sum()
                missing_pct = (missing_count / len(df)) * 100
                print(f"  {col}: {missing_count} missing ({missing_pct:.1f}%)")
            raise ValueError(f"Missing values found in columns: {missing_cols}")

        # Check for infinite values
        infinite_cols = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if not np.isfinite(df[col]).all():
                infinite_cols.append(col)
                inf_count = (~np.isfinite(df[col])).sum()
                print(f"  {col}: {inf_count} non-finite values")
    
        if infinite_cols:
            raise ValueError(f"Non-finite values found in columns: {infinite_cols}")

        # Check minimum dataset size
        if len(df) < 50:
            raise ValueError(f"Dataset is too small ({len(df)} rows). Minimum 50 rows required for DSPO attack.")
        elif len(df) < 100:
            warnings.warn(f"Dataset is small ({len(df)} rows). DSPO attack may not be very effective.")

        # Check if target has sufficient variation
        if config.task_type == 'regression':
            if df[config.target_column].std() == 0:
                raise ValueError("Target column has no variation (all values are identical)")
            elif df[config.target_column].nunique() < 3:
                warnings.warn("Target column has very low variation. Results may not be meaningful.")
        else:  # classification
            unique_classes = df[config.target_column].nunique()
            if unique_classes < 2:
                raise ValueError("Classification target must have at least 2 classes")
            elif unique_classes == 2:
                # Check class balance for binary classification
                class_counts = df[config.target_column].value_counts()
                min_class_pct = class_counts.min() / len(df)
                if min_class_pct < 0.05:
                    warnings.warn(f"Highly imbalanced classes detected. Minority class: {min_class_pct:.1%}")
    
        # Final verification - ensure all dtypes are numeric
        all_numeric = all(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns)
        if not all_numeric:
            non_numeric_final = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
            raise ValueError(f"Final validation failed. Non-numeric columns remain: {non_numeric_final}")
    
        print("Dataset validation passed. Ready for DSPO attack.")
        print(f"  Shape: {df.shape}")
        print(f"  Target: {config.target_column} ({config.task_type})")
        print(f"  Features: {len(df.columns) - 1}")
    
        if config.task_type == 'classification':
            print(f"  Classes: {df[config.target_column].nunique()}")
        else:
            print(f"  Target range: [{df[config.target_column].min():.3f}, {df[config.target_column].max():.3f}]")  
            


def create_flexible_dataset_config(csv_path: str, target_column: Optional[str] = None) -> Tuple[AutoDatasetConfig, pd.DataFrame]:
    """
    Convenience function to analyze and prepare any dataset for DSPO attack
    
    Args:
        csv_path: Path to CSV file
        target_column: Target column name (if None, will be auto-detected)
    
    Returns:
        Tuple of (dataset configuration, processed dataframe)
    """
    loader = FlexibleDatasetLoader()
    
    # Analyze dataset
    config = loader.analyze_dataset(csv_path, target_column)
    
    # Load and preprocess
    processed_df = loader.load_and_preprocess_dataset(config)
    
    return config, processed_df


# Example usage and integration with existing system
def integrate_flexible_loader_with_existing_system():
    """
    Example of how to integrate flexible loader with existing ExperimentConfig
    """
    
    # Example: Load any CSV file
    csv_path = "path/to/your/dataset.csv"
    target_col = None  # Auto-detect or specify manually
    
    # Create flexible configuration
    config, processed_df = create_flexible_dataset_config(csv_path, target_col)
    
    # Convert to existing DatasetConfig format
    dataset_config = InteractiveDatasetConfig(
        path=config.path,
        task_type=config.task_type,
        target_column=config.target_column
    )
    
    print(f"Dataset ready for DSPO attack:")
    print(f"  Path: {dataset_config.path}")
    print(f"  Task: {dataset_config.task_type}")
    print(f"  Target: {dataset_config.target_column}")
    print(f"  Shape: {processed_df.shape}")
    
    return dataset_config, processed_df


def interactive_dataset_analyzer(csv_path: str) -> Tuple[AutoDatasetConfig, pd.DataFrame]:
    """
    Interactive dataset analyzer that lets user choose target column
    """
    print("="*60)
    print(f"ANALYZING DATASET: {csv_path}")
    print("="*60)
    
    # Load dataset for analysis
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded successfully!")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print()
    
    # Show basic info about columns
    print("COLUMN ANALYSIS:")
    print("-" * 50)
    
    loader = FlexibleDatasetLoader()
    datetime_columns = loader._detect_datetime_columns(df)
    categorical_columns = loader._detect_categorical_columns(df)
    numeric_columns = loader._detect_numeric_columns(df)
    
    print(f" Datetime columns ({len(datetime_columns)}): {datetime_columns}")
    print(f"  Categorical columns ({len(categorical_columns)}): {categorical_columns}")  
    print(f" Numeric columns ({len(numeric_columns)}): {numeric_columns}")
    print()
    
    # Show detailed column info
    print("DETAILED COLUMN INFORMATION:")
    print("-" * 50)
    for i, col in enumerate(df.columns, 1):
        col_type = "numeric" if col in numeric_columns else ("categorical" if col in categorical_columns else "datetime")
        unique_vals = df[col].nunique()
        missing_pct = (df[col].isnull().sum() / len(df)) * 100
        
        print(f"{i:2d}. {col:<25} | Type: {col_type:<12} | Unique: {unique_vals:<8} | Missing: {missing_pct:.1f}%")
        
        # Show sample values for better understanding
        if col_type == "numeric":
            print(f"     Stats: Min={df[col].min():.2f}, Max={df[col].max():.2f}, Mean={df[col].mean():.2f}")
        elif unique_vals <= 10:
            sample_vals = df[col].value_counts().head(5).index.tolist()
            print(f"     Sample values: {sample_vals}")
        print()
    
    # Interactive target column selection
    print("="*60)
    print("TARGET COLUMN SELECTION")
    print("="*60)
    
    # Auto-suggest target column
    suggested_target = loader._auto_detect_target(df, numeric_columns)
    print(f" Suggested target column: '{suggested_target}'")
    
    # Show why this column was suggested
    if suggested_target in numeric_columns:
        unique_ratio = df[suggested_target].nunique() / len(df)
        if unique_ratio > 0.05:
            suggested_task = "regression"
        else:
            suggested_task = "classification"
        print(f"   Suggested task type: {suggested_task}")
        print(f"   Reason: {'High cardinality numeric column' if suggested_task == 'regression' else 'Low cardinality numeric column'}")
    
    print()
    print("Available target column options:")
    
    # Show only numeric columns as potential targets
    target_options = []
    for i, col in enumerate(numeric_columns, 1):
        target_options.append(col)
        unique_vals = df[col].nunique()
        unique_ratio = unique_vals / len(df)
        
        # Determine likely task type
        if unique_ratio > 0.05 and unique_vals > 20:
            likely_task = "regression"
        else:
            likely_task = "classification"
        
        marker = "" if col == suggested_target else "  "
        print(f"{marker} {i:2d}. {col:<25} | Unique values: {unique_vals:<8} | Likely task: {likely_task}")
    
    print()
    
    # User input loop
    while True:
        try:
            user_input = input(f"Enter target column number (1-{len(target_options)}) or column name [default: {suggested_target}]: ").strip()
            
            if not user_input:
                # Use suggested target
                selected_target = suggested_target
                break
            elif user_input.isdigit():
                # User entered a number
                col_index = int(user_input)
                if 1 <= col_index <= len(target_options):
                    selected_target = target_options[col_index - 1]
                    break
                else:
                    print(f" Please enter a number between 1 and {len(target_options)}")
            else:
                # User entered column name
                if user_input in df.columns:
                    if user_input in numeric_columns:
                        selected_target = user_input
                        break
                    else:
                        print(f" Column '{user_input}' is not numeric. Please choose a numeric column for target.")
                else:
                    print(f" Column '{user_input}' not found in dataset.")
        except KeyboardInterrupt:
            print("\n\n Operation cancelled by user.")
            raise
        except Exception as e:
            print(f" Invalid input: {e}")
    
    print(f"\n Selected target column: '{selected_target}'")
    
    # Determine final task type
    final_task_type = loader._determine_task_type(df, selected_target)
    print(f" Determined task type: {final_task_type}")
    
    # Confirm with user
    print(f"\n SUMMARY:")
    print(f"   Dataset: {Path(csv_path).name}")
    print(f"   Target column: {selected_target}")
    print(f"   Task type: {final_task_type}")
    print(f"   Dataset shape: {df.shape}")
    
    confirm = input(f"\nProceed with these settings? [Y/n]: ").strip().lower()
    if confirm in ['n', 'no']:
        print(" Operation cancelled by user.")
        return None, None
    
    print(f"\n Processing dataset with target column: '{selected_target}'...")
    
    # Create config and process dataset
    config = AutoDatasetConfig(
        path=csv_path,
        task_type=final_task_type,
        target_column=selected_target
    )
    
    processed_df = loader.load_and_preprocess_dataset(config)
    
    print(f" Dataset processing completed!")
    print(f"   Final shape: {processed_df.shape}")
    print(f"   Ready for DSPO attack!")
    
    return config, processed_df


def interactive_universal_loader(csv_path: str) -> pd.DataFrame:
    """
    Interactive universal dataset loader for use in DATASET_LOADERS
    """
    config, processed_df = interactive_dataset_analyzer(csv_path)
    if processed_df is None:
        raise ValueError("Dataset processing was cancelled by user")
    return processed_df


def run_interactive_dataset_setup():
    """
    Main function to run interactive dataset setup
    """
    print(" FLEXIBLE DATASET ANALYZER")
    print("="*60)
    
    while True:
        try:
            csv_path = input("Enter path to your CSV dataset: ").strip()
            
            if not csv_path:
                print(" Please enter a valid path")
                continue
                
            csv_path = Path(csv_path)
            if not csv_path.exists():
                print(f" File not found: {csv_path}")
                continue
                
            if not csv_path.suffix.lower() == '.csv':
                print(" File must be a CSV file")
                continue
                
            break
            
        except KeyboardInterrupt:
            print("\n\n Operation cancelled by user.")
            return None, None
        except Exception as e:
            print(f" Error: {e}")
    
    return interactive_dataset_analyzer(str(csv_path))


# Create a universal dataset loader function
def universal_dataset_loader(csv_path: str, target_column: Optional[str] = None) -> pd.DataFrame:
    """
    Universal dataset loader function that can replace all specific loaders
    This function can be used directly in the DATASET_LOADERS dictionary
    """
    if target_column is None:
        # Use interactive mode
        return interactive_universal_loader(csv_path)
    else:
        # Use specified target column
        loader = FlexibleDatasetLoader()
        config = loader.analyze_dataset(csv_path, target_column)
        processed_df = loader.load_and_preprocess_dataset(config)
        return processed_df

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
    # Add parameter information to the summary table
    summary_with_params = pd.DataFrame({
        'Dataset': [dataset_name] * len(summary),
        'Attack': summary['attack'],
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

def run_interactive_single_experiment():
    """
    Updated interactive experiment with attack selection
    """
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║               Interactive Data Poisoning Toolkit                 ║
    ║         A Comprehensive Framework for ML Security Research       ║
    ║                                                                  ║
    ║  Authors: M.A. Yagiz, M.A. Dayi, Dr. P. Goktas                   ║
    ║  BlackHat Europe 2025 | IEEE Conference Submission               ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    print("INTERACTIVE MULTI-ATTACK EXPERIMENT")
    print("="*80)
    print("Compare Random, Greedy Residue, and DSPO attacks on any CSV dataset")
    print("="*80)
    
    try:
        # Step 1: Interactive dataset setup (mevcut kod)
        print("\n STEP 1: DATASET SELECTION AND ANALYSIS")
        print("-" * 60)
        
        config, df_clean = run_interactive_dataset_setup()
        
        if config is None or df_clean is None:
            print(" Dataset setup cancelled. Exiting...")
            return
        
        dataset_name = Path(config.path).stem
        
        # Create interactive dataset config
        interactive_config = InteractiveDatasetConfig(
            path=config.path,
            task_type=config.task_type,
            target_column=config.target_column,
            dataset_name=dataset_name
        )
        
        print(f"   Dataset '{dataset_name}' loaded and preprocessed successfully!")
        print(f"   Shape: {df_clean.shape}")
        print(f"   Target: {config.target_column} ({config.task_type})")
        
        # Step 2: Attack configuration with selection (updated function)
        attack_config = _configure_attack_parameters_with_selection(df_clean, config)
        
        # Step 3: Get existing system config for defenses
        cfg = get_config()
        np.random.seed(cfg.random_seed)
        torch.manual_seed(cfg.random_seed)
        
        # Step 4: Run experiments with multiple attacks (updated function)
        _run_experiment_with_multiple_attacks(
            interactive_config, df_clean, attack_config, cfg
        )
        
    except KeyboardInterrupt:
        print("\n\n Experiment cancelled by user.")
        return
    except Exception as e:
        print(f"\n Error during experiment: {e}")
        import traceback
        traceback.print_exc()
        return


def _configure_attack_parameters_with_selection(df_clean: pd.DataFrame, config: AutoDatasetConfig) -> Dict[str, Any]:
    """Configure attack parameters and select which attacks to run"""
    print(f"\n STEP 2: ATTACK CONFIGURATION")
    print("-" * 60)
    
    print(f"Dataset size: {len(df_clean)} rows")
    print(f"Target column: {config.target_column}")
    print(f"Task type: {config.task_type}")
    print()
    
    # Attack selection
    available_attacks = {
        'random': 'Random Attack (baseline)',
        'greedy_residue': 'Greedy Residue Attack', 
        'dspo': 'D-SPO Attack (sophisticated)',
        'old_dspo': 'D-SPO Attack (Legacy)'
    }
    
    print("AVAILABLE ATTACKS:")
    for key, desc in available_attacks.items():
        print(f"  {key}: {desc}")
    print()
    
    # Let user select attacks
    selected_attacks = []
    while True:
        print("Select attacks to run:")
        print("  1. All attacks (recommended for comparison)")
        print("  2. Only DSPO attack")
        print("  3. Custom selection")
        
        try:
            choice = input("Enter your choice (1-3): ").strip()
            
            if choice == '1':
                selected_attacks = list(available_attacks.keys())
                break
            elif choice == '2':
                selected_attacks = ['dspo']
                break
            elif choice == '3':
                print("\nCustom selection - enter attack names separated by commas")
                print(f"Available: {', '.join(available_attacks.keys())}")
                custom_input = input("Attacks: ").strip().lower()
                
                custom_attacks = [att.strip() for att in custom_input.split(',')]
                valid_attacks = [att for att in custom_attacks if att in available_attacks]
                
                if valid_attacks:
                    selected_attacks = valid_attacks
                    break
                else:
                    print(" No valid attacks selected. Please try again.")
            else:
                print(" Invalid choice. Please enter 1, 2, or 3.")
                
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f" Error: {e}")
    
    print(f"\n Selected attacks: {', '.join(selected_attacks)}")
    
    # Mevcut parametreleri al (aynı kod)
    print("\nATTACK PARAMETERS:")
    print("(Press Enter to use default values)")
    print()
    
    # Budget percentage
    while True:
        try:
            budget_input = input(f" Budget percentage (% of dataset to poison) [default: 10%]: ").strip()
            if not budget_input:
                budget_pct = 0.1
                break
            budget_pct = float(budget_input) / 100
            if 0 < budget_pct <= 1:
                break
            else:
                print(" Budget must be between 0 and 100%")
        except ValueError:
            print(" Please enter a valid number")
    
    # Perturbation strength
    while True:
        try:
            perturb_input = input(f" Perturbation strength (0.1 = weak, 1.0 = strong) [default: 0.5]: ").strip()
            if not perturb_input:
                perturb_strength = 0.5
                break
            perturb_strength = float(perturb_input)
            if 0.1 <= perturb_strength <= 2.0:
                break
            else:
                print(" Perturbation strength should be between 0.1 and 2.0")
        except ValueError:
            print(" Please enter a valid number")
    
    # Number of iterations (only for DSPO)
    num_iterations = 100
    if 'dspo' in selected_attacks:
        while True:
            try:
                iter_input = input(f" DSPO optimization iterations [default: 100]: ").strip()
                if not iter_input:
                    num_iterations = 100
                    break
                num_iterations = int(iter_input)
                if 10 <= num_iterations <= 1000:
                    break
                else:
                    print(" Number of iterations should be between 10 and 1000")
            except ValueError:
                print(" Please enter a valid integer")
    
    # Learning rate (only for DSPO)
    learning_rate = 0.01
    if 'dspo' in selected_attacks:
        while True:
            try:
                lr_input = input(f" DSPO learning rate [default: 0.01]: ").strip()
                if not lr_input:
                    learning_rate = 0.01
                    break
                learning_rate = float(lr_input)
                if 0.001 <= learning_rate <= 0.1:
                    break
                else:
                    print(" Learning rate should be between 0.001 and 0.1")
            except ValueError:
                print(" Please enter a valid number")
    
    # Batch size
    suggested_batch_size = min(5000, len(df_clean))
    while True:
        try:
            batch_input = input(f" Batch size [default: {suggested_batch_size}]: ").strip()
            if not batch_input:
                batch_size = suggested_batch_size
                break
            batch_size = int(batch_input)
            if 100 <= batch_size <= len(df_clean):
                break
            else:
                print(f" Batch size should be between 100 and {len(df_clean)}")
        except ValueError:
            print(" Please enter a valid integer")
    
    attack_config = {
        'selected_attacks': selected_attacks,
        'budget_pct': budget_pct,
        'perturb_strength': perturb_strength,
        'num_iterations': num_iterations,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'lambda_impact': 1.0,
        'lambda_stealth': 0.3,
        'lambda_defense': 0.7
    }
    
    print(f"\n Attack configuration completed!")
    print(f"   Selected attacks: {', '.join(selected_attacks)}")
    print(f"   Budget: {budget_pct*100:.1f}% ({int(len(df_clean) * budget_pct)} points)")
    print(f"   Perturbation strength: {perturb_strength}")
    if 'dspo' in selected_attacks:
        print(f"   DSPO iterations: {num_iterations}")
        print(f"   DSPO learning rate: {learning_rate}")
    
    return attack_config


def _run_experiment_with_multiple_attacks(
    dataset_config: InteractiveDatasetConfig, 
    df_clean: pd.DataFrame, 
    attack_config: Dict[str, Any],
    cfg
):
    """Run experiments with multiple selected attacks"""
    
    print(f"\n STEP 3: RUNNING ATTACKS")
    print("-" * 60)
    
    all_results = []
    selected_attacks = attack_config['selected_attacks']
    
    print(f"Starting experiments for '{dataset_config.dataset_name.upper()}'")
    print(f"Target column: {dataset_config.target_column}")
    print(f"Task type: {dataset_config.task_type}")
    print(f"Dataset shape: {df_clean.shape}")
    print(f"Attacks to run: {', '.join(selected_attacks)}")
    
    # UPDATED: Create a temporary config that uses the interactive parameters
    # This ensures save_results uses the correct values
    temp_cfg = cfg
    
    # Update the cfg.attacks with interactive parameters for save_results
    for attack_name in selected_attacks:
        if attack_name == 'dspo':
            temp_attack_config = base_attack.DSPOConfig(
                budget_pct=attack_config['budget_pct'],
                perturb_strength=attack_config['perturb_strength'],
                num_iterations=attack_config['num_iterations'],
                learning_rate=attack_config['learning_rate'],
                lambda_impact=attack_config.get('lambda_impact', 1.0),
                lambda_stealth=attack_config.get('lambda_stealth', 0.3),
                lambda_defense=attack_config.get('lambda_defense', 0.7),
                batch_size=attack_config.get('batch_size', None)
            )
        else:
            temp_attack_config = base_attack.BaseAttackConfig(
                budget_pct=attack_config['budget_pct'],
                perturb_strength=attack_config['perturb_strength']
            )
        
        # Update cfg.attacks to use interactive parameters
        temp_cfg.attacks[attack_name] = temp_attack_config
    
    # Get defenses
    defenses = {name: model_detectors.get_defense(name) for name in cfg.active_defenses}
    print(f"Active defenses: {', '.join(cfg.active_defenses)}")
    
    # Prepare base model
    X_clean_scaled = StandardScaler().fit_transform(df_clean.drop(columns=[dataset_config.target_column]))
    y_clean = df_clean[dataset_config.target_column]
    
    if dataset_config.task_type == 'classification':
        base_model = RandomForestClassifier(n_estimators=100, random_state=0)
    else:
        base_model = RandomForestRegressor(n_estimators=100, random_state=0)
    
    print(f"Training base {dataset_config.task_type} model...")
    base_model.fit(X_clean_scaled, y_clean)
    print("✓ Base model trained successfully!")
    
    # Run each selected attack
    for attack_name in selected_attacks:
        print(f"\n--- Running {attack_name.upper()} Attack ---")
        print(f"Budget: {attack_config['budget_pct']*100:.1f}% ({int(len(df_clean) * attack_config['budget_pct'])} points)")
        
        try:
            # Create attack instance based on type
            if attack_name == 'random':
                # Random attack config - defenses parametresini kaldır
                base_config = base_attack.BaseAttackConfig(
                    budget_pct=attack_config['budget_pct'],
                    perturb_strength=attack_config['perturb_strength']
                )
                attack_instance = RandomAttack(
                    config=base_config,
                    df=df_clean,
                    target_col=dataset_config.target_column,
                    task_type=dataset_config.task_type
                )
                
            elif attack_name == 'greedy_residue':
                # Greedy residue attack config - defenses parametresini kaldır
                base_config = base_attack.BaseAttackConfig(
                    budget_pct=attack_config['budget_pct'],
                    perturb_strength=attack_config['perturb_strength']
                )
                attack_instance = GreedyResidueAttack(
                    config=base_config,
                    df=df_clean,
                    target_col=dataset_config.target_column,
                    task_type=dataset_config.task_type
                )
                
            elif attack_name in ['dspo', 'old_dspo']:
                # DSPO attack (mevcut kodu kullan)
                attack_instance = base_attack.get_attack(
                    attack_name, 
                    None,
                    df_clean, 
                    dataset_config.target_column, 
                    dataset_config.task_type, 
                    defenses,
                    interactive_config=attack_config
                )
            else:
                print(f" Unknown attack: {attack_name}")
                continue
            
            print(f" Crafting poisoned dataset with {attack_name} attack...")
            df_poisoned = attack_instance.craft(base_model)
            print(f" {attack_name} poisoned dataset created successfully!")
            
            # Get poison indices
            poison_indices = attack_instance.poison_indices
            
            # Evaluate impact and stealth
            print(f" Evaluating {attack_name} attack impact and stealth...")
            impact_metrics = stat_detectors.evaluate_impact(
                df_clean, df_poisoned, dataset_config.target_column, dataset_config.task_type
            )
            stealth_metrics = stat_detectors.evaluate_stealth(
                df_clean[dataset_config.target_column], 
                df_poisoned[dataset_config.target_column]
            )
            
            print(f"✓ {attack_name} impact and stealth evaluation completed!")
            
            # Test against defenses
            print(f" Testing {attack_name} against {len(defenses)} defense mechanisms...")
            
            for defense_name, defense_instance in defenses.items():
                print(f"  Testing {attack_name} against: {defense_name}")
                
                try:
                    anomaly_scores = defense_instance.detect(df_poisoned[dataset_config.target_column])
                    
                    if isinstance(anomaly_scores, pd.Series):
                        threshold = anomaly_scores.quantile(1 - attack_config['budget_pct'])
                        detected_indices = anomaly_scores[anomaly_scores >= threshold].index
                    else:
                        detected_indices = anomaly_scores[anomaly_scores == -1].index
                    
                    defense_perf = stat_detectors.evaluate_defense_performance(
                        df_clean.index, poison_indices, detected_indices
                    )
                    
                    result_entry = {
                        "dataset": dataset_config.dataset_name,
                        "task": dataset_config.task_type,
                        "attack": attack_name,
                        "defense": defense_name,
                        "poison_budget": attack_config['budget_pct'],
                        **impact_metrics,
                        **defense_perf,
                        **stealth_metrics
                    }
                    all_results.append(result_entry)
                    
                    precision_val = defense_perf.get('precision', 0)
                    recall_val = defense_perf.get('recall', 0)
                    print(f"     {defense_name}: Precision={precision_val:.3f}, Recall={recall_val:.3f}")
                    
                except Exception as e:
                    print(f"     {defense_name}: Error - {e}")
                    
        except Exception as e:
            print(f" Error running {attack_name} attack: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create results DataFrame
    df_results = pd.DataFrame(all_results)
    
    if df_results.empty:
        print(" No results generated. Check for errors above.")
        return
    
    # Save results with the UPDATED config that has interactive parameters
    results_dir = cfg.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # IMPORTANT: Use temp_cfg instead of cfg to ensure correct parameters are saved
    file_prefix = save_results(df_results, results_dir, dataset_config.dataset_name, temp_cfg)
    
    # Plot results
    try:
        plot_results(df_results, results_dir, dataset_config.dataset_name, temp_cfg, file_prefix)
        print(f" Results visualization saved!")
    except Exception as e:
        print(f" Could not create plots: {e}")
    
    # Display final results with comparison
    print(f"\n STEP 4: EXPERIMENT RESULTS")
    print("="*80)
    print(f"ATTACK COMPARISON FOR '{dataset_config.dataset_name.upper()}'")
    print("="*80)
    
    # Group by attack and show comparison
    if len(selected_attacks) > 1:
        print(f"\n ATTACK EFFECTIVENESS COMPARISON:")
        print("-" * 60)
        
        attack_summary = df_results.groupby('attack').agg({
            'precision': 'mean',
            'recall': 'mean', 
            'f1_score': 'mean'
        }).round(3)
        
        print(f"{'ATTACK':<15} | {'AVG PRECISION':<12} | {'AVG RECALL':<10} | {'AVG F1':<8}")
        print("-" * 60)
        for attack, row in attack_summary.iterrows():
            print(f"{attack.upper():<15} | {row['precision']:<12.3f} | {row['recall']:<10.3f} | {row['f1_score']:<8.3f}")
    
    # Detailed results by defense
    print(f"\n DETAILED RESULTS BY DEFENSE:")
    print("-" * 80)
    for attack in selected_attacks:
        print(f"\n{attack.upper()} ATTACK RESULTS:")
        attack_results = df_results[df_results['attack'] == attack]
        for _, row in attack_results.iterrows():
            precision_val = row.get('precision', 0)
            recall_val = row.get('recall', 0) 
            f1_val = row.get('f1_score', 0)
            print(f"  {row['defense']:<15} | Precision: {precision_val:.3f} | Recall: {recall_val:.3f} | F1: {f1_val:.3f}")
    
    print(f"\n All experiments completed successfully!")
    print(f" Results saved in: {results_dir}")



def main():
    """
    Enhanced main entry point with interactive mode support
    """
    parser = argparse.ArgumentParser(
        description="Data Poisoning Toolkit - Interactive ML Security Research Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiment_runner.py                              # Interactive mode - choose your own dataset
  python experiment_runner.py --dataset smart_grid         # Use predefined dataset
  python experiment_runner.py --list-datasets              # Show available predefined datasets
  python experiment_runner.py --interactive                # Force interactive mode
  python experiment_runner.py --dataset water_quality --attack dspo  # Run specific attack

Interactive Mode:
  In interactive mode, you can load any CSV file and choose the target column.
  The system will automatically analyze your dataset and prepare it for DSPO attack.

For academic citation and more information, visit:
https://github.com/[username]/data-poison-toolkit
        """
    )
    
    parser.add_argument(
        "--dataset", 
        type=str,
        help="Predefined dataset to use for experiments"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Force interactive mode (choose your own CSV dataset)"
    )
    
    parser.add_argument(
        "--attack",
        type=str,
        help="Specific attack to run (only works with predefined datasets)"
    )
    
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List all predefined datasets and exit"
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
    
    # Get configuration for predefined datasets
    try:
        cfg = get_config()
    except Exception as e:
        print(f" Could not load predefined configuration: {e}")
        print("Falling back to interactive mode...")
        args.interactive = True
        cfg = None
    
    # Handle list operations
    if args.list_datasets and cfg:
        print("\n Available Predefined Datasets:")
        print("=" * 60)
        for name, config in cfg.datasets.items():
            print(f"  • {name:<25} | {config.task_type:<12} | {config.target_column}")
        print("\n Use --dataset <name> to select a specific dataset")
        print(" Use --interactive to use your own CSV dataset")
        sys.exit(0)
    
    if args.list_attacks and cfg:
        print("\n Available Attacks:")
        print("=" * 40)
        for name in cfg.attacks.keys():
            print(f"  • {name}")
        print("\n Use --attack <name> to select a specific attack (only with predefined datasets)")
        sys.exit(0)
    
    # Decide between interactive and predefined mode
    if args.interactive or args.dataset is None:
        # Interactive mode
        print(" Starting in INTERACTIVE mode...")
        print("You can choose any CSV dataset and target column.")
        run_interactive_single_experiment()
        
    else:
        # Predefined dataset mode
        if not cfg:
            print(" Error: Cannot use predefined datasets without configuration.")
            print(" Use --interactive to choose your own CSV dataset.")
            sys.exit(1)
            
        # Validate dataset
        if args.dataset not in cfg.datasets:
            print(f" Error: Dataset '{args.dataset}' not found.")
            print(f"Available datasets: {', '.join(cfg.datasets.keys())}")
            print(" Use --list-datasets to see all options or --interactive to use your own dataset.")
            sys.exit(1)
        
        # Filter attacks if specified
        if args.attack:
            if args.attack not in cfg.attacks:
                print(f" Error: Attack '{args.attack}' not found.")
                print(f"Available attacks: {', '.join(cfg.attacks.keys())}")
                print("Use --list-attacks to see all options.")
                sys.exit(1)
            
            # Keep only the specified attack
            cfg.attacks = {args.attack: cfg.attacks[args.attack]}
            print(f" Running only '{args.attack}' attack")
        
        print(f" Starting in PREDEFINED mode...")
        print(f" Using dataset: {args.dataset}")
        print(f" Active defenses: {', '.join(cfg.active_defenses)}")
        print(f" Attacks to test: {', '.join(cfg.attacks.keys())}")
        
        try:
            # Use the original run_single_experiment function
            from demo_runner import run_single_experiment  # Import your original function
            run_single_experiment(args.dataset)
            print("\n Experiment completed successfully!")
        except KeyboardInterrupt:
            print("\n Experiment interrupted by user.")
            sys.exit(1)
        except Exception as e:
            print(f"\n Experiment failed with error: {e}")
            print(" Check dataset paths and configuration.")
            sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main()
    else:
        # Default behavior when run without arguments - interactive mode
        print(" No arguments provided. Starting in INTERACTIVE mode...")
        run_interactive_single_experiment()