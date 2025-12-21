import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datapoisontoolkit.detectors import base_defense
from typing import Dict, Optional

@dataclass
class BaseAttackConfig:
    """Base configuration class for all attack types."""
    budget_pct: float = 0.02  # Percentage of data to poison
    perturb_strength: float = 0.2  # Maximum strength of perturbations
    
    def update_from_interactive_config(self, interactive_config: Dict[str, any]):
        """Update configuration from interactive user input"""
        if 'budget_pct' in interactive_config:
            self.budget_pct = interactive_config['budget_pct']
        if 'perturb_strength' in interactive_config:
            self.perturb_strength = interactive_config['perturb_strength']

@dataclass
class DSPOConfig(BaseAttackConfig):
    """Configuration specifically for D-SPO (Differentiable Shielded Poisoning Oracle) attack."""
    num_iterations: int = 200
    learning_rate: float = 0.05
    lambda_defense: float = 0.1
    lambda_stealth: float = 0.05
    lambda_impact: float = 5.0
    lambda_impact_mean: float = 5.0
    lambda_impact_std: float = 4.0
    lambda_impact_extremeness: float = 5.0
    lambda_impact_skew: float = 3.0
    lambda_impact_kurt: float = 2.0
    lambda_impact_variance: float = 4.0
    use_stealth_scheduler: bool = False
    
    batch_size: Optional[int] = None  # New field for batch processing
    
    def update_from_interactive_config(self, interactive_config: Dict[str, any]):
        """Update DSPO-specific configuration from interactive input"""
        # First update base parameters
        super().update_from_interactive_config(interactive_config)
        
        # Then update DSPO-specific parameters
        if 'num_iterations' in interactive_config:
            self.num_iterations = interactive_config['num_iterations']
        if 'learning_rate' in interactive_config:
            self.learning_rate = interactive_config['learning_rate']
        if 'batch_size' in interactive_config:
            self.batch_size = interactive_config['batch_size']
        if 'lambda_impact' in interactive_config:
            self.lambda_impact = interactive_config['lambda_impact']
        if 'lambda_stealth' in interactive_config:
            self.lambda_stealth = interactive_config['lambda_stealth']
        if 'lambda_impact_mean' in interactive_config:
            self.lambda_impact_mean = interactive_config['lambda_impact_mean']
        if 'lambda_impact_std' in interactive_config:
            self.lambda_impact_std = interactive_config['lambda_impact_std']
        if 'lambda_impact_extremeness' in interactive_config:
            self.lambda_impact_extremeness = interactive_config['lambda_impact_extremeness']
        if 'lambda_impact_skew' in interactive_config:
            self.lambda_impact_skew = interactive_config['lambda_impact_skew']
        if 'use_stealth_scheduler' in interactive_config:
            self.use_stealth_scheduler = interactive_config['use_stealth_scheduler']
        if 'lambda_impact_kurt' in interactive_config:
            self.lambda_impact_kurt = interactive_config['lambda_impact_kurt']
        if 'lambda_impact_variance' in interactive_config:
            self.lambda_impact_variance = interactive_config['lambda_impact_variance']
        if 'use_stealth_scheduler' in interactive_config:
            self.use_stealth_scheduler = interactive_config['use_stealth_scheduler']
    
    @classmethod
    def from_interactive_config(cls, interactive_config: Dict[str, any]) -> 'DSPOConfig':
        """Create DSPOConfig directly from interactive configuration"""
        return cls(
            budget_pct=interactive_config.get('budget_pct', 0.02),
            perturb_strength=interactive_config.get('perturb_strength', 0.2),
            num_iterations=interactive_config.get('num_iterations', 200),
            learning_rate=interactive_config.get('learning_rate', 0.05),
            lambda_defense=interactive_config.get('lambda_defense', 0.1),
            lambda_stealth=interactive_config.get('lambda_stealth', 0.05),
            lambda_impact=interactive_config.get('lambda_impact', 5.0),
            lambda_impact_mean=interactive_config.get('lambda_impact_mean', 5.0),
            lambda_impact_std=interactive_config.get('lambda_impact_std', 4.0),
            lambda_impact_extremeness=interactive_config.get('lambda_impact_extremeness', 5.0),
            lambda_impact_skew=interactive_config.get('lambda_impact_skew', 3.0),
            lambda_impact_kurt=interactive_config.get('lambda_impact_kurt', 2.0),
            lambda_impact_variance=interactive_config.get('lambda_impact_variance', 4.0),
            batch_size=interactive_config.get('batch_size', None)
        )

class Attack(ABC):
    """
    Abstract base class for all data poisoning attacks.
    
    This class defines the common interface that all attack implementations
    must follow. Each attack should implement the 'craft' method to generate
    poisoned data.
    """
    
    def __init__(self, config: BaseAttackConfig, df: pd.DataFrame, target_col: str, task_type: str):
        """
        Initialize the attack base class.
        
        Args:
            config: Attack configuration parameters
            df: Input dataset as pandas DataFrame
            target_col: Name of the target column
            task_type: Type of machine learning task ('classification' or 'regression')
        """
        self.config = config
        self.df = df.copy()
        self.target_col = target_col
        self.task_type = task_type
        self.sigmas = self.df.drop(self.target_col, axis=1).std()
        self.poison_indices = None

    @abstractmethod
    def craft(self, model) -> pd.DataFrame:
        """
        Abstract method to be implemented by concrete attack classes.
        
        This method should generate and return a poisoned version of the dataset.
        
        Args:
            model: Trained model (may be used by some attacks)
            
        Returns:
            DataFrame with poisoned data
        """
        pass

    def get_poison_indices(self, poisoned_df: pd.DataFrame) -> pd.Index:
        """
        Identify which data points were actually poisoned.
        
        Compares the original and poisoned datasets to find modified rows.
        
        Args:
            poisoned_df: The poisoned dataset
            
        Returns:
            Index of poisoned data points
        """
        poisoned_features = poisoned_df.drop(self.target_col, axis=1)
        clean_features = self.df.drop(self.target_col, axis=1)
        diff = (poisoned_features != clean_features).any(axis=1)
        return self.df.index[diff]

def create_attack_config_from_interactive(
    attack_name: str, 
    interactive_config: Dict[str, any]
) -> BaseAttackConfig:
    """
    Create appropriate attack configuration from interactive parameters
    
    Args:
        attack_name: Name of the attack
        interactive_config: Dictionary containing user-provided parameters
        
    Returns:
        Configured attack config object
    """
    if attack_name == "dspo":
        return DSPOConfig.from_interactive_config(interactive_config)
    elif attack_name == "random":
        config = BaseAttackConfig()
        config.update_from_interactive_config(interactive_config)
        return config
    elif attack_name == "greedy_residue":
        config = BaseAttackConfig()
        config.update_from_interactive_config(interactive_config)
        return config
    else:
        # Default fallback
        config = BaseAttackConfig()
        config.update_from_interactive_config(interactive_config)
        return config
    
def get_attack(
    name: str, 
    cfg: BaseAttackConfig, 
    df: pd.DataFrame, 
    target_col: str, 
    task_type: str, 
    defenses: Dict[str, base_defense.Defense],
    interactive_config: Optional[Dict[str, any]] = None
) -> Attack:
    """
    Factory function to instantiate attack objects.
    
    This function serves as a factory pattern implementation, creating
    appropriate attack instances based on the attack name.
    
    Args:
        name: Name of the attack to instantiate
        cfg: Configuration for the attack (can be overridden by interactive_config)
        df: Input dataset
        target_col: Target column name
        task_type: ML task type
        defenses: Dictionary of defense mechanisms
        interactive_config: Optional interactive configuration to override cfg
        
    Returns:
        Instance of the requested attack class
        
    Raises:
        ValueError: If an unknown attack name is provided
    """
    
    # If interactive config is provided, create new config from it
    if interactive_config is not None:
        cfg = create_attack_config_from_interactive(name, interactive_config)
    
    if name == "random": 
        return RandomAttack(cfg, df, target_col, task_type)
    elif name == "greedy_residue":
        from datapoisontoolkit.attacks import greedy_residue_mock
        return greedy_residue_mock.GreedyResidueAttack(cfg, df, target_col, task_type)
    elif name == "dspo":
        from datapoisontoolkit.attacks import dspo
        
        # Ensure we have DSPOConfig
        if not isinstance(cfg, DSPOConfig):
            if interactive_config:
                cfg = DSPOConfig.from_interactive_config(interactive_config)
            else:
                # Convert BaseAttackConfig to DSPOConfig with defaults
                cfg = DSPOConfig(
                    budget_pct=cfg.budget_pct,
                    perturb_strength=cfg.perturb_strength,
                    num_iterations=100,
                    learning_rate=0.01,
                    lambda_defense=0.5,
                    lambda_stealth=0.1,
                    lambda_impact=2.0
                )
        
        return dspo.DSPOAttack(cfg, df, target_col, task_type, defenses)
        
    elif name == "old_dspo":
        from datapoisontoolkit.attacks import old_dspo
        
        # Old dspo uses same config structure (mostly) but let's ensure it gets what it needs
        if not isinstance(cfg, DSPOConfig):
            if interactive_config:
                cfg = DSPOConfig.from_interactive_config(interactive_config)
            else:
                 cfg = DSPOConfig(
                    budget_pct=cfg.budget_pct,
                    perturb_strength=cfg.perturb_strength
                )
        
        return old_dspo.DSPOAttack(cfg, df, target_col, task_type, defenses)
    else:
        raise ValueError(f"Unknown attack: {name}")


# Helper function for the demorunner integration
def create_dspo_config_from_demo_runner(attack_config: Dict[str, any]) -> DSPOConfig:
    """
    Helper function to create DSPOConfig from demorunner attack_config
    
    Args:
        attack_config: Dictionary from _configure_attack_parameters in demorunner
        
    Returns:
        Properly configured DSPOConfig object
    """
    return DSPOConfig(
        budget_pct=attack_config['budget_pct'],
        perturb_strength=attack_config['perturb_strength'],
        num_iterations=attack_config['num_iterations'],
        learning_rate=attack_config['learning_rate'],
        lambda_impact=attack_config.get('lambda_impact', 1.0),
        lambda_stealth=attack_config.get('lambda_stealth', 0.3),
        lambda_defense=attack_config.get('lambda_defense', 0.7),
        batch_size=attack_config.get('batch_size', None)
    )
import numpy as np

class RandomAttack(Attack):
    """
    Simple random poisoning attack.
    
    This baseline attack randomly selects data points and applies
    random perturbations to them. It serves as a comparison baseline
    for more sophisticated attacks.
    """
    
    def craft(self, model) -> pd.DataFrame:
        """
        Apply random perturbations to randomly selected data points.
        
        For classification tasks:
        - Flip class labels of selected points
        - Add random noise to feature values
        
        For regression tasks:
        - Add random noise to target values
        """
        budget = int(self.config.budget_pct * len(self.df))
        poisoned_df = self.df.copy()
        
        # Randomly select indices to poison
        self.poison_indices = np.random.choice(self.df.index, budget, replace=False)
        
        if self.task_type == 'classification':
            # Flip class labels
            poisoned_df.loc[self.poison_indices, self.target_col] = 1 - poisoned_df.loc[self.poison_indices, self.target_col]
            
            # Add random noise to features
            numeric_cols = poisoned_df.select_dtypes(include=[np.number]).columns
            numeric_cols = numeric_cols.drop(self.target_col, errors='ignore')
            
            for col in numeric_cols:
                sigma = self.df[col].std()
                deltas = np.random.uniform(
                    -self.config.perturb_strength * sigma,
                    self.config.perturb_strength * sigma,
                    size=len(self.poison_indices)
                )
                poisoned_df.loc[self.poison_indices, col] += deltas
                
        elif self.task_type == 'regression':
            # Add random noise to target values
            perturbations = np.random.uniform(
                -self.config.perturb_strength * self.df[self.target_col].std(),
                self.config.perturb_strength * self.df[self.target_col].std(),
                size=budget
            )
            poisoned_df.loc[self.poison_indices, self.target_col] += perturbations
            
        return poisoned_df