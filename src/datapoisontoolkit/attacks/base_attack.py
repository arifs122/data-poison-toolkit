import pandas as pd

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datapoisontoolkit.detectors import base_defense
from typing import Dict

@dataclass
class BaseAttackConfig:
    """Base configuration class for all attack types."""
    budget_pct: float = 0.02  # Percentage of data to poison
    perturb_strength: float = 0.2 # Maximum strength of perturbations (in terms of standard deviation)

@dataclass
class DSPOConfig(BaseAttackConfig):
    """Configuration specifically for D-SPO (Differentiable Shielded Poisoning Oracle) attack."""
    num_iterations: int = 200           # Increased number of optimization iterations
    learning_rate: float = 0.05          # Much higher learning rate for faster convergence
    lambda_defense: float = 0.1          # Lower weight for defense evasion (less emphasis)
    lambda_stealth: float = 0.05         # Very low weight for stealth loss (minimal emphasis)
    lambda_impact: float = 5.0           # HIGH weight for impact loss (maximum emphasis)

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
    
def get_attack(name: str, cfg: BaseAttackConfig, df: pd.DataFrame, target_col: str, task_type: str, defenses: Dict[str, base_defense.Defense]) -> Attack:
    """
    Factory function to instantiate attack objects.
    
    This function serves as a factory pattern implementation, creating
    appropriate attack instances based on the attack name.
    
    Args:
        name: Name of the attack to instantiate
        cfg: Configuration for the attack
        df: Input dataset
        target_col: Target column name
        task_type: ML task type
        defenses: Dictionary of defense mechanisms
        
    Returns:
        Instance of the requested attack class
        
    Raises:
        ValueError: If an unknown attack name is provided
    """
    if name == "random": 
        return RandomAttack(cfg, df, target_col, task_type)
    elif name == "greedy_residue":
        from datapoisontoolkit.attacks import greedy_residue_mock
        return greedy_residue_mock.GreedyResidueAttack(cfg, df, target_col, task_type)
    elif name == "dspo":
        from datapoisontoolkit.attacks import dspo_improved
        # Convert to DSPOConfig if needed
        if not isinstance(cfg, DSPOConfig):
            cfg = DSPOConfig(
                budget_pct=cfg.budget_pct,
                perturb_strength=cfg.perturb_strength,
                num_iterations=100,
                learning_rate=0.01,
                lambda_defense=0.5,
                lambda_stealth=0.1,
                lambda_impact=2.0
            )
        return dspo_improved.DSPOAttack(cfg, df, target_col, task_type, defenses)
    else:
        raise ValueError(f"Unknown attack: {name}")

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