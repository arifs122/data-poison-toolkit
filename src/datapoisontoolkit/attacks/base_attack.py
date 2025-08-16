import pandas as pd

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datapoisontoolkit.detectors import base_defense
from typing import Dict

@dataclass
class BaseAttackConfig:
	budget_pct: float = 0.03  # Zehirlenecek veri yüzdesi
	perturb_strength: float = 0.5 # Pertürbasyonun maksimum gücü (std dev cinsinden)

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
    
def get_attack(name: str, cfg: BaseAttackConfig, df: pd.DataFrame, target_col: str, task_type: str, defenses: Dict[str, base_defense.Defense]) -> Attack:
    if name == "random": 
        return RandomAttack(cfg, df, target_col, task_type)
    elif name == "greedy_residue":
        from datapoisontoolkit.attacks import greedy_residue_mock
        return greedy_residue_mock.GreedyResidueAttack(cfg, df, target_col, task_type)
    elif name == "dspo":
        from datapoisontoolkit.attacks import dspo_mock
        return dspo_mock.DSPOAttack(cfg, df, target_col, task_type, defenses)
    else:
        raise ValueError(f"Unknown attack: {name}")

import numpy as np

class RandomAttack(Attack):
    def craft(self, model) -> pd.DataFrame:
        budget = int(self.config.budget_pct * len(self.df))
        poisoned_df = self.df.copy()
        
        # Rastgele indeksler seç
        self.poison_indices = np.random.choice(self.df.index, budget, replace=False)
        
        if self.task_type == 'classification':
            # Sınıf etiketlerini değiştir
            poisoned_df.loc[self.poison_indices, self.target_col] = 1 - poisoned_df.loc[self.poison_indices, self.target_col]
            
            # Özelliklere rastgele gürültü ekle
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
            # Hedef değişkene rastgele gürültü ekle
            perturbations = np.random.uniform(
                -self.config.perturb_strength * self.df[self.target_col].std(),
                self.config.perturb_strength * self.df[self.target_col].std(),
                size=budget
            )
            poisoned_df.loc[self.poison_indices, self.target_col] += perturbations
            
        return poisoned_df