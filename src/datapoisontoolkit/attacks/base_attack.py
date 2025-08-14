import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datapoisontoolkit.attacks import dspo_mock

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
    
def get_attack(name: str, cfg: BaseAttackConfig, df: pd.DataFrame, target_col: str, task_type: str, defenses: Dict[str, Defense]) -> Attack:
    if name == "random": return RandomAttack(cfg, df, target_col, task_type)
    if name == "greedy_residue": return GreedyResidueAttack(cfg, df, target_col, task_type)
    if name == "dspo" : return dspo_mock.DSPOAttack(cfg, df, target_col, task_type, defenses)
    raise ValueError(f"Unknown attack: {name}")