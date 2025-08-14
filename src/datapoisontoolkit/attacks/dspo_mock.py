import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple, List
from dataclasses import dataclass
from datapoisontoolkit.attacks import base_attack
from tqdm.notebook import tqdm
from datapoisontoolkit.detectors import base_defense

@dataclass
class DSPOConfig:
    num_iterations: int = 100
    learning_rate: float = 0.01
    budget_pct: float = 0.03
    perturb_strength: float = 0.5
    lambda_defense: float = 1.0

class DSPOAttack(base_attack.Attack):
    def __init__(self, config: DSPOConfig, df: pd.DataFrame, target_col: str, task_type: str, defenses: Dict[str, base_defense.Defense]):
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