import numpy as np
import pandas as pd
from dataclasses import dataclass
from datapoisontoolkit.attacks import base_attack

class GreedyResidueAttack(base_attack.Attack):
    """
    Greedy attack that targets points with highest residues/errors.
    """
    
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
            
            # SADECE SAYISAL SÜTUNLARI AL
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