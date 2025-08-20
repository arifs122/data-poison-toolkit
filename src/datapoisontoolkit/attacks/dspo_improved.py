import numpy as np
import pandas as pd
import torch
from typing import Dict
from datapoisontoolkit.attacks import base_attack
from tqdm import tqdm
from datapoisontoolkit.detectors import base_defense
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

class DSPOAttack(base_attack.Attack):
    """
    Memory-efficient version of D-SPO Attack that processes data in batches
    to avoid RAM overflow on large datasets.
    """
    
    def __init__(self, config: base_attack.DSPOConfig, df: pd.DataFrame, target_col: str, task_type: str, defenses: Dict[str, base_defense.Defense]):
        super().__init__(config, df, target_col, task_type)
        self.config = config
        self.df = df.copy()  # DataFrame'in bir kopyasını al
        self.target_col = target_col
        self.task_type = task_type
        
        # DataFrame'in integer index kullandığından emin ol
        if not isinstance(self.df.index, pd.RangeIndex):
            print("DataFrame indexini integer indexe çeviriliyor...")
            self.df = self.df.reset_index(drop=True)
        
        # Device selection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Batch size for processing (configurable)
        self.batch_size = getattr(config, 'batch_size', 5000)  # Default 5k rows per batch
        
        # Prepare differentiable defense proxies
        self.diff_defenses = self._prepare_defense_proxies(defenses)
        
        if not self.diff_defenses:
            raise ValueError("DSPOAttack requires at least one defense with a 'get_differentiable_proxy' method.")
        
        # Calculate dataset statistics
        self.sigma = self.df[target_col].std()
        self.clean_mean = self.df[target_col].mean()
        self.clean_std = self.df[target_col].std()
        
        # Select points to poison (strategic selection)
        self.original_indices_to_poison, self.poison_locations = self._select_poison_indices()
        
        # Create batch information for poisoned points
        self._create_batch_mapping()
    
    def _create_batch_mapping(self):
        """Create mapping of which batch each poisoned point belongs to."""
        self.batch_info = {}
        total_rows = len(self.df)
        
        # original_indices_to_poison artık pandas Index objesi, listeye çevir
        poison_indices_list = self.original_indices_to_poison.tolist()
        
        for i, idx in enumerate(poison_indices_list):
            batch_num = idx // self.batch_size
            local_idx = idx % self.batch_size  # Index within the batch
            
            if batch_num not in self.batch_info:
                self.batch_info[batch_num] = {
                    'global_indices': [],
                    'local_indices': [],
                    'poison_mask': []
                }
            
            self.batch_info[batch_num]['global_indices'].append(idx)
            self.batch_info[batch_num]['local_indices'].append(local_idx)
        
        # Calculate batch ranges
        self.num_batches = (total_rows + self.batch_size - 1) // self.batch_size
        print(f"Processing {total_rows} rows in {self.num_batches} batches of size {self.batch_size}")
    
    def _get_batch_data(self, batch_num: int) -> tuple:
        """Get data for a specific batch."""
        start_idx = batch_num * self.batch_size
        end_idx = min((batch_num + 1) * self.batch_size, len(self.df))
        
        batch_data = self.df.iloc[start_idx:end_idx][self.target_col].values
        batch_tensor = torch.tensor(batch_data, dtype=torch.float32, device=self.device)
        
        # Create poison mask for this batch
        poison_mask = torch.zeros(len(batch_tensor), dtype=torch.bool, device=self.device)
        if batch_num in self.batch_info:
            local_indices = self.batch_info[batch_num]['local_indices']
            poison_mask[local_indices] = True
        
        return batch_tensor, poison_mask, start_idx, end_idx
    
    def _compute_batch_attack_loss(self, deltas: torch.Tensor, poison_mask: torch.Tensor) -> torch.Tensor:
        """Compute stealth loss for current batch."""
        if poison_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)
        return torch.abs(deltas[poison_mask]).mean()
    
    def _compute_batch_defense_loss(self, poisoned_data: torch.Tensor, poison_mask: torch.Tensor) -> torch.Tensor:
        """Compute defense evasion loss for current batch."""
        if poison_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)
        
        total_defense_loss = torch.tensor(0.0, device=self.device)
        
        for defense_name, defense_proxy in self.diff_defenses.items():
            # Only process poisoned points to save memory
            poisoned_points = poisoned_data[poison_mask]
            if len(poisoned_points) > 0:
                try:
                    # Defense proxy için k değerini kontrol et ve ayarla
                    if hasattr(defense_proxy, 'k'):
                        # k değerini poisoned points sayısından küçük olacak şekilde ayarla
                        original_k = defense_proxy.k
                        max_k = len(poisoned_points) - 1
                        if max_k > 0:
                            defense_proxy.k = min(original_k, max_k)
                        else:
                            # Tek bir point varsa, k=1 olarak ayarla
                            defense_proxy.k = 1
                    
                    anomaly_scores = defense_proxy(poisoned_points.unsqueeze(1))
                    defense_loss = anomaly_scores.mean()
                    total_defense_loss += defense_loss
                    
                    # k değerini geri yükle
                    if hasattr(defense_proxy, 'k'):
                        defense_proxy.k = original_k
                        
                except RuntimeError as e:
                    if "kthvalue" in str(e) or "out of range" in str(e):
                        # k-NN hatası durumunda bu defense'i atla
                        print(f"Warning: Skipping {defense_name} due to insufficient points in batch")
                        continue
                    else:
                        raise e
        
        return total_defense_loss
    
    def _compute_batch_impact_loss(self, clean_data: torch.Tensor, poisoned_data: torch.Tensor, poison_mask: torch.Tensor) -> torch.Tensor:
        """Compute model impact loss for current batch."""
        if poison_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)
        
        clean_subset = clean_data[poison_mask]
        poisoned_subset = poisoned_data[poison_mask]
        
        clean_mean = torch.mean(clean_subset)
        poisoned_mean = torch.mean(poisoned_subset)
        
        clean_std = torch.std(clean_subset)
        poisoned_std = torch.std(poisoned_subset)
        
        # Statistical changes
        mean_change = torch.abs(poisoned_mean - clean_mean) / (clean_std + 1e-8)
        std_change = torch.abs(poisoned_std - clean_std) / (clean_std + 1e-8)
        
        # Push toward extreme values
        extreme_target = clean_mean + 10 * clean_std
        extremeness_loss = torch.mean(torch.abs(poisoned_subset - extreme_target))
        
        # Distribution changes
        clean_skew = torch.mean(((clean_subset - clean_mean) / (clean_std + 1e-8))**3)
        poisoned_skew = torch.mean(((poisoned_subset - poisoned_mean) / (poisoned_std + 1e-8))**3)
        skew_change = torch.abs(poisoned_skew - clean_skew)
        
        clean_kurt = torch.mean(((clean_subset - clean_mean) / (clean_std + 1e-8))**4) - 3
        poisoned_kurt = torch.mean(((poisoned_subset - poisoned_mean) / (poisoned_std + 1e-8))**4) - 3
        kurt_change = torch.abs(poisoned_kurt - clean_kurt)
        
        # Variance amplification
        variance_amplification = torch.var(poisoned_subset) / (torch.var(clean_subset) + 1e-8)
        
        # Combine all factors (negate for maximization)
        impact_loss = -(5.0 * mean_change + 4.0 * std_change + 5.0 * extremeness_loss + 
                        3.0 * skew_change + 2.0 * kurt_change + 4.0 * variance_amplification)
        
        return impact_loss
    
    def craft(self, model=None) -> pd.DataFrame:
        """
        Execute memory-efficient D-SPO attack optimization using batch processing.
        """
        print(f"--- Memory-Efficient D-SPO Optimization Starting ({self.device}) ---")
        
        # Initialize global delta storage (only for poisoned points)
        num_poison_points = len(self.original_indices_to_poison)
        global_deltas = torch.zeros(num_poison_points, device=self.device, requires_grad=True)
        max_perturb = self.config.perturb_strength * self.sigma
        
        # Setup optimizer
        optimizer = torch.optim.Adam([global_deltas], lr=self.config.learning_rate, betas=(0.9, 0.999), weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.9)
        
        pbar = tqdm(range(self.config.num_iterations), desc="Memory-Efficient D-SPO")
        
        for iteration in pbar:
            optimizer.zero_grad()
            
            total_attack_loss = 0.0
            total_defense_loss = 0.0  
            total_impact_loss = 0.0
            batch_count = 0
            
            # Process each batch
            for batch_num in range(self.num_batches):
                # Get current batch data
                batch_clean, poison_mask, start_idx, end_idx = self._get_batch_data(batch_num)
                
                # Skip batches with no poisoned points
                if poison_mask.sum() == 0:
                    continue
                
                # Create batch deltas tensor
                batch_deltas = torch.zeros_like(batch_clean, device=self.device)
                
                # Map global deltas to batch deltas
                if batch_num in self.batch_info:
                    local_indices = self.batch_info[batch_num]['local_indices']
                    global_poison_indices = []
                    
                    # Find corresponding global delta indices
                    poison_indices_list = self.original_indices_to_poison.tolist()
                    for local_idx in local_indices:
                        global_idx = start_idx + local_idx
                        poison_position = poison_indices_list.index(global_idx)
                        global_poison_indices.append(poison_position)
                    
                    batch_deltas[local_indices] = global_deltas[global_poison_indices]
                
                # Apply perturbations
                batch_poisoned = batch_clean + batch_deltas
                
                # Compute batch losses
                attack_loss = self._compute_batch_attack_loss(batch_deltas, poison_mask)
                defense_loss = self._compute_batch_defense_loss(batch_poisoned, poison_mask)
                impact_loss = self._compute_batch_impact_loss(batch_clean, batch_poisoned, poison_mask)
                
                # Accumulate losses
                total_attack_loss += attack_loss
                total_defense_loss += defense_loss
                total_impact_loss += impact_loss
                batch_count += 1
                
                # Free memory
                del batch_clean, batch_poisoned, batch_deltas, poison_mask
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Average losses across batches
            if batch_count > 0:
                total_attack_loss /= batch_count
                total_defense_loss /= batch_count
                total_impact_loss /= batch_count
            
            # Compute total loss
            total_loss = (-self.config.lambda_impact * total_impact_loss + 
                         self.config.lambda_stealth * total_attack_loss + 
                         self.config.lambda_defense * total_defense_loss)
            
            # Backpropagation
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Clamp perturbations
            with torch.no_grad():
                global_deltas.data = torch.clamp(global_deltas.data, -max_perturb, max_perturb)
            
            # Update progress
            if iteration % 10 == 0 or iteration == self.config.num_iterations - 1:
                pbar.set_postfix({
                    "Total Loss": f"{total_loss.item():.4f}",
                    "Impact": f"{-total_impact_loss.item():.4f}",
                    "Stealth": f"{total_attack_loss.item():.4f}",
                    "Defense": f"{total_defense_loss.item():.4f}",
                    "Batches": batch_count
                })
        
        # Apply final perturbations to create poisoned dataset
        final_deltas = global_deltas.detach().cpu().numpy()
        poisoned_df = self.df.copy()
        
        # Apply deltas to original indices
        poison_indices_list = self.original_indices_to_poison.tolist()
        for i, idx in enumerate(poison_indices_list):
            poisoned_df.loc[idx, self.target_col] += final_deltas[i]
        
        # Round for classification
        if self.task_type == 'classification':
            poisoned_df[self.target_col] = poisoned_df[self.target_col].round().astype(int)
        
        self.poison_indices = self.original_indices_to_poison
        
        print(f"--- Memory-Efficient D-SPO Optimization Completed ---")
        return poisoned_df
    
    def _prepare_defense_proxies(self, defenses: Dict[str, base_defense.Defense]) -> Dict[str, base_defense.DifferentiableDefense]:
        """Prepare differentiable proxies for defense mechanisms."""
        diff_defenses = {}
        for name, defense in defenses.items():
            if hasattr(defense, 'get_differentiable_proxy'):
                proxy = defense.get_differentiable_proxy()
                if "NoOp" not in proxy.__class__.__name__:
                    diff_defenses[name] = proxy
        return diff_defenses
    
    def _select_poison_indices(self):
        """Strategically select points to poison."""
        # Train baseline model
        if self.task_type == 'classification':
            model = RandomForestClassifier(n_estimators=10, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Calculate selection criteria
        if self.task_type == 'classification':
            misclassified = (y != y_pred)
            residues = misclassified.astype(int)
        else:
            residues = np.abs(y - y_pred)
            
        feature_vars = X.var(axis=1)
        z_scores = np.abs((y - self.clean_mean) / self.clean_std)
        outlier_scores = (z_scores > 2).astype(int)
        
        # Combined scoring - Series olarak oluştur
        residues_series = pd.Series(residues, index=self.df.index)
        feature_vars_series = pd.Series(feature_vars, index=self.df.index)
        outlier_scores_series = pd.Series(outlier_scores, index=self.df.index)
        
        combined_score = residues_series + 0.1 * feature_vars_series + 0.2 * outlier_scores_series
        
        # Select top points
        budget = int(self.config.budget_pct * len(self.df))
        original_indices_to_poison = combined_score.nlargest(budget).index
        poison_locations = self.df.index.get_indexer(original_indices_to_poison)
        
        return original_indices_to_poison, poison_locations