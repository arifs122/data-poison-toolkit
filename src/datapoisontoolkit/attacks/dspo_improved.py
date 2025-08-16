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
    D-SPO (Differentiable Shielded Poisoning Oracle) Attack Implementation.
    
    This class implements a sophisticated data poisoning attack that aims to:
    1. Maximize the degradation of model performance (high positive impact score)
    2. Minimize detection by defensive mechanisms (low recall)
    3. Maintain stealth to avoid suspicion (low perturbation magnitude)
    
    The attack works by optimizing perturbations using gradient-based methods,
    taking into account both the impact on the target model and the responses
    of various defense mechanisms.
    """
    
    def __init__(self, config: base_attack.DSPOConfig, df: pd.DataFrame, target_col: str, task_type: str, defenses: Dict[str, base_defense.Defense]):
        """
        Initialize the D-SPO attack.
        
        Args:
            config: Configuration parameters for the attack
            df: Input dataset as a pandas DataFrame
            target_col: Name of the target column to be poisoned
            task_type: Type of ML task ('classification' or 'regression')
            defenses: Dictionary of defense mechanisms to evade
        """
        # Call parent constructor
        super().__init__(config, df, target_col, task_type)
        self.config = config
        self.df = df
        self.target_col = target_col
        self.task_type = task_type
        
        # Device selection (use GPU if available)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Prepare differentiable defense proxies
        self.diff_defenses = self._prepare_defense_proxies(defenses)
        
        if not self.diff_defenses:
            raise ValueError("DSPOAttack requires at least one defense with a 'get_differentiable_proxy' method.")
        
        # Calculate dataset statistics
        self.sigma = df[target_col].std()
        
        # Clean data statistics for reference
        self.clean_mean = df[target_col].mean()
        self.clean_std = df[target_col].std()
        
        # Select points to poison (strategic selection)
        self.original_indices_to_poison, self.poison_locations = self._select_poison_indices()
        
    def _prepare_defense_proxies(self, defenses: Dict[str, base_defense.Defense]) -> Dict[str, base_defense.DifferentiableDefense]:
        """
        Prepare differentiable proxies for defense mechanisms.
        
        Only defenses that have a differentiable proxy and are not NoOp proxies
        are selected for the attack optimization.
        
        Args:
            defenses: Dictionary of defense mechanisms
            
        Returns:
            Dictionary of differentiable defense proxies
        """
        diff_defenses = {}
        for name, defense in defenses.items():
            if hasattr(defense, 'get_differentiable_proxy'):
                proxy = defense.get_differentiable_proxy()
                # Exclude NoOp proxies which don't contribute to the optimization
                if "NoOp" not in proxy.__class__.__name__:
                    diff_defenses[name] = proxy
        return diff_defenses
    
    def _select_poison_indices(self):
        """
        Strategically select points to poison.
        
        This method identifies the most impactful points to target for poisoning:
        1. Points where the current model performs poorly (high residuals)
        2. Points with high variance (more sensitive to perturbations)
        3. Outlier points (more likely to affect model behavior)
        
        Returns:
            Tuple of (original_indices, poison_locations)
        """
        # Train a baseline model to identify poorly predicted points
        if self.task_type == 'classification':
            model = RandomForestClassifier(n_estimators=10, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Calculate residuals (model prediction errors)
        if self.task_type == 'classification':
            # For classification, misclassified points = 1, correctly classified = 0
            misclassified = (y != y_pred)
            residues = misclassified.astype(int)
        else:
            # For regression, use absolute prediction errors
            residues = np.abs(y - y_pred)
            
        # Feature variance for sensitivity analysis
        feature_vars = X.var(axis=1)
        
        # Outlier detection (points more than 2 standard deviations from mean)
        z_scores = np.abs((y - self.clean_mean) / self.clean_std)
        outlier_scores = (z_scores > 2).astype(int)
        
        # Combined scoring function prioritizing high-impact targets
        combined_score = residues + 0.1 * feature_vars + 0.2 * outlier_scores
        
        # Select top scoring points within the budget constraint
        budget = int(self.config.budget_pct * len(self.df))
        original_indices_to_poison = combined_score.nlargest(budget).index
        poison_locations = self.df.index.get_indexer(original_indices_to_poison)
        
        return original_indices_to_poison, poison_locations
    
    def _compute_attack_loss(self, deltas: torch.Tensor) -> torch.Tensor:
        """
        Compute stealth loss (magnitude of perturbations).
        
        This loss term encourages small perturbations to maintain stealth.
        Lower values indicate better stealth (harder to detect).
        
        Args:
            deltas: Perturbation tensor
            
        Returns:
            Stealth loss value (should be minimized)
        """
        return torch.abs(deltas[self.poison_locations]).mean()
    
    def _compute_defense_loss(self, poisoned_data: torch.Tensor) -> torch.Tensor:
        """
        Compute defense evasion loss.
        
        This loss term measures how well the attack evades detection by
        various defense mechanisms. Lower values indicate better evasion.
        
        Args:
            poisoned_data: Tensor of poisoned data
            
        Returns:
            Defense loss value (should be minimized)
        """
        total_defense_loss = torch.tensor(0.0, device=self.device)
        
        # Aggregate anomaly scores from all defense proxies
        for defense_name, defense_proxy in self.diff_defenses.items():
            anomaly_scores = defense_proxy(poisoned_data.unsqueeze(1))
            defense_loss = anomaly_scores[self.poison_locations].mean()
            total_defense_loss += defense_loss
            
        return total_defense_loss
    
    def _compute_impact_loss(self, clean_data: torch.Tensor, poisoned_data: torch.Tensor) -> torch.Tensor:
        """
        Compute model impact loss.
        
        This loss measures how much the attack degrades model performance.
        POSITIVE VALUES INDICATE SUCCESSFUL ATTACK (this is what we want to MAXIMIZE).
        
        We want poisoned points to significantly affect the model training.
        """
        # Statistics of the poisoned subset
        clean_subset = clean_data[self.poison_locations]
        poisoned_subset = poisoned_data[self.poison_locations]
        
        clean_mean = torch.mean(clean_subset)
        poisoned_mean = torch.mean(poisoned_subset)
        
        clean_std = torch.std(clean_subset)
        poisoned_std = torch.std(poisoned_subset)
        
        # Statistical changes - we want LARGE changes to disrupt the model
        mean_change = torch.abs(poisoned_mean - clean_mean) / (clean_std + 1e-8)
        std_change = torch.abs(poisoned_std - clean_std) / (clean_std + 1e-8)
        
        # Push poisoned points toward EXTREME values (much more disruptive)
        # Target: 10 standard deviations above the clean mean (very extreme!)
        extreme_target = clean_mean + 10 * clean_std
        extremeness_loss = torch.mean(torch.abs(poisoned_subset - extreme_target))
        
        # Distribution shape disruption (skewness and kurtosis changes)
        clean_skew = torch.mean(((clean_subset - clean_mean) / (clean_std + 1e-8))**3)
        poisoned_skew = torch.mean(((poisoned_subset - poisoned_mean) / (poisoned_std + 1e-8))**3)
        skew_change = torch.abs(poisoned_skew - clean_skew)
        
        clean_kurt = torch.mean(((clean_subset - clean_mean) / (clean_std + 1e-8))**4) - 3
        poisoned_kurt = torch.mean(((poisoned_subset - poisoned_mean) / (poisoned_std + 1e-8))**4) - 3
        kurt_change = torch.abs(poisoned_kurt - clean_kurt)
        
        # Variance amplification - make poisoned points MUCH more variable
        variance_amplification = torch.var(poisoned_subset) / (torch.var(clean_subset) + 1e-8)
        
        # COMBINE ALL FACTORS - Note: We NEGATE this because we want to MAXIMIZE impact
        # But the optimizer MINIMIZES the total loss
        # Much higher weights for impact factors
        impact_loss = -(5.0 * mean_change + 4.0 * std_change + 5.0 * extremeness_loss + 
                        3.0 * skew_change + 2.0 * kurt_change + 4.0 * variance_amplification)
        
        return impact_loss
    
    def _compute_total_loss(self, attack_loss: torch.Tensor, defense_loss: torch.Tensor, impact_loss: torch.Tensor) -> torch.Tensor:
        """
        Compute the total combined loss function.
        
        The total loss combines three objectives:
        1. Maximize impact (negative impact_loss because we want to maximize it)
        2. Minimize stealth loss (lower = more stealthy)
        3. Minimize defense loss (lower = better evasion)
        
        Args:
            attack_loss: Stealth loss component
            defense_loss: Defense evasion loss component  
            impact_loss: Model impact loss component
            
        Returns:
            Total loss value to be minimized by the optimizer
        """
        # Note: We negate impact_loss because we want to MAXIMIZE impact
        # But the optimizer MINIMIZES the total loss
        return (-self.config.lambda_impact * impact_loss + 
                self.config.lambda_stealth * attack_loss + 
                self.config.lambda_defense * defense_loss)
    
    def craft(self, model=None) -> pd.DataFrame:
        """
        Execute the D-SPO attack optimization.
        
        This method performs gradient-based optimization to find optimal
        perturbations that maximize attack impact while maintaining stealth
        and evading defenses.
        
        Args:
            model: Optional pre-trained model (not used in current implementation)
            
        Returns:
            DataFrame with poisoned data
        """
        print(f"--- D-SPO Optimization Starting ({self.device}) ---")
        
        # Convert clean data to tensor
        clean_data = torch.tensor(self.df[self.target_col].values, dtype=torch.float32, device=self.device)
        
        # Initialize perturbations as learnable parameters
        deltas = torch.zeros(len(self.df), device=self.device, requires_grad=True)
        max_perturb = self.config.perturb_strength * self.sigma
        
        # Setup optimizer with momentum and weight decay for better convergence
        # Use higher learning rate and momentum for aggressive optimization
        optimizer = torch.optim.Adam([deltas], lr=self.config.learning_rate, betas=(0.9, 0.999), weight_decay=1e-5)
        
        # More aggressive learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.9)
        
        # Progress tracking
        pbar = tqdm(range(self.config.num_iterations), desc="D-SPO Optimization")
        
        # Main optimization loop
        for iteration in pbar:
            # Reset gradients
            optimizer.zero_grad()
            
            # Apply current perturbations
            poisoned_data = clean_data + deltas
            
            # Calculate individual loss components
            attack_loss = self._compute_attack_loss(deltas)
            defense_loss = self._compute_defense_loss(poisoned_data)
            impact_loss = self._compute_impact_loss(clean_data, poisoned_data)
            
            # Compute total loss
            total_loss = self._compute_total_loss(attack_loss, defense_loss, impact_loss)
            
            # Backpropagation
            total_loss.backward()
            optimizer.step()
            
            # Update learning rate
            scheduler.step()
            
            # Clamp perturbations to stay within bounds
            with torch.no_grad():
                deltas.data = torch.clamp(deltas.data, -max_perturb, max_perturb)
            
            # Update progress bar every 10 iterations
            if iteration % 10 == 0 or iteration == self.config.num_iterations - 1:
                pbar.set_postfix({
                    "Total Loss": f"{total_loss.item():.4f}",
                    "Impact": f"{-impact_loss.item():.4f}",  # Negate for intuitive display
                    "Stealth": f"{attack_loss.item():.4f}",
                    "Defense": f"{defense_loss.item():.4f}",
                    "LR": f"{scheduler.get_last_lr()[0]:.6f}"
                })
        
        # Apply final perturbations to create poisoned dataset
        final_deltas = deltas.detach().cpu().numpy()
        poisoned_df = self.df.copy()
        poisoned_df.loc[self.original_indices_to_poison, self.target_col] += final_deltas[self.poison_locations]
        
        # Round to integers for classification tasks
        if self.task_type == 'classification':
            poisoned_df[self.target_col] = poisoned_df[self.target_col].round().astype(int)
        
        # Set poison indices for compatibility
        self.poison_indices = self.original_indices_to_poison
        
        print(f"--- D-SPO Optimization Completed ---")
        return poisoned_df