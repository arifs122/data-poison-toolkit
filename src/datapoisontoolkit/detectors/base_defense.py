import numpy as np
import pandas as pd
import torch
from abc import ABC, abstractmethod


class DifferentiableDefense(ABC):
    """Main class for differentiable defense proxies to be used by the D-SPO attack."""
    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass

class Defense(ABC):
	@abstractmethod
	def detect(self, series: pd.Series) -> pd.Series:
		pass