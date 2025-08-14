import numpy as np
import pandas as pd
import torch
from abc import ABC, abstractmethod


class DifferentiableDefense(ABC):
    """D-SPO saldırısının kullanacağı, türevlenebilir savunma proxy'leri için ana sınıf."""
    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass

class Defense(ABC):
	@abstractmethod
	def detect(self, series: pd.Series) -> pd.Series:
		pass